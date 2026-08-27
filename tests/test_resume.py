"""Guards on the news backfill's crash-resume behaviour.

The collector runs for the better part of an hour against an unauthenticated API over a
connection that drops. The property that matters is not throughput but this: whatever has
been fetched is on disk, and re-running the identical command continues instead of
restarting. Every test here kills a run partway through and checks what survived.

No network. The BSE endpoint is replaced by a fake that can be told to fail on demand.
"""
import os
import sys

import pandas as pd
import pytest
import requests

sys.path.insert(0, ".")
import src.data.news_sources as NS

SYMS = ["AAA.NS", "BBB.NS", "CCC.NS"]


class FakeAPI:
    """Two pages of announcements per scrip, and an optional list of scrips that explode."""

    def __init__(self, fail_on=()):
        self.fail_on, self.calls = set(fail_on), []

    def __call__(self, url, params, timeout=60, retries=NS.RETRIES, log=None):
        scrip = params.get("strScrip")
        self.calls.append(scrip)
        if scrip in self.fail_on:
            raise requests.RequestException("simulated drop")
        page = params["pageno"]
        if page > 1:
            return _Resp({"Table": []})
        return _Resp({"Table": [
            {"DT_TM": f"2024-01-0{page}T10:00:00", "NEWSSUB": f"{scrip} headline",
             "MORE": "body", "ATTACHMENTNAME": "", "ANNOUNCEMENT_TYPE": "C"}]})


class _Resp:
    def __init__(self, payload):
        self.payload, self.status_code = payload, 200

    def json(self):
        return self.payload


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """collect_bse pointed at a temp news.csv, with the network stubbed out."""
    api = FakeAPI()
    monkeypatch.setattr(NS, "_get", api)
    monkeypatch.setattr(NS, "scrip_map", lambda *a, **k: {s.replace(".NS", ""): s[0] * 3
                                                          for s in SYMS})
    return api, str(tmp_path / "news.csv")


def test_each_symbol_is_on_disk_before_the_next_one_starts(wired):
    """The failure this whole mechanism exists for: 40 minutes of fetching held in RAM."""
    api, csv = wired
    seen = []

    def spy(url, params, **kw):
        # At the moment symbol k is requested, symbols 0..k-1 must already be readable.
        partial, done = NS.partial_paths(csv)
        seen.append(len(NS._done_symbols(done)))
        return FakeAPI.__call__(api, url, params, **kw)

    orig = NS._get
    NS._get = spy
    try:
        NS.collect_bse(SYMS, "2024-01-01", "2024-12-31", news_csv=csv)
    finally:
        NS._get = orig
    # first request for each symbol sees one more completed symbol than the last
    assert seen[0] == 0 and max(seen) == len(SYMS) - 1


def test_resume_skips_what_is_already_done(wired):
    api, csv = wired
    NS.collect_bse(SYMS, "2024-01-01", "2024-12-31", news_csv=csv)
    n_first = len(api.calls)
    api.calls.clear()

    again = NS.collect_bse(SYMS, "2024-01-01", "2024-12-31", news_csv=csv)
    assert api.calls == [], "a completed backfill must not re-fetch anything"
    assert len(again) == len(SYMS), "resume must still return the full corpus"
    assert n_first > 0


def test_an_interrupted_run_keeps_what_it_had(tmp_path, monkeypatch):
    """Symbol 2 fails. Symbol 1 must survive, and symbol 2 must be retried on resume."""
    csv = str(tmp_path / "news.csv")
    monkeypatch.setattr(NS, "scrip_map", lambda *a, **k: {s.replace(".NS", ""): s[0] * 3
                                                          for s in SYMS})
    monkeypatch.setattr(NS, "_get", FakeAPI(fail_on={"BBB"}))
    first = NS.collect_bse(SYMS, "2024-01-01", "2024-12-31", news_csv=csv)
    assert set(first["symbol"]) == {"AAA.NS", "CCC.NS"}

    # network comes back
    api = FakeAPI()
    monkeypatch.setattr(NS, "_get", api)
    second = NS.collect_bse(SYMS, "2024-01-01", "2024-12-31", news_csv=csv)
    assert api.calls and all(c == "BBB" for c in api.calls), "only the failed symbol retries"
    assert set(second["symbol"]) == set(SYMS)


def test_a_symbol_written_twice_does_not_duplicate_rows(wired):
    """A crash between the row-append and the done-marker re-fetches the symbol. That is the
    safe direction only because append_csv dedupes; this pins that assumption down."""
    api, csv = wired
    NS.collect_bse(SYMS, "2024-01-01", "2024-12-31", news_csv=csv)
    partial, done = NS.partial_paths(csv)
    with open(done) as f:
        kept = f.read().splitlines()[:-1]          # forget the last symbol was finished
    open(done, "w").write("\n".join(kept) + "\n")

    df = NS.collect_bse(SYMS, "2024-01-01", "2024-12-31", news_csv=csv)
    n = NS.append_csv(df, csv)
    assert n == len(SYMS), f"expected one row per symbol after dedupe, got {n}"


def test_a_missing_scrip_code_is_never_retried(tmp_path, monkeypatch):
    """Unlike a dropped connection, this never resolves itself - retrying it every resume
    would mean the collector re-walks the same dead symbol forever."""
    csv = str(tmp_path / "news.csv")
    monkeypatch.setattr(NS, "scrip_map", lambda *a, **k: {"AAA": "AAA"})   # BBB, CCC unknown
    api = FakeAPI()
    monkeypatch.setattr(NS, "_get", api)
    NS.collect_bse(SYMS, "2024-01-01", "2024-12-31", news_csv=csv)
    _, done = NS.partial_paths(csv)
    assert set(NS._done_symbols(done)) == set(SYMS)


def test_transient_failures_are_retried_before_giving_up(monkeypatch):
    """_get must not surface a blip. Three failures then a success is one successful call."""
    calls = []

    def flaky(url, headers=None, timeout=None, params=None):
        calls.append(1)
        if len(calls) < 3:
            raise requests.RequestException("blip")
        return _Resp({"Table": []})

    monkeypatch.setattr(NS.requests, "get", flaky)
    monkeypatch.setattr(NS.time, "sleep", lambda s: None)      # no real backoff in tests
    r = NS._get("u", {})
    assert r.status_code == 200 and len(calls) == 3


def test_read_partial_returns_tz_aware_timestamps(wired):
    """The summariser buckets by IST hour. A column of naive or mixed-offset objects would
    silently mis-bucket every post-close filing."""
    api, csv = wired
    NS.collect_bse(SYMS, "2024-01-01", "2024-12-31", news_csv=csv)
    ts = NS.read_partial(csv)["timestamp"]
    assert isinstance(ts.dtype, pd.DatetimeTZDtype)
    assert str(ts.dt.tz) == "Asia/Kolkata"


def test_scrip_map_is_cached_so_a_resume_needs_no_master_download(tmp_path, monkeypatch):
    cache = str(tmp_path / "scrip.csv")
    monkeypatch.setattr(NS, "_get", FakeAPI())

    def master(url, params, timeout=60, retries=4, log=None):
        return _Resp([{"scrip_id": "AAA", "SCRIP_CD": "500001"}])

    monkeypatch.setattr(NS, "_get", master)
    assert NS.scrip_map(cache=cache) == {"AAA": "500001"}

    def boom(*a, **k):
        raise AssertionError("scrip master re-downloaded despite a warm cache")

    monkeypatch.setattr(NS, "_get", boom)
    assert NS.scrip_map(cache=cache) == {"AAA": "500001"}


def test_append_csv_is_idempotent_across_a_reload(wired):
    """Regression. Rows already on disk read back as strings while incoming rows are tz-aware
    Timestamps; deduping across that boundary matched nothing and doubled news.csv on every
    re-run. Caught live, not by the fakes - the fakes never wrote the file twice."""
    api, csv = wired
    df = NS.collect_bse(SYMS, "2024-01-01", "2024-12-31", news_csv=csv)
    first = NS.append_csv(df, csv)
    again = NS.append_csv(NS.read_partial(csv), csv)
    assert again == first == len(SYMS)


def test_boilerplate_filter_drops_paperwork_and_keeps_filings():
    """The filter earns its place only if it is blunt in one direction: administrative
    notices out, anything that could move a price left alone."""
    df = pd.DataFrame({"headline": [
        "Loss of Share Certificate",
        "Closure of Trading Window",
        "Analyst / Investor Meet Intimation",
        "Board approves acquisition of XYZ Ltd for Rs 4,200 crore",
        "Unaudited Financial Results for the quarter ended June 30",
    ]})
    kept = NS.drop_boilerplate(df, log=None)["headline"].tolist()
    assert kept == df["headline"].tolist()[3:]
