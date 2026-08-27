"""Indian news collection for the hierarchical summarisation stage.

Two complementary sources, because no Indian equivalent of the CMIN news dump exists:

  * BSE corporate announcements - the historical backbone. Exchange-official, filed with
    an exact IST timestamp, and queryable by date range back to ~2018. This is what makes
    the paper's intraday-bucket -> daily summarisation possible at all: without real
    timestamps there is nothing to bucket.
  * RSS feeds (Economic Times, Business Standard, ...) - forward collection only.
    Back-archives are thin, so this accumulates coverage from the day you start.

Both normalise to the news.csv schema: symbol, timestamp, headline, body.
"""
import io
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Referer": "https://www.bseindia.com/"}
SCRIP_MASTER = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
ANNOUNCEMENTS = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
ATTACH_BASE = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/"
SCRIP_CACHE = "data/raw/bse_scrip_map.csv"

# NSE ticker != BSE scrip_id for a handful of names; patch them explicitly.
# TATAMOTORS -> TMCV reflects the 2025 commercial/passenger-vehicle demerger: the NSE
# ticker's pre-2025 history and the post-demerger BSE filings are not the same entity,
# so treat that symbol's 2018-2025 series as a known discontinuity.
ALIASES = {"BAJAJ-AUTO": "BAJAJAUTO", "TATAMOTORS": "TMCV"}


# ------------------------------------------------------- BSE (historical backfill)
#
# Everything below assumes the connection will drop, because on this link it does. Nothing
# is held in memory that we would mind losing: each symbol is fsync'd to a partial CSV the
# moment it is fetched, and a sidecar file records which symbols are finished. Re-running
# the collector after a drop resumes at the first unfinished symbol instead of starting over.
#
# The previous version accumulated all 50 symbols in a list and wrote once at the end, so
# forty minutes of fetching could sit in RAM with nothing on disk behind it.

RETRIES = 4


def _get(url, params, timeout=60, retries=RETRIES, log=None):
    """One HTTP GET, retried with exponential backoff. A flaky link is the normal case here,
    not the exception, so a transient failure must not be allowed to cost a whole symbol."""
    last = "?"
    for a in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout, params=params)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = type(e).__name__
        if a < retries - 1:
            wait = 3 * 2 ** a                    # 3s, 6s, 12s
            if log:
                log(f"      {last}; retry {a + 1}/{retries - 1} in {wait}s")
            time.sleep(wait)
    raise requests.RequestException(f"{retries} attempts failed, last: {last}")


def scrip_map(timeout=60, cache=SCRIP_CACHE) -> dict:
    """NSE-style symbol -> BSE numeric scrip code, from the live BSE scrip master.

    Cached to disk after the first success. The master is a multi-megabyte download that
    changes rarely, and re-fetching it at the top of every resume would mean a bad connection
    could fail the run before it even reaches the symbol it was meant to continue from.
    """
    if cache and os.path.exists(cache):
        c = pd.read_csv(cache, dtype=str)
        return dict(zip(c["scrip_id"], c["scrip_cd"]))
    r = _get(SCRIP_MASTER, {"Group": "", "Scripcode": "", "industry": "",
                            "segment": "Equity", "status": "Active"}, timeout)
    m = {str(row["scrip_id"]).strip().upper(): str(row["SCRIP_CD"])
         for row in r.json() if row.get("scrip_id")}
    if cache and m:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        pd.DataFrame({"scrip_id": list(m), "scrip_cd": list(m.values())}).to_csv(cache, index=False)
    return m


def fetch_bse(scrip_cd: str, start: str, end: str, pause=0.4, max_pages=40,
              log=None) -> pd.DataFrame:
    """All announcements for one scrip in [start, end]. Dates as 'YYYYMMDD'."""
    rows, page = [], 1
    while page <= max_pages:
        r = _get(ANNOUNCEMENTS, {"pageno": page, "strCat": "-1", "strPrevDate": start,
                                 "strScrip": scrip_cd, "strSearch": "P", "strToDate": end,
                                 "strType": "C", "subcategory": ""}, log=log)
        table = r.json().get("Table") or []
        if not table:
            break
        rows += table
        page += 1
        time.sleep(pause)                       # be polite; the API is unauthenticated
    if not rows:
        return pd.DataFrame(columns=["timestamp", "headline", "body", "attachment", "category"])
    df = pd.DataFrame(rows)
    # DT_TM is the filing time; NEWS_DT the announcement date. Prefer the former.
    ts = pd.to_datetime(df.get("DT_TM", df.get("NEWS_DT")), errors="coerce")
    out = pd.DataFrame({
        "timestamp": ts.dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT"),
        "headline": df.get("NEWSSUB", "").astype(str).str.strip(),
        "body": df.get("MORE", "").fillna("").astype(str).str.strip(),
        "attachment": df.get("ATTACHMENTNAME", "").fillna("").astype(str),
        "category": df.get("ANNOUNCEMENT_TYPE", "").fillna("").astype(str)})
    return out.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp", "headline"])


def partial_paths(news_csv: str):
    """The two checkpoint files that sit beside news.csv: the rows, and the done-marker."""
    base = os.path.splitext(news_csv)[0]
    return base + ".bse.partial.csv", base + ".bse.done.txt"


def _done_symbols(done_path: str) -> set:
    if not os.path.exists(done_path):
        return set()
    with open(done_path, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def _checkpoint(df: pd.DataFrame, sym: str, partial: str, done_path: str):
    """Append one symbol's rows and mark it finished - both fsync'd before returning.

    Order matters. Rows are written first, the done-marker second. A crash between the two
    causes the symbol to be re-fetched on resume, and the dedupe in append_csv absorbs the
    repeat. The reverse order would silently lose the symbol forever.
    """
    os.makedirs(os.path.dirname(partial) or ".", exist_ok=True)
    if len(df):
        with open(partial, "a", newline="", encoding="utf-8") as f:
            df.to_csv(f, index=False, header=f.tell() == 0)
            f.flush()
            os.fsync(f.fileno())
    with open(done_path, "a", encoding="utf-8") as f:
        f.write(sym + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_partial(news_csv: str) -> pd.DataFrame:
    """Everything checkpointed so far, whether or not the run that wrote it ever finished."""
    partial, _ = partial_paths(news_csv)
    cols = ["symbol", "timestamp", "headline", "body"]
    if not os.path.exists(partial):
        return pd.DataFrame(columns=cols)
    out = pd.read_csv(partial)
    # Written with a +05:30 offset; parse through UTC so the column comes back as a single
    # tz-aware dtype rather than a column of mixed-offset objects.
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce"
                                      ).dt.tz_convert("Asia/Kolkata")
    # A symbol re-fetched after a crash between the row-write and the done-marker appends a
    # second copy of its rows. append_csv would collapse them on the way out, but the partial
    # file itself would keep growing across resumes - so dedupe on the way in too.
    return (out.dropna(subset=["timestamp"])
               .drop_duplicates(subset=["symbol", "timestamp", "headline"])
               .sort_values(["symbol", "timestamp"]))


def collect_bse(symbols, start: str, end: str, suffix=".NS", log=print,
                news_csv="data/raw/news.csv", resume=True) -> pd.DataFrame:
    """Historical backfill across a universe. start/end as 'YYYY-MM-DD'.

    Returns everything on disk plus whatever this call adds, so an interruption costs at most
    the one symbol that was in flight. Kill it and re-run the identical command to continue.
    """
    partial, done_path = partial_paths(news_csv)
    done = _done_symbols(done_path) if resume else set()
    if done:
        log(f"resuming: {len(done)} symbols already checkpointed, "
            f"{max(0, len(symbols) - len(done))} left")
    smap = scrip_map()
    s0, s1 = start.replace("-", ""), end.replace("-", "")
    got = 0
    for i, sym in enumerate(symbols, 1):
        base = sym.replace(suffix, "").replace(".BO", "").upper()
        full = base + suffix
        if full in done:
            continue
        code = smap.get(ALIASES.get(base, base)) or smap.get(base)
        if not code:
            log(f"  [{i}/{len(symbols)}] {base}: no BSE scrip code, skipped")
            _checkpoint(pd.DataFrame(), full, partial, done_path)   # permanent, never retry
            continue
        try:
            df = fetch_bse(code, s0, s1, log=log)
        except Exception as e:
            # Deliberately NOT marked done: a network failure should be retried by the next
            # run, unlike a missing scrip code, which never resolves itself.
            log(f"  [{i}/{len(symbols)}] {base}: {type(e).__name__}, left for the next resume")
            continue
        df["symbol"] = full
        _checkpoint(df, full, partial, done_path)
        got += len(df)
        log(f"  [{i}/{len(symbols)}] {base} ({code}): {len(df)} announcements, saved")
    log(f"this run added {got:,} announcements")
    return read_partial(news_csv)


# ---------------------------------------------------------------- RSS (forward collection)

FEEDS = {
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "bs_markets": "https://www.business-standard.com/rss/markets-106.rss",
    "mc_business": "https://www.moneycontrol.com/rss/business.xml",
}


def fetch_rss(url: str, timeout=30) -> pd.DataFrame:
    """Parse an RSS feed into timestamped items. stdlib only - no feedparser dependency."""
    import xml.etree.ElementTree as ET
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    rows = []
    for item in root.iter("item"):
        get = lambda tag: (item.findtext(tag) or "").strip()
        ts = pd.to_datetime(get("pubDate"), errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        rows.append({"timestamp": ts.tz_convert("Asia/Kolkata"),
                     "headline": re.sub(r"<[^>]+>", "", get("title")),
                     "body": re.sub(r"<[^>]+>", "", get("description")),
                     "link": get("link")})
    return pd.DataFrame(rows)


def tag_symbols(df: pd.DataFrame, names: dict, suffix=".NS") -> pd.DataFrame:
    """Attach headlines to tickers by company-name matching.

    RSS is market-wide, so an article must be mapped to a symbol before it can be summarised
    per-stock. Matching is deliberately conservative - a whole-word company-name hit - since a
    false attribution injects another company's news into this stock's daily summary.
    """
    pats = {sym: re.compile(r"\b" + re.escape(n) + r"\b", re.I) for sym, n in names.items()}
    out = []
    for _, row in df.iterrows():
        blob = f"{row['headline']} {row['body']}"
        for sym, pat in pats.items():
            if pat.search(blob):
                out.append({**row, "symbol": sym if sym.endswith(suffix) else sym + suffix})
    return pd.DataFrame(out)


def append_csv(df: pd.DataFrame, path: str):
    """Idempotent append - reruns of the collector must not duplicate rows."""
    cols = ["symbol", "timestamp", "headline", "body"]
    df = df.reindex(columns=cols)
    try:
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass
    # The rows already on disk come back as strings; the incoming ones are tz-aware
    # datetimes. Deduping across that boundary compares a str to a Timestamp, matches
    # nothing, and silently doubles the file on every re-run - which is exactly the
    # idempotency R6 promises. Normalise both sides to one canonical ISO string first.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce"
                                     ).dt.tz_convert("Asia/Kolkata")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["symbol", "timestamp", "headline"]).sort_values(
        ["symbol", "timestamp"])
    df.to_csv(path, index=False)
    return len(df)


def company_names(symbols, suffix=".NS") -> dict:
    """symbol -> short company name, harvested from the BSE scrip master, for RSS tagging.

    The trailing 'Ltd'/'Limited'/'India' tokens are stripped so that headlines saying
    "Reliance posts Q3 profit" still match "Reliance Industries Ltd".
    """
    r = requests.get(SCRIP_MASTER, params={"Group": "", "Scripcode": "", "industry": "",
                                           "segment": "Equity", "status": "Active"},
                     headers=UA, timeout=60)
    r.raise_for_status()
    by_id = {str(row["scrip_id"]).strip().upper(): str(row.get("Scrip_Name") or "")
             for row in r.json() if row.get("scrip_id")}
    out = {}
    for sym in symbols:
        base = sym.replace(suffix, "").replace(".BO", "").upper()
        name = by_id.get(ALIASES.get(base, base)) or by_id.get(base)
        if not name:
            continue
        short = re.sub(r"\s+(Ltd|Limited|Corporation|Corp|Company|Co)\.?$", "", name.strip(),
                       flags=re.I).strip()
        out[sym] = short if len(short) > 3 else name.strip()
    return out


# ------------------------------------------------------- boilerplate filter
#
# 42.8% of the BSE corpus (26,451 of 61,791 filings) is administrative paperwork that no
# price could plausibly react to: lost share certificates, newspaper-publication copies,
# trading-window closures, scheduled investor-meet notices. Feeding it to the summariser
# costs GPU hours and dilutes whatever signal the real filings carry.
#
# Matched on the headline only - the body is often an attachment name and says nothing.
# Kept as one flat list rather than a category taxonomy: the goal is to drop noise, not to
# classify it, and configs/india.yaml can switch the whole thing off so filtered-vs-raw
# stays available as a cheap ablation.

BOILERPLATE = [
    r"loss of (?:share )?certificate|duplicate share|issue of duplicate",
    r"newspaper publication|publication of (?:extract|notice)",
    r"closure of trading window|trading window",
    r"compliances?[- ]certificate|certificate under (?:reg|regulation)",
    r"compliances?[- ]reg\.? *7|reg\.? *74 *\(5\)",
    r"reconciliation of share capital",
    r"grievance redressal|investor complaint|statement of investor complaints",
    r"analyst / ?investor meet|investor meet|schedule of (?:analyst|investor)",
    r"shareholding pattern",
    r"disclosure of related party transaction",
]
_BOILER_RE = re.compile("|".join(BOILERPLATE), re.I)


def drop_boilerplate(df: pd.DataFrame, log=None) -> pd.DataFrame:
    """Remove administrative filings. Returns a copy; leaves df alone."""
    if "headline" not in df.columns or not len(df):
        return df
    noise = df["headline"].fillna("").str.contains(_BOILER_RE)
    if log:
        log(f"boilerplate filter: dropped {int(noise.sum()):,} of {len(df):,} filings "
            f"({noise.mean():.1%})")
    return df.loc[~noise].copy()
