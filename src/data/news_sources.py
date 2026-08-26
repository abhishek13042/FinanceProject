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

# NSE ticker != BSE scrip_id for a handful of names; patch them explicitly.
# TATAMOTORS -> TMCV reflects the 2025 commercial/passenger-vehicle demerger: the NSE
# ticker's pre-2025 history and the post-demerger BSE filings are not the same entity,
# so treat that symbol's 2018-2025 series as a known discontinuity.
ALIASES = {"BAJAJ-AUTO": "BAJAJAUTO", "TATAMOTORS": "TMCV"}


def scrip_map(timeout=60) -> dict:
    """NSE-style symbol -> BSE numeric scrip code, from the live BSE scrip master."""
    r = requests.get(SCRIP_MASTER, params={"Group": "", "Scripcode": "", "industry": "",
                                           "segment": "Equity", "status": "Active"},
                     headers=UA, timeout=timeout)
    r.raise_for_status()
    return {str(row["scrip_id"]).strip().upper(): str(row["SCRIP_CD"])
            for row in r.json() if row.get("scrip_id")}


def fetch_bse(scrip_cd: str, start: str, end: str, pause=0.4, max_pages=40) -> pd.DataFrame:
    """All announcements for one scrip in [start, end]. Dates as 'YYYYMMDD'."""
    rows, page = [], 1
    while page <= max_pages:
        r = requests.get(ANNOUNCEMENTS, headers=UA, timeout=60, params={
            "pageno": page, "strCat": "-1", "strPrevDate": start, "strScrip": scrip_cd,
            "strSearch": "P", "strToDate": end, "strType": "C", "subcategory": ""})
        if r.status_code != 200:
            break
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


def collect_bse(symbols, start: str, end: str, suffix=".NS", log=print) -> pd.DataFrame:
    """Historical backfill across a universe. start/end as 'YYYY-MM-DD'."""
    smap = scrip_map()
    s0, s1 = start.replace("-", ""), end.replace("-", "")
    frames = []
    for i, sym in enumerate(symbols, 1):
        base = sym.replace(suffix, "").replace(".BO", "").upper()
        code = smap.get(ALIASES.get(base, base)) or smap.get(base)
        if not code:
            log(f"  [{i}/{len(symbols)}] {base}: no BSE scrip code, skipped")
            continue
        try:
            df = fetch_bse(code, s0, s1)
        except Exception as e:                  # one bad symbol must not kill a long backfill
            log(f"  [{i}/{len(symbols)}] {base}: {type(e).__name__}, skipped")
            continue
        df["symbol"] = base + suffix
        frames.append(df)
        log(f"  [{i}/{len(symbols)}] {base} ({code}): {len(df)} announcements")
    if not frames:
        return pd.DataFrame(columns=["symbol", "timestamp", "headline", "body"])
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])


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
