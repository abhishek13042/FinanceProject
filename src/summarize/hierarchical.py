"""Section 3.2 - hierarchical (intraday-bucket -> daily) summarisation + embedding.

India-specific: the NSE session runs 09:15-15:30 IST, so a trading "day" of news is
bucketed into pre-open (< 09:15), six hourly intraday slots, and post-close (>= 15:30).
News timestamped after the close is attributed to the NEXT trading day - attributing it
to the same day would leak information the market could not have traded on.
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROMPT_HOUR = ("summarize the market-moving facts in this Indian stock news for {sym}: "
               "earnings, orders, regulatory or SEBI/RBI action, promoter or FII/DII flows, "
               "guidance and analyst actions. omit boilerplate. ")
PROMPT_DAY = ("summarize the day's key drivers for {sym} in Indian markets, ranked by "
              "likely price impact. ")


def bucket_by_session(ts: pd.Series, cfg: dict) -> pd.Series:
    """Map IST timestamps to an ordered intraday bucket id (0 = pre-open ... last = post-close)."""
    t = pd.to_datetime(ts, utc=True).dt.tz_convert(cfg["market"]["tz"])
    start = pd.to_datetime(cfg["market"]["session_start"]).time()
    end = pd.to_datetime(cfg["market"]["session_end"]).time()
    hour = t.dt.hour + t.dt.minute / 60.0
    b = np.clip(np.floor(hour - start.hour), 0, cfg["market"]["hourly_buckets"] - 1) + 1
    b = pd.Series(b, index=t.index)
    b[t.dt.time < start] = 0
    b[t.dt.time >= end] = cfg["market"]["hourly_buckets"] - 1
    return b.astype(int)


def attribution_date(ts: pd.Series, trading_days: pd.DatetimeIndex, cfg: dict) -> pd.Series:
    """News after the close belongs to the next trading session (no look-ahead)."""
    t = pd.to_datetime(ts, utc=True).dt.tz_convert(cfg["market"]["tz"])
    end = pd.to_datetime(cfg["market"]["session_end"]).time()
    day = t.dt.normalize().dt.tz_localize(None)
    day = day.where(t.dt.time < end, day + pd.Timedelta(days=1))
    idx = trading_days.searchsorted(day.values, side="left").clip(0, len(trading_days) - 1)
    return pd.Series(trading_days[idx], index=ts.index)


def dedup(df: pd.DataFrame, ratio: float, min_chars: int) -> pd.DataFrame:
    """Drop syndicated wire duplicates; keep the earliest version of near-identical stories."""
    import difflib
    df = df[df["text"].str.len() >= min_chars].copy()
    keep = []
    for (_, _), g in df.groupby(["symbol", "date"]):
        seen = []
        for i, row in g.sort_values("timestamp").iterrows():
            if any(difflib.SequenceMatcher(None, row["text"][:400], s).ratio() > ratio for s in seen):
                continue
            seen.append(row["text"][:400])
            keep.append(i)
    return df.loc[keep]


class HierarchicalSummarizer:
    """Frozen LLM, two passes (Algorithm 1). Never fine-tuned - only the encoder,
    fusion and classifier downstream are trained end to end."""

    def __init__(self, cfg):
        s = cfg["summarizer"]
        self.cfg, self.s = cfg, s
        self.dev = s["device"] if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(s["model"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(s["model"]).to(self.dev).eval()

    @torch.no_grad()
    def _gen(self, texts, max_new):
        outs = []
        for i in range(0, len(texts), self.s["batch_size"]):
            b = self.tok(texts[i:i + self.s["batch_size"]], return_tensors="pt", truncation=True,
                         max_length=self.s["max_input_tokens"], padding=True).to(self.dev)
            y = self.model.generate(**b, max_new_tokens=max_new, num_beams=2, no_repeat_ngram_size=3)
            outs += self.tok.batch_decode(y, skip_special_tokens=True)
        return outs

    def summarize(self, news: pd.DataFrame) -> pd.DataFrame:
        """news: symbol, date, bucket, text  ->  symbol, date, summary."""
        news = news.copy()
        news["prompt"] = [PROMPT_HOUR.format(sym=s) + t for s, t in zip(news["symbol"], news["text"])]
        news["hourly"] = self._gen(news["prompt"].tolist(), self.s["max_summary_tokens"])  # pass 1
        grp = (news.sort_values(["symbol", "date", "bucket", "timestamp"])
                   .groupby(["symbol", "date"])["hourly"].apply(" ".join).reset_index())
        grp["prompt"] = [PROMPT_DAY.format(sym=s) + t for s, t in zip(grp["symbol"], grp["hourly"])]
        grp["summary"] = self._gen(grp["prompt"].tolist(), self.s["daily_summary_tokens"])  # pass 2
        return grp[["symbol", "date", "summary"]]

    @torch.no_grad()
    def embed(self, summaries: list[str]) -> np.ndarray:
        """Mean-pooled mT5 encoder states - the x_e of Algorithm 1, line 10."""
        vecs = []
        for i in range(0, len(summaries), self.s["batch_size"]):
            b = self.tok(summaries[i:i + self.s["batch_size"]], return_tensors="pt", truncation=True,
                         max_length=self.s["max_input_tokens"], padding=True).to(self.dev)
            h = self.model.get_encoder()(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).float()
            vecs.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
        return np.concatenate(vecs, 0)
