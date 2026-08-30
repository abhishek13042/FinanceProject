"""Section 3.2 - hierarchical (intraday-bucket -> daily) summarisation + embedding.

India-specific: the NSE session runs 09:15-15:30 IST, so a trading "day" of news is
bucketed into pre-open (< 09:15), six hourly intraday slots, and post-close (>= 15:30).
News timestamped after the close is attributed to the NEXT trading day - attributing it
to the same day would leak information the market could not have traded on.

Two models, deliberately split (see spec 03):

  generation  an instruction-following LLM. The original choice, mT5-XLSum, is fine-tuned on
              BBC articles and does not summarise Indian filings so much as re-imagine them:
              measured on 8 real filings it named the correct company 2/8 and emitted
              "Scottish Stock Exchange", "British Stock Exchange" and "Financial Conduct
              Authority" for Asian Paints, ITC and Cipla. Qwen2.5-1.5B-Instruct named the
              company 8/8, with correct dates and amounts. The paper's method assumes an LLM
              that summarises what it is handed; using one that does is what makes this a
              replication rather than a re-enactment.
  embedding   a frozen encoder, still mT5. Keeps x_e at d = 768 exactly as configured, and an
              encoder is a better sentence embedder than a decoder-only model's last hidden
              state. The two are never resident at once - `embed()` releases the generator
              first, which is what keeps this inside 6 GB.

Both stay frozen; only the alignment, fusion and classifier downstream are trained.
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# Instruction-model prompts. The constraints matter more than the verbs: naming the symbol and
# forbidding other company names is what stopped the entity substitution in testing.
INSTR_ITEM = ("Summarise this Indian stock exchange filing for {sym} in one sentence. "
              "State only facts present in the filing - the action taken, amounts, and dates. "
              "Do not add background, do not name any company other than {sym}.\n\nFiling:\n{txt}")
INSTR_DAY = ("Summarise the day's developments for {sym} in one or two sentences, most "
             "market-relevant first. Use only the facts below. Do not add background.\n\n{txt}")

# Seq2seq fallback (mT5-XLSum and friends). A fine-tuned summariser has never seen an
# instruction, so a natural-language directive is just more article for it to summarise - and
# being first, it dominates. Measured: the instruction prefix cut content-word grounding in the
# source from 31.1% to 13.4%. The bare T5 sentinel is the correct way to address such a model.
PROMPT_HOUR = "summarize: "
PROMPT_DAY = "summarize: "


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
    """Frozen LLM, two passes (Algorithm 1). Never fine-tuned."""

    def __init__(self, cfg):
        s = cfg["summarizer"]
        self.cfg, self.s = cfg, s
        self.dev = s["device"] if torch.cuda.is_available() else "cpu"
        self.causal = bool(s.get("causal", False))
        self._gtok = self._gmodel = None
        self._etok = self._emodel = None

    # ------------------------------------------------------------------ model handling
    def _generator(self):
        """Loaded on first use, so `embed` never pays for a generator it will not call."""
        if self._gmodel is None:
            name = self.s["model"]
            self._gtok = AutoTokenizer.from_pretrained(name)
            if self.causal:
                # Left padding: a decoder-only model continues from the LAST token, so right
                # padding would have it continue from <pad> for every short prompt in a batch.
                self._gtok.padding_side = "left"
                if self._gtok.pad_token is None:
                    self._gtok.pad_token = self._gtok.eos_token
                m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
            else:
                m = AutoModelForSeq2SeqLM.from_pretrained(name)   # fp32: mT5 is unstable in fp16
            self._gmodel = m.to(self.dev).eval()
        return self._gtok, self._gmodel

    def _embedder(self):
        """The frozen encoder that produces x_e. Releases the generator first: both resident at
        once is ~5.6 GB and this is expected to run on a 6 GB laptop card."""
        if self._emodel is None:
            if self._gmodel is not None:
                self._gmodel, self._gtok = None, None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            name = self.s.get("embed_model", self.s["model"])
            self._etok = AutoTokenizer.from_pretrained(name)
            self._emodel = AutoModelForSeq2SeqLM.from_pretrained(name).to(self.dev).eval()
        return self._etok, self._emodel

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def _gen(self, texts, max_new):
        tok, model = self._generator()
        bs, outs = self.s["batch_size"], []
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            if self.causal:
                prompts = [tok.apply_chat_template([{"role": "user", "content": c}],
                                                   tokenize=False, add_generation_prompt=True)
                           for c in chunk]
                b = tok(prompts, return_tensors="pt", truncation=True, padding=True,
                        max_length=self.s["max_input_tokens"]).to(self.dev)
                y = model.generate(**b, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
                outs += [tok.decode(o[b["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                         for o in y]
            else:
                b = tok(chunk, return_tensors="pt", truncation=True, padding=True,
                        max_length=self.s["max_input_tokens"]).to(self.dev)
                y = model.generate(**b, max_new_tokens=max_new, num_beams=2, no_repeat_ngram_size=3)
                outs += tok.batch_decode(y, skip_special_tokens=True)
        return outs

    def _item_prompt(self, sym, txt):
        bare = str(sym).split(".")[0]
        return INSTR_ITEM.format(sym=bare, txt=txt[:4000]) if self.causal else PROMPT_HOUR + txt

    def _day_prompt(self, sym, txt):
        bare = str(sym).split(".")[0]
        return INSTR_DAY.format(sym=bare, txt=txt[:4000]) if self.causal else PROMPT_DAY + txt

    def summarize(self, news: pd.DataFrame) -> pd.DataFrame:
        """news: symbol, date, bucket, text  ->  symbol, date, summary."""
        news = news.copy()
        news["prompt"] = [self._item_prompt(s, t) for s, t in zip(news["symbol"], news["text"])]
        news["hourly"] = self._gen(news["prompt"].tolist(), self.s["max_summary_tokens"])  # pass 1
        # Ordered by timestamp, not by bucket id. `bucket` is a time-of-day code, and post-close
        # news has already been re-attributed to the NEXT session, where it is the earliest
        # information of that day - a bucket sort files it last instead.
        grp = (news.sort_values(["symbol", "date", "timestamp"])
                   .groupby(["symbol", "date"])["hourly"].apply(" ".join).reset_index())
        grp["prompt"] = [self._day_prompt(s, t) for s, t in zip(grp["symbol"], grp["hourly"])]
        grp["summary"] = self._gen(grp["prompt"].tolist(), self.s["daily_summary_tokens"])  # pass 2
        return grp[["symbol", "date", "summary"]]

    # ------------------------------------------------------------------ embedding
    @torch.no_grad()
    def embed(self, summaries: list[str]) -> np.ndarray:
        """Mean-pooled encoder states - the x_e of Algorithm 1, line 10."""
        tok, model = self._embedder()
        n = int(self.s.get("embed_max_tokens", self.s["max_input_tokens"]))
        vecs = []
        for i in range(0, len(summaries), self.s["batch_size"]):
            b = tok(summaries[i:i + self.s["batch_size"]], return_tensors="pt", truncation=True,
                    max_length=n, padding=True).to(self.dev)
            h = model.get_encoder()(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).float()
            vecs.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
        return np.concatenate(vecs, 0)
