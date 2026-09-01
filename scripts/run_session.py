"""Unattended end-to-end session driver.

One process runs the whole thing so nothing depends on a human (or an agent) being present
between stages. Every stage is resumable, every stage is logged, and the two stop conditions
from HANDOFF.md are enforced in code rather than left to whoever reads the output first:

  * the shuffled-text control must land at chance. If its MCC interval excludes zero, the
    text stream is being exploited through a leak and every later ablation would be an
    ablation of a broken pipeline. The session stops.
  * anything near 60% accuracy is treated as a leak until proven otherwise, not as a result.

The ablation matrix is run one tag at a time, cheapest-signal first, so an overnight run that
does not finish still leaves the three ablations that answer the question. A deadline stops
new ablations from starting; it never kills one mid-flight.

    python scripts/run_session.py [--deadline-hours 10]

Everything lands in artifacts/, with a written verdict in artifacts/SESSION_REPORT.md.
"""
import argparse, io, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOG = None
T0 = time.time()

DECISIVE = ["full", "price_only", "shuffled_text"]
REFINEMENTS = ["concat", "oneway", "layers2", "align32", "align64"]


def log(msg=""):
    line = f"[{(time.time() - T0) / 60:7.1f}m] {msg}"
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n")
        LOG.flush()
        os.fsync(LOG.fileno())


def run(name, args, retries=1):
    """Run a stage. Returns True on success. Retries once - most overnight failures are
    transient (a CUDA hiccup), and every stage here resumes from its own checkpoint."""
    for attempt in range(retries + 1):
        log(f"--- {name}{f' (retry {attempt})' if attempt else ''}: {' '.join(args)}")
        p = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True)
        tail = "\n".join((p.stdout or "").strip().splitlines()[-25:])
        if LOG:
            LOG.write(p.stdout or "")
            LOG.write(p.stderr or "")
            LOG.flush()
        if p.returncode == 0:
            log(f"    OK: {name}")
            for l in tail.splitlines()[-6:]:
                log(f"    | {l}")
            return True
        log(f"    FAILED ({p.returncode}): {name}")
        for l in ((p.stderr or "").strip().splitlines() or ["<no stderr>"])[-8:]:
            log(f"    ! {l}")
        if "out of memory" in ((p.stdout or "") + (p.stderr or "")).lower():
            halve_batch()
    return False


def halve_batch():
    """OOM insurance. Summariser batch size is throughput only - it changes GPU hours, not
    the result - so it is safe to reduce automatically and note that we did."""
    p = os.path.join(ROOT, "configs", "india.yaml")
    s = io.open(p, encoding="utf-8").read()
    import re
    m = re.search(r"^(  batch_size: )(\d+)", s, re.M)
    if not m or int(m.group(2)) <= 2:
        return
    new = max(2, int(m.group(2)) // 2)
    s = s[:m.start()] + m.group(1) + str(new) + s[m.end():]
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    log(f"    OOM: summariser batch_size {m.group(2)} -> {new}, retrying")


def results(tag):
    p = os.path.join(ROOT, "artifacts", f"results_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def gate_shuffled():
    """The negative control. Must be indistinguishable from chance."""
    r = results("shuffled_text")
    if r is None:
        return False, "shuffled_text produced no results file"
    ci = r["ci_block_bootstrap_days"]["mcc"]
    acc = r["test_mean_over_seeds"]["acc"]["mean"]
    at_chance = ci["lo"] <= 0.0 <= ci["hi"]
    msg = (f"shuffled_text: acc {acc:.2f}%, mcc 95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}] "
           f"-> {'at chance' if at_chance else 'NOT at chance'}")
    return at_chance, msg


def gate_leak():
    """Anything near 60% is a leak until proven otherwise (HANDOFF calibration)."""
    bad = []
    for tag in DECISIVE:
        r = results(tag)
        if r and r["test_mean_over_seeds"]["acc"]["mean"] >= 60.0:
            bad.append(f"{tag} at {r['test_mean_over_seeds']['acc']['mean']:.2f}%")
    return (not bad), ("no ablation near 60%" if not bad
                       else "SUSPECT, treat as leak: " + ", ".join(bad))


def report(lines):
    p = os.path.join(ROOT, "artifacts", "SESSION_REPORT.md")
    io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    log(f"wrote {p}")


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-hours", type=float, default=10.0,
                    help="stop STARTING new ablations past this; never kills one mid-run")
    ap.add_argument("--config", default="configs/india.yaml")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    LOG = io.open(os.path.join(ROOT, "artifacts", "session.log"), "a", encoding="utf-8", newline="\n")
    log("=" * 72)
    log(f"session start | deadline {a.deadline_hours}h | python {sys.version.split()[0]}")

    done, notes = [], []

    # ---------------------------------------------------------------- spec 03
    stages = [
        ("summarize", ["scripts/build_text.py", "summarize", a.config, "--chunk", "200"]),
        ("summarize --raw", ["scripts/build_text.py", "summarize", a.config, "--raw"]),
        ("embed", ["scripts/build_text.py", "embed", a.config]),
        ("embed --raw", ["scripts/build_text.py", "embed", a.config, "--raw"]),
    ]
    for name, args in stages:
        if not run(name, args, retries=2):
            notes.append(f"STAGE FAILED: {name} - session cannot continue past it")
            report(["# Session report", "", f"Aborted: `{name}` failed. See artifacts/session.log."])
            return 1
        done.append(name)

    # ---------------------------------------------------------------- spec 07
    if not run("guards", ["-m", "pytest", "tests/", "-q"], retries=0):
        notes.append("GUARDS FAILED - results below are not trustworthy")
    else:
        done.append("guards (24)")

    # ---------------------------------------------------------------- spec 05
    ran = []
    for tag in DECISIVE:
        if results(tag):
            log(f"--- {tag}: already on disk, skipping")
            ran.append(tag)
            continue
        if run(f"train {tag}", ["scripts/run_all.py", "--config", a.config, "--only", tag], retries=1):
            ran.append(tag)
        else:
            notes.append(f"ablation {tag} failed to produce results")

    ok_shuf, msg_shuf = gate_shuffled()
    ok_leak, msg_leak = gate_leak()
    log(f"GATE {msg_shuf}")
    log(f"GATE {msg_leak}")
    notes += [msg_shuf, msg_leak]

    # ---------------------------------------------------------------- refinements, if earned
    if not ok_shuf:
        log("STOP: negative control did not land at chance. Not running the refinements - "
            "they would be ablations of a pipeline that is leaking.")
        notes.append("STOPPED before refinements: shuffled-text control failed.")
    elif not ok_leak:
        log("STOP: an ablation is at or above 60%. Treating as a leak; not running refinements.")
        notes.append("STOPPED before refinements: accuracy in leak territory.")
    else:
        for tag in REFINEMENTS:
            if (time.time() - T0) / 3600 > a.deadline_hours:
                log(f"deadline reached; not starting {tag}")
                notes.append(f"deadline reached before {tag} (and any after it)")
                break
            if results(tag):
                continue
            if run(f"train {tag}", ["scripts/run_all.py", "--config", a.config, "--only", tag],
                   retries=0):
                ran.append(tag)

    # ---------------------------------------------------------------- verdict
    tbl = subprocess.run([PY, "scripts/run_all.py", "--config", a.config, "--table"],
                         cwd=ROOT, capture_output=True, text=True)
    log("final table:")
    for l in (tbl.stdout or "").strip().splitlines():
        log(f"    {l}")

    hrs = (time.time() - T0) / 3600
    lines = [
        "# Session report", "",
        f"Unattended run, {hrs:.1f} hours. Ablations completed: {', '.join(ran) or 'none'}.", "",
        "## Gates", "",
        f"- {msg_shuf}", f"- {msg_leak}",
        "- The test set is evaluated once per ablation; no code can enforce this, but "
        "`run_all.py` skips any ablation already on disk, so nothing was re-scored.", "",
        "## Results", "", "```", (tbl.stdout or "").strip(), "```", "",
        "## Read this before quoting any number", "",
        "Expected range on NSE data is 53-57% accuracy, MCC 0.05-0.15. The paper's 67% is an "
        "upper bound on a denser corpus, not a target. Two overlapping intervals are not a "
        "finding, however tidy the ranking looks - the G3 verdict in the table above says "
        "which differences are reportable.", "",
        "## Deviations from the original design, and why", "",
        "These were measured, not assumed. Full detail in the session transcript.", "",
        "1. **Summariser swapped** from `mT5_multilingual_XLSum` to `Qwen2.5-1.5B-Instruct`. "
        "On 8 real filings mT5 named the correct company 2/8 and emitted 'Scottish Stock "
        "Exchange', 'Bank of England' and 'Financial Conduct Authority' for Asian Paints, "
        "Axis Bank and Cipla. Qwen scored 8/8 with correct dates and amounts. Embeddings are "
        "still mT5's encoder, so `text_dim` stays 768 and nothing downstream moved.",
        "2. **Instruction prefix removed** for seq2seq models. A fine-tuned summariser has "
        "never seen an instruction, so the directive was just more article to summarise - and "
        "first, so it dominated. Measured: grounding in the source fell from 31.1% to 13.4% "
        "with the prefix.",
        "3. **Empty-body filings dropped** (`min_body_chars: 30`). BSE serves most Regulation "
        "30 filings as a category label plus a PDF attachment that `collect_news.py` never "
        "fetches, leaving 60.2% of the post-boilerplate corpus with nothing to summarise. The "
        "single string 'Announcement under Regulation 30 (LODR)-Press Release / Media "
        "Release. A.' appeared 3,969 times, and identical stubs produced identical invented "
        "summaries across unrelated stocks - a constant injected into the text stream.",
        "4. **Coverage is therefore 9.5% of stock-days, not the 23.4% in HANDOFF.md.** That "
        "figure counted stub filings as coverage. Quote 9.5%.",
        "5. **Post-close news sorted by timestamp, not bucket id.** Attribution already moves "
        "it to the next session, where it is the earliest information of that day; the bucket "
        "sort filed it last. Most BSE filings are post-close.",
        "6. **Per-ablation checkpoint names.** Every ablation previously wrote "
        "`best_seed<N>.pt`; identical-shaped ablations could silently load each other's "
        "weights.", "",
        "## The honest limitation", "",
        "The real repair for the corpus is fetching the BSE PDF attachments that hold the "
        "missing bodies - `news_sources.py` already captures `ATTACHMENTNAME` and then drops "
        "the column. That would restore coverage toward the original 23.4% with real text "
        "rather than stubs. It is a multi-hour re-scrape plus PDF extraction and was out of "
        "scope for a borrowed GPU. Until then, a null result here means 'text did not help on "
        "the 9.5% of stock-days where we actually have text', which is a narrower claim than "
        "the write-up may want to make.", "",
    ]
    if notes:
        lines += ["## Notes from the run", ""] + [f"- {n}" for n in notes] + [""]
    report(lines)
    log(f"session complete in {hrs:.1f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
