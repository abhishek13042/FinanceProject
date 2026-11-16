"""Run the whole ablation matrix, then print the comparison table with G3 applied.

Built for a borrowed GPU: the matrix is 8 ablations x 5 seeds, and the session will be
interrupted. Any ablation whose `results_<tag>.json` already exists is skipped, so re-running
this script resumes rather than restarts. Nothing here touches the test set more than once
per ablation.

    python scripts/run_all.py                  # everything not yet done
    python scripts/run_all.py --only full,price_only,shuffled_text
    python scripts/run_all.py --table          # no training; just re-print the table

`--table` is the one to run on the laptop after copying results back.
"""
import argparse, json, os, sys, time
sys.path.insert(0, ".")

import yaml

import src.train as T
from src.eval import intervals_overlap

# Deliberate order: the three that decide whether anything else is worth reading come first.
# If the shuffled-text control does not land at chance, every later row is uninterpretable
# and the session should stop rather than spend GPU hours on ablations of a broken pipeline.
ORDER = ["full", "price_only", "shuffled_text", "concat", "oneway", "layers2", "align32", "align64",
         "opt_price_only", "opt_full"]


def load_results(out_dir):
    res = {}
    for tag in ORDER:
        p = os.path.join(out_dir, f"results_{tag}.json")
        if os.path.exists(p):
            res[tag] = json.load(open(p))
    return res


def table(res):
    if not res:
        print("no results yet")
        return
    print(f"\n{'ablation':<16} {'acc':>16} {'mcc':>18} {'95% CI (mcc)':>20} {'beta gap':>9}")
    print("-" * 82)
    for tag, r in res.items():
        m, ci = r["test_mean_over_seeds"], r["ci_block_bootstrap_days"]
        gap = r["beta_contamination"].get("mcc_gap")
        print(f"{tag:<16} {m['acc']['mean']:>8.2f} +/- {m['acc']['std']:<4.2f} "
              f"{m['mcc']['mean']:>9.3f} +/- {m['mcc']['std']:<5.3f} "
              f"{ci['mcc']['lo']:>9.3f} to {ci['mcc']['hi']:<7.3f} "
              f"{'' if gap is None else f'{gap:+.3f}'}")
    if "full" in res:
        floor = res["full"]["baseline_constant_up"]["acc"]
        print(f"\nconstant-up floor: {floor:.2f}% acc, 0.000 mcc (spec 05 R5)")

    # Spec 06 G3: a difference whose intervals overlap is not a finding. Printed as a verdict
    # rather than left to the reader, because the temptation arrives after the numbers do.
    if "full" in res:
        print("\nG3 - is the difference from `full` reportable?")
        for tag, r in res.items():
            if tag == "full":
                continue
            ov = intervals_overlap(res["full"]["ci_block_bootstrap_days"],
                                   r["ci_block_bootstrap_days"])
            print(f"  full vs {tag:<16} {'NO  - intervals overlap, this is noise' if ov else 'yes - intervals are disjoint'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/india.yaml")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--only", default=None, help="comma-separated subset of the matrix")
    ap.add_argument("--table", action="store_true", help="print the table and exit")
    ap.add_argument("--force", action="store_true", help="re-run ablations already on disk")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    out = cfg["train"]["out_dir"]
    os.makedirs(out, exist_ok=True)
    if a.table:
        return table(load_results(out))

    todo = [t for t in (a.only.split(",") if a.only else ORDER)
            if a.force or not os.path.exists(os.path.join(out, f"results_{t}.json"))]
    seeds = [int(s) for s in a.seeds.split(",")]
    asked = a.only.split(",") if a.only else ORDER
    print(f"matrix: {len(todo)} to run, {len(asked) - len(todo)} of {len(asked)} "
          f"already on disk -> {todo}")

    t0 = time.time()
    for i, tag in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {tag} === ({(time.time()-t0)/60:.0f} min elapsed)")
        try:
            T.main(a.config, tag, seeds, a.n_boot, tag)
        except Exception as e:                  # one bad ablation must not lose the session
            print(f"!! {tag} FAILED: {type(e).__name__}: {e}")
    table(load_results(out))


if __name__ == "__main__":
    main()
