"""
Tables 1 and 2 of the MEDES 2026 paper from the runs in results/. Standard library only.

    python experiments/analyze_medes.py [results]

Per backbone and configuration: mean and 95% CI (Student t) over seeds or repetitions of each
scoring dimension, the number of agent errors and the mean latency. Agent errors (rows without
a result) count as incorrect on every dimension. DeepSeek-V3 has one run per configuration, so
no interval is printed. Also: the paired Full - No-Knowledge gateway difference per seed, and
how often the temperature-0 baseline reproduces its text across repetitions.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import score_instance, load_ground_truth, GT_PATH  # noqa: E402

DIMS = ["diagnosis_ok", "substance_ok", "dose_ok", "gateway_ok", "escalation_ok", "tools_ok"]
SHORT = {"diagnosis_ok": "diag", "substance_ok": "subst", "dose_ok": "dose",
         "gateway_ok": "gateway", "escalation_ok": "esc", "tools_ok": "tools"}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def rows(path: Path) -> dict:
    """{instance_id: row}; a successful retry overrides an earlier failed row."""
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("result") or r["instance_id"] not in out:
                out[r["instance_id"]] = r
    return out


def scores(run: dict, gt: dict) -> dict:
    acc = defaultdict(list)
    for iid, r in run.items():
        s = score_instance(r["result"], gt[iid]) if r.get("result") else {d: False for d in DIMS}
        for d in DIMS:
            acc[d].append(float(s[d]))
    return {d: statistics.fmean(v) for d, v in acc.items()}


def ci(xs: list) -> str:
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return f"{m:.3f}"
    return f"{m:.3f}±{T95[len(xs) - 1] * statistics.stdev(xs) / len(xs) ** 0.5:.3f}"


def main(root: Path) -> None:
    gt_prod = load_ground_truth(GT_PATH)
    gt_tf = load_ground_truth(GT_PATH.parent / "ground_truth_toolforcing_n100.jsonl")
    cells = defaultdict(dict)  # (backbone, config) -> replicate -> {instance: row}
    for f in sorted(root.glob("*/*/results.jsonl")):
        backbone, config, rep = f.parent.name.split("__")
        cells[(backbone, config.replace("_skipprefetch", ""))][rep] = rows(f)

    for (backbone, config), reps in sorted(cells.items()):
        gt = gt_tf if config == "toolforcing" else gt_prod
        per = {rep: scores(run, gt) for rep, run in reps.items()}
        errors = sum(1 for run in reps.values() for r in run.values() if not r.get("result"))
        lat = [r["latency_s"] for run in reps.values() for r in run.values() if r.get("result") and r.get("latency_s")]
        print(f"\n{backbone} {config}  ({len(reps)} run(s), {errors} agent errors"
              + (f", latency {statistics.fmean(lat):.1f}s)" if lat else ")"))
        dims = [d for d in DIMS if not (config == "toolforcing" and d == "gateway_ok")]
        print("  " + "  ".join(f"{SHORT[d]} {ci([per[r][d] for r in per])}" for d in dims))
        if config == "monolithic" and len(reps) > 1:
            ids = set.intersection(*(set(run) for run in reps.values()))
            text = lambda run, i: (run[i].get("result") or {}).get("treatment")
            same = sum(1 for i in ids if len({text(run, i) for run in reps.values()}) == 1)
            print(f"  identical treatment text across all repetitions: {same}/{len(ids)}")

    for backbone in sorted({b for b, _ in cells}):
        f, k = cells.get((backbone, "full")), cells.get((backbone, "no_knowledge"))
        if f and k and len(f) > 1:
            d = [scores(f[s], gt_prod)["gateway_ok"] - scores(k[s], gt_prod)["gateway_ok"]
                 for s in sorted(f) if s in k]
            print(f"\n{backbone} gateway Full - No-Knowledge per seed: "
                  + ", ".join(f"{x:+.2f}" for x in d) + f"   mean {ci(d)}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
