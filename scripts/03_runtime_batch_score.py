"""Example 3 — production-side scoring with the stdlib-only runtime.

This script deliberately imports nothing from numpy, pandas, sklearn, or
catboost. It takes the artifact produced by example 1, scores a CSV with the
csv module, writes decisions out, renders one waterfall SVG, and reports timing.
This is the "deployment stack is not the modeling stack" claim, tested.

Run:  uv run python scripts/03_runtime_batch_score.py [artifact.json] [input.csv]
      (defaults: artifacts/insurance_fraud.json and the insurance cold-start CSV)
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

from compileml.runtime import decide, load_artifact
from compileml.viz import waterfall_svg

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else ROOT / "artifacts" / "insurance_fraud.json"
)
INPUT = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else ROOT / "data" / "Insurance_FraudulentAutoInsuranceClaims_2K_coldstart.csv"
)
OUT = ROOT / "artifacts"


def to_float(v: str):
    v = v.strip()
    return None if v == "" else float(v)


def main() -> None:
    forbidden = {
        m for m in ("numpy", "pandas", "sklearn", "catboost") if m in sys.modules
    }
    art = load_artifact(ARTIFACT)
    names = art["features"]["names"]
    print(
        f"artifact={ARTIFACT.name}  hash={art['artifact_hash'][:16]}…  features={len(names)}  bands={art['bands']['labels']}"
    )

    with INPUT.open(newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        missing_cols = [n for n in names if n not in header]
        if missing_cols:
            raise SystemExit(f"input CSV lacks artifact features: {missing_cols}")
        rows = list(reader)

    # fast path: score + band + PD, no explanation
    t0 = time.perf_counter()
    fast = [decide(art, [to_float(r[n]) for n in names], explain=False) for r in rows]
    t_fast = (time.perf_counter() - t0) / len(rows) * 1e3

    # full path: exact attribution + reason codes
    lat = []
    full = []
    for r in rows:
        t1 = time.perf_counter()
        full.append(
            decide(
                art,
                [to_float(r[n]) for n in names],
                top_k=3,
                include_contributions=True,
            )
        )
        lat.append((time.perf_counter() - t1) * 1e3)

    # the explained path must land on exactly the same integers as the fast path
    assert all(
        f["latent_int"] == d["latent_int"] and f["band"] == d["band"]
        for f, d in zip(fast, full, strict=True)
    )

    out_path = OUT / f"{ARTIFACT.stem}_decisions.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["row", "band", "latent_int", "pd", "top_negative_codes", "artifact_hash"]
        )
        for i, d in enumerate(full):
            w.writerow(
                [
                    i,
                    d["band"],
                    d["latent_int"],
                    d["pd"],
                    "|".join(r["code"] for r in d["reasons_negative"]),
                    d["artifact_hash"][:12],
                ]
            )

    # reconciliation: contributions must add back to the score with zero residual at depth<=2
    worst_resid = max(
        abs(int(d.get("residual_half_micro", d.get("residual", 0)) or 0)) for d in full
    )

    counts: dict[str, int] = {}
    for d in full:
        counts[d["band"]] = counts.get(d["band"], 0) + 1
    print(f"\nscored {len(rows)} rows -> {out_path.relative_to(ROOT)}")
    print(f"band counts: {dict(sorted(counts.items()))}")
    print(
        f"timing per row: score-only {t_fast:.3f} ms   full explain median {statistics.median(lat):.2f} ms  p95 {sorted(lat)[int(0.95 * len(lat)) - 1]:.2f} ms"
    )
    print(f"max attribution residual across rows: {worst_resid}")
    print(f"ML libraries loaded in this process: {forbidden or 'none'}")

    worst = max(range(len(full)), key=lambda i: full[i]["latent_int"])
    svg_path = OUT / f"{ARTIFACT.stem}_waterfall_row{worst}.svg"
    svg_path.write_text(waterfall_svg(full[worst], max_features=8))
    print(
        f"highest-risk row #{worst}: band={full[worst]['band']} pd={full[worst]['pd']:.4f} -> {svg_path.relative_to(ROOT)}"
    )
    for r in full[worst]["reasons_negative"]:
        print(f"   - {r['code']:<15} impact={r['impact_int']:>5}  {r['message']}")


if __name__ == "__main__":
    main()
