"""Example 1 — end-to-end CompileML pipeline on the insurance fraud cold-start set.

Data: data/Insurance_FraudulentAutoInsuranceClaims_2K_coldstart.csv (2,000 claims,
12 numeric features, EVENT_LABEL fraud/legit).

Steps
  1. train a CatBoost teacher (CompileML cannot compile CatBoost, so this is a
     good test of the "any teacher, distilled" claim)
  2. distill it into a depth-2 whitebox with train_whitebox
  3. build monotone quantile bands + isotonic calibration
  4. compile everything into one hashed JSON artifact
  5. run the stdlib-only runtime: decide() with reason codes
  6. run the 10-check validation framework against the holdout
  7. collapse the artifact to an exact scorecard and re-sum it
  8. export SQL, execute it in SQLite, and check parity with the Python runtime
  9. compare Gini: teacher -> whitebox float -> integer artifact -> band ordinal

Run:  uv run python scripts/01_insurance_fraud_end_to_end.py
"""

from __future__ import annotations

import json
import sqlite3
from itertools import pairwise
from pathlib import Path

import _tracking
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from compileml.artifact import build_artifact, save_artifact
from compileml.bands import band_efficiency, monotone_quantile_bands, semantic_bands
from compileml.compile import rha, train_whitebox
from compileml.export import export_sql
from compileml.reference import fit_reference, reference_gini
from compileml.runtime import decide, load_artifact
from compileml.scorecard import build_scorecard, score_from_scorecard, scorecard_to_csv
from compileml.validate import validate_artifact
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Insurance_FraudulentAutoInsuranceClaims_2K_coldstart.csv"
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

FEATURES = [
    "incident_severity",
    "num_vehicles_involved",
    "num_injuries",
    "num_witnesses",
    "police_report_available",
    "injury_claim",
    "vehicle_claim",
    "incident_hour",
    "customer_age",
    "policy_deductable",
    "policy_annual_premium",
    "num_claims_past_year",
]

# Reason dictionary: the institution's language, not the library's.
REASONS = {
    "incident_severity": {
        "code": "SEVERITY",
        "negative": "Incident severity is high relative to similar claims.",
        "positive": "Incident severity is in the normal range.",
    },
    "num_vehicles_involved": {
        "code": "VEHICLES",
        "negative": "Unusual number of vehicles involved.",
        "positive": "Typical number of vehicles involved.",
    },
    "num_injuries": {
        "code": "INJURIES",
        "negative": "Number of reported injuries is unusual for the incident.",
        "positive": "Reported injuries are consistent with the incident.",
    },
    "num_witnesses": {
        "code": "WITNESSES",
        "negative": "Few or no independent witnesses.",
        "positive": "Independent witnesses are available.",
    },
    "police_report_available": {
        "code": "POLICE_REPORT",
        "negative": "No police report was filed.",
        "positive": "A police report is on file.",
    },
    "injury_claim": {
        "code": "INJURY_AMOUNT",
        "negative": "Injury claim amount is high.",
        "positive": "Injury claim amount is moderate.",
    },
    "vehicle_claim": {
        "code": "VEHICLE_AMOUNT",
        "negative": "Vehicle claim amount is high.",
        "positive": "Vehicle claim amount is moderate.",
    },
    "incident_hour": {
        "code": "TIME_OF_DAY",
        "negative": "Incident occurred at an unusual hour.",
        "positive": "Incident occurred during typical hours.",
    },
    "customer_age": {
        "code": "CUSTOMER_AGE",
        "negative": "Customer age profile is associated with higher claim risk.",
        "positive": "Customer age profile is associated with lower claim risk.",
    },
    "policy_deductable": {
        "code": "DEDUCTIBLE",
        "negative": "Policy deductible is low.",
        "positive": "Policy deductible is in the standard range.",
    },
    "policy_annual_premium": {
        "code": "PREMIUM",
        "negative": "Annual premium is atypical for the policy.",
        "positive": "Annual premium is typical for the policy.",
    },
    "num_claims_past_year": {
        "code": "PRIOR_CLAIMS",
        "negative": "Multiple claims filed in the past year.",
        "positive": "No recent claim history.",
    },
}


def widest_clean_bands(latent, y, max_bands: int = 8, scale: int = 1000):
    """Largest fixed-K monotone-quantile ladder whose edges survive fixed-point conversion.

    With a ~5% bad rate a large share of whitebox latents clip to 0, so quantile
    edges repeat at 0.0 and build_artifact rejects the ladder (edges collide at
    scale=1000). allow_merge only merges bad-rate inversions, not duplicate
    edges, so we step K down until the integer ladder is strictly increasing.
    """
    for k in range(max_bands, 1, -1):
        spec = monotone_quantile_bands(latent, y, n_bands=k, allow_merge=True)
        ints = [rha(float(e) * scale) for e in spec.edges]
        if all(b > a for a, b in pairwise(ints)):
            return spec
    raise ValueError("no band ladder with >= 2 distinct integer edges")


def gini(y, s) -> float:
    return float(2.0 * roc_auc_score(y, s) - 1.0)


def main() -> None:
    df = pd.read_csv(DATA)
    X = df[FEATURES].astype(float)
    y = (df["EVENT_LABEL"] == "fraud").astype(int)
    print(f"rows={len(df)}  fraud rate={y.mean():.3f}")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_tr_np, X_va_np = X_tr.to_numpy(), X_va.to_numpy()
    y_tr_np, y_va_np = y_tr.to_numpy(), y_va.to_numpy()

    # 1. teacher
    teacher = CatBoostClassifier(
        iterations=200,
        depth=3,
        learning_rate=0.05,
        l2_leaf_reg=10,
        verbose=0,
        random_seed=42,
    )
    teacher.fit(X_tr, y_tr)
    mlflow.log_params(
        {
            "teacher": "CatBoostClassifier",
            **{
                k: teacher.get_params()[k]
                for k in ("iterations", "depth", "learning_rate", "l2_leaf_reg")
            },
        }
    )
    t_tr = teacher.predict_proba(X_tr)[:, 1]
    t_va = teacher.predict_proba(X_va)[:, 1]
    print(
        f"\n[teacher] CatBoost Gini  train={gini(y_tr, t_tr):.3f}  val={gini(y_va, t_va):.3f}"
    )
    mlflow.log_metrics(
        {"teacher_gini_train": gini(y_tr, t_tr), "teacher_gini_val": gini(y_va, t_va)}
    )

    # 2. distill
    whitebox, fidelity = train_whitebox(X_tr_np, t_tr, n_estimators=60, max_depth=2)
    mlflow.log_params({"whitebox_n_estimators": 60, "whitebox_max_depth": 2})
    mlflow.log_metrics(
        {f"fidelity_{k}": v for k, v in fidelity.items() if isinstance(v, float)}
    )
    print(
        "[whitebox] fidelity to teacher:",
        {k: round(v, 4) if isinstance(v, float) else v for k, v in fidelity.items()},
    )
    latent_tr = whitebox.predict(X_tr_np).clip(0, 1)
    latent_va = whitebox.predict(X_va_np).clip(0, 1)
    print(
        f"[whitebox] Gini  train={gini(y_tr, latent_tr):.3f}  val={gini(y_va, latent_va):.3f}"
    )
    mlflow.log_metrics(
        {
            "whitebox_gini_train": gini(y_tr, latent_tr),
            "whitebox_gini_val": gini(y_va, latent_va),
        }
    )

    # 3. bands
    sem = semantic_bands(latent_tr, y_tr_np, max_bands=8, min_band_size=100)
    print(
        f"\n[bands] semantic_bands certifies {sem.n_bands} separable band(s) on {len(latent_tr)} rows (too few rows to certify more)"
    )
    bands = widest_clean_bands(latent_tr, y_tr_np, max_bands=8)
    print(
        f"[bands] widest clean monotone-quantile ladder: {bands.n_bands} bands, edges={np.round(bands.edges, 4).tolist()}"
    )
    print(
        f"[bands] counts={bands.metadata['counts']}  empirical bad rate={np.round(bands.metadata['empirical_bad_rate'], 3).tolist()}"
    )
    eff = band_efficiency(latent_va, y_va_np, bands)
    print(
        f"[bands] continuous Gini={eff['continuous_gini']:.3f}  band-ordinal Gini={eff['band_ordinal_gini']:.3f}  gap={eff['gini_gap_pct']:.1f}%"
    )
    mlflow.log_param("n_bands", bands.n_bands)
    mlflow.log_metrics(
        {
            "band_ordinal_gini_val": eff["band_ordinal_gini"],
            "band_gini_gap_pct": eff["gini_gap_pct"],
        }
    )

    # 4. compile
    medians = X_tr.median().tolist()
    artifact = build_artifact(
        whitebox,
        FEATURES,
        baseline=medians,
        band_edges=bands,
        calibration_latent=latent_tr,
        calibration_y=y_tr_np,
        reasons=REASONS,
        X_sample=X_tr_np,
        metadata={
            "dataset": DATA.name,
            "teacher": "CatBoostClassifier",
            "example": "01",
        },
    )
    path = OUT / "insurance_fraud.json"
    save_artifact(artifact, path)
    print(
        f"\n[artifact] saved {path.relative_to(ROOT)}  hash={artifact['artifact_hash'][:16]}…  size={path.stat().st_size / 1024:.0f} KB"
    )
    mlflow.set_tag("artifact_hash", artifact["artifact_hash"])
    mlflow.log_artifact(str(path))

    # 5. runtime
    art = load_artifact(path)  # verifies hash
    row = X_va_np[0].tolist()
    d = decide(art, row, top_k=3)
    print("\n[decide] first holdout row:")
    print(
        json.dumps(
            {k: d[k] for k in ("band", "pd", "latent_int", "artifact_hash")}, indent=2
        )
    )
    for r in d["reasons_negative"]:
        print(f"   - {r['code']:<15} impact={r['impact_int']:>5}  {r['message']}")
    d_missing = decide(art, [None] * len(FEATURES), explain=False)
    print(
        f"[decide] all-missing row -> baseline imputation: band={d_missing['band']} pd={d_missing['pd']:.4f}"
    )

    # 6. validate
    ref = fit_reference(X_tr_np, y_tr_np, feature_names=FEATURES)
    print(
        f"\n[reference] WOE-logit Gini on val = {reference_gini(ref, X_va_np, y_va_np):.3f}"
    )
    report = validate_artifact(
        path, X_va_np, y_va_np, model=whitebox, latent_train=latent_tr, reference=ref
    )
    print(f"[validate] all_pass={report['all_pass']}")
    _tracking.log_validation(report)
    for name, chk in report["checks"].items():
        status = "SKIP" if chk.get("skipped") else ("PASS" if chk["pass"] else "FAIL")
        extra = {
            k: v
            for k, v in chk.items()
            if k not in ("pass", "skipped") and isinstance(v, (int, float, str, bool))
        }
        print(f"   {status:<4} {name:<28} {json.dumps(extra)[:110]}")

    # 7. scorecard
    sc = build_scorecard(art)
    sc_path = OUT / "insurance_fraud_scorecard.csv"
    sc_path.write_text(scorecard_to_csv(sc))
    mismatches = sum(
        score_from_scorecard(sc, X_va_np[i].tolist())
        != decide(art, X_va_np[i].tolist(), explain=False)["raw_micro"]
        for i in range(len(X_va_np))
    )
    print(
        f"\n[scorecard] {len(sc['main_effects'])} main effects, {len(sc['interactions'])} interaction grids -> {sc_path.name}"
    )
    print(
        f"[scorecard] re-sum mismatches vs runtime on {len(X_va_np)} holdout rows: {mismatches}"
    )
    mlflow.log_metric("scorecard_resum_mismatches", mismatches)
    mlflow.log_artifact(str(sc_path))

    # 8. SQL parity
    sql = export_sql(art, table="features", dialect="sqlite")
    (OUT / "insurance_fraud.sqlite.sql").write_text(sql)
    con = sqlite3.connect(":memory:")
    cols = ", ".join(f'"{c}" REAL' for c in FEATURES)
    con.execute(f"CREATE TABLE features (row_id INTEGER, {cols})")
    con.executemany(
        f"INSERT INTO features VALUES ({', '.join('?' * (len(FEATURES) + 1))})",
        [(i, *map(float, r)) for i, r in enumerate(X_va_np)],
    )
    sql_rows = con.execute(sql).fetchall()
    colnames = [c[0] for c in con.execute(sql).description]
    ix = {n: colnames.index(n) for n in ("row_id", "latent_int", "band", "pd_ppm")}
    py = {
        i: decide(art, X_va_np[i].tolist(), explain=False) for i in range(len(X_va_np))
    }
    sql_mismatch = sum(
        (r[ix["latent_int"]], r[ix["band"]], r[ix["pd_ppm"]])
        != (
            py[r[ix["row_id"]]]["latent_int"],
            py[r[ix["row_id"]]]["band"],
            py[r[ix["row_id"]]]["pd_ppm"],
        )
        for r in sql_rows
    )
    print(
        f"[sql] SQLite vs Python runtime mismatches on {len(sql_rows)} rows: {sql_mismatch}"
    )
    mlflow.log_metric("sql_parity_mismatches", sql_mismatch)
    mlflow.log_artifact(str(OUT / "insurance_fraud.sqlite.sql"))

    # 9. Gini ladder
    art_latent = np.array([py[i]["latent_int"] for i in range(len(X_va_np))])
    art_band = np.array([py[i]["band_idx"] for i in range(len(X_va_np))])
    g_t = gini(y_va_np, t_va)
    print("\n[gini ladder on holdout]")
    for name, s in [
        ("teacher (CatBoost)", t_va),
        ("whitebox float", latent_va),
        ("artifact integer", art_latent),
        ("band ordinal", art_band),
    ]:
        g = gini(y_va_np, s)
        print(f"   {name:<20} {g:.3f}   ({100 * g / g_t:5.1f}% of teacher)")
        mlflow.log_metric("gini_val_" + name.split(" (")[0].replace(" ", "_"), g)
    mlflow.log_metric(
        "artifact_gini_retention_pct", 100 * gini(y_va_np, art_latent) / g_t
    )


if __name__ == "__main__":
    with _tracking.start("01-insurance-end-to-end", dataset=DATA.name):
        main()
