"""Example 2 — feature engineering, capacity sweeps, and recalibration on the
transaction fraud set.

Data: data/transaction_data_100K_full.csv (~117K card transactions). The raw
columns are mostly strings, so this script builds a numeric feature frame first,
then uses a time split: earlier events train the model, later events are the
holdout and the recalibration sample.

Steps
  1. engineer numeric features from the raw event columns
  2. CatBoost teacher on the early window
  3. sweep_whitebox: what do trees and depth buy, and what does each cost
  4. sweep_bands: how many bands the score can support
  5. compile an artifact from the chosen configuration and validate it
  6. recalibrate on the later window: PD table changes, model + edges do not

Run:  uv run python scripts/02_transaction_fraud_tuning.py
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import _tracking
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from compileml.artifact import build_artifact, recalibrate_artifact, save_artifact
from compileml.bands import band_efficiency, monotone_quantile_bands, semantic_bands
from compileml.compile import rha, train_whitebox
from compileml.runtime import decide, verify_artifact
from compileml.tune import sweep_bands, sweep_whitebox
from compileml.validate import validate_artifact
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transaction_data_100K_full.csv"
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

FREE_MAIL = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["EVENT_TIMESTAMP"], utc=True)
    email_domain = df["customer_email"].str.split("@").str[-1].str.lower()
    ua = df["user_agent"].fillna("")
    f = pd.DataFrame(
        {
            "order_price": df["order_price"].astype(float),
            "log_order_price": np.log1p(df["order_price"].astype(float)),
            "event_hour": ts.dt.hour.astype(float),
            "event_dow": ts.dt.dayofweek.astype(float),
            "card_bin": pd.to_numeric(df["card_bin"], errors="coerce"),
            "billing_lat": df["billing_latitude"].astype(float),
            "billing_lon": df["billing_longitude"].astype(float),
            "billing_zip": pd.to_numeric(df["billing_zip"], errors="coerce"),
            "is_us_billing": (df["billing_country"] == "US").astype(float),
            "is_usd": (df["payment_currency"] == "USD").astype(float),
            "free_mail_domain": email_domain.isin(FREE_MAIL).astype(float),
            "email_local_len": df["customer_email"].str.split("@").str[0].str.len().astype(float),
            "ua_is_mobile": ua.str.contains("Mobile|iPhone|Android", regex=True).astype(float),
            "product_category": df["product_category"].astype("category").cat.codes.astype(float),
        }
    )
    return f


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
    df = df.sort_values("EVENT_TIMESTAMP").reset_index(drop=True)
    X = engineer(df)
    y = df["EVENT_LABEL"].astype(int).to_numpy()
    names = list(X.columns)
    print(f"rows={len(df)}  fraud rate={y.mean():.4f}  features={len(names)}")
    print(f"missing values per feature: {X.isna().sum().to_dict()}")

    cut = int(len(df) * 0.7)
    X_tr, X_va = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_va = y[:cut], y[cut:]
    print(f"time split: train={len(X_tr)} ({df['EVENT_TIMESTAMP'].iloc[0][:10]}..{df['EVENT_TIMESTAMP'].iloc[cut-1][:10]})  "
          f"holdout={len(X_va)} (..{df['EVENT_TIMESTAMP'].iloc[-1][:10]})")

    # sklearn's classic GBM will not accept NaN: impute at the training medians,
    # which is also what the artifact's missing_policy="baseline" does at runtime.
    medians = X_tr.median()
    X_tr_np = X_tr.fillna(medians).to_numpy()
    X_va_np = X_va.fillna(medians).to_numpy()

    # 2. teacher --------------------------------------------------------------
    teacher = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.08, verbose=0, random_seed=42)
    teacher.fit(X_tr_np, y_tr)
    mlflow.log_params({"teacher": "CatBoostClassifier", **{k: teacher.get_params()[k] for k in ("iterations", "depth", "learning_rate")}, "n_features": len(names), "split": "time_70_30"})
    t_tr = teacher.predict_proba(X_tr_np)[:, 1]
    t_va = teacher.predict_proba(X_va_np)[:, 1]
    print(f"\n[teacher] CatBoost Gini  train={gini(y_tr, t_tr):.3f}  holdout={gini(y_va, t_va):.3f}")
    mlflow.log_metrics({"teacher_gini_train": gini(y_tr, t_tr), "teacher_gini_val": gini(y_va, t_va), "fraud_rate_train": float(y_tr.mean()), "fraud_rate_val": float(y_va.mean())})

    # 3. sweep whitebox capacity ----------------------------------------------
    print("\n[sweep_whitebox] trees x depth (alpha=0 -> pure distillation of the teacher)")
    results = sweep_whitebox(
        X_tr_np, t_tr, y_tr,
        trees_grid=(20, 40, 80),
        depth_grid=(1, 2, 3),
        X_val=X_va_np, y_val=y_va, teacher_latent_val=t_va,
    )
    mlflow.log_table(pd.DataFrame(results), "sweep_whitebox.json")
    keys = [k for k in results[0] if isinstance(results[0][k], (int, float, bool, str))]
    print("   columns:", keys)
    for r in results:
        print("   " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in keys}))

    # pick: best holdout Gini among depth<=2 (keeps exact attribution + scorecard)
    def val_gini(r):
        for k in ("val_gini", "gini_val", "artifact_gini_val", "gini"):
            if k in r:
                return r[k]
        return -1

    cand = [r for r in results if r.get("max_depth", r.get("depth")) <= 2]
    best = max(cand, key=val_gini)
    n_trees = best.get("n_estimators", best.get("trees"))
    depth = best.get("max_depth", best.get("depth"))
    print(f"\n[choice] depth<=2 config with best holdout Gini: trees={n_trees} depth={depth}")
    mlflow.log_params({"whitebox_n_estimators": n_trees, "whitebox_max_depth": depth})

    whitebox, fidelity = train_whitebox(X_tr_np, t_tr, n_estimators=n_trees, max_depth=depth)
    latent_tr = whitebox.predict(X_tr_np).clip(0, 1)
    latent_va = whitebox.predict(X_va_np).clip(0, 1)
    print(f"[whitebox] spearman to teacher={fidelity['spearman']:.4f}  Gini train={gini(y_tr, latent_tr):.3f} holdout={gini(y_va, latent_va):.3f}")
    mlflow.log_metrics({"fidelity_spearman": fidelity["spearman"], "whitebox_gini_train": gini(y_tr, latent_tr), "whitebox_gini_val": gini(y_va, latent_va)})

    # 4. sweep bands ------------------------------------------------------------
    print("\n[sweep_bands] K -> band-ordinal Gini, gap, worst within-band AUC")
    band_sweep = sweep_bands(latent_tr, y_tr, k_grid=(4, 6, 8, 10, 12))
    mlflow.log_table(pd.DataFrame([{k: v for k, v in r.items() if isinstance(v, (int, float, bool, str))} for r in band_sweep]), "sweep_bands.json")
    for r in band_sweep:
        flat = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if isinstance(v, (int, float, bool, str))}
        print("   " + json.dumps(flat))

    sem = semantic_bands(latent_tr, y_tr, max_bands=10)
    print(f"\n[bands] semantic_bands certifies {sem.n_bands} separable band(s); edges={np.round(sem.edges, 4).tolist()}")
    bands = sem if sem.n_bands >= 3 else widest_clean_bands(latent_tr, y_tr, max_bands=10)
    eff = band_efficiency(latent_va, y_va, bands)
    print(f"[bands] using {bands.metadata['method']} ladder with {bands.n_bands} bands; holdout continuous Gini={eff['continuous_gini']:.3f} "
          f"band-ordinal={eff['band_ordinal_gini']:.3f} gap={eff['gini_gap_pct']:.1f}%")
    mlflow.log_params({"band_method": bands.metadata["method"], "n_bands": bands.n_bands})
    mlflow.log_metrics({"band_ordinal_gini_val": eff["band_ordinal_gini"], "band_gini_gap_pct": eff["gini_gap_pct"]})
    for b in eff["per_band"]:
        flat = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items() if isinstance(v, (int, float, str))}
        print("   " + json.dumps(flat))

    # 5. compile + validate ----------------------------------------------------
    artifact = build_artifact(
        whitebox, names,
        baseline=medians.tolist(),
        band_edges=bands,
        calibration_latent=latent_tr, calibration_y=y_tr,
        X_sample=X_tr_np,
        metadata={"dataset": DATA.name, "teacher": "CatBoostClassifier", "example": "02", "split": "time"},
    )
    path = OUT / "transaction_fraud.json"
    save_artifact(artifact, path)
    print(f"\n[artifact] saved {path.relative_to(ROOT)}  hash={artifact['artifact_hash'][:16]}…")
    cov = artifact.get("metadata", {}).get("reason_coverage")
    print(f"[artifact] reason coverage (no dictionary supplied on purpose): {cov}")

    report = validate_artifact(artifact, X_va_np, y_va, model=whitebox, latent_train=latent_tr)
    print(f"[validate] all_pass={report['all_pass']}")
    _tracking.log_validation(report)
    mlflow.set_tag("artifact_hash", artifact["artifact_hash"])
    mlflow.log_artifact(str(path))
    for name, chk in report["checks"].items():
        status = "SKIP" if chk.get("skipped") else ("PASS" if chk["pass"] else "FAIL")
        print(f"   {status:<4} {name}")

    # 6. recalibrate on the later window ---------------------------------------
    # The holdout's own latents, as the deployed runtime computes them.
    dec_va = [decide(artifact, r.tolist(), explain=False) for r in X_va_np]
    latent_va_rt = np.array([d["latent_micro"] for d in dec_va]) / artifact["model"]["micro_scale"]
    recal = recalibrate_artifact(artifact, latent_va_rt, y_va)
    save_artifact(recal, OUT / "transaction_fraud_recalibrated.json")

    same_model = json.dumps(artifact["model"], sort_keys=True) == json.dumps(recal["model"], sort_keys=True)
    same_edges = artifact["bands"]["edges_int"] == recal["bands"]["edges_int"]
    moved = sum(
        decide(artifact, r.tolist(), explain=False)["band"] != decide(recal, r.tolist(), explain=False)["band"]
        for r in X_va_np[:2000]
    )
    pd_shift = np.mean([abs(decide(recal, r.tolist(), explain=False)["pd"] - d["pd"]) for r, d in zip(X_va_np[:2000], dec_va[:2000])])
    print("\n[recalibrate] on the later window")
    print(f"   model bytes identical: {same_model}   band edges identical: {same_edges}   hash verified: {verify_artifact(recal)}")
    print(f"   rows that changed band (first 2000): {moved}   mean |ΔPD|: {pd_shift:.4f}")
    mlflow.log_metrics({"recal_band_churn_rows": moved, "recal_mean_abs_pd_shift": float(pd_shift), "recal_model_identical": float(same_model), "recal_edges_identical": float(same_edges)})
    mlflow.log_artifact(str(OUT / "transaction_fraud_recalibrated.json"))
    lineage = {k: (str(v)[:16] + "…") for k, v in recal["metadata"].items() if "recalibrat" in k}
    print(f"   lineage metadata={lineage}  new hash={recal['artifact_hash'][:16]}…")


if __name__ == "__main__":
    with _tracking.start("02-transaction-tuning", dataset=DATA.name):
        main()
