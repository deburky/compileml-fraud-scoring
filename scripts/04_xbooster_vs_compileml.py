"""Example 4 — xbooster CatBoost scorecard vs compileml artifact, same teacher.

Mirrors the two xBooster example notebooks
(examples/catboost-getting-started.ipynb and examples/shap-scorecard-examples.ipynb)
on the insurance fraud data, then distills the same CatBoost model through compileml
and compares:

  * rule parsing: xbooster's leaf-level rule table (Tree, LeafIndex, DetailedSplit,
    WOE, points) vs compileml's exact depth-2 scorecard
  * scoring: xbooster raw / WOE / PDO points / SHAP points vs compileml integer latent
  * explanations: xbooster per-feature SHAP score decomposition vs compileml reason codes
    (top-driver agreement per row on the holdout)

Run:  uv run python scripts/04_xbooster_vs_compileml.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from compileml.artifact import build_artifact
from compileml.compile import train_whitebox
from compileml.runtime import decide
from compileml.scorecard import build_scorecard
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from xbooster.cb_constructor import (
    CBScorecardConstructor,  # xbooster.constructor pulls in xgboost, which needs libomp
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tracking

ex1 = importlib.import_module("01_insurance_fraud_end_to_end")  # FEATURES, REASONS, gini, widest_clean_bands

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)
FEATURES, REASONS, gini = ex1.FEATURES, ex1.REASONS, ex1.gini


def main() -> None:
    df = pd.read_csv(ex1.DATA)
    X = df[FEATURES].astype(float)
    y = (df["EVENT_LABEL"] == "fraud").astype(int)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_tr, X_va = X_tr.reset_index(drop=True), X_va.reset_index(drop=True)
    y_tr, y_va = y_tr.reset_index(drop=True), y_va.reset_index(drop=True)

    # --- teacher, as in catboost-getting-started.ipynb ------------------------
    pool = Pool(X_tr, y_tr)
    model = CatBoostClassifier(
        iterations=100, depth=3, learning_rate=0.1, l2_leaf_reg=10,
        allow_writing_files=False, verbose=0, random_seed=42,
    )
    model.fit(pool)
    cb_raw_va = model.predict(X_va, prediction_type="RawFormulaVal")
    cb_prob_tr = model.predict_proba(X_tr)[:, 1]
    print(f"[teacher] CatBoost Gini train={gini(y_tr, model.predict_proba(X_tr)[:, 1]):.3f}  holdout={gini(y_va, cb_raw_va):.3f}")
    mlflow.log_params({"teacher": "CatBoostClassifier", "iterations": 100, "depth": 3, "learning_rate": 0.1, "xbooster_version": __import__("xbooster").__version__})

    # --- xbooster: parse rules ---------------------------------------------------
    constructor = CBScorecardConstructor(model, pool)
    scorecard = constructor.construct_scorecard()
    print(f"\n[xbooster] rule table: {scorecard.shape[0]} leaves x {scorecard.shape[1]} cols")
    print(f"[xbooster] columns: {scorecard.columns.tolist()}")
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(scorecard.loc[scorecard.Tree == 0, ["Tree", "LeafIndex", "Feature", "Sign", "Split", "Count", "EventRate", "XAddEvidence", "WOE", "DetailedSplit"]].to_string(index=False))
    points = constructor.create_points(pdo=50, target_points=600, target_odds=19, precision_points=0)
    print(f"[xbooster] create_points -> columns {points.columns.tolist()[-4:]}")
    scorecard_path = OUT / "xbooster_catboost_scorecard.csv"
    points.to_csv(scorecard_path, index=False)
    imp = constructor.get_feature_importance()
    print("[xbooster] feature importance:", {k: round(v, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])})
    mlflow.log_artifact(str(scorecard_path))
    mlflow.log_metrics({f"xb_importance_{k}": v for k, v in imp.items()})
    mlflow.log_metric("xb_n_leaves", scorecard.shape[0])

    # --- xbooster: score, as in shap-scorecard-examples.ipynb --------------------
    xb = {
        "raw": constructor.predict_score(X_va, method="raw"),
        "woe": constructor.predict_score(X_va, method="woe"),
        "pdo": constructor.predict_score(X_va, method="pdo"),
        "shap": constructor.predict_score(X_va, method="shap"),
    }
    shap_dec = constructor.predict_scores(X_va, method="shap")
    print(f"\n[xbooster] SHAP decomposition columns: {shap_dec.columns.tolist()}")
    print(f"[xbooster] max |sum(feature scores) - score| = {np.abs(shap_dec.drop(columns='score').sum(axis=1) - shap_dec['score']).max():.3f}")
    np.testing.assert_allclose(xb["raw"], cb_raw_va, rtol=1e-2, atol=1e-2)
    print("[xbooster] raw leaf-sum scores match CatBoost RawFormulaVal (atol 1e-2)")

    # --- compileml: distill the same teacher --------------------------------------
    whitebox, fidelity = train_whitebox(X_tr.to_numpy(), cb_prob_tr, n_estimators=60, max_depth=2)
    latent_tr = whitebox.predict(X_tr.to_numpy()).clip(0, 1)
    bands = ex1.widest_clean_bands(latent_tr, y_tr.to_numpy(), max_bands=8)
    artifact = build_artifact(
        whitebox, FEATURES, baseline=X_tr.median().tolist(), band_edges=bands,
        calibration_latent=latent_tr, calibration_y=y_tr.to_numpy(), reasons=REASONS,
    )
    cml = [decide(artifact, r.tolist(), top_k=3) for r in X_va.to_numpy()]
    cml_latent = np.array([d["latent_int"] for d in cml])
    sc = build_scorecard(artifact)
    print(f"\n[compileml] whitebox spearman to teacher={fidelity['spearman']:.3f}; scorecard has {len(sc['main_effects'])} main effects + {len(sc['interactions'])} pairwise grids over {len(artifact['model']['trees'])} trees")

    # --- compare: ranking power ---------------------------------------------------
    print("\n[gini on holdout]  (xbooster points are 'higher = safer', so their Gini is negated)")
    rows = [
        ("CatBoost raw", cb_raw_va, 1),
        ("xbooster raw (leaf sum)", xb["raw"].to_numpy(), 1),
        ("xbooster WOE", xb["woe"].to_numpy(), 1),
        ("xbooster PDO points", xb["pdo"].to_numpy(), -1),
        ("xbooster SHAP points", xb["shap"].to_numpy(), -1),
        ("compileml integer latent", cml_latent, 1),
    ]
    for name, s, sign in rows:
        g = gini(y_va, sign * np.asarray(s, dtype=float))
        print(f"   {name:<26} {g:.3f}")
        mlflow.log_metric("gini_val_" + name.split(" (")[0].replace(" ", "_"), g)
    rho = spearmanr(-xb["shap"].to_numpy(), cml_latent).correlation
    print(f"\n[agreement] Spearman(xbooster SHAP points, compileml latent) = {rho:.3f}")

    # --- compare: per-row top driver -------------------------------------------------
    # xbooster: most negative feature score = biggest risk-increasing driver.
    # compileml: reasons_negative[0].feature. Baselines differ (SHAP expected value vs
    # median row), so exact agreement is not expected; this measures how often they align.
    feat_cols = [f"{f}_score" for f in FEATURES]
    xb_rank = shap_dec[feat_cols].to_numpy().argsort(axis=1)  # ascending: most negative first
    xb_top1 = [FEATURES[i] for i in xb_rank[:, 0]]
    xb_top3 = [{FEATURES[i] for i in r[:3]} for r in xb_rank]
    cml_top1 = [d["reasons_negative"][0]["feature"] if d["reasons_negative"] else None for d in cml]
    cml_top3 = [{r["feature"] for r in d["reasons_negative"][:3]} for d in cml]
    n = len(cml)
    top1 = sum(a == b for a, b in zip(xb_top1, cml_top1, strict=True)) / n
    in3 = sum(b in a for a, b in zip(xb_top3, cml_top1, strict=True)) / n
    ov3 = np.mean([len(a & b) / 3 for a, b in zip(xb_top3, cml_top3, strict=True)])
    print(f"[agreement] top-1 driver identical: {top1:.1%}   compileml top-1 in xbooster top-3: {in3:.1%}   mean top-3 overlap: {ov3:.1%}")
    mlflow.log_metrics({"spearman_xb_shap_vs_cml": float(rho), "top1_driver_agreement": top1, "cml_top1_in_xb_top3": in3, "top3_overlap": float(ov3)})

    hi = int(np.argmax(cml_latent))
    print(f"\n[row {hi}] highest compileml risk: latent={cml_latent[hi]} band={cml[hi]['band']}  xbooster SHAP points={int(xb['shap'].iloc[hi])}")
    print("   xbooster per-feature SHAP points (most negative first):")
    for i in xb_rank[hi, :3]:
        print(f"      {FEATURES[i]:<24} {shap_dec.iloc[hi][feat_cols[i]]:>7.0f}")
    print("   compileml reasons_negative:")
    for r in cml[hi]["reasons_negative"]:
        print(f"      {r['feature']:<24} {r['impact_int']:>7}  {r['code']}")
    print(f"\n[files] {scorecard_path.relative_to(ROOT)}")


if __name__ == "__main__":
    with _tracking.start("04-xbooster-vs-compileml", dataset=ex1.DATA.name):
        main()
