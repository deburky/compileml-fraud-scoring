# CompileML examples

Scripts that exercise [compileml](https://github.com/orgoca/CompileML) against the
AWS Fraud Detector sample data in `data/`. Outputs land in `artifacts/`.

| Script | Data | What it tests |
| --- | --- | --- |
| `01_insurance_fraud_end_to_end.py` | insurance cold-start (2K, 12 numeric features) | teacher → whitebox → bands → artifact → decide → validate → scorecard → SQL parity |
| `02_transaction_fraud_tuning.py` | transactions (117K, engineered features, time split) | `sweep_whitebox`, `sweep_bands`, `band_efficiency`, validate, `recalibrate_artifact` |
| `03_runtime_batch_score.py` | any artifact + CSV | stdlib-only batch scoring, reason codes, timing, `waterfall_svg` |
| `04_xbooster_vs_compileml.py` | insurance cold-start | xbooster CatBoost rule table + SHAP points vs compileml artifact on the same teacher: Gini, rank agreement, top-driver agreement |

```bash
uv run python scripts/01_insurance_fraud_end_to_end.py
uv run python scripts/02_transaction_fraud_tuning.py
uv run python scripts/03_runtime_batch_score.py            # needs artifacts/insurance_fraud.json from 01
uv run python scripts/04_xbooster_vs_compileml.py

make mlflow-ui                                             # local SQLite store, experiment "compileml-tests"


# or log to the SageMaker MLflow App (stack in cloudformation/mlflow-app, see Makefile):
# .env holds AWS_PROFILE and MLFLOW_TRACKING_URI=<App ARN from `make arn`>; scripts load it.
# Offline: MLFLOW_LOCAL=1 forces the local SQLite store; add `uv run --offline --no-sync` if uv complains
uv run python scripts/01_insurance_fraud_end_to_end.py
```

The CLI covers the same runtime surface without Python code:

```bash
uv run compileml inspect  artifacts/insurance_fraud.json
uv run compileml verify   artifacts/insurance_fraud.json
uv run compileml score    artifacts/insurance_fraud.json --csv data/Insurance_FraudulentAutoInsuranceClaims_2K_coldstart.csv --out artifacts/cli_scores.csv
uv run compileml score    artifacts/insurance_fraud.json --features "6,2,8,2,0,8157,3755,7,54,453,2113,4" --explain
uv run compileml scorecard artifacts/insurance_fraud.json --format markdown
uv run compileml export   artifacts/insurance_fraud.json --target sql --dialect sqlite --out artifacts/scorer.sql
uv run compileml export   artifacts/insurance_fraud.json --target cobol --out artifacts/scorer.cob
uv run compileml validate artifacts/insurance_fraud.json --csv <holdout.csv> --y-col <label_col>
```

Notes

- The teacher is CatBoost in both scripts. CompileML can only *compile* sklearn,
  XGBoost, and LightGBM trees, but the teacher can be anything: only its
  predicted probabilities are used, via `train_whitebox`.
- xgboost needs `libomp` on Apple Silicon (`brew install libomp`, done 2026-09-04); noted in
  compileml's README since 0.5.2.
- The repo tracks compileml 0.8.0 (bumped 2026-09-15); `FINDINGS.md` ends with the upstream
  status of every item reported against 0.4.3.
  lightgbm is not installed.
- xbooster: `import xbooster` fails outright when xgboost is installed but its
  shared library cannot load. `_try_import` in `xbooster/shap_scorecard.py` catches
  only `ImportError`/`AttributeError`; `XGBoostError` escapes. Catching `Exception`
  there would let the CatBoost path work without a healthy xgboost.
- Scripts 01, 02 and 04 log params, metrics and artifacts via `scripts/_tracking.py`:
  to the SageMaker MLflow App when `.env` sets `MLFLOW_TRACKING_URI`, else to a local
  SQLite store (`mlflow.db`, artifacts in `mlruns/`). mlflow is pinned to 3.10.1 to
  match the App's server version. Script 03 stays untracked on purpose
  so its "no ML libraries loaded" check means something.
- `registration_data_*` and `ato_data_800K_full` are mostly string / entity
  columns (IP, email, user agent, session). They need feature engineering or
  per-entity aggregation before they fit a tabular scorecard, so no example
  uses them yet.

SageMaker deployment (BYOC image, ModelBuilder local mode, batch transform, model-server matrix) lives in
`sagemaker-ai/`; see `sagemaker-ai/README.md`.

Offline reading: the CompileML repo (docs/, examples/ notebooks, src/) is cloned at
`.tmp/CompileML` (gitignored). The four example notebooks are self-contained and offline.

Study notebook with the same pipeline, explanations and plots, pre-executed:
`notebooks/compileml_insurance_fraud.ipynb`.
