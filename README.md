# compileml-fraud-scoring

Experiments with [CompileML](https://github.com/orgoca/CompileML) on the AWS Fraud Detector
sample data: distil a CatBoost teacher into a white-box scorecard, compile it to a JSON decision
artifact that a stdlib-only runtime scores with reason codes, and serve that artifact through
SageMaker, locally and offline first.

## Layout

| path | what is there |
| --- | --- |
| `scripts/` | the experiments (`01` insurance end-to-end, `02` transaction tuning, `03` stdlib batch scoring, `04` xbooster vs compileml) and `FINDINGS.md`, the notes on compileml 0.4.3. See `scripts/README.md`. |
| `notebooks/compileml_insurance_fraud.ipynb` | the insurance pipeline as a pre-executed study notebook |
| `sagemaker-ai/` | serving the artifact with SageMaker SDK v3: BYOC image (`container/`), `deploy_modelbuilder.ipynb` (ModelBuilder and core `Endpoint` resources, local), local Batch Transform, the model-server matrix and SMD/CustomOrchestrator examples (`model_servers/`). Runs with no AWS reachable. See `sagemaker-ai/README.md`. |
| `cloudformation/mlflow-app/` | SAM stack for a SageMaker MLflow App the scripts can log to (`make deploy`, `make url`) |
| `data/` | AWS Fraud Detector samples; only the two files the scripts read are committed (see `data/README.md`) |
| `artifacts/` | outputs of the scripts (compiled artifact, scorecards, decisions); generated, not committed |

## Setup

```bash
uv sync                                   # Python 3.12, deps from uv.lock
cp .env.example .env                      # AWS_PROFILE, region, optional MLFLOW_TRACKING_URI
uv run python scripts/01_insurance_fraud_end_to_end.py
uv run python scripts/03_runtime_batch_score.py
```

The scripts log to the MLflow App when `.env` sets `MLFLOW_TRACKING_URI`, otherwise to a local
SQLite store (`make mlflow-ui`). `MLFLOW_LOCAL=1` forces local. Offline: `uv run --offline --no-sync`.

## Serving (SageMaker, local mode)

```bash
make image                                # build the BYOC scorer image, compileml-scorer:local
uv run python sagemaker-ai/local_byoc.py --model-server TORCHSERVE   # or MMS, SMD
uv run python sagemaker-ai/local_transform.py                        # local Batch Transform, 2,000 claims
uv run jupyter nbconvert --to webpdf sagemaker-ai/deploy_modelbuilder.ipynb --output-dir .tmp/pdf
```

Everything under `sagemaker-ai/` runs with `SM_OFFLINE=1` by default, which stubs the three
control-plane calls the SDK makes even in local mode (see `sagemaker-ai/sagemaker_offline.py`).
`make image-push` publishes the image to ECR for a real endpoint.

## Notes

- The compileml runtime is standard library only; the serving image installs `compileml` with
  `--no-deps` and carries no numpy, scikit-learn or CatBoost.
- `sagemaker-ai/README.md` records what was measured about SDK 3.21: `model_server` semantics
  for BYOC, the local-mode quirks, and the SMD / CustomOrchestrator incompatibility reported in
  [aws/sagemaker-python-sdk#6200](https://github.com/aws/sagemaker-python-sdk/issues/6200).
