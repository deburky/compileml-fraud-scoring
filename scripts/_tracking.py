"""Local MLflow tracking for the example scripts.

By default runs land in ./mlflow.db (SQLite, gitignored; MLflow 3.x refuses the
plain file store) with artifacts under ./mlruns. Browse them with `make mlflow-ui`.

To log to the SageMaker MLflow App instead, put the App ARN and profile in ./.env
(gitignored; `make arn` prints the ARN). It is loaded here without overriding
variables already exported, and the sagemaker-mlflow plugin resolves the ARN:

    AWS_PROFILE=aws-free-tier
    AWS_DEFAULT_REGION=us-east-1
    MLFLOW_TRACKING_URI=arn:aws:sagemaker:...:mlflow-app/app-...

Set MLFLOW_LOCAL=1 (or remove MLFLOW_TRACKING_URI from .env) to force the local
SQLite store, e.g. when offline:

    MLFLOW_LOCAL=1 uv run python scripts/01_insurance_fraud_end_to_end.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
EXPERIMENT = "compileml-tests"


def start(run_name: str, **params):
    """Open an MLflow run in the local file store; use as a context manager."""
    local = f"sqlite:///{ROOT / 'mlflow.db'}"
    remote = os.environ.get("MLFLOW_TRACKING_URI")
    uri = local if os.environ.get("MLFLOW_LOCAL") == "1" or not remote else remote
    print(f"[mlflow] tracking -> {'local sqlite' if uri == local else uri}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    run = mlflow.start_run(run_name=run_name)
    if params:
        mlflow.log_params(params)
    return run


def log_validation(report: dict, prefix: str = "validate") -> None:
    """Log the compileml 10-check report: one 0/1 metric per check plus the JSON."""
    mlflow.log_metric(f"{prefix}_all_pass", float(report["all_pass"]))
    for name, chk in report["checks"].items():
        mlflow.log_metric(f"{prefix}_{name}", -1.0 if chk.get("skipped") else float(chk["pass"]))
    mlflow.log_dict(report, f"{prefix}_report.json")
