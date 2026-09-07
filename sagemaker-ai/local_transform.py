# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "botocore[crt]", "boto3"]
# ///
"""Batch-score a claims CSV with a SageMaker Batch Transform job in LOCAL mode.

A real SageMaker Transform job on the compileml-scorer BYOC container, in local mode
(instance_type="local"): the SDK packs the artifact as model.tar.gz, brings the container up
via docker compose with the tar unpacked at /opt/ml/model, asks GET /execution-parameters,
posts the input and writes <input>.out exactly as a cloud job would (swap instance_type and
point data/output at S3 for the cloud). The whole CSV goes as one payload (split_type=None);
the container answers text/csv in the layout of scripts/03_runtime_batch_score.py, so the
.out file is compared with artifacts/insurance_fraud_decisions.csv when present.

The image is built locally (not in ECR), so this runs with SM_OFFLINE=1 (default): it stubs
the SDK's account and role checks (see sagemaker_offline), needing no account.

Env: MODEL_IMAGE (default compileml-scorer:local), SAGEMAKER_ROLE_ARN (any arn),
     ARTIFACT (default artifacts/insurance_fraud.json), INPUT (default the cold-start CSV),
     SM_OFFLINE=1 to run with no account (default).
Run: uv run python sagemaker-ai/local_transform.py
"""

import csv
import glob
import os
import sys
import tarfile
import tempfile
from pathlib import Path

from sagemaker.core.local import LocalSession
from sagemaker.core.transformer import Transformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# Like the v3 examples, import ModelBuilder before anything local-mode: core/local/image.py
# reads sagemaker.serve.model_builder.DIR_PARAM_NAME without importing that module itself.
from sagemaker.serve.model_builder import ModelBuilder  # noqa: F401

# Offline only: with no account there is no STS or IAM to reach, so stub the SDK's
# account and role checks. SM_OFFLINE=0 leaves the SDK untouched.
if os.environ.get("SM_OFFLINE", "1") == "1":
    from sagemaker_offline import use_local_stubs

    use_local_stubs()

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
IMAGE = os.environ.get("MODEL_IMAGE", "compileml-scorer:local")
ROLE = os.environ.get(
    "SAGEMAKER_ROLE_ARN",
    "arn:aws:iam::823613469927:role/service-role/AmazonSageMaker-ExecutionRole-20260529T194818",
)
ARTIFACT = Path(
    os.environ.get("ARTIFACT", ROOT / "artifacts" / "insurance_fraud.json")
).resolve()
INPUT = Path(
    os.environ.get(
        "INPUT",
        ROOT / "data" / "Insurance_FraudulentAutoInsuranceClaims_2K_coldstart.csv",
    )
).resolve()
EXPECTED = (
    ROOT / "artifacts" / f"{ARTIFACT.stem}_decisions.csv"
)  # written by scripts/03

sess = LocalSession()
sess.config = {"local": {"local_code": True}}

# package the artifact into model.tar.gz (files at the tar root, which SageMaker unpacks
# into /opt/ml/model where serve.py picks up the first *.json)
staging = tempfile.mkdtemp(prefix="compileml-transform-")
tar_path = os.path.join(staging, "model.tar.gz")
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(ARTIFACT, arcname=ARTIFACT.name)

MODEL_NAME = "compileml-scorer-transform"
sess.create_model(
    name=MODEL_NAME,
    role=ROLE,
    # local mode reads primary_container["Environment"] unconditionally, so set it
    container_defs={
        "Image": IMAGE,
        "ModelDataUrl": f"file://{tar_path}",
        "Environment": {"ARTIFACT_PATH": f"/opt/ml/model/{ARTIFACT.name}"},
    },
)

out_dir = tempfile.mkdtemp(prefix="compileml-transform-out-")
transformer = Transformer(
    model_name=MODEL_NAME,
    instance_count=1,
    instance_type="local",
    strategy="SingleRecord",
    output_path=f"file://{out_dir}",
    accept="text/csv",
    max_payload=100,
    sagemaker_session=sess,
)
# job_name pins the run so the SDK does not call the real DescribeModel; wait=False
# keeps it from polling a real DescribeTransformJob (there is none in local mode).
transformer.transform(
    data=f"file://{INPUT}",
    content_type="text/csv",
    split_type=None,
    job_name="compileml-scorer-transform-local",
    wait=False,
)

outs = glob.glob(os.path.join(out_dir, "**", "*.out"), recursive=True)
print("output files:", outs)
if not outs:
    raise SystemExit("no .out produced")
with open(outs[0], newline="") as f:
    rows = list(csv.DictReader(f))
print(f"scored {len(rows)} rows; first 3:")
for r in rows[:3]:
    print("  ", r)
bands = {}
for r in rows:
    bands[r["band"]] = bands.get(r["band"], 0) + 1
print("band counts:", dict(sorted(bands.items())))

if EXPECTED.exists():
    with EXPECTED.open(newline="") as f:
        expected = list(csv.DictReader(f))
    same = len(expected) == len(rows) and all(
        (a["band"], a["latent_int"], a["top_negative_codes"])
        == (b["band"], b["latent_int"], b["top_negative_codes"])
        for a, b in zip(expected, rows)
    )
    print(
        f"matches {EXPECTED.relative_to(ROOT)} (band, latent_int, reason codes) on all rows: {same}"
    )
