# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "botocore[crt]", "boto3"]
# ///
"""Serve the compileml-scorer BYOC container as a SageMaker LOCAL endpoint (Mode.LOCAL_CONTAINER).

The same custom container run through SDK v3 ModelBuilder's local mode, with the compileml
artifact at /opt/ml/model; swapping to Mode.SAGEMAKER_ENDPOINT deploys the identical image
serverless. The image is built locally (not in ECR) and ModelBuilder would pull the serving
image, so this runs with SM_OFFLINE=1 (default): it stubs the SDK's account, role, and
image-pull calls (see sagemaker_offline), needing no account. On an ARM Mac the amd64 image
runs under emulation, so the first /ping is slow.

model_server=MMS (default; MODEL_SERVER=TORCHSERVE also works) is the generic
serve/ping/invocations runner. Unlike TORCHSERVE it mounts <model_path>/code (not <model_path>)
at /opt/ml/model, so the artifact is staged in both places. ModelBuilder still needs a model
or an inference_spec to build; the spec's prepare() is a no-op because staging is done up
front. The MMS invoke path wraps the response body in a one-element list.

Usage:
  uv run python sagemaker-ai/local_endpoint.py
  MODEL_IMAGE=<tag> SAGEMAKER_ROLE_ARN=<any-arn> ARTIFACT=<path.json> SM_OFFLINE=0 ... for the native path
"""

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.utils.types import ModelServer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# Offline only: with no account there is no STS or IAM to reach, so stub the SDK's
# account, role, and image-pull calls. SM_OFFLINE=0 leaves the SDK untouched.
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
INPUT_CSV = ROOT / "data" / "Insurance_FraudulentAutoInsuranceClaims_2K_coldstart.csv"

# Local runners differ in what they mount at /opt/ml/model: TorchServe mounts <model_path>,
# MMS mounts <model_path>/code. Stage the artifact in both (absolute path: a relative path
# is read as a Docker volume name) so MODEL_SERVER=TORCHSERVE or MMS both find it.
staging = tempfile.mkdtemp(prefix="compileml-endpoint-")
for d in (Path(staging), Path(staging) / "code"):
    d.mkdir(exist_ok=True)
    shutil.copy(ARTIFACT, d / ARTIFACT.name)
MODEL_SERVER = ModelServer[os.environ.get("MODEL_SERVER", "MMS")]


class StagedArtifactSpec(InferenceSpec):
    """Satisfies ModelBuilder's 'model or inference_spec' rule; the artifact is already staged."""

    def prepare(self, model_dir: str, *args, **kwargs) -> None:
        """Prepare step."""

    def load(self, model_dir: str):  # SDK-handler hook, unused with a BYOC image
        """Load artifact."""
        from compileml.runtime import load_artifact

        return load_artifact(Path(model_dir) / ARTIFACT.name)

    def invoke(self, input_object, model):  # SDK-handler hook, unused with a BYOC image
        """Invoke endpoint."""
        from compileml.runtime import decide

        names = model["features"]["names"]
        return decide(model, [input_object.get(n) for n in names], top_k=3)


# one claim, the container's request contract (a record it scores)
names = json.loads(ARTIFACT.read_text())["features"]["names"]
with INPUT_CSV.open(newline="") as fh:
    first = next(csv.DictReader(fh))
sample = {k: (float(first[k]) if first[k].strip() else None) for k in names}
schema = SchemaBuilder(
    sample_input=sample,
    sample_output={
        "band": "G01",
        "latent_int": 0,
        "pd": 0.05,
        "reasons": [],
        "artifact_hash": "0" * 12,
    },
)


builder = ModelBuilder(
    inference_spec=StagedArtifactSpec(),
    image_uri=IMAGE,
    model_server=MODEL_SERVER,  # MMS: the generic serve/ping/invocations runner; TORCHSERVE also works
    model_path=staging,
    schema_builder=schema,
    role_arn=ROLE,
    mode=Mode.LOCAL_CONTAINER,  # Switch here to SAGEMAKER_ENDPOINT
    dependencies={"auto": False},
    env_vars={"ARTIFACT_PATH": f"/opt/ml/model/{ARTIFACT.name}"},
)
builder.build()
endpoint = builder.deploy_local(
    endpoint_name="compileml-scorer-endpoint",
    container_timeout_in_seconds=1200,
)
try:
    resp = endpoint.invoke(body=json.dumps(sample), content_type="application/json")
    body = json.loads(resp.body.read())
    if isinstance(body, list) and len(body) == 1:  # MMS path: the SDK wraps the response in a list
        body = body[0]
    print("prediction:", json.dumps(body))
finally:  # Tear down the endpoint and running containers
    endpoint.delete()
    import docker

    for c in docker.from_env().containers.list(
        all=True
    ):  # delete() leaves the containers behind
        ports = c.attrs.get("HostConfig", {}).get("PortBindings") or {}
        if IMAGE in (c.image.tags or []) or "8080/tcp" in ports or "algo-1" in c.name:
            c.remove(force=True)
