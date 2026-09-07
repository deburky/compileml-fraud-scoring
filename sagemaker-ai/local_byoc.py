"""Run the BYOC scorer image through SageMaker SDK v3 ModelBuilder in local container mode.

    ModelBuilder(
        inference_spec=CompileMLSpec(...),
        model_server=ModelServer.TORCHSERVE,
        schema_builder=SchemaBuilder(sample_input, sample_output),
        image_uri="compileml-scorer:local",
        mode=Mode.LOCAL_CONTAINER,
    )

Why inference_spec and not model=: ModelBuilder(model=obj) maps obj to a framework by its class
name (torch / xgb / keras / tensorflow / sklearn) and raises for anything else, image_uri or
not. The compileml artifact is a JSON dict, so the v3 hook for it is InferenceSpec. With a
BYOC image only prepare() has an effect (it stages the artifact into model_path); load() and
invoke() are what the SDK's own handler would run inside an AWS DLC and never execute here.

Offline: SM_OFFLINE=1 (the default here) applies sagemaker_offline.use_local_stubs(), which
neutralises the three control-plane calls the SDK makes even in local mode: execution-role
validation (IAM SimulatePrincipalPolicy inside build()), the default S3 bucket lookup (STS),
and the image pull. With the local image tag nothing then needs the network. SM_OFFLINE=0
keeps the SDK intact (needs AWS credentials and, for an ECR image, a pull).

Switching model servers (--model-server or MODEL_SERVER=...): with a BYOC image the value
does not change what serves (our serve.py does), only how the SDK stages and launches. Three
things have to line up, all handled below:
  1. mount: the TorchServe runner mounts <model_path> at /opt/ml/model, the MMS runner mounts
     <model_path>/code. CompileMLSpec.prepare() stages the artifact in both.
  2. response: the MMS invoke path wraps the body in a one-element list. as_json() unwraps it.
  3. launcher: only TORCHSERVE and MMS work with this image in local mode. TENSORFLOW_SERVING,
     DJL_SERVING, TGI, TRITON reject our sample input/output at build; TEI asks Docker for a
     GPU; SMD, VLLM, SGLANG, VLLM_OMNI, LLAMACPP have no local launcher in sagemaker 3.21
     (see model_servers/). SMD works through model_servers/smd_local.py, with our image or
     with the real SageMaker Distribution image (IMAGE_URI=public.ecr.aws/sagemaker/sagemaker-distribution:3.2.0-cpu),
     in which case the spec stages code/inference.py (model_servers/smd_handler.py) and a
     vendored compileml. On a real endpoint SageMaker runs `serve` on 8080 with /opt/ml/model
     regardless of the value.

Run:  uv run python sagemaker-ai/local_byoc.py [--model-server TORCHSERVE|MMS] [--artifact path]
      MODEL_SERVER=MMS uv run python sagemaker-ai/local_byoc.py
"""

from __future__ import annotations

import argparse
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
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))
DEFAULT_ARTIFACT = ROOT / "artifacts" / "insurance_fraud.json"
INPUT_CSV = ROOT / "data" / "Insurance_FraudulentAutoInsuranceClaims_2K_coldstart.csv"


def load_dotenv() -> None:
    """Load dotenv from root."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_dotenv()
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ACCOUNT = "823613469927"
IMAGE_URI = os.environ.get("IMAGE_URI", "compileml-scorer:local")
# Validated by the SDK (IAM SimulatePrincipalPolicy) unless SM_OFFLINE stubs that out.
# Servers whose local-container launcher works with this image (measured, see docstring).
LOCAL_OK = {ModelServer.TORCHSERVE, ModelServer.MMS, ModelServer.SMD}  # SMD via model_servers/smd_local.py
# What each local runner mounts at /opt/ml/model, relative to model_path.
MOUNTS = {ModelServer.TORCHSERVE: ".", ModelServer.MMS: "code", ModelServer.SMD: "."}
ROLE_ARN = os.environ.get(
    "SAGEMAKER_ROLE_ARN",
    f"arn:aws:iam::{ACCOUNT}:role/service-role/AmazonSageMaker-ExecutionRole-20260529T194818",
)


class CompileMLSpec(InferenceSpec):
    """compileml artifact for ModelBuilder. BYOC: only prepare() matters."""

    def __init__(self, artifact_src: str) -> None:
        self.artifact_src = artifact_src
        self.artifact_name = Path(artifact_src).name

    def prepare(self, model_dir: str, *args, **kwargs) -> None:
        """Build time, on the host: stage the artifact where the container will find it.

        TorchServe's local runner mounts <model_path> at /opt/ml/model, MMS's mounts
        <model_path>/code there. Stage in both so --model-server TORCHSERVE and MMS both work.
        """
        for d in (Path(model_dir), Path(model_dir) / "code"):
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.artifact_src, d / self.artifact_name)

    # -------------------------------------------------------------------------------
    # SDK-handler hooks: run only inside an AWS DLC with the SDK's inference.py
    # -------------------------------------------------------------------------------
    def load(self, model_dir: str):
        from compileml.runtime import load_artifact

        return load_artifact(Path(model_dir) / self.artifact_name)

    def invoke(self, input_object, model):
        from compileml.runtime import decide

        names = model["features"]["names"]
        rows = (
            input_object["instances"]
            if isinstance(input_object, dict)
            else [input_object]
        )
        return {
            "predictions": [
                decide(model, [r.get(n) for n in names], top_k=3) for r in rows
            ]
        }


def sample_payload(artifact_path: Path, n: int = 3) -> tuple[dict, dict]:
    """Prepare sample payload."""
    names = json.loads(artifact_path.read_text())["features"]["names"]
    with INPUT_CSV.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [next(reader) for _ in range(n)]
    instances = [
        {k: (float(r[k]) if r[k].strip() else None) for k in names} for r in rows
    ]
    sample_input = {"instances": instances, "explain": True, "top_k": 3}
    sample_output = {
        "predictions": [{"band": "G01", "latent_int": 0, "pd": 0.05, "reasons": []}],
        "artifact_hash": "0" * 12,
    }
    return sample_input, sample_output


def build_model_builder(
    artifact_path: Path, model_server: ModelServer, mode: Mode
) -> ModelBuilder:
    """Build model builder."""
    if os.environ.get("SM_OFFLINE", "1") == "1" and mode == Mode.LOCAL_CONTAINER:
        from sagemaker_offline import use_local_stubs

        use_local_stubs()
    spec: InferenceSpec = CompileMLSpec(str(artifact_path))
    if model_server == ModelServer.SMD and mode == Mode.LOCAL_CONTAINER:
        sys.path.insert(0, str(HERE / "model_servers"))
        from smd_local import SMD_IMAGE, enable_smd_local_mode, stage_smd_code

        enable_smd_local_mode()  # 3.21 has no local launcher for SMD
        if IMAGE_URI == SMD_IMAGE or "sagemaker-distribution" in IMAGE_URI:
            # the real SageMaker Distribution image: it needs code/inference.py (the handler)
            # and the compileml package next to it; our BYOC image needs neither.
            class SMDSpec(CompileMLSpec):
                def prepare(self, model_dir: str, *args, **kwargs) -> None:
                    stage_smd_code(model_dir, self.artifact_src)

            spec = SMDSpec(str(artifact_path))
    sample_input, sample_output = sample_payload(artifact_path)
    return ModelBuilder(
        inference_spec=spec,
        model_server=model_server,
        schema_builder=SchemaBuilder(
            sample_input=sample_input, sample_output=sample_output
        ),
        image_uri=IMAGE_URI,
        mode=mode,
        model_path=tempfile.mkdtemp(prefix="compileml-mb-"),
        role_arn=ROLE_ARN,
        dependencies={"auto": False},  # skip the pickle dependency detector
        env_vars={"ARTIFACT_PATH": f"/opt/ml/model/{artifact_path.name}"},
    )

    # So here's the dilemma.
    # When setting model server to MMS, I get this. With TORCHSERVE works just fine.
    # Pinging local endpoint.. and then never starts. With TORCHSERVE it's less than 5 sec.


def as_json(body):
    """Read as JSON. LocalEndpoint.invoke() returns a BytesIO of json.dumps(deserialized);
    on the MMS path the deserialized value is a one-element list, so unwrap after parsing."""
    if hasattr(body, "read"):
        body = body.read()
    if isinstance(body, (bytes, bytearray)):
        body = body.decode()
    if isinstance(body, str):
        body = json.loads(body)
    if isinstance(body, list) and len(body) == 1 and isinstance(body[0], dict):
        body = body[0]
    return body


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-server",
        default=os.environ.get("MODEL_SERVER", "TORCHSERVE"),
        choices=[m.name for m in ModelServer],
        help="TORCHSERVE or MMS work locally with this image; the rest are listed for the matrix",
    )
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = ap.parse_args(argv)

    server = ModelServer[args.model_server]
    if server not in LOCAL_OK:
        print(
            f"warning: {server.name} is not known to work in local mode with this image; "
            f"expected to work: {sorted(m.name for m in LOCAL_OK)}"
        )
    mb = build_model_builder(args.artifact, server, Mode.LOCAL_CONTAINER)
    mb.build()
    if server == ModelServer.SMD:
        req = Path(mb.model_path) / "code" / "requirements.txt"  # SDK writes an empty one; the SMD
        if req.exists() and req.stat().st_size == 0:              # image would micromamba-install it
            req.unlink()
    print(f"model_server={server.name}: local runner mounts model_path/{MOUNTS.get(server, '?')} at /opt/ml/model")
    print("staged model_path:", mb.model_path)
    for p in sorted(Path(mb.model_path).rglob("*")):
        print("   ", p.relative_to(mb.model_path))

    endpoint = mb.deploy_local(
        endpoint_name="compileml-byoc-local", container_timeout_in_seconds=300
    )
    try:
        sample_input, _ = sample_payload(args.artifact)
        resp = endpoint.invoke(body=sample_input)
        print("response:", json.dumps(as_json(resp.body), indent=2))
    finally:
        endpoint.delete()


if __name__ == "__main__":
    main(sys.argv[1:])
