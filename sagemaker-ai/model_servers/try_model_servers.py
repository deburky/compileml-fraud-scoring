"""Does `model_server` change what gets deployed when image_uri is our own BYOC image?

For every ModelServer value, with the compileml fraud artifact and the compileml-scorer image:
build, deploy_local, inspect the container Docker actually started (command, ports, mounts,
SageMaker env vars), invoke with three claims, tear down. Prints one block per server and
writes model_server_matrix.json next to this file.

Offline: SM_OFFLINE=1 (default) applies sagemaker_offline.use_local_stubs() so no IAM/STS
call or image pull happens; the local image tag is used as-is.

Run:  uv run python sagemaker-ai/try_model_servers.py            # all servers
      uv run python sagemaker-ai/try_model_servers.py TORCHSERVE MMS
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import docker
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.utils.types import ModelServer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # sagemaker-ai/model_servers -> repo root
sys.path.insert(0, str(HERE.parent))  # sagemaker_offline lives in sagemaker-ai/
ARTIFACT = ROOT / "artifacts" / "insurance_fraud.json"
INPUT_CSV = ROOT / "data" / "Insurance_FraudulentAutoInsuranceClaims_2K_coldstart.csv"
IMAGE_URI = os.environ.get("IMAGE_URI", "compileml-scorer:local")
ROLE_ARN = os.environ.get(
    "SAGEMAKER_ROLE_ARN",
    "arn:aws:iam::823613469927:role/service-role/AmazonSageMaker-ExecutionRole-20260529T194818",
)
CONTAINER_TIMEOUT = int(os.environ.get("CONTAINER_TIMEOUT", "120"))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


# -------------------------------------------------------------------------------
# The model
# -------------------------------------------------------------------------------
class CompileMLSpec(InferenceSpec):
    """compileml artifact for ModelBuilder. With the BYOC image only prepare() has an effect."""

    def __init__(self, artifact_src: str) -> None:
        self.artifact_src = artifact_src
        self.artifact_name = Path(artifact_src).name

    def prepare(self, model_dir: str, *args, **kwargs) -> None:
        # TorchServe/TF mount <model_path> at /opt/ml/model, MMS mounts <model_path>/code:
        # stage the artifact in both so every runner finds it.
        for d in (Path(model_dir), Path(model_dir) / "code"):
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.artifact_src, d / self.artifact_name)

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


def sample_payload(n: int = 3) -> tuple[dict, dict]:
    names = json.loads(ARTIFACT.read_text())["features"]["names"]
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


# -------------------------------------------------------------------------------
# Local mode patch
# -------------------------------------------------------------------------------
if os.environ.get("SM_OFFLINE", "1") == "1":
    from sagemaker_offline import use_local_stubs

    use_local_stubs()


# -------------------------------------------------------------------------------
# Docker inspection
# -------------------------------------------------------------------------------
def containers_for_image():
    """Containers from our image or on our port."""
    client = docker.from_env()
    out = []
    for c in client.containers.list(all=True):
        ports = c.attrs.get("HostConfig", {}).get("PortBindings") or {}
        if (
            IMAGE_URI in (c.image.tags or [])
            or "8080/tcp" in ports
            or "algo-1" in c.name
        ):
            out.append(c)
    return out


def describe(c) -> dict:
    a = c.attrs
    env = dict(kv.split("=", 1) for kv in a["Config"]["Env"] if "=" in kv)
    return {
        "name": c.name,
        "cmd": " ".join(a["Config"]["Cmd"] or []),
        "ports": {
            k: v[0]["HostPort"]
            for k, v in (a["HostConfig"].get("PortBindings") or {}).items()
        },
        "mounts": [
            f"{Path(m['Source']).name}->{m['Destination']}" for m in a.get("Mounts", [])
        ],
        "sm_env": {
            k: v
            for k, v in env.items()
            if k.startswith(
                ("SAGEMAKER_", "ARTIFACT", "HF_", "OPTION_", "TS_", "MODEL_", "SM_")
            )
        },
    }


def cleanup() -> None:
    for c in containers_for_image():
        try:
            c.remove(force=True)
        except Exception:
            pass


def as_json(body):
    if hasattr(body, "read"):
        body = body.read()
    if isinstance(body, (bytes, bytearray)):
        body = body.decode()
    if isinstance(body, str):
        body = json.loads(body)
    if (
        isinstance(body, list) and len(body) == 1 and isinstance(body[0], dict)
    ):  # MMS path
        body = body[0]
    return body


# -------------------------------------------------------------------------------
# One server
# -------------------------------------------------------------------------------
def try_server(name: str) -> dict:
    row = {
        "server": name,
        "build": "",
        "staged": [],
        "containers": [],
        "invoke": "",
        "seconds": 0.0,
    }
    t0 = time.perf_counter()
    cleanup()
    sample_input, sample_output = sample_payload()
    try:
        mb = ModelBuilder(
            inference_spec=CompileMLSpec(str(ARTIFACT)),
            model_server=ModelServer[name],
            schema_builder=SchemaBuilder(
                sample_input=sample_input, sample_output=sample_output
            ),
            image_uri=IMAGE_URI,
            mode=Mode.LOCAL_CONTAINER,
            model_path=tempfile.mkdtemp(prefix=f"compileml-{name.lower()}-"),
            role_arn=ROLE_ARN,
            dependencies={"auto": False},
            env_vars={"ARTIFACT_PATH": f"/opt/ml/model/{ARTIFACT.name}"},
        )
        mb.build()
        row["build"] = "ok"
        row["staged"] = sorted(
            str(p.relative_to(mb.model_path))
            for p in Path(mb.model_path).rglob("*")
            if p.is_file()
        )
    except Exception as exc:
        row["build"] = f"FAIL {type(exc).__name__}: {str(exc)[:200]}"
        row["seconds"] = round(time.perf_counter() - t0, 1)
        return row

    endpoint = None
    try:
        endpoint = mb.deploy_local(
            endpoint_name=f"try-{name.lower()}",
            container_timeout_in_seconds=CONTAINER_TIMEOUT,
        )
        row["containers"] = [describe(c) for c in containers_for_image()]
        body = as_json(endpoint.invoke(body=sample_input).body)
        preds = body["predictions"]
        row["invoke"] = "ok " + ", ".join(f"{p['band']}/{p['pd']}" for p in preds)
    except Exception as exc:
        row["containers"] = row["containers"] or [
            describe(c) for c in containers_for_image()
        ]
        row["invoke"] = f"FAIL {type(exc).__name__}: {str(exc)[:200]}"
    finally:
        try:
            if endpoint:
                endpoint.delete()
        except Exception:
            pass
        cleanup()
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


HARD_TIMEOUT = (
    CONTAINER_TIMEOUT + 180
)  # the SDK's ping loop does not always honour its timeout


def try_server_isolated(name: str) -> dict:
    """Run try_server in a subprocess so a wedged SDK ping loop cannot stall the matrix."""
    result_file = Path(tempfile.mkstemp(suffix=".json")[1])
    cmd = [sys.executable, "-W", "ignore", __file__, "--one", name, str(result_file)]
    try:
        subprocess.run(
            cmd,
            check=False,
            timeout=HARD_TIMEOUT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result_file.exists() and result_file.stat().st_size:
            return json.loads(result_file.read_text())
        return {
            "server": name,
            "build": "FAIL (subprocess produced no result)",
            "staged": [],
            "containers": [],
            "invoke": "",
            "seconds": HARD_TIMEOUT,
        }
    except subprocess.TimeoutExpired:
        cs = [describe(c) for c in containers_for_image()]
        return {
            "server": name,
            "build": "ok (assumed)",
            "staged": [],
            "containers": cs,
            "invoke": f"TIMEOUT after {HARD_TIMEOUT}s (ping never healthy)",
            "seconds": HARD_TIMEOUT,
        }
    finally:
        cleanup()
        result_file.unlink(missing_ok=True)


def main(names: list[str]) -> None:
    if (
        len(names) == 3 and names[0] == "--one"
    ):  # child: one server, write the row, exit
        Path(names[2]).write_text(json.dumps(try_server(names[1])))
        return
    names = names or [m.name for m in ModelServer]
    rows = [try_server_isolated(n) for n in names]
    out = HERE / "model_server_matrix.json"
    out.write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(
            f"\n== {r['server']}  ({r['seconds']}s)\n  build:   {r['build']}\n  staged:  {r['staged']}"
        )
        for c in r["containers"]:
            print(
                f"  docker:  cmd={c['cmd']!r} ports={c['ports']} mounts={c['mounts']}\n           env={c['sm_env']}"
            )
        print(f"  invoke:  {r['invoke']}")
    print("\nwritten:", out)


if __name__ == "__main__":
    main(sys.argv[1:])
