"""CustomOrchestrator on the SageMaker Distribution (SMD) image, run locally.

The SDK's own SMD flow, made visible:

  1. You write a CustomOrchestrator: one class with handle(data) -> response. `self.client` is a
     ready-made sagemaker-runtime boto3 client, because the point of an orchestrator is to sit in
     front of *other* endpoints (call a champion and a challenger, combine, route) rather than to
     host a framework model. Here handle() scores the compileml artifact itself and, when
     CHALLENGER_ENDPOINT is set, also fans out to that endpoint through self.client.
  2. ModelBuilder cloudpickles (orchestrator, schema_builder) into code/serve.pkl and copies its
     generic handler, custom_execution_inference.py, to code/inference.py. That handler unpickles
     serve.pkl at import time and forwards every request to orchestrator.handle(request.body).
  3. The SMD image's Tornado server imports inference.handler and serves it on 8080.

ModelBuilder refuses to do step 2 outside Mode.SAGEMAKER_ENDPOINT ("Custom orchestrator
deployment is only supported for SageMaker Endpoint Mode"), so this script calls the same two
SDK functions it would call, save_pkl() and prepare_for_smd(), then runs the SMD image with the
SMD env vars. Nothing here is a reimplementation; the handler that runs is the SDK's.

One mismatch, measured: sagemaker 3.21 (host) writes code/metadata.json with a plain SHA-256 of
serve.pkl and passes no key, while the SMD 3.2.0 image bundles sagemaker 2.245, whose handler
verifies an HMAC-SHA256 keyed by SAGEMAKER_SERVE_SECRET_KEY and crashes at import when the
variable is missing. So the v3 SDK's own orchestrator flow does not work against this image
until either side updates. stage() below re-signs metadata.json the v2 way and the container
gets the key, which is exactly what SDK v2 used to do.

Run (needs the SMD image locally, see README):
    uv run python sagemaker-ai/model_servers/custom_orchestrator.py
Cloud (not run here): deploy_cloud() below is the ModelBuilder call for a real endpoint.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path

import docker
import requests
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.detector.pickler import save_pkl
from sagemaker.serve.model_server.smd.prepare import prepare_for_smd
from sagemaker.serve.spec.inference_base import CustomOrchestrator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from smd_local import SMD_ENV, SMD_IMAGE, stage_smd_code

ARTIFACT = ROOT / "artifacts" / "insurance_fraud.json"


class FraudOrchestrator(CustomOrchestrator):
    """Scores claims with the compileml artifact; optionally fans out to a challenger endpoint."""

    def __init__(
        self, artifact_name: str, challenger_endpoint: str | None = None
    ) -> None:
        super().__init__()  # sets self._client = None; the boto3 client is created lazily
        self.artifact_name = artifact_name
        self.challenger_endpoint = challenger_endpoint
        self._artifact = None  # loaded on first request inside the container

    def _load(self):
        if self._artifact is None:
            from compileml.runtime import (
                load_artifact,  # vendored under /opt/ml/model/code
            )

            base = Path(
                os.environ.get("SAGEMAKER_INFERENCE_BASE_DIRECTORY", "/opt/ml/model")
            )
            self._artifact = load_artifact(base / self.artifact_name)
        return self._artifact

    def handle(self, data, context=None):
        """data is the raw request body (bytes). Return a dict; the SMD server writes it as JSON."""
        from compileml.runtime import decide

        art = self._load()
        names = art["features"]["names"]
        body = json.loads(data or b"{}")
        rows = (
            body["instances"]
            if isinstance(body, dict) and "instances" in body
            else [body]
        )
        preds = []
        for row in rows:
            feats = [None if row.get(n) is None else float(row[n]) for n in names]
            d = decide(art, feats, top_k=3)
            preds.append(
                {
                    "band": d["band"],
                    "latent_int": d["latent_int"],
                    "pd": round(d["pd"], 6),
                    "reasons": [r["code"] for r in d["reasons_negative"]],
                }
            )
        out = {
            "predictions": preds,
            "artifact_hash": art["artifact_hash"][:12],
            "orchestrator": type(self).__name__,
        }
        if (
            self.challenger_endpoint
        ):  # the orchestrator part: call another endpoint and attach its answer
            resp = self.client.invoke_endpoint(
                EndpointName=self.challenger_endpoint,
                ContentType="application/json",
                Body=data,
            )
            out["challenger"] = json.loads(resp["Body"].read())
        return out


def sign_v2(code_dir: Path) -> str:
    """Re-sign code/serve.pkl the way sagemaker 2.x does (HMAC-SHA256 with a secret key), so the
    2.245 handler inside the SMD 3.2.0 image accepts a pickle produced by sagemaker 3.21."""
    from sagemaker.core.remote_function.core.serialization import _MetaData

    secret = secrets.token_hex(32)
    digest = hmac.new(
        secret.encode(), (code_dir / "serve.pkl").read_bytes(), hashlib.sha256
    ).hexdigest()
    (code_dir / "metadata.json").write_bytes(_MetaData(digest).to_json())
    return secret


def stage(
    model_path: Path, orchestrator: CustomOrchestrator, schema: SchemaBuilder
) -> tuple[Path, str]:
    """Exactly what ModelBuilder._build_for_smd does, plus our artifact, vendored compileml, v2 signing."""
    stage_smd_code(
        model_path, ARTIFACT
    )  # artifact + code/compileml (+ a handler we overwrite next)
    # ModelBuilder._save_model_inference_spec pickles (orchestrator, schema_builder). The SDK's
    # handler only unpacks the first element, and a 3.21 SchemaBuilder references sagemaker.core.*
    # modules the image's 2.245 SDK does not have, so the schema is left out of the pickle.
    save_pkl(model_path / "code", (orchestrator, None))
    prepare_for_smd(
        model_path=str(model_path),
        shared_libs=[],
        dependencies={"auto": False},
        inference_spec=orchestrator,
    )
    req = (
        model_path / "code" / "requirements.txt"
    )  # empty file from dependencies={"auto": False}
    if req.exists() and req.stat().st_size == 0:
        req.unlink()
    secret = sign_v2(model_path / "code")
    for p in [model_path, *model_path.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)  # the SMD image runs as sagemaker-user
    return model_path, secret


def run_local(
    model_path: Path, sample_input: dict, secret: str, image: str = SMD_IMAGE
) -> dict:
    client = docker.from_env()
    for c in client.containers.list(all=True):
        if "8080/tcp" in (c.attrs.get("HostConfig", {}).get("PortBindings") or {}):
            c.remove(force=True)
    container = client.containers.run(
        image,
        "serve",
        detach=True,
        ports={"8080/tcp": 8080},
        volumes={str(model_path): {"bind": "/opt/ml/model", "mode": "ro"}},
        environment={
            **SMD_ENV,
            "ARTIFACT_PATH": f"/opt/ml/model/{ARTIFACT.name}",
            "SAGEMAKER_SERVE_SECRET_KEY": secret,
        },
    )
    try:
        for _ in range(120):
            try:
                if (
                    requests.get("http://localhost:8080/ping", timeout=2).status_code
                    == 200
                ):
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            print(container.logs().decode()[-3000:])
            raise SystemExit("SMD server never became healthy")
        r = requests.post(
            "http://localhost:8080/invocations", json=sample_input, timeout=60
        )
        r.raise_for_status()
        return r.json()
    finally:
        print("--- container log tail")
        print("\n".join(container.logs().decode().splitlines()[-6:]))
        container.remove(force=True)


def deploy_cloud(
    orchestrator: CustomOrchestrator,
    schema: SchemaBuilder,
    model_path: Path,
    role_arn: str,
    endpoint_name: str,
):
    """The SDK's own path for an orchestrator (not run here): endpoint mode only, SMD image chosen by the SDK."""
    from sagemaker.serve.mode.function_pointers import Mode
    from sagemaker.serve.model_builder import ModelBuilder

    mb = ModelBuilder(
        inference_spec=orchestrator,  # a CustomOrchestrator: ModelBuilder switches to SMD + SAGEMAKER_ENDPOINT itself
        schema_builder=schema,
        model_path=str(
            model_path
        ),  # pre-staged: artifact + vendored compileml (prepare() is not called for orchestrators)
        role_arn=role_arn,
        mode=Mode.SAGEMAKER_ENDPOINT,
        dependencies={"auto": False},
        env_vars={"CHALLENGER_ENDPOINT": ""},
    )
    mb.build()
    return mb.deploy(
        endpoint_name=endpoint_name,
        instance_type="ml.c5.xlarge",
        initial_instance_count=1,
    )


def main() -> None:
    import csv

    names = json.loads(ARTIFACT.read_text())["features"]["names"]
    with (
        ROOT / "data" / "Insurance_FraudulentAutoInsuranceClaims_2K_coldstart.csv"
    ).open(newline="") as fh:
        rows = [next(r) for r in [csv.DictReader(fh)] * 1 for _ in range(3)]
    instances = [
        {k: (float(r[k]) if r[k].strip() else None) for k in names} for r in rows
    ]
    sample_input = {"instances": instances}
    schema = SchemaBuilder(
        sample_input=sample_input,
        sample_output={"predictions": [], "artifact_hash": ""},
    )

    orchestrator = FraudOrchestrator(
        artifact_name=ARTIFACT.name,
        challenger_endpoint=os.environ.get("CHALLENGER_ENDPOINT"),
    )
    model_path, secret = stage(
        Path(tempfile.mkdtemp(prefix="compileml-orchestrator-")), orchestrator, schema
    )
    print("staged", model_path)
    for p in sorted(model_path.rglob("*")):
        if p.is_file() and "compileml/" not in str(p.relative_to(model_path)):
            print("   ", p.relative_to(model_path), f"({p.stat().st_size} B)")
    print("--- code/inference.py is the SDK's generic handler:")
    print(
        "\n".join(
            l
            for l in (model_path / "code" / "inference.py").read_text().splitlines()
            if "custom_orchestrator" in l
            or "def handler" in l
            or "cloudpickle.load" in l
        )
    )

    result = run_local(model_path, sample_input, secret)
    print("--- response from orchestrator.handle() via the SMD image:")
    print(json.dumps(result, indent=2)[:900])


if __name__ == "__main__":
    main()
