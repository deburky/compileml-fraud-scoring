"""Give SageMaker SDK v3 local container mode an SMD launcher.

sagemaker 3.21 builds for ModelServer.SMD (SageMaker Distribution) but its local runner has no
branch for it: LocalContainerMode.create_server() starts nothing and LocalEndpoint.invoke()
raises "Unsupported model server". The SMD serving contract is the plain SageMaker one
(`docker run <image> serve`, port 8080, /ping, /invocations, model at /opt/ml/model) plus two
env vars telling the SageMaker Distribution image which handler to import:

    SAGEMAKER_INFERENCE_CODE_DIRECTORY=/opt/ml/model/code
    SAGEMAKER_INFERENCE_CODE=inference.handler      # async def handler(request) in code/inference.py

That is exactly what the TorchServe launcher already does (minus the env vars), so SMD is routed
through it with the SMD env vars merged in. Works with our BYOC image (which ignores the env)
and with the real sagemaker-distribution-prod image (which honours it).

    from smd_local import enable_smd_local_mode
    enable_smd_local_mode()
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from sagemaker.serve import local_resources
from sagemaker.serve.mode import local_container_mode
from sagemaker.serve.utils.types import ModelServer

SMD_ENV = {
    "SAGEMAKER_INFERENCE_CODE_DIRECTORY": "/opt/ml/model/code",
    "SAGEMAKER_INFERENCE_CODE": "inference.handler",
    "LOCAL_PYTHON": platform.python_version(),
}


def enable_smd_local_mode() -> None:
    """Patch create_server() and LocalEndpoint.invoke() to treat SMD like TorchServe + SMD env."""
    Mode = local_container_mode.LocalContainerMode
    if getattr(Mode, "_smd_patched", False):
        return
    original_create_server = Mode.create_server
    original_invoke = local_resources.LocalEndpoint.invoke

    def create_server(
        self,
        image,
        container_timeout_seconds,
        secret_key,
        container_config,
        ping_fn=None,
        env_vars=None,
        model_path=None,
        jumpstart=False,
    ):
        if self.model_server != ModelServer.SMD:
            return original_create_server(
                self,
                image,
                container_timeout_seconds,
                secret_key,
                container_config,
                ping_fn,
                env_vars,
                model_path,
                jumpstart,
            )
        merged = {**SMD_ENV, **(env_vars if env_vars else self.env_vars or {})}
        self.model_server = (
            ModelServer.TORCHSERVE
        )  # borrow its docker-run launcher and wait loop
        try:
            return original_create_server(
                self,
                image,
                container_timeout_seconds,
                secret_key,
                container_config,
                ping_fn,
                merged,
                model_path,
                jumpstart,
            )
        finally:
            self.model_server = ModelServer.SMD

    def invoke(
        self, body, content_type="application/json", accept="application/json", **kwargs
    ):
        if self.model_server != ModelServer.SMD:
            return original_invoke(self, body, content_type, accept, **kwargs)
        self.model_server = (
            ModelServer.TORCHSERVE
        )  # same /invocations POST, same unwrapping
        try:
            return original_invoke(self, body, content_type, accept, **kwargs)
        finally:
            self.model_server = ModelServer.SMD

    Mode.create_server = create_server
    local_resources.LocalEndpoint.invoke = invoke
    Mode._smd_patched = True


# -------------------------------------------------------------------------------
# Staging for the real SageMaker Distribution image
# -------------------------------------------------------------------------------
SMD_IMAGE = "public.ecr.aws/sagemaker/sagemaker-distribution:3.2.0-cpu"  # == 885854791233...sagemaker-distribution-prod:3.2.0-cpu


def stage_smd_code(
    model_dir: str | os.PathLike, artifact_src: str | os.PathLike
) -> Path:
    """Lay out /opt/ml/model for the SMD image: the artifact at the root, code/inference.py (the
    handler in smd_handler.py) and a vendored copy of the compileml package so no install is needed.
    Everything is made world-readable: the image runs as the non-root sagemaker-user."""
    import compileml

    model_dir = Path(model_dir)
    code = model_dir / "code"
    code.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact_src, model_dir / Path(artifact_src).name)
    shutil.copy2(Path(__file__).with_name("smd_handler.py"), code / "inference.py")
    pkg_src = Path(compileml.__file__).parent
    if (code / "compileml").exists():
        shutil.rmtree(code / "compileml")
    shutil.copytree(
        pkg_src,
        code / "compileml",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for p in [model_dir, *model_dir.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    return model_dir
