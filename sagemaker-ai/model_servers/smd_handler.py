"""SageMaker Distribution (SMD) inference handler for the compileml artifact.

Staged as /opt/ml/model/code/inference.py and named by SAGEMAKER_INFERENCE_CODE=inference.handler.
The image's Tornado server imports this module once (so the artifact loads once), puts the code
directory on sys.path (so the vendored compileml package next to this file imports), and calls
handler(request) per POST /invocations in a thread. request is a tornado HTTPServerRequest:
request.body is bytes. A dict return value is written as JSON.

Same request and response shapes as container/serve.py, so the notebook comparisons hold.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))  # vendored compileml (stdlib-only runtime)
from compileml.runtime import decide, load_artifact

MODEL_DIR = CODE_DIR.parent
_path = os.environ.get("ARTIFACT_PATH") or str(next(MODEL_DIR.glob("*.json")))
ARTIFACT = load_artifact(_path)
NAMES = ARTIFACT["features"]["names"]
print(
    f"smd handler: loaded {_path} hash={ARTIFACT['artifact_hash'][:16]} features={NAMES}",
    flush=True,
)


def _decision(row, *, explain: bool, top_k: int) -> dict:
    feats = [row.get(n) for n in NAMES] if isinstance(row, dict) else list(row)
    d = decide(
        ARTIFACT,
        [None if v is None else float(v) for v in feats],
        explain=explain,
        top_k=top_k if explain else None,
    )
    out = {"band": d["band"], "latent_int": d["latent_int"], "pd": round(d["pd"], 6)}
    if explain:
        out["reasons"] = [
            {"code": r["code"], "impact_int": r["impact_int"], "message": r["message"]}
            for r in d["reasons_negative"]
        ]
    return out


def handler(request):
    """Sync handler: Tornado runs it in an executor. Return a dict -> JSON response."""
    body = json.loads(request.body or b"{}")
    explain = bool(body.get("explain", True)) if isinstance(body, dict) else True
    top_k = int(body.get("top_k", 3)) if isinstance(body, dict) else 3
    if isinstance(body, dict) and "instances" in body:
        return {
            "predictions": [
                _decision(r, explain=explain, top_k=top_k) for r in body["instances"]
            ],
            "artifact_hash": ARTIFACT["artifact_hash"][:12],
        }
    out = _decision(body, explain=explain, top_k=top_k)
    out["artifact_hash"] = ARTIFACT["artifact_hash"][:12]
    return out
