"""SageMaker BYOC inference server for a compileml decision artifact.

Contract: GET /ping (200 once the artifact is loaded) and POST /invocations.
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html

The artifact is looked up at ARTIFACT_PATH, else the first *.json directly under
/opt/ml/model (where SageMaker unpacks model.tar.gz, and where ModelBuilder mounts
model_path in local container mode).

Request body (Content-Type application/json), either shape:
  {"feature_a": 1.0, "feature_b": 2.0, ...}                 -> one decision
  {"instances": [{...}, {...}], "explain": true, "top_k": 3} -> {"predictions": [...]}
A row may also be a list of floats in the artifact's feature order. Missing or null
values are passed through as None, which the runtime treats as missing.

Content-Type text/csv (Batch Transform): a CSV with a header row naming at least the
artifact's features; extra columns are ignored. Accept text/csv returns one CSV row per
input row in the layout of scripts/03_runtime_batch_score.py
(row,band,latent_int,pd,top_negative_codes,artifact_hash); Accept application/json
returns {"predictions": [...]}. GET /execution-parameters advertises the batch settings.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

from compileml.runtime import decide, load_artifact
from fastapi import FastAPI, Request, Response, status

MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH")

app = FastAPI(title="compileml-scorer")
_artifact: dict | None = None
_load_error: str | None = None


def _find_artifact() -> Path:
    if ARTIFACT_PATH:
        return Path(ARTIFACT_PATH)
    candidates = sorted(MODEL_DIR.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no *.json artifact under {MODEL_DIR}")
    return candidates[0]


@app.on_event("startup")
def _load() -> None:
    global _artifact, _load_error
    try:
        path = _find_artifact()
        _artifact = load_artifact(path)  # verify=True: hash check before serving
        print(f"loaded {path} hash={_artifact['artifact_hash'][:16]} features={_artifact['features']['names']}")
    except Exception as exc:  # surface on /ping instead of crashing the worker
        _load_error = f"{type(exc).__name__}: {exc}"
        print("artifact load failed:", _load_error)


def _decision(row, *, explain: bool, top_k: int) -> dict:
    names = _artifact["features"]["names"]
    if isinstance(row, dict):
        feats = [None if row.get(n) is None else float(row[n]) for n in names]
    else:
        feats = [None if v is None else float(v) for v in row]
    d = decide(_artifact, feats, explain=explain, top_k=top_k if explain else None)
    out = {"band": d["band"], "latent_int": d["latent_int"], "pd": round(d["pd"], 6)}
    if explain:
        out["reasons"] = [
            {"code": r["code"], "impact_int": r["impact_int"], "message": r["message"]}
            for r in d["reasons_negative"]
        ]
    return out


@app.get("/execution-parameters")
def execution_parameters() -> Response:
    """Batch Transform asks the container for its preferred batching before it starts."""
    return Response(
        content=json.dumps({"BatchStrategy": "MultiRecord", "MaxPayloadInMB": 100}),
        media_type="application/json",
    )


@app.get("/ping")
def ping() -> Response:
    if _artifact is None:
        return Response(content=_load_error or "loading", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return Response(status_code=status.HTTP_200_OK)


def _csv_rows(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    missing = [n for n in _artifact["features"]["names"] if n not in (reader.fieldnames or [])]
    if missing:
        raise KeyError(f"CSV header lacks artifact features: {missing}")
    return [{k: (None if (v is None or v.strip() == "") else v) for k, v in r.items()} for r in reader]


def _csv_out(decisions: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["row", "band", "latent_int", "pd", "top_negative_codes", "artifact_hash"])
    for i, d in enumerate(decisions):
        w.writerow([i, d["band"], d["latent_int"], d["pd"], "|".join(r["code"] for r in d.get("reasons", [])), _artifact["artifact_hash"][:12]])
    return buf.getvalue()


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    if _artifact is None:
        return Response(content=_load_error or "not ready", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    content_type = (request.headers.get("content-type") or "application/json").split(";")[0].strip().lower()
    accept = (request.headers.get("accept") or "").split(",")[0].split(";")[0].strip().lower()
    raw = await request.body()

    try:
        if content_type == "text/csv":
            rows = _csv_rows(raw.decode("utf-8-sig"))
            decisions = [_decision(r, explain=True, top_k=3) for r in rows]
            if accept in ("", "*/*", "text/csv"):
                return Response(content=_csv_out(decisions), media_type="text/csv")
            payload = {"predictions": decisions, "artifact_hash": _artifact["artifact_hash"][:12]}
        else:
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as exc:
                return Response(content=f"invalid JSON: {exc}", status_code=status.HTTP_400_BAD_REQUEST)
            explain = bool(body.get("explain", True)) if isinstance(body, dict) else True
            top_k = int(body.get("top_k", 3)) if isinstance(body, dict) else 3
            if isinstance(body, dict) and "instances" in body:
                payload = {
                    "predictions": [_decision(r, explain=explain, top_k=top_k) for r in body["instances"]],
                    "artifact_hash": _artifact["artifact_hash"][:12],
                }
            else:
                payload = _decision(body, explain=explain, top_k=top_k)
                payload["artifact_hash"] = _artifact["artifact_hash"][:12]
            if accept == "text/csv":
                preds = payload["predictions"] if "predictions" in payload else [payload]
                return Response(content=_csv_out(preds), media_type="text/csv")
    except (KeyError, TypeError, ValueError) as exc:
        return Response(content=f"bad input: {type(exc).__name__}: {exc}", status_code=status.HTTP_400_BAD_REQUEST)
    return Response(content=json.dumps(payload), media_type="application/json")
