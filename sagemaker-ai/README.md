# SageMaker SDK v3 with our own container (BYOC), offline

Everything here runs the compileml fraud artifact (`artifacts/insurance_fraud.json`) inside
our own image through the SageMaker Python SDK v3 (`sagemaker` 3.21, `sagemaker-serve` 1.21)
in **local mode**, with no AWS account reachable. Verified 2026-09-06 with all HTTPS traffic
routed to a dead proxy.

| file | what it does |
| --- | --- |
| `container/` | the BYOC image: python:3.12-slim + FastAPI/uvicorn, `compileml` installed `--no-deps` (runtime is stdlib-only; no numpy/sklearn/catboost, 212 MB). `serve.py`: `GET /ping`, `GET /execution-parameters`, `POST /invocations` (JSON one record or `{"instances": [...]}`; `text/csv` in and out for Batch Transform, same columns as `scripts/03`). |
| `sagemaker_offline.py` | `use_local_stubs()`: replaces the three control-plane calls the SDK makes even in local mode (IAM `SimulatePrincipalPolicy` inside `build()`, STS for the default bucket, `docker pull`) with local no-ops. Applied when `SM_OFFLINE=1` (default in every script here). |
| `deploy_modelbuilder.ipynb` | notebook, pre-executed airgapped. Part 1: `ModelBuilder(inference_spec=..., model_server=TORCHSERVE, image_uri=compileml-scorer:local, mode=LOCAL_CONTAINER)` → `build()` → `deploy_local()` → `LocalEndpoint.invoke()` and `sagemaker.core.resources.Endpoint(name).invoke()`. Part 2: no ModelBuilder: `Model.create()` → `EndpointConfig.create()` → `Endpoint.create()` → `invoke()` and the boto3-shaped `invoke_endpoint()`, all local by swapping both clients of the `SageMakerClient` singleton for `LocalSession`'s. Endpoint-mode cell at the end, not executed. |
| `local_byoc.py` | script form of the notebook's build/deploy/invoke, `--model-server` selectable. |
| `local_endpoint.py` | same through `model_server=MMS` (the generic serve/ping/invocations runner). |
| `local_transform.py` | a SageMaker **Batch Transform** job in local mode (`instance_type="local"`): model.tar.gz → docker compose → `/execution-parameters` → whole CSV posted as one `text/csv` payload → `<input>.out`. 2,000 claims scored; matches `artifacts/insurance_fraud_decisions.csv` on every row. |
| `model_servers/try_model_servers.py` | runs every `ModelServer` value against the image and records what Docker actually started (command, ports, mounts, env) and whether invoke worked → `model_servers/model_server_matrix.json`. `model_servers/smd_local.py` + `smd_handler.py`: SMD support; `custom_orchestrator.py`: the SDK's CustomOrchestrator flow on the SMD image, locally (see below). |
| `deploy_modelbuilder.py` | the original MLflow-registry script this was adapted from (other project). |

```bash
make image                                   # build compileml-scorer:local (once, online)
uv run python sagemaker-ai/local_byoc.py     # or local_endpoint.py / local_transform.py
uv run jupyter execute sagemaker-ai/deploy_modelbuilder.ipynb   # or open it
docker rm -f $(docker ps -aq --filter publish=8080)              # if a run left port 8080 busy
```

## What we learned about ModelBuilder (3.21)

- **`model=` needs a framework object.** It maps the object's class or base-class name to
  torch / xgboost / tensorflow / sklearn and raises otherwise, even with `image_uri`. The
  compileml artifact is a JSON dict, so the v3 hook is **`inference_spec=`**.
- **`InferenceSpec` with a BYOC image:** `load()`/`invoke()` are what the SDK's own handler
  runs inside an AWS DLC (`code/inference.py` + cloudpickled `code/serve.pkl`). Our image
  runs `serve.py` and never touches them. `prepare(model_dir)` runs on the host at build time
  and is the place to stage the artifact into `model_path`.
- **`model_server` is mandatory with `image_uri`, and it names a serving *toolkit contract*,
  not a type.** Each value tells the SDK which serving stack it should package for and, in
  local mode, how to launch and talk to it. Measured against our image
  (`model_server_matrix.json`, sagemaker 3.21):

  | value | build | local launch | result with our BYOC image |
  | --- | --- | --- | --- |
  | TORCHSERVE | ok: `code/inference.py`, `serve.pkl`, `requirements.txt` | `docker run <image> serve`, port 8080, mounts **`model_path`** at `/opt/ml/model` | works |
  | MMS | ok: same files | same, but mounts **`model_path/code`** at `/opt/ml/model`; `invoke()` returns the response wrapped in a one-element list (inside the JSON) | works once the artifact is staged under `code/` too |
  | TENSORFLOW_SERVING | fails: "only supported for mlflow models" | port 8501 | n/a |
  | DJL_SERVING, TGI | fails: sample input must be `{"inputs": str, "parameters": dict}` (LLM contract) | posts to `/predictions/model`, `/generate` | n/a |
  | TRITON | fails: sample input must be ndarray/torch/tf | `tritonserver --model-repository=/models`, port 8000 | n/a |
  | TEI | ok | requests a GPU device; Docker on a Mac refuses | n/a |
  | SMD | ok | **no local launcher in 3.21** (`None.logs`); added by `model_servers/smd_local.py` | works, with our image or the real SMD image + `smd_handler.py` |
  | VLLM, SGLANG, VLLM_OMNI, LLAMACPP | ok | no local launcher in 3.21 | endpoint mode only |

  On a real endpoint SageMaker itself runs `docker run <image> serve` on 8080 with
  `model.tar.gz` unpacked at `/opt/ml/model`, whatever the value; there the difference is
  only the staged scaffold and env vars, which our `serve.py` ignores. So for BYOC:
  TORCHSERVE or MMS, both fine, and stage the artifact at the root *and* under `code/`.
- **SMD (SageMaker Distribution), measured.** The SDK's `ModelServer.SMD` targets the Studio
  image `public.ecr.aws/sagemaker/sagemaker-distribution:3.2.0-cpu` (identical layers to the
  private `885854791233...sagemaker-distribution-prod:3.2.0-cpu` the SDK names; Public ECR pulls
  ~8x faster here). Its server is Python Tornado on 8080 (`/etc/sagemaker-inference-server`):
  it installs `code/requirements.txt` (micromamba, then pip), puts the code dir on `sys.path`,
  imports `SAGEMAKER_INFERENCE_CODE=inference.handler`, and calls `handler(request)` (sync or
  async, `request.body` bytes) per `/invocations`; a dict return is written as JSON. `/ping`
  only answers once the handler is loaded. Runs as non-root `sagemaker-user`. sagemaker 3.21
  has **no local launcher** for it (`create_server` has no SMD branch, `deploy_local` dies on
  `None.logs`) and only writes a handler for `CustomOrchestrator` specs, which it forces to
  endpoint mode. `model_servers/smd_local.py` adds the launcher (SMD env vars over the
  TorchServe docker-run path) and `smd_handler.py` is the handler for the compileml artifact,
  staged as `code/inference.py` with a vendored `compileml` (stdlib runtime, nothing to
  install). Verified airgapped, both with our BYOC image and with the real SMD image:
  `IMAGE_URI=public.ecr.aws/sagemaker/sagemaker-distribution:3.2.0-cpu uv run python sagemaker-ai/local_byoc.py --model-server SMD`.
- **CustomOrchestrator, the SDK's own SMD flow, measured** (`model_servers/custom_orchestrator.py`).
  An orchestrator is one class with `handle(data) -> response` plus a ready-made
  `sagemaker-runtime` client (`self.client`) for calling *other* endpoints; it is Lambda-shaped
  on purpose because it sits in front of models rather than hosting one. ModelBuilder pickles
  `(orchestrator, schema_builder)` into `code/serve.pkl`, copies its generic
  `custom_execution_inference.py` to `code/inference.py` (unpickles at import, forwards
  `request.body` to `handle()`), forces `Mode.SAGEMAKER_ENDPOINT` and `ModelServer.SMD`, and
  picks the SMD image. The script does the same staging with the SDK's `save_pkl` and
  `prepare_for_smd` and runs the SMD image locally. Two version mismatches between the 3.21
  host SDK and the 2.245 SDK bundled in the SMD 3.2.0 image had to be bridged: (1) 3.21 writes
  a plain SHA-256 into `metadata.json`; the image's handler verifies an HMAC keyed by
  `SAGEMAKER_SERVE_SECRET_KEY` and crashes without it, so the script re-signs the pickle the
  v2 way and passes the key; (2) a 3.21 `SchemaBuilder` references `sagemaker.core.*`, which
  the image lacks, so the pickle carries `(orchestrator, None)`. Until either side updates,
  the unmodified v3 flow against this image fails on a real endpoint the same way.
- **Which value for BYOC, then.** None of TORCHSERVE, MMS or SMD needs an `inference.py` from
  *you* with our image: `serve.py` is the handler and reads the artifact itself. The SDK writes
  a `code/inference.py` for all three regardless; it only matters when the image's server looks
  for it (DLC toolkits, the SMD Tornado server). TORCHSERVE is the convenient value for BYOC
  because its local launcher mounts `model_path` as-is and returns the response unwrapped.
- **Control-plane calls in local mode** (why `sagemaker_offline` exists): `ModelBuilder(...)`
  resolves a role via IAM when `role_arn` is missing; `build()` on the TorchServe path calls
  `_create_model` → `resolve_and_validate_role` → IAM `SimulatePrincipalPolicy`;
  `build()` → `_get_serve_setting` → `Session.default_bucket()` → STS; `deploy_local()`
  → `docker pull image_uri` (ECR login when the name looks like ECR).
- `dependencies={"auto": False}`: the default `{"auto": True}` shells out to the *system*
  python to introspect the pickle and fails.
- `deploy()` / `deploy_local()` in local mode return a **`LocalEndpoint`** (not a Predictor):
  `invoke(body=...)` → `InvokeEndpointOutput` whose `.body` is a `BytesIO` of the raw JSON.
  `delete()` leaves the containers behind (one from `docker run`, one from the docker compose
  pass) → next deploy fails with `port 8080 already allocated`.
- **`sagemaker.core.resources.Endpoint` is a pydantic mirror of the SageMaker API resource.**
  `Endpoint(endpoint_name=...)` only builds the object (no API call): a handle whose methods
  then call the API with that name. `Endpoint.get(name)` = DescribeEndpoint. `Endpoint.create(...)`
  = CreateEndpoint (control plane; there is no local implementation, which is why it "reaches
  SageMaker immediately"; local mode's stand-in is `ModelBuilder.deploy_local()` →
  `LocalEndpoint`). `invoke()` = runtime `InvokeEndpoint`; in 3.21 there is no
  `invoke_endpoint()` method on the resource (that name is the boto3 client call; older
  sagemaker-core READMEs show it). `invoke()` takes its client from the `SageMakerClient`
  singleton; setting `SageMakerClient().sagemaker_runtime_client = LocalSagemakerRuntimeClient()`
  routes the identical call to `localhost:8080`. **The control plane swaps the same way:**
  `SageMakerClient().sagemaker_client = LocalSession().sagemaker_client` makes
  `Model.create()`, `EndpointConfig.create()`, `Endpoint.create()`, `describe`/`delete` run
  against local mode's in-memory registry and docker compose (`instance_type="local"`,
  `model_data_url="file://<dir or model.tar.gz>"`). Import `sagemaker.serve.model_builder`
  first; `core/local/image.py` uses it without importing it. Local `delete_endpoint` stops
  the container but leaves the name in the registry.
- Apple Silicon: the image is linux/amd64 and runs under emulation; first `/ping` takes a few
  seconds. A single-arch manifest list in ECR fails the SDK's pull on arm64 (`no matching
  manifest`); `make image-push` builds amd64 + arm64.
- DLC path (not used): `pytorch-inference:2.6.0-cpu-py312` ships neither `cloudpickle` nor
  `sagemaker`, both imported by the generated handler; the SDK writes `SAGEMAKER_PROGRAM` and
  `SAGEMAKER_SUBMIT_DIRECTORY` as empty strings and lets `env_vars` override its own defaults;
  under Rosetta the JVM needs `-Djdk.lang.Process.launchMechanism=FORK` and TorchServe system
  metrics must be off or the SDK aborts on the first `[ERROR]` log line.
