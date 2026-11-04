# EdgeMind — Stage 11: Built with Llama

Meta Llama 3.2 1B Instruct (INT4 OpenVINO conversion by
[llmware](https://huggingface.co/llmware/llama-3.2-1b-instruct-ov)) is now the
preselected local demonstration model. Qwen 2.5 1.5B remains available. Both
use the same `openvino_genai` adapter, LangGraph workflow, RAG, structural
verification, recovery and profiling. CPU and GPU are supported; NPU generation
is not. Cloud inference is **NOT IMPLEMENTED**.

Mock mode remains the safe startup default. Select **Real local OpenVINO** and
**Meta Llama 3.2 1B** or **Qwen 2.5 1.5B** to run real generation. The local model
selector applies to real mode; mock routing still uses fictional profiles.

```powershell
$env:EDGEMIND_ENABLE_REAL_LOCAL = '1'
$env:EDGEMIND_RETRIEVAL_LOCAL_ONLY = '1'
python run_web.py
```

Existing model files must already be present. Llama defaults to
`models/llama-3.2-1b`; override only on the server with
`EDGEMIND_LLAMA_MODEL_PATH`. Qwen keeps `models/qwen-1.5b` and
`EDGEMIND_MODEL_PATH`. Requests never accept filesystem paths. No model weights
are downloaded by the inference adapter. The initial Llama download was explicitly
approved; existing Qwen weights were preserved.

CLI (real inference, no generated-code execution):

```powershell
python main.py --real-local --model llama --local-device CPU --max-new-tokens 32 --profile --query "Explain local AI briefly"
python main.py --real-local --model qwen --local-device GPU --max-new-tokens 32 --profile --query "Explain local AI briefly"
```

API execution modes are `mock`, `local_llama`, and the backward-compatible
`local_qwen`. Real modes require server opt-in, `privacy_mode=local_only`, and
text/code tasks. All real models share the same nonblocking admission lock.
The service checks model identity at routing, coordination and verification.
The Llama adapter checks its local configuration before loading the pipeline.
Llama uses the bundled instruct chat template; reference text stays untrusted.

**Auto device routing requires matching measurements for the selected model.**
Qwen uses `benchmarks_local.json`; Llama uses `benchmarks_llama.json` (override
with `EDGEMIND_LLAMA_BENCHMARK_FILE`). Fingerprints, model ID, machine, freshness
and token count must match. A Qwen observation is never eligible for Llama,
even when its path is reused. To produce short local Llama observations:

```powershell
python benchmark_devices.py --model llama --model-path models/llama-3.2-1b --devices CPU GPU --runs 1 --max-new-tokens 32
```

Select Auto and set the UI token count to **32** for that file. Missing or stale
measurements defer safely. RAM admission budgets (Llama 1800 MB, Qwen 2300 MB)
are conservative configuration assumptions, not measured peaks. Quality scores
remain unvalidated placeholders. Reports distinguish observed load/generation/RSS
measurements from these estimates. One cold request is a smoke measurement,
not a representative performance study.

## Llama attribution and terms

**Built with Llama.** Base model: Meta Llama 3.2 1B Instruct; conversion: llmware.
Llama is governed by Meta's custom
[Llama 3.2 Community License](LICENSE-Llama-3.2.txt) and
[Acceptable Use Policy](LLAMA_ACCEPTABLE_USE_POLICY.md), not the application’s
or Qwen’s license. Official sources:
[license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE),
[policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md).

When distributing Llama materials, include the agreement and the attribution
in `NOTICE`; retain **Built with Llama** in the interface/documentation. Use must
follow the acceptable-use policy and applicable laws. The license also contains
additional commercial terms for the specified 700-million-monthly-user threshold,
and naming requirements if Llama materials or outputs are used to improve and
distribute another AI model. This application does not train or distribute a new
model. Review the complete bundled terms for your intended use. Keep these files
with any redistributed model bundle; model weights are ignored by Git.

## Stage 10 compatibility reference

EdgeMind now includes a **FastAPI backend and an interactive browser UI**. This package is built on the working Stage 9 project. It retains the original `run_task()` service, CLI, OpenVINO adapter, benchmarks, LangGraph orchestration, RAG, verification, and recovery.

**Default = mock mode.** The browser clearly labels its generated output as simulated. Actual local Qwen inference is *opt-in* and performs no cloud-model calls. The current app is a **local demonstration**, not an internet-facing production service.

## Fast installation (Windows PowerShell)

1. Back up your working Stage 9 project. Unzip `EdgeMind-Stage10-Upgrade.zip` **into that same folder**, overwriting files. Do not overwrite your `venv`, `models` or benchmark data.
2. From the project root, with your existing virtual environment activated:

```powershell
python -m pip install -r requirements-web.txt
python -m unittest discover -s tests -v
python run_web.py
```

3. Open **http://127.0.0.1:8000**. The default mode is **Mock / safe demo**. Try the starter prompt and inspect **Plan**, **Trace** and **Details**.
4. To use actual Qwen inference in the browser, **stop the server** and, in the same PowerShell terminal, configure the local model *before* starting it:

```powershell
$env:EDGEMIND_ENABLE_REAL_LOCAL = '1'
$env:EDGEMIND_MODEL_PATH = 'C:\Users\LENOVO\OneDrive\Desktop\projects\meta.ai\YOUR_PROJECT\models\qwen-1.5b'
$env:EDGEMIND_BENCHMARK_FILE = 'benchmarks_local.json'
python run_web.py
```

Change `EDGEMIND_MODEL_PATH` to the **actual existing directory** containing the downloaded OpenVINO Qwen files. If you keep the model at `models/qwen-1.5b` relative to the current project, you can omit the path variable entirely. In the UI, choose **Real local Qwen**, then **GPU** or **Auto** (matching benchmark data is required for Auto). Stage 9 benchmark data is valid only for the same model/machine/token limit; set the web UI token limit to match your saved benchmark (`64` by default).

**Optional**: install `requirements-local.txt` if OpenVINO is not present, or `requirements-rag.txt` for local FastEmbed semantic retrieval. BM25 mode works without FastEmbed. Hybrid retrieval may download embedding/reranker weights on first use, so it is not strictly network-free until cached.

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Browser application |
| `GET /api/health` | No-inference status (without exposing model paths) |
| `POST /api/tasks` | Runs a task through existing `service.run_task()` |
| `GET /docs` | FastAPI interactive API docs |

Example request (mock):

```powershell
$payload = @{ query = 'Explain what an NPU does'; task_type = 'text' } | ConvertTo-Json
Invoke-RestMethod 'http://127.0.0.1:8000/api/tasks' -Method POST -ContentType 'application/json' -Body $payload
```

Requests may include: `query`, `execution_mode` (`mock` or `local_qwen`), `task_type`, `privacy_mode`, `network`, `complexity`, `strategy`, `local_device` (`GPU`, `CPU` or `auto`), `max_new_tokens`, `rag_mode` (`bm25` or `hybrid`), `rerank`, `rag_top_k`, and up to six `{source_id,text}` documents. Inputs have strict lengths/allowlists. A prompt must be supplied. The API returns status, content, plan steps, selected model, trace, evidence, validation and observed performance if available.

## Privacy and limitations

- **Bind to `127.0.0.1`, not `0.0.0.0`.** The demo has no authentication or user sessions. Do not expose it publicly, tunnel it, or share access to the local server.
- The frontend uses only local assets; the app doesn't persist prompts, documents or responses. Uploaded files and arbitrary path access are not supported.
- **Cloud model execution is not implemented.** `cloud_allowed` only affects the *mock* routing policy; it never invokes a real cloud service.
- Real generation is allowlisted to local OpenVINO Qwen on CPU/GPU, not NPU. Generated code is never executed. Classification remains mock-only.
- Only **one real inference task** runs at a time (concurrent real requests return HTTP 409). Long model loads may take time.
- The Router's quality estimates are not validated quality scores. Auto routing uses Stage 9 observed cold-start benchmark latencies; manual device selection is not benchmark-aware. The verifier checks structure, **not factual correctness**.
- `network=offline` is a routing flag, not a system-wide firewall. Choose BM25 and already-installed local models when physically offline.

## New files

- `edgemind/api.py` — validated FastAPI HTTP adapter; delegates to `service.run_task()`.
- `edgemind/web/index.html`, `style.css`, `app.js` — responsive UI with safe text-only rendering.
- `run_web.py` — loopback-only server launcher.
- `requirements-web.txt` — FastAPI, Uvicorn and test client.
- `tests/test_stage10_web_api.py` — HTTP and safety-contract tests.

## Testing

Run the audit suite and generate Markdown/JSON reports locally:

```powershell
python run_all_tests.py              # unit, regression, workflow, API, benchmarks
python run_all_tests.py --quick      # unit/regression and mocked benchmark checks
python run_all_tests.py --integration # workflow and API tests
python run_all_tests.py --browser    # optional installed Playwright + Chromium
python run_all_tests.py --hardware   # existing Qwen CPU/GPU, 12-token requests
python run_all_tests.py --retrieval  # actual cached FastEmbed + cross-encoder
python run_all_tests.py --all        # all groups; no downloads or cloud calls
```

Use the project's existing virtual environment. Reports are written to
`reports/EDGEMIND_TEST_REPORT.md` and `reports/TEST_SUMMARY.json` for the selected
groups. Missing browser tooling is skipped as NOT TESTED; unavailable hardware,
model files, RAM or inference timeout is BLOCKED. Actual inference failures fail
the suite. Hardware tests use child processes with 180-second limits. Existing
benchmark records are preserved. The feature inventory is in
`reports/FEATURE_INVENTORY.md`. A passing subset does not establish MVP readiness.

Browser tests start and stop a separate loopback Uvicorn process and exercise
the actual HTTP application, including short real CPU/GPU Qwen requests.
Playwright and Chromium must already be installed; the runner never installs
them. Cached retrieval tests use `EDGEMIND_RETRIEVAL_LOCAL_ONLY=1` and offline
loading. Missing weights are BLOCKED rather than downloaded. Set that variable
when running the application to forbid automatic retrieval model downloads.

```powershell
python -m unittest discover -s tests -v
python -m unittest tests.test_stage10_web_api -v
```

All Stage 9 regression tests should remain unchanged. At least one real local Qwen request through the browser is required to validate the Stage 10 integration on your hardware. For the final milestone, Stage 11 will add error-handling polish, demo scenarios, packaging and deployment planning.
