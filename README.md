# EdgeMind — Stage 10: Local Web Application

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

```powershell
python -m unittest discover -s tests -v
python -m unittest tests.test_stage10_web_api -v
```

All Stage 9 regression tests should remain unchanged. At least one real local Qwen request through the browser is required to validate the Stage 10 integration on your hardware. For the final milestone, Stage 11 will add error-handling polish, demo scenarios, packaging and deployment planning.
