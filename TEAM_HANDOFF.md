# EdgeMind — Stage 10 Team Handoff

## Architectural boundary

`Browser → POST /api/tasks → edgemind.api → edgemind.service.run_task → LangGraph → planner / RAG / router / coordinator / verifier / recovery → JSON → browser`

**Layer 2 APIs are unchanged.** The web layer collects user inputs, validates them, maps them into the existing `run_task` contract, and formats the returned state into an API response. CLI continues to work.

## What teammates can work on independently

- **Frontend:** `edgemind/web/*`. Safe DOM output is produced using `textContent`, never injected response HTML. Local static files only.
- **Backend/UI:** `edgemind/api.py`. Add web-safe endpoints here, do not move Layer 2 decision logic into HTTP handlers.
- **Model infrastructure:** `edgemind/adapters/openvino_qwen.py`, Stage 9 benchmark module. This stage does not introduce new models.
- **Testing:** `tests/test_stage10_web_api.py`; preserve mock mode and assert no private user inputs are written or sent to cloud endpoints.

## Contract considerations

- Default `execution_mode = mock`; `is_real_inference=true` only if real mode and verifier returned `COMPLETED`.
- Real mode requires explicit environment configuration, `local_only` privacy, and text/code task type.
- Model path and benchmark file path come **only from server environment**, never user JSON.
- One real inference request at a time, with HTTP 409 for another simultaneous request.
- Browser task text is not persisted; server memory is process-scoped. A production deployment would need consent, authentication, resource isolation, logging policies, and stronger tool sandboxing.
- The frontend's cloud-eligible mock setting is a **demonstration only**; no real cloud API adapter exists.

## API example

`POST /api/tasks` with JSON `{"query":"Explain an NPU","privacy_mode":"local_only","execution_mode":"mock"}` responds with fields `status`, `execution_mode`, `is_real_inference`, `result`, `error`, `model`, `plan`, `executed_steps`, `evidence`, `verification`, `performance`, `trace`, `notice`.

## Ready-for-Stage-11 definition

- All existing tests pass on Windows.
- Mock request works through the browser.
- Real local Qwen request works through browser with configured `EDGEMIND_MODEL_PATH`.
- No public network binding; privacy policy enforced; benchmark-aware `auto` tested only with matching observations.
