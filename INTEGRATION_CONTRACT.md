# EdgeMind Web Integration Contract (Stage 10)

- **Source of truth:** Existing Stage 9 `edgemind.service.run_task` callable. The web adapter does not modify the orchestration graph, routing policy, or tool execution adapters.
- **HTTP adapter:** `edgemind.api.app` exposes read-only health/config information and a validated `POST /api/tasks` operation; `/` and `/assets/*` serve local frontend files.
- **Mode defaults:** `mock`; simulated results must not be advertised as real inference. Cloud routing in mock mode does not imply a cloud API call.
- **Real-mode admission:** server env `EDGEMIND_ENABLE_REAL_LOCAL=1`, a server-configured model directory, privacy `local_only`, text or code operation, no caller-supplied model paths. `CPU`, `GPU`, or benchmark-aware `auto` devices only. Real mode executes no generated code.
- **Concurrency:** real requests are serialized via nonblocking `threading.Lock`. Concurrent real requests receive HTTP 409; mock requests remain available.
- **Input safety:** Pydantic forbids additional fields and validates lengths, enums and document IDs. No arbitrary file reads/writes from request JSON. No CORS enabled, no session/authentication.
- **Output schema:** response contains `status`, `execution_mode`, `is_real_inference`, `result`, `error`, `model`, `plan`, `executed_steps`, `verification`, `performance`, `trace`, `notice`; optionally evidence/diagnostics. Frontend displays returned values as text rather than trusted HTML.
- **Local server only:** use `python run_web.py`. Do not deploy this server publicly or use `--host 0.0.0.0`; authentication and isolation belong to a later production track.
- **Optional retrieval:** BM25 works with the core requirements; FastEmbed is required for hybrid and reranking and may download local model weights initially.
