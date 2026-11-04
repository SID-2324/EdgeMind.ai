# EdgeMind feature inventory

## Stage 11 extension (2026-11-05)

Meta Llama 3.2 1B Instruct INT4 (`llmware/llama-3.2-1b-instruct-ov`) is added
beside Qwen through `edgemind/local_models.py`, a fixed metadata allowlist.
`local_llama` and legacy `local_qwen` share the existing OpenVINO adapter,
graph, RAG, structural verifier and recovery. Llama is the preselected real
model in the browser; mock remains the startup execution mode. CLI adds
`--model llama|qwen`. CPU/GPU and model-specific benchmark-aware Auto are
supported; NPU generation remains unsupported.

Server paths and benchmark files are separate for each model. Llama uses its
bundled instruct chat template, checks local configuration before loading, and
has a conservative 1800 MB available-RAM admission budget; Qwen retains
2300 MB. These budgets and quality estimates are assumptions. Observed timing
and RSS measurements are in MODEL_PERFORMANCE.json; one-run CPU/GPU Llama
benchmarks are in LLAMA_BENCHMARK.json. A Qwen measurement can never satisfy
Llama's model identity check. Existing Qwen records and weights are preserved.

The Llama download was approved explicitly. Official Llama 3.2 license, policy,
NOTICE and Built with Llama UI attribution are included. No cloud adapter or
paid service was added. The 223-test Stage 10 baseline is preserved separately
and compared by test ID in TEST_SUMMARY.json. See the latest report for actual
Llama CPU/GPU, code, RAG, switching, browser and regression outcomes.

## Stage 10 audit

Audit date: 2026-11-04 (Asia/Kolkata). Source: actual repository, README,
requirements, INTEGRATION_CONTRACT.md, TEAM_HANDOFF.md and existing tests.

## Repository and dependencies

`edgemind/` contains the API, service, CLI, graph, state/contracts, modules,
planning helpers, retrieval algorithms/adapters, OpenVINO adapter, benchmark
implementation and three local web assets. `tests/` originally contained 193
unittest cases spanning Stages 3–10 plus NumPy regression coverage.
Top-level launchers are `main.py`, `run_web.py`, `benchmark_devices.py`,
`smoke_retrieval.py`, `local_llm.py`, and `download_qwen.py`.
The last two can download weights and were not executed during this audit.

Core requirement: LangGraph >=1,<2. Optional web dependencies: FastAPI,
Uvicorn, HTTPX. Optional inference: OpenVINO, OpenVINO GenAI, psutil.
Optional hybrid retrieval: FastEmbed >=0.7,<1.0. The existing Python 3.13 venv
has these installed. Python 3.14 outside that environment lacks LangGraph.
During the initial audit Playwright was absent. In the follow-up, Playwright and
Chromium were installed with explicit user approval. The missing configured
cross-encoder weights were downloaded with separate approval. Existing Qwen
weights and production benchmark records were preserved.

## Interfaces

| HTTP endpoint | Implementation |
|---|---|
| GET / | Local HTML application |
| GET /assets/{path} | Local JS/CSS (StaticFiles) |
| GET /api/health | Status/config existence, no model load or paths |
| POST /api/tasks | Validated task, synchronous graph run in threadpool |
| GET /docs, /redoc, /openapi.json | FastAPI generated documentation/schema |

CLI entry: `python main.py`. Controls include `--query`, `--privacy`,
`--network`, `--complexity`, `--task-type`, `--demo-docs`, `--json`;
RAG mode/rerank/candidate/top-k/chunk/overlap; simulated failure/permanent
failure/invalid result; RAM/GPU/NPU/CPU simulation; strategy, latency, cost and
quality restrictions; `--real-local`, model path, local device, benchmarks,
benchmark age, token limit, profiling and metrics output. Invalid combinations
are rejected by argparse. `python run_web.py` binds only to 127.0.0.1:8000.
`benchmark_devices.py` supports explicit existing model path, devices, runs,
warmups, tokens and output. Test runner never calls download scripts.

## Execution graph

`START → planner → dispatch`. Dispatch chooses retrieval, routing/generation,
or verification from a validated dependency-ordered plan:

- `rag → record_retrieval → dispatch`
- `router → tools → record_tool → dispatch`
- `verifier → record_verification → END` on simulated/completed output
- Failed/blocked verification goes to recovery. One transient retry is allowed:
  `recovery → prepare_retry → dispatch`; otherwise END with DEFERRED.

Retries retain successful retrieval, clear generation/verification step records,
and append verification history and trace. Recovery recognizes TIMEOUT,
RATE_LIMIT and TEMPORARY_UNAVAILABLE; permanent failures, invalid output and
privacy blocks defer. It does not implement persisted checkpoints or hard
model-call deadlines. The test harness bounds real inference in a child process.

## Implemented features and boundaries

| Component | Implemented behavior | Boundary |
|---|---|---|
| Planner | Deterministic text/code/classification plans; optional document retrieval; prior-step dependencies; final verifier | No LLM planning or arbitrary tools; document analysis uses a text task plus documents |
| Router | Privacy/offline, capability, complexity, device, RAM and hard constraints before ranking; four strategies | Quality/cost are assumptions; simulated NPU classification only |
| RAG | Word windows, BM25, cosine embeddings, RRF, optional cross-encoder, top-k, source IDs | In-memory, re-embeds corpus; injected adapters test algorithms; real model quality not validated |
| Coordinator | Executes only validated primary operation; mock envelopes or allowlisted local Qwen | Never executes generated code/shell; no live cloud adapter |
| Verifier | Status, content, planned-operation/model provenance, execution-mode checks; history | Structural validity only; no factual correctness assessment |
| Recovery | At most one transient retry; no unsafe cloud fallback | No durable recovery across process restarts |
| Real adapter | Existing OpenVINO Qwen2.5-1.5B int4 on CPU/GPU; text/code; optional observed timing/RSS | No NPU generation; fresh pipeline per call; model identity assumed from configured allowlist |
| API | Strict request fields/allowlists, document limits, generic HTTP errors; server-only paths; real opt-in | No auth/sessions; real lock returns 409 for concurrent real request |
| UI | Task form, privacy/mode/device, RAG/document controls, response/plan/trace/details/performance, copy/errors | Browser interaction evidence requires Playwright; HTTP asset tests alone do not prove behavior |
| Benchmarks | Cold load + generation samples, medians, atomic JSON save, freshness/model/machine/token matching | No quality calibration; production benchmark records preserved |

## Hardware and model inventory

OpenVINO runtime probe detected CPU, GPU and NPU. Available RAM at initial probe:
2905 MB (dynamic, not installed RAM). Existing `models/qwen-1.5b` includes
model XML/BIN, tokenizer and detokenizer XML/BIN, config and tokenizer assets.
No weights were downloaded or overwritten. Dedicated tests probe RAM/devices
again, run CPU/GPU text/code through the real HTTP adapter with 12-token limits,
and report numeric observed metrics. See TEST_SUMMARY.json for actual outcomes.

## Benchmark storage

`benchmarks_local.json` is the configured default observation file;
`metrics_gpu.json` is an existing profiling artifact. Eligibility requires
schema v1, model ID, prompt profile, measurement scope, fingerprints, timezone,
age (default 168 hours), exact token limit and median agreement with samples.
Missing/stale/mismatched observations fail closed; hardware constraints still
apply. New benchmark tests use temporary directories and injected timings.
Observed inference measurements in TEST_SUMMARY.json are separate from
production routing records. No synthetic reading is claimed as real hardware.

## Coverage and unimplemented work

The original suite covers planner dependencies, routing strategies and privacy,
BM25 and injected hybrid/reranker behavior, NumPy cosine, plan dispatch,
verification, bounded recovery, mocked real adapters, metrics, benchmark
identity/freshness and FastAPI contracts. Added regressions exercise concurrent
HTTP state isolation, returned diagnostic privacy, validation, document/code
end-to-end execution and repeated timeout exhaustion. Dedicated real inference
and optional browser suites are separate from fast tests.

Real cloud inference is NOT IMPLEMENTED. Follow-up automated tests now exercise
actual cached BAAI semantic embeddings, Xenova cross-encoder scores, hybrid
retrieval/reranking and the running FastAPI application through Chromium,
including real CPU/GPU Qwen generation. Tests set explicit offline-only
retrieval loading; see the latest JSON report for actual outcomes. NPU generation, authentication,
durable task history/checkpoints, production isolation, factual verification and
public deployment are outside the current implemented MVP.
