# EdgeMind test report

Generated: 2026-11-05T00:02:37.959409+05:30

## Executive summary

Executed 223 test cases: 223 passed, 0 failed, 0 errors, 0 skipped, 0 blocked.

193 existing tests passed before changes. New regressions reproduced four failures (one diagnostic leak and three invalid-setting cases) before fixes.

Stage 10 implemented local features are fully verified; ready for Stage 11 review.
This local demonstration is not approved for public deployment.

## Feature validation matrix

| Feature | Status | Evidence | Notes |
|---|---|---|---|
| Planner | PASS | 51/51 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Router | PASS | 54/54 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| RAG algorithms (injected semantic adapters) | PASS | 33/33 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Coordinator | PASS | 46/46 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Verifier | PASS | 45/45 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Recovery | PASS | 36/36 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| FastAPI | PASS | 23/23 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Web UI browser interactions | PASS | 8/8 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Local Qwen CPU/GPU | PASS | 4/4 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Real FastEmbed semantic retrieval and cross-encoder | PASS | 4/4 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Adaptive routing and benchmark validation | PASS | 19/19 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Real cloud inference | NOT IMPLEMENTED | No authenticated adapter in coordinator | Cloud routing is simulated |

## Audit and fixes

- Feature inventory and architecture: [FEATURE_INVENTORY.md](FEATURE_INVENTORY.md).
- Fixed returned OpenVINO exception text leaking through API error, verifier explanation, and trace. The API now replaces failed real-adapter diagnostics with safe text.
- Fixed hybrid retrieval accepting invalid top-k, candidate-k and similarity with empty documents. Settings are validated before the empty-corpus branch.
- Failed real requests now state that no verified result was produced.
- Added concurrent API state isolation, validation, full HTTP document/code smoke, persistent timeout exhaustion and privacy matrix regressions.
- Added a runner with quick, integration, browser, hardware and all groups; JSON records include individual outcomes and observed hardware metrics when successful.

- Follow-up: installed Playwright/Chromium after user approval; browser tests start an actual loopback Uvicorn process and submit mock, retrieval and real CPU/GPU Qwen tasks. Browser requests outside the local server are blocked and treated as failures.
- Follow-up: actual BAAI embedding cache loaded successfully offline. The configured Xenova cross-encoder was missing; its weights were downloaded with explicit user approval. Subsequent tests require local_files_only and offline mode. Existing Qwen weights were preserved.
- Added EDGEMIND_RETRIEVAL_LOCAL_ONLY=1 for explicit no-download retrieval model loading.

## Environment and exact errors

- Sandbox launch of the existing venv failed: `The file cannot be accessed by the system.` Running the same existing environment with approved escalation succeeded; venv was preserved.
- Default Python 3.14 failed workflow imports: `ModuleNotFoundError: No module named langgraph`. Use the existing project venv (Python 3.13). Only the approved browser tooling was installed.

- Initial runner report generation found missing Windows timezone data (`No time zone found with key Asia/Kolkata`). Fixed by using the standard-library fixed IST offset; no tzdata install required.

## Failures, skipped and blocked tests

## Remaining risks, manual attention and deployment blockers

- Real cloud inference is NOT IMPLEMENTED; cloud routing is simulated.
- Verifier checks structure, not factual correctness.
- Passing tests do not establish public deployment readiness.
- NPU generation is unsupported; NPU eligibility/classification is simulated.
- No authentication, durable checkpoints, hard inference cancellation, production isolation or persistent task history.
- Network offline is a routing flag; hybrid retrieval can download weights on first use.
- Benchmarks test cold adapter latency, not factual quality or GPU kernel timings. Production benchmark JSON and model weights were not changed.
- Hardware child processes have a 180-second test timeout; a timeout is BLOCKED, never PASS.
- Real inference errors are failures; unavailable hardware/models/RAM are BLOCKED.

## Observed dedicated hardware measurements

| Test | Load ms | Generate ms | Adapter total ms | Workflow ms |
|---|---|---|---|---|
| test_cpu_code | 2082.69 | 2909.13 | 4991.93 | 5200.1 |
| test_cpu_text | 1933.83 | 2886.8 | 4820.73 | 5054.26 |
| test_gpu_code | 4971.08 | 364.55 | 5335.72 | 5499.6 |
| test_gpu_text | 4718.17 | 323.51 | 5041.81 | 5182.13 |

These are short, 12-token cold requests, not statistically representative benchmarks.

## Browser and cached-model evidence

Each test outcome in TEST_SUMMARY.json contains numeric/model evidence when available. Browser tests connect over real loopback HTTP; model output is not mocked. Real inference screenshots are in reports/browser/.

### tests.test_browser_ui.BrowserTests.test_actual_hybrid_and_reranking_in_browser

```json
{
  "transport": "real loopback HTTP",
  "browser": "156.0.8078.4",
  "status": "SIMULATED",
  "is_real_inference": false,
  "device": "GPU",
  "performance": {},
  "rag_diagnostics": {
    "method": "hybrid",
    "input_documents": 1,
    "indexed_passages": 1,
    "retrieved_passages": 1,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "reranker_requested": true,
    "bm25_hits": 1,
    "semantic_hits": 1,
    "fused_candidates": 1,
    "reranked": true,
    "semantic_model_used": true,
    "top_semantic_similarity": 0.7295
  }
}
```

### tests.test_browser_ui.BrowserTests.test_bm25_document_retrieval

```json
{
  "transport": "real loopback HTTP",
  "browser": "156.0.8078.4",
  "status": "SIMULATED",
  "is_real_inference": false,
  "device": "GPU",
  "performance": {},
  "rag_diagnostics": {
    "method": "bm25",
    "input_documents": 1,
    "indexed_passages": 1,
    "retrieved_passages": 1,
    "scores": [
      0.575364
    ]
  }
}
```

### tests.test_browser_ui.BrowserTests.test_invalid_input_and_api_error_display

```json
{
  "transport": "real loopback HTTP",
  "browser": "156.0.8078.4",
  "status": "SIMULATED",
  "is_real_inference": false,
  "device": "GPU",
  "performance": {},
  "rag_diagnostics": {}
}
```

### tests.test_browser_ui.BrowserTests.test_mock_code_and_classification_selection

```json
{
  "transport": "real loopback HTTP",
  "browser": "156.0.8078.4",
  "status": "SIMULATED",
  "is_real_inference": false,
  "device": "NPU",
  "performance": {},
  "rag_diagnostics": {}
}
```

### tests.test_browser_ui.BrowserTests.test_mock_text_plan_trace_copy_and_safe_rendering

```json
{
  "transport": "real loopback HTTP",
  "browser": "156.0.8078.4",
  "status": "SIMULATED",
  "is_real_inference": false,
  "device": "GPU",
  "performance": {},
  "rag_diagnostics": {}
}
```

### tests.test_browser_ui.BrowserTests.test_privacy_policy_and_real_mode_controls

```json
{
  "transport": "real loopback HTTP",
  "browser": "156.0.8078.4",
  "status": "SIMULATED",
  "is_real_inference": false,
  "device": "CLOUD",
  "performance": {},
  "rag_diagnostics": {}
}
```

### tests.test_browser_ui.BrowserTests.test_real_cpu_text

```json
{
  "transport": "real loopback HTTP",
  "browser": "156.0.8078.4",
  "status": "COMPLETED",
  "is_real_inference": true,
  "device": "CPU",
  "performance": {
    "measurement_type": "observed",
    "model_device": "CPU",
    "model_load_ms": 2026.25,
    "generation_ms": 2985.47,
    "adapter_total_ms": 5011.82,
    "process_rss_before_mb": 400.2,
    "process_rss_after_mb": 2258.44,
    "process_rss_delta_mb": 1858.25,
    "output_characters": 34,
    "memory_note": "RSS is process RAM before/after, NOT peak RAM or GPU memory.",
    "latency_note": "Cold/load cost may vary; GPU execution is asynchronous internally.",
    "workflow_total_ms": 5312.49,
    "workflow_note": "Whole LangGraph workflow including planning, retrieval, routing, model load, generation, and verification as applicable."
  },
  "rag_diagnostics": {}
}
```

### tests.test_browser_ui.BrowserTests.test_real_gpu_code

```json
{
  "transport": "real loopback HTTP",
  "browser": "156.0.8078.4",
  "status": "COMPLETED",
  "is_real_inference": true,
  "device": "GPU",
  "performance": {
    "measurement_type": "observed",
    "model_device": "GPU",
    "model_load_ms": 5934.41,
    "generation_ms": 362.53,
    "adapter_total_ms": 6297.03,
    "process_rss_before_mb": 435.08,
    "process_rss_after_mb": 2339.03,
    "process_rss_delta_mb": 1903.95,
    "output_characters": 19,
    "memory_note": "RSS is process RAM before/after, NOT peak RAM or GPU memory.",
    "latency_note": "Cold/load cost may vary; GPU execution is asynchronous internally.",
    "workflow_total_ms": 6383.74,
    "workflow_note": "Whole LangGraph workflow including planning, retrieval, routing, model load, generation, and verification as applicable."
  },
  "rag_diagnostics": {}
}
```

### tests.test_live_retrieval.CachedRetrievalTests.test_actual_cross_encoder_scores

```json
{
  "model": "Xenova/ms-marco-MiniLM-L-6-v2",
  "scores": [
    8.089582443237305,
    -11.127354621887207,
    -11.154939651489258
  ],
  "local_files_only": true
}
```

### tests.test_live_retrieval.CachedRetrievalTests.test_actual_embeddings_and_numpy_cosine

```json
{
  "model": "BAAI/bge-small-en-v1.5",
  "dimension": 384,
  "semantic_scores": [
    0.808261422783362,
    0.33563066042742784,
    0.3110574387872746
  ],
  "source_ids": [
    "capital",
    "fruit",
    "compute"
  ],
  "local_files_only": true
}
```

### tests.test_live_retrieval.CachedRetrievalTests.test_actual_hybrid_retrieval

```json
{
  "bm25_hits": 1,
  "semantic_hits": 3,
  "fused_candidates": 3,
  "reranked": false,
  "semantic_model_used": true,
  "top_semantic_similarity": 0.8083
}
```

### tests.test_live_retrieval.CachedRetrievalTests.test_actual_hybrid_with_cross_encoder

```json
{
  "method": "hybrid",
  "input_documents": 3,
  "indexed_passages": 3,
  "retrieved_passages": 1,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "reranker_requested": true,
  "bm25_hits": 1,
  "semantic_hits": 3,
  "fused_candidates": 3,
  "reranked": true,
  "semantic_model_used": true,
  "top_semantic_similarity": 0.8158
}
```
