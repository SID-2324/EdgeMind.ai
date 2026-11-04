# EdgeMind test report

Generated: 2026-11-05T01:03:49.190368+05:30

## Executive summary

Executed 251 test cases: 251 passed, 0 failed, 0 errors, 0 skipped, 0 blocked.

193 existing tests passed before changes. New regressions reproduced four failures (one diagnostic leak and three invalid-setting cases) before fixes.

Stage 11 Llama local integration verified alongside Stage 10 regressions.
This local demonstration is not approved for public deployment.

## Feature validation matrix

| Feature | Status | Evidence | Notes |
|---|---|---|---|
| Planner | PASS | 51/51 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Router | PASS | 55/55 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| RAG algorithms (injected semantic adapters) | PASS | 36/36 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Coordinator | PASS | 46/46 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Verifier | PASS | 45/45 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Recovery | PASS | 36/36 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| FastAPI | PASS | 23/23 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Web UI browser interactions | PASS | 12/12 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Local Qwen CPU/GPU | PASS | 4/4 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Llama CPU/GPU, code and RAG | PASS | 24/24 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Real FastEmbed semantic retrieval and cross-encoder | PASS | 4/4 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
| Adaptive routing and benchmark validation | PASS | 21/21 matching outcomes in TEST_SUMMARY.json | Evidence groups overlap |
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

- Stage 11: Llama 3.2 1B INT4 added to the fixed allowlist, shared OpenVINO adapter, service, coordinator and verifier. Legacy local_qwen mode and model paths are preserved.
- Llama is preselected in the local-model UI. Real mode remains opt-in. CPU/GPU and model-specific Auto routing are available. Qwen observations cannot satisfy Llama identity checks.
- Llama download explicitly approved; existing Qwen weights were preserved. Official license, acceptable-use policy and NOTICE accompany Built with Llama attribution.
- Original 223-test Stage 10 baseline preserved in STAGE10_BASELINE.json; per-ID regression comparison is in TEST_SUMMARY.json. Model timing/RSS evidence is also exported to MODEL_PERFORMANCE.json.

- Initial Stage 11 full run blocked three browser cases under low RAM after hybrid model loading. The browser fixture now clears its parent preflight model caches and restarts only its own loopback server before real-model scenarios. Application memory safeguards are unchanged.

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
- Benchmarks test cold adapter latency, not factual quality or GPU kernel timings. Existing Qwen benchmark JSON and weights were not changed; Llama measurements use a separate file.
- Hardware child processes have a 180-second test timeout; a timeout is BLOCKED, never PASS.
- Real inference errors are failures; unavailable hardware/models/RAM are BLOCKED.

## Observed dedicated hardware measurements

| Model | Test | Load ms | Generate ms | Adapter total ms | Workflow ms |
|---|---|---|---|---|---|
| openvino-qwen2.5-1.5b-instruct-int4 | test_cpu_code | 2092.51 | 2988.92 | 5081.54 | 5310.0 |
| openvino-qwen2.5-1.5b-instruct-int4 | test_cpu_text | 1982.07 | 2628.19 | 4610.37 | 4817.47 |
| openvino-qwen2.5-1.5b-instruct-int4 | test_gpu_code | 5245.73 | 366.32 | 5612.16 | 5748.86 |
| openvino-qwen2.5-1.5b-instruct-int4 | test_gpu_text | 4871.59 | 319.12 | 5190.81 | 5327.37 |
| openvino-llama3.2-1b-instruct-int4 | test_cpu_code | 1557.9 | 2269.41 | 3827.45 | 4021.4 |
| openvino-llama3.2-1b-instruct-int4 | test_cpu_text | 1420.62 | 1435.41 | 2856.19 | 3054.36 |
| openvino-llama3.2-1b-instruct-int4 | test_gpu_code | 5002.24 | 511.59 | 5513.97 | 6236.98 |
| openvino-llama3.2-1b-instruct-int4 | test_gpu_text | 4737.58 | 201.99 | 4939.79 | 5469.81 |
| openvino-llama3.2-1b-instruct-int4 | test_llama_auto_from_real_benchmarks | 1461.06 | 1618.54 | 3079.76 | 3281.55 |
| openvino-llama3.2-1b-instruct-int4 | test_llama_rag_summary_cpu | 1425.14 | 2294.98 | 3720.32 | 3984.53 |

These are short, 12–32-token cold requests, not statistically representative benchmarks.

## Stage 10 baseline preservation

223/223 original test IDs passed in this run.

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

### tests.test_browser_ui.BrowserTests.test_llama_auto_uses_own_observations

```json
{
  "transport": "real loopback HTTP",
  "browser": "156.0.8078.4",
  "status": "COMPLETED",
  "is_real_inference": true,
  "device": "CPU",
  "performance": {
    "measurement_type": "observed",
    "model_id": "openvino-llama3.2-1b-instruct-int4",
    "model_device": "CPU",
    "model_load_ms": 3986.53,
    "generation_ms": 7374.47,
    "adapter_total_ms": 11361.24,
    "process_rss_before_mb": 129.17,
    "process_rss_after_mb": 1903.59,
    "process_rss_delta_mb": 1774.41,
    "output_characters": 175,
    "memory_note": "RSS is process RAM before/after, NOT peak RAM or GPU memory.",
    "latency_note": "Cold/load cost may vary; GPU execution is asynchronous internally.",
    "workflow_total_ms": 11872.5,
    "workflow_note": "Whole LangGraph workflow including planning, retrieval, routing, model load, generation, and verification as applicable."
  },
  "rag_diagnostics": {}
}
```

### tests.test_browser_ui.BrowserTests.test_llama_gpu_generation

```json
{
  "transport": "real loopback HTTP",
  "browser": "156.0.8078.4",
  "status": "COMPLETED",
  "is_real_inference": true,
  "device": "GPU",
  "performance": {
    "measurement_type": "observed",
    "model_id": "openvino-llama3.2-1b-instruct-int4",
    "model_device": "GPU",
    "model_load_ms": 8732.72,
    "generation_ms": 938.92,
    "adapter_total_ms": 9671.94,
    "process_rss_before_mb": 128.19,
    "process_rss_after_mb": 2189.01,
    "process_rss_delta_mb": 2060.82,
    "output_characters": 92,
    "memory_note": "RSS is process RAM before/after, NOT peak RAM or GPU memory.",
    "latency_note": "Cold/load cost may vary; GPU execution is asynchronous internally.",
    "workflow_total_ms": 10123.21,
    "workflow_note": "Whole LangGraph workflow including planning, retrieval, routing, model load, generation, and verification as applicable."
  },
  "rag_diagnostics": {}
}
```

### tests.test_browser_ui.BrowserTests.test_llama_qwen_switching_and_rag_real_inference

```json
{
  "transport": "real loopback HTTP",
  "switch_sequence": [
    {
      "model_id": "openvino-llama3.2-1b-instruct-int4",
      "performance": {
        "measurement_type": "observed",
        "model_id": "openvino-llama3.2-1b-instruct-int4",
        "model_device": "CPU",
        "model_load_ms": 3902.59,
        "generation_ms": 8425.19,
        "adapter_total_ms": 12328.01,
        "process_rss_before_mb": 118.51,
        "process_rss_after_mb": 1907.91,
        "process_rss_delta_mb": 1789.41,
        "output_characters": 175,
        "memory_note": "RSS is process RAM before/after, NOT peak RAM or GPU memory.",
        "latency_note": "Cold/load cost may vary; GPU execution is asynchronous internally.",
        "workflow_total_ms": 12842.09,
        "workflow_note": "Whole LangGraph workflow including planning, retrieval, routing, model load, generation, and verification as applicable."
      }
    },
    {
      "model_id": "openvino-qwen2.5-1.5b-instruct-int4",
      "performance": {
        "measurement_type": "observed",
        "model_id": "openvino-qwen2.5-1.5b-instruct-int4",
        "model_device": "CPU",
        "model_load_ms": 3799.3,
        "generation_ms": 11967.2,
        "adapter_total_ms": 15766.63,
        "process_rss_before_mb": 177.8,
        "process_rss_after_mb": 2015.06,
        "process_rss_delta_mb": 1837.26,
        "output_characters": 191,
        "memory_note": "RSS is process RAM before/after, NOT peak RAM or GPU memory.",
        "latency_note": "Cold/load cost may vary; GPU execution is asynchronous internally.",
        "workflow_total_ms": 16145.48,
        "workflow_note": "Whole LangGraph workflow including planning, retrieval, routing, model load, generation, and verification as applicable."
      }
    },
    {
      "model_id": "openvino-llama3.2-1b-instruct-int4",
      "performance": {
        "measurement_type": "observed",
        "model_id": "openvino-llama3.2-1b-instruct-int4",
        "model_device": "CPU",
        "model_load_ms": 4016.42,
        "generation_ms": 7359.99,
        "adapter_total_ms": 11376.54,
        "process_rss_before_mb": 170.98,
        "process_rss_after_mb": 1911.61,
        "process_rss_delta_mb": 1740.63,
        "output_characters": 175,
        "memory_note": "RSS is process RAM before/after, NOT peak RAM or GPU memory.",
        "latency_note": "Cold/load cost may vary; GPU execution is asynchronous internally.",
        "workflow_total_ms": 11802.64,
        "workflow_note": "Whole LangGraph workflow including planning, retrieval, routing, model load, generation, and verification as applicable."
      }
    }
  ]
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
    "model_id": "openvino-qwen2.5-1.5b-instruct-int4",
    "model_device": "CPU",
    "model_load_ms": 1937.31,
    "generation_ms": 2792.75,
    "adapter_total_ms": 4730.17,
    "process_rss_before_mb": 128.92,
    "process_rss_after_mb": 2008.71,
    "process_rss_delta_mb": 1879.79,
    "output_characters": 34,
    "memory_note": "RSS is process RAM before/after, NOT peak RAM or GPU memory.",
    "latency_note": "Cold/load cost may vary; GPU execution is asynchronous internally.",
    "workflow_total_ms": 4943.32,
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
    "model_id": "openvino-qwen2.5-1.5b-instruct-int4",
    "model_device": "GPU",
    "model_load_ms": 5849.13,
    "generation_ms": 334.26,
    "adapter_total_ms": 6183.52,
    "process_rss_before_mb": 129.52,
    "process_rss_after_mb": 2157.91,
    "process_rss_delta_mb": 2028.39,
    "output_characters": 19,
    "memory_note": "RSS is process RAM before/after, NOT peak RAM or GPU memory.",
    "latency_note": "Cold/load cost may vary; GPU execution is asynchronous internally.",
    "workflow_total_ms": 6324.04,
    "workflow_note": "Whole LangGraph workflow including planning, retrieval, routing, model load, generation, and verification as applicable."
  },
  "rag_diagnostics": {}
}
```

### tests.test_live_hardware.LocalInferenceTests.test_cpu_code

```json
{
  "model_id": "openvino-qwen2.5-1.5b-instruct-int4",
  "device": "CPU",
  "task_type": "code",
  "max_new_tokens": 12,
  "evidence_sources": []
}
```

### tests.test_live_hardware.LocalInferenceTests.test_cpu_text

```json
{
  "model_id": "openvino-qwen2.5-1.5b-instruct-int4",
  "device": "CPU",
  "task_type": "text",
  "max_new_tokens": 12,
  "evidence_sources": []
}
```

### tests.test_live_hardware.LocalInferenceTests.test_gpu_code

```json
{
  "model_id": "openvino-qwen2.5-1.5b-instruct-int4",
  "device": "GPU",
  "task_type": "code",
  "max_new_tokens": 12,
  "evidence_sources": []
}
```

### tests.test_live_hardware.LocalInferenceTests.test_gpu_text

```json
{
  "model_id": "openvino-qwen2.5-1.5b-instruct-int4",
  "device": "GPU",
  "task_type": "text",
  "max_new_tokens": 12,
  "evidence_sources": []
}
```

### tests.test_live_llama.LlamaInferenceTests.test_cpu_code

```json
{
  "model_id": "openvino-llama3.2-1b-instruct-int4",
  "device": "CPU",
  "task_type": "code",
  "max_new_tokens": 12,
  "evidence_sources": []
}
```

### tests.test_live_llama.LlamaInferenceTests.test_cpu_text

```json
{
  "model_id": "openvino-llama3.2-1b-instruct-int4",
  "device": "CPU",
  "task_type": "text",
  "max_new_tokens": 12,
  "evidence_sources": []
}
```

### tests.test_live_llama.LlamaInferenceTests.test_gpu_code

```json
{
  "model_id": "openvino-llama3.2-1b-instruct-int4",
  "device": "GPU",
  "task_type": "code",
  "max_new_tokens": 12,
  "evidence_sources": []
}
```

### tests.test_live_llama.LlamaInferenceTests.test_gpu_text

```json
{
  "model_id": "openvino-llama3.2-1b-instruct-int4",
  "device": "GPU",
  "task_type": "text",
  "max_new_tokens": 12,
  "evidence_sources": []
}
```

### tests.test_live_llama.LlamaInferenceTests.test_llama_auto_from_real_benchmarks

```json
{
  "model_id": "openvino-llama3.2-1b-instruct-int4",
  "device": "CPU",
  "task_type": "text",
  "max_new_tokens": 32,
  "evidence_sources": []
}
```

### tests.test_live_llama.LlamaInferenceTests.test_llama_rag_summary_cpu

```json
{
  "model_id": "openvino-llama3.2-1b-instruct-int4",
  "device": "CPU",
  "task_type": "text",
  "max_new_tokens": 32,
  "evidence_sources": [
    "gpu-guide#chunk-001"
  ]
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
