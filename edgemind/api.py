"""Stage 10: opt-in local browser/API facade over the *unchanged* EdgeMind service.

Run with: python -m uvicorn edgemind.api:app --host 127.0.0.1 --port 8000
This app is for one local user. No authentication, no CORS, no cloud adapter,
no file uploads or persistent task history. Do not expose it on a public host.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from .service import run_task
from .local_models import REAL_MODES, MODELS, LLAMA, for_mode

WEB_ROOT = Path(__file__).resolve().parent / 'web'
REAL_LOCK = threading.Lock()

app = FastAPI(
    title='EdgeMind Local API', version='0.11.0',
    description='Local-first orchestration demo. Mock by default; local Llama/Qwen opt-in. No cloud execution.',
)
app.mount('/assets', StaticFiles(directory=str(WEB_ROOT)), name='assets')


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    source_id: str = Field(min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9._/:-]+$')
    text: str = Field(min_length=1, max_length=12000)


class TaskRequest(BaseModel):
    """The browser deliberately offers a narrow, allowlisted subset of Layer 2."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=2000)
    execution_mode: Literal['mock', 'local_qwen', 'local_llama'] = 'mock'
    task_type: Literal['text', 'code', 'classification'] = 'text'
    privacy_mode: Literal['local_only', 'cloud_allowed'] = 'local_only'
    network: Literal['online', 'offline'] = 'online'
    complexity: Literal['simple', 'complex'] = 'simple'
    strategy: Literal['balanced', 'fastest', 'cheapest', 'highest_quality'] = 'balanced'
    local_device: Literal['GPU', 'CPU', 'auto'] = 'GPU'
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    rag_mode: Literal['bm25', 'hybrid'] = 'bm25'
    rerank: bool = False
    rag_top_k: int = Field(default=3, ge=1, le=8)
    documents: list[DocumentInput] = Field(default_factory=list, max_length=6)

    @model_validator(mode='after')
    def check_combinations(self):
        if len({doc.source_id for doc in self.documents}) != len(self.documents):
            raise ValueError('Document source IDs must be unique')
        if sum(len(doc.text) for doc in self.documents) > 36000:
            raise ValueError('Combined document text exceeds 36,000 characters')
        if self.rerank and self.rag_mode != 'hybrid':
            raise ValueError('Reranking requires hybrid retrieval')
        if self.execution_mode in REAL_MODES:
            if self.privacy_mode != 'local_only':
                raise ValueError('Real local execution requires local_only privacy')
            if self.task_type not in ('text', 'code'):
                raise ValueError('Real local models do not support the classification adapter')
        return self


def real_enabled() -> bool:
    return os.getenv('EDGEMIND_ENABLE_REAL_LOCAL', '0') == '1'


def real_model_path(mode='local_qwen') -> str:
    variable = 'EDGEMIND_LLAMA_MODEL_PATH' if mode == LLAMA.mode else 'EDGEMIND_MODEL_PATH'
    return os.getenv(variable, for_mode(mode).directory)


def benchmark_path(mode='local_qwen') -> str:
    variable = 'EDGEMIND_LLAMA_BENCHMARK_FILE' if mode == LLAMA.mode else 'EDGEMIND_BENCHMARK_FILE'
    return os.getenv(variable, for_mode(mode).benchmark_file)


@app.get('/', include_in_schema=False)
def index():
    return FileResponse(WEB_ROOT / 'index.html')


@app.get('/licenses/llama', include_in_schema=False)
def llama_license():
    return FileResponse(WEB_ROOT.parents[1] / 'LICENSE-Llama-3.2.txt', media_type='text/plain')


@app.get('/licenses/llama-use-policy', include_in_schema=False)
def llama_use_policy():
    return FileResponse(WEB_ROOT.parents[1] / 'LLAMA_ACCEPTABLE_USE_POLICY.md', media_type='text/plain')


@app.get('/api/health')
def health():
    """No hardware probe or model load; does not reveal filesystem paths."""
    return {
        'status': 'ok', 'app': 'EdgeMind', 'stage': 11,
        'default_mode': 'mock',
        'default_demo_model': 'llama',
        'local_models': [{'key': model.key, 'label': model.label, 'execution_mode': model.mode,
                          'model_id': model.model_id,
                          'directory_exists': Path(real_model_path(model.mode)).is_dir(),
                          'benchmark_exists': Path(benchmark_path(model.mode)).is_file()}
                         for model in MODELS],
        'real_local_enabled': real_enabled(),
        'model_directory_exists': Path(real_model_path()).is_dir(),
        'benchmark_file_exists': Path(benchmark_path()).is_file(),
        'cloud_execution_implemented': False,
        'network_scope': 'local-only server; bind to 127.0.0.1',
    }


def _run_task_sync(payload: TaskRequest):
    """Run the synchronous LangGraph workflow in a thread, not on the event loop."""
    is_real = payload.execution_mode in REAL_MODES
    if is_real and not REAL_LOCK.acquire(blocking=False):
        raise RuntimeError('REAL_BUSY')
    try:
        result = run_task(
            query=payload.query,
            execution_mode=payload.execution_mode,
            task_type=payload.task_type,
            privacy_mode=payload.privacy_mode,
            network=payload.network,
            complexity=payload.complexity,
            documents=[doc.model_dump() for doc in payload.documents],
            routing_preferences={'strategy': payload.strategy},
            rag_settings={
                'mode': payload.rag_mode,
                'top_k': payload.rag_top_k,
                'enable_reranker': payload.rerank,
                'candidate_k': 10,
            },
            local_model_path=real_model_path(payload.execution_mode) if is_real else None,
            local_device=payload.local_device,
            benchmark_file=benchmark_path(payload.execution_mode) if is_real else None,
            max_new_tokens=payload.max_new_tokens,
            profile_local=is_real,
        )
        decision = result.get('decision') or {}
        verification = result.get('verification') or {}
        outcome = result.get('tool_result') or {}
        trace = result.get('trace') or []
        if is_real and outcome.get('status') == 'failed':
            # Adapter exceptions are untrusted diagnostics and can contain paths,
            # prompts or credentials. They also propagate into verifier/trace.
            safe_error = 'Local inference failed. Check the configured model and local runtime.'
            outcome = {**outcome, 'error': safe_error}
            verification = {**verification, 'explanation': safe_error}
            trace = ['Local inference failed; adapter diagnostics omitted for privacy.']
        # Explicitly distinguish real generation from mock mode in *every* response.
        return {
            'status': result.get('status', 'UNKNOWN'),
            'execution_mode': payload.execution_mode,
            'is_real_inference': bool(is_real
                                      and verification.get('status') == 'COMPLETED'),
            'result': outcome.get('content', ''),
            'error': outcome.get('error'),
            'model': {
                'model_id': decision.get('model_id'),
                'provider': decision.get('provider'),
                'device': decision.get('device'),
                'outcome': decision.get('outcome'),
                'reason': decision.get('reason'),
                'latency_source': decision.get('latency_source'),
                'observed_median_adapter_total_ms': decision.get('observed_median_adapter_total_ms'),
                'estimates': decision.get('estimates'),
            },
            'plan': result.get('plan_steps') or [],
            'executed_steps': result.get('executed_steps') or [],
            'evidence': [
                {'source_id': entry.get('source_id'), 'text': str(entry.get('text', ''))[:1200]}
                for entry in (result.get('evidence') or [])[:8]
            ],
            'verification': verification,
            'recovery_action': result.get('recovery_action'),
            'performance': result.get('performance') or {},
            'rag_diagnostics': result.get('rag_diagnostics') or {},
            'benchmark_diagnostics': result.get('benchmark_diagnostics'),
            'trace': trace[:80],
            'notice': ('Real local inference; checks verify result structure, not factual accuracy.'
                       if is_real and verification.get('status') == 'COMPLETED' else
                       'Real local inference requested; no verified result was produced.' if is_real else
                       'SIMULATED generation. Model names, quality and latency are not measured.'),
        }
    finally:
        if is_real:
            REAL_LOCK.release()


@app.post('/api/tasks')
async def execute_task(payload: TaskRequest):
    if payload.execution_mode in REAL_MODES:
        if not real_enabled():
            raise HTTPException(status_code=403, detail='Real local mode is disabled. Set EDGEMIND_ENABLE_REAL_LOCAL=1 on the server.')
        if not Path(real_model_path(payload.execution_mode)).is_dir():
            raise HTTPException(status_code=409, detail='Selected local model folder not found. Configure its server model path.')
    try:
        return await run_in_threadpool(_run_task_sync, payload)
    except RuntimeError as exc:
        if str(exc) == 'REAL_BUSY':
            raise HTTPException(status_code=409, detail='Another local model task is running. Retry after it finishes.') from exc
        raise HTTPException(status_code=500, detail='Task failed. Check the local server console for details.') from exc
    except (ValueError, OSError) as exc:
        # Some lower-layer errors may contain sensitive file paths; don't echo them.
        raise HTTPException(status_code=422, detail='Workflow could not run with these settings. Check server logs.') from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail='Unexpected local execution error. Check the server console.') from exc
