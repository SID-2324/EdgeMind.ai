"""Stable entry point Layer 1 can call without knowing about LangGraph internals.

Mock mode is the default. Real local Qwen inference is opt-in only.
"""
from .contracts import Evidence, HardwareProfile, ModelCapability, RoutingPreferences, mock_hardware_profile, mock_model_registry
from .graph import make_graph
from .adapters.openvino_qwen import runtime_hardware_profile
from .state import EdgeState
import time
from .local_models import REAL_MODES, for_mode


def run_task(
    query: str,
    privacy_mode: str = 'local_only',
    network: str = 'online',
    complexity: str = 'simple',
    task_type: str = 'text',
    documents: list[Evidence] | None = None,
    available_models: list[ModelCapability] | None = None,
    hardware_profile: HardwareProfile | None = None,
    routing_preferences: RoutingPreferences | None = None,
    simulate_failure: bool = False,
    simulate_permanent_failure: bool = False,
    simulate_invalid_result: bool = False,
    rag_settings: dict | None = None,
    execution_mode: str = 'mock',
    local_model_path: str | None = None,
    local_device: str = 'GPU',
    max_new_tokens: int = 160,
    profile_local: bool = False,
    benchmark_file: str | None = None,
    benchmark_max_age_hours: int = 168,
) -> EdgeState:
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Query cannot be empty')
    if execution_mode != 'mock' and execution_mode not in REAL_MODES:
        raise ValueError('execution_mode must be mock, local_qwen or local_llama')
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
        raise ValueError('max_new_tokens must be between 1 and 512')
    if type(profile_local) is not bool:
        raise ValueError('profile_local must be a boolean')
    if profile_local and execution_mode not in REAL_MODES:
        raise ValueError('Performance profiling requires real local inference')
    if execution_mode in REAL_MODES:
        model = for_mode(execution_mode)
        if privacy_mode != 'local_only':
            raise ValueError('Real local execution requires local_only privacy')
        if local_device not in ('GPU', 'CPU', 'auto'):
            raise ValueError('OpenVINO Qwen generation supports GPU, CPU, or benchmark-aware auto selection')
        if not local_model_path:
            raise ValueError('local_model_path required for real local Qwen inference')
        if available_models is not None or hardware_profile is not None:
            raise ValueError('Real mode probes actual hardware and uses a fixed allowlisted model')
        if simulate_failure or simulate_permanent_failure or simulate_invalid_result:
            raise ValueError('Simulated failure flags require mock mode')
        hardware_profile = runtime_hardware_profile()
        if local_device == 'auto':
            # Only locally measured and matching devices enter the real registry.
            # Missing/stale data => no model selected, not a guessed GPU choice.
            from .benchmark.benchmarks import eligible_observations
            observed, diagnostic = eligible_observations(
                benchmark_file=benchmark_file or model.benchmark_file,
                model_path=local_model_path, max_new_tokens=max_new_tokens,
                max_age_hours=benchmark_max_age_hours, model_id=model.model_id)
            available_models = [{
                'model_id': model.model_id,
                'provider': 'local', 'device': device,
                'complexities': ['simple'], 'task_types': ['text', 'code'],
                'available': True, 'required_memory_mb': model.required_memory_mb,
                'estimated_latency_ms': max(1, round(latency)),
                'estimated_cost_usd': 0.0, 'estimated_quality_score': model.quality_estimate,
            } for device, latency in observed.items()]
        else:
            diagnostic = 'manual device: no benchmark-based routing'
            available_models = [{
                'model_id': model.model_id,
                'provider': 'local', 'device': local_device,
                'complexities': ['simple'], 'task_types': ['text', 'code'],
                'available': True, 'required_memory_mb': model.required_memory_mb,
                'estimated_latency_ms': 650, 'estimated_cost_usd': 0.0,
                'estimated_quality_score': model.quality_estimate,
            }]
    initial: EdgeState = {
        'query': query,
        'execution_mode': execution_mode,
        'local_model_path': local_model_path,
        'local_device_preference': local_device,
        'benchmark_diagnostics': diagnostic if execution_mode in REAL_MODES else 'mock estimates only',
        'max_new_tokens': max_new_tokens,
        'profile_local': profile_local,
        'privacy_mode': privacy_mode,
        'network': network,
        'complexity': complexity,
        'task_type': task_type,
        'available_models': available_models if available_models is not None else mock_model_registry(),
        'hardware_profile': hardware_profile if hardware_profile is not None else mock_hardware_profile(),
        'routing_preferences': routing_preferences if routing_preferences is not None else {},
        'documents': documents if documents is not None else [],
        'simulate_failure': simulate_failure,
        'simulate_permanent_failure': simulate_permanent_failure,
        'simulate_invalid_result': simulate_invalid_result,
        'verification_history': [],
        'rag_settings': rag_settings if rag_settings is not None else {},
        'attempts': 0,
        'step_results': {},
        'executed_steps': [],
        'trace': [],
    }
    start = time.perf_counter() if profile_local else None
    result = make_graph().invoke(initial, config={'recursion_limit': 60})
    if profile_local:
        performance = dict(result.get('performance') or {})
        performance['measurement_type'] = 'observed'
        performance['workflow_total_ms'] = round((time.perf_counter() - start) * 1000, 2)
        performance['workflow_note'] = ('Whole LangGraph workflow including planning, retrieval, routing, '
                                        'model load, generation, and verification as applicable.')
        result['performance'] = performance
    return result
