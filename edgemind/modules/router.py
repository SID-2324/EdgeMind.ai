"""Stage 4 router: strict eligibility rules, then estimated performance optimization.

This is a PURE function using SIMULATED model/hardware/benchmark profiles.
No real models, drivers, web requests, or device probing happen here.

Order is critical: policy -> capability -> hardware -> constraints -> ranking.
Never relax a hard policy just to improve a score.
"""
import math
from ..local_models import REAL_MODES

from ..contracts import (
    HardwareProfile, ModelCapability, ModelDecision, RoutingPreferences,
    mock_hardware_profile,
)
from ..state import EdgeState

VALID_TASK_TYPES = {'text', 'code', 'classification'}
LOCAL_DEVICES = {'CPU', 'GPU', 'NPU'}
VALID_STRATEGIES = {'balanced', 'fastest', 'cheapest', 'highest_quality'}


def validate_hardware(profile: HardwareProfile) -> None:
    if not isinstance(profile, dict):
        raise ValueError('Hardware profile must be a dictionary')
    memory = profile.get('available_memory_mb')
    if type(memory) is not int or memory < 0:
        raise ValueError('available_memory_mb must be a nonnegative integer')
    devices = profile.get('devices')
    if not isinstance(devices, dict) or any(type(v) is not bool for v in devices.values()):
        raise ValueError('devices must map device names to booleans')
    if any(name not in LOCAL_DEVICES for name in devices):
        raise ValueError('Unexpected local hardware device')


def _number(value: object) -> bool:
    return type(value) in {float, int} and math.isfinite(value)


def _valid_benchmark(model: dict) -> bool:
    latency = model.get('estimated_latency_ms')
    cost = model.get('estimated_cost_usd')
    quality = model.get('estimated_quality_score')
    return (type(latency) is int and latency > 0
            and _number(cost) and cost >= 0
            and _number(quality) and 0 <= quality <= 1)


def validate_preferences(pref: RoutingPreferences) -> RoutingPreferences:
    if not isinstance(pref, dict):
        raise ValueError('Routing preferences must be a dictionary')
    if set(pref).difference({'strategy','max_latency_ms','max_cost_usd','min_quality_score'}):
        raise ValueError('Unexpected routing preference')
    strategy = pref.get('strategy', 'balanced')
    if strategy not in VALID_STRATEGIES:
        raise ValueError('Invalid routing strategy')
    latency = pref.get('max_latency_ms')
    cost = pref.get('max_cost_usd')
    quality = pref.get('min_quality_score')
    if latency is not None and (type(latency) is not int or latency <= 0):
        raise ValueError('max_latency_ms must be a positive integer')
    if cost is not None and (not _number(cost) or cost < 0):
        raise ValueError('max_cost_usd must be nonnegative')
    if quality is not None and (not _number(quality) or not 0 <= quality <= 1):
        raise ValueError('min_quality_score must be between 0 and 1')
    return pref


def _balanced_score(m: ModelCapability) -> float:
    """Simple heuristic on fictional scales; never presented as calibrated quality."""
    quality = m['estimated_quality_score']
    latency = m['estimated_latency_ms']
    cost = m['estimated_cost_usd']
    return (0.65 * quality
            + 0.25 / (1 + latency / 1000)
            + 0.10 / (1 + cost / 0.002))


def choose_model(
    privacy_mode: str,
    network: str,
    complexity: str,
    registry: list[ModelCapability],
    task_type: str = 'text',
    hardware: HardwareProfile | None = None,
    preferences: RoutingPreferences | None = None,
) -> ModelDecision:
    if privacy_mode not in {'local_only', 'cloud_allowed'}:
        raise ValueError('Invalid privacy mode')
    if network not in {'online', 'offline'}:
        raise ValueError('Invalid network status')
    if complexity not in {'simple', 'complex'}:
        raise ValueError('Invalid complexity')
    if task_type not in VALID_TASK_TYPES:
        raise ValueError('Unsupported task type')
    if not isinstance(registry, list):
        raise ValueError('Model registry must be a list')

    hw = hardware if hardware is not None else mock_hardware_profile()
    validate_hardware(hw)
    pref: RoutingPreferences = preferences if preferences is not None else {}
    validate_preferences(pref)
    strategy = pref.get('strategy', 'balanced')
    eligible: list[ModelCapability] = []
    rejected: list[str] = []

    for model in registry:
        name = model.get('model_id', '(unknown)')
        provider = model.get('provider')
        device = model.get('device')
        memory = model.get('required_memory_mb')
        reason: str | None = None

        if not model.get('available', False):
            reason = 'model unavailable'
        elif complexity not in model.get('complexities', []):
            reason = 'complexity unsupported'
        elif task_type not in model.get('task_types', []):
            reason = 'task type unsupported'
        elif provider not in {'local', 'cloud'}:
            reason = 'invalid provider'
        elif provider == 'cloud' and privacy_mode == 'local_only':
            reason = 'cloud forbidden by privacy policy'
        elif provider == 'cloud' and network != 'online':
            reason = 'cloud unavailable while offline'
        elif provider == 'cloud' and device != 'CLOUD':
            reason = 'invalid cloud device declaration'
        elif provider == 'local' and device not in LOCAL_DEVICES:
            reason = 'unsupported local execution device'
        elif provider == 'local' and not hw['devices'].get(device, False):
            reason = f'{device} device unavailable'
        elif (type(memory) is not int or
              (provider == 'local' and memory <= 0) or
              (provider == 'cloud' and memory < 0)):
            reason = 'missing or invalid memory requirement'
        elif provider == 'local' and memory > hw['available_memory_mb']:
            reason = f'insufficient free RAM ({memory}MB required, {hw["available_memory_mb"]}MB available)'
        elif not _valid_benchmark(model):
            reason = 'missing or invalid performance estimates'
        elif pref.get('max_latency_ms') is not None and model['estimated_latency_ms'] > pref['max_latency_ms']:
            reason = f'estimated latency exceeds {pref["max_latency_ms"]}ms limit'
        elif pref.get('max_cost_usd') is not None and model['estimated_cost_usd'] > pref['max_cost_usd']:
            reason = f'estimated cost exceeds ${pref["max_cost_usd"]} limit'
        elif pref.get('min_quality_score') is not None and model['estimated_quality_score'] < pref['min_quality_score']:
            reason = f'estimated quality below {pref["min_quality_score"]} threshold'

        if reason:
            rejected.append(f'{name}: {reason}')
        else:
            eligible.append(model)

    if not eligible:
        return {
            'outcome': 'deferred', 'model_id': None, 'provider': None,
            'device': None, 'strategy': strategy, 'estimates': None,
            'reason': f'No permitted, capable, hardware-compatible model meets constraints for {task_type}/{complexity}.',
            'rejected': rejected,
        }

    if strategy == 'fastest':
        key = lambda m: (m['estimated_latency_ms'], -m['estimated_quality_score'],
                         m['estimated_cost_usd'], m['model_id'])
    elif strategy == 'cheapest':
        key = lambda m: (m['estimated_cost_usd'], -m['estimated_quality_score'],
                         m['estimated_latency_ms'], m['model_id'])
    elif strategy == 'highest_quality':
        key = lambda m: (-m['estimated_quality_score'], m['estimated_cost_usd'],
                         m['estimated_latency_ms'], m['model_id'])
    else:
        key = lambda m: (-_balanced_score(m), m['estimated_cost_usd'],
                         m['estimated_latency_ms'], m['model_id'])

    selected = min(eligible, key=key)
    estimates = {
        'latency_ms': selected['estimated_latency_ms'],
        'cost_usd': selected['estimated_cost_usd'],
        'quality_score': selected['estimated_quality_score'],
    }
    return {
        'outcome': 'selected', 'model_id': selected['model_id'],
        'provider': selected['provider'], 'device': selected['device'],
        'strategy': strategy, 'estimates': estimates,
        'reason': (f'All hard checks passed. Ranked {len(eligible)} simulated eligible '
                   f'model(s) by {strategy}. Estimates are fictional, not observed performance.'),
        'rejected': rejected,
    }


def route_model(state: EdgeState) -> dict:
    result = choose_model(
        state['privacy_mode'], state['network'], state['complexity'],
        state['available_models'], task_type=state.get('task_type', 'text'),
        hardware=state.get('hardware_profile'),
        preferences=state.get('routing_preferences'),
    )
    if state.get('execution_mode') in REAL_MODES:
        if state.get('local_device_preference') == 'auto':
            if result['outcome'] == 'selected':
                # The Router keeps its historic 'estimates' contract for existing
                # UIs; this extra field explains the observed source of latency.
                result['latency_source'] = 'observed_cold_adapter_median'
                result['observed_median_adapter_total_ms'] = result['estimates']['latency_ms']
                result['reason'] = (
                    'Privacy/capability/hardware checks passed. Selected from matching '
                    'local CPU/GPU cold-start benchmark medians; quality and cost '
                    'are assumptions, not measured benchmarks.')
            else:
                result['latency_source'] = 'none'
                result['reason'] += ' ' + state.get('benchmark_diagnostics', '')
        else:
            result['latency_source'] = 'unmeasured_manual_device'
            result['reason'] = ('Explicit manual local device; router latency/quality '
                                'values are placeholder estimates. ' + result['reason'])
    label = f"{result['model_id']} ({result['device']})" if result['outcome'] == 'selected' else 'DEFERRED'
    return {'decision': result, 'trace': [f"ROUTER: {label} — {result['reason']}"]}
