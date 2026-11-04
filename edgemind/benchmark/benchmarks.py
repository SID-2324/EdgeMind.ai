"""Stage 9: reproducible, local-only cold-start OpenVINO microbenchmarks.

Each sample includes a fresh pipeline construction, because the current local
adapter does that on *every* request. These are not model-quality scores,
GPU-kernel timings, or statistically representative production benchmarks.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
from ..local_models import QWEN, for_id

from ..adapters.openvino_qwen import (
    generate_local_profiled, runtime_hardware_profile, LocalModelError,
)

SCHEMA_VERSION = 1
MODEL_ID = 'openvino-qwen2.5-1.5b-instruct-int4'
PROMPT_PROFILE = 'edgemind-stage9-cold-request-v1'
PROMPT = 'Explain the difference between a CPU and GPU in exactly one concise sentence.'
SUPPORTED_DEVICES = ('CPU', 'GPU')
MIN_AVAILABLE_MEMORY_MB = 2300  # conservative requirement, not a measured peak


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not-installed'


def machine_fingerprint() -> str:
    """One-way identifier; no raw host name or account name in the report."""
    return _sha256('|'.join((platform.node(), platform.machine(),
                            platform.processor(), platform.system(),
                            _package_version('openvino'),
                            _package_version('openvino-genai'))))


def model_fingerprint(model_path: str) -> str:
    """Detect path or core configuration changes without exposing the path."""
    folder = Path(model_path).expanduser().resolve()
    parts = [str(folder)]
    for filename in ('config.json', 'openvino_model.xml', 'openvino_model.bin',
                     'generation_config.json', 'tokenizer_config.json'):
        child = folder / filename
        if child.is_file():
            st = child.stat()
            parts.append(f'{filename}:{st.st_size}:{st.st_mtime_ns}')
    return _sha256('|'.join(parts))


def _valid_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _median(samples, field):
    return round(statistics.median(sample[field] for sample in samples), 2)


def run_benchmarks(*, model_path: str, devices=('GPU', 'CPU'), runs=2, warmups=0,
                   max_new_tokens=64, runner=None, hardware=None, now=None,
                   model_id=MODEL_ID) -> dict:
    """Perform real profiling, or use an injected runner in deterministic tests.

    Failed device runs never become eligible routing observations. Reports
    contain timings and hashes only, not prompts, answers or model paths.
    """
    model = for_id(model_id)
    if type(runs) is not int or not 1 <= runs <= 20:
        raise ValueError('runs must be an integer in 1..20')
    if type(warmups) is not int or not 0 <= warmups <= 5:
        raise ValueError('warmups must be an integer in 0..5')
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
        raise ValueError('max_new_tokens must be in 1..512')
    devices = tuple(devices)
    if not devices or len(set(devices)) != len(devices) or any(d not in SUPPORTED_DEVICES for d in devices):
        raise ValueError('devices must be distinct CPU/GPU choices')
    if not Path(model_path).expanduser().is_dir():
        raise ValueError('Model directory not found; provide --model-path to existing OpenVINO Qwen files')
    runtime = hardware if hardware is not None else runtime_hardware_profile()
    runner = runner if runner is not None else generate_local_profiled
    date = now if now is not None else utc_now()
    if date.tzinfo is None:
        raise ValueError('benchmark date must have a timezone')
    report = {
        'schema_version': SCHEMA_VERSION,
        'created_utc': date.astimezone(timezone.utc).isoformat(),
        'machine_fingerprint': machine_fingerprint(),
        'model_fingerprint': model_fingerprint(model_path),
        'model_id': model_id,
        'prompt_profile': PROMPT_PROFILE,
        'max_new_tokens': max_new_tokens,
        'runs_per_device': runs,
        'warmups_per_device': warmups,
        'measurement_scope': 'cold-adapter-total-includes-model-load',
        'devices': {},
    }
    for device in devices:
        if not runtime['devices'].get(device, False):
            report['devices'][device] = {'status': 'unavailable'}
            continue
        if runtime['available_memory_mb'] < model.required_memory_mb:
            report['devices'][device] = {'status': 'insufficient_ram'}
            continue
        observations = []
        try:
            for attempt in range(warmups + runs):
                # Each call creates a fresh OpenVINO pipeline by design.
                identity = {'model_id': model_id} if model_id != QWEN.model_id else {}
                _result, performance = runner(
                    query=PROMPT, evidence=[], operation='generate_text',
                    model_path=model_path, device=device, max_new_tokens=max_new_tokens, **identity)
                if not isinstance(_result, str) or not _result.strip() or not isinstance(performance, dict):
                    raise ValueError('Adapter returned invalid benchmark result')
                sample = {key: performance.get(key) for key in
                          ('model_load_ms', 'generation_ms', 'adapter_total_ms', 'output_characters')}
                if any(not _valid_number(sample[key]) for key in sample):
                    raise ValueError('Adapter returned invalid benchmark measurements')
                if sample['adapter_total_ms'] < sample['generation_ms']:
                    raise ValueError('Adapter total cannot be less than generation duration')
                if attempt >= warmups:
                    observations.append(sample)
            report['devices'][device] = {
                'status': 'ok', 'count': len(observations),
                'median_adapter_total_ms': _median(observations, 'adapter_total_ms'),
                'median_generation_ms': _median(observations, 'generation_ms'),
                'median_model_load_ms': _median(observations, 'model_load_ms'),
                'samples': observations,
            }
        except (LocalModelError, ValueError, RuntimeError, OSError):
            # Do not persist error messages, which may contain paths/prompts.
            report['devices'][device] = {'status': 'failed'}
    return report


def save_report(report: dict, output_path: str) -> None:
    """Atomic JSON write, using a unique temp file in the same directory."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', prefix='.edgemind-',
                                         dir=target.parent, encoding='utf-8', delete=False) as handle:
            name = handle.name
            json.dump(report, handle, indent=2, allow_nan=False)
        os.replace(name, target)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def eligible_observations(*, benchmark_file: str, model_path: str,
                          max_new_tokens: int, max_age_hours: int = 168,
                          now=None, model_id=MODEL_ID) -> tuple[dict[str, float], str]:
    """Fail closed on missing, old, malformed or mismatched measurements.

    Valid readings are returned per device, never inferred for an unmeasured
    device. The caller must still apply hardware/privacy/capability filters.
    """
    for_id(model_id)
    if type(max_age_hours) is not int or not 1 <= max_age_hours <= 24 * 365:
        raise ValueError('max_age_hours must be in 1..8760')
    try:
        obj = json.loads(Path(benchmark_file).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, 'benchmark file missing or unreadable; run benchmark_devices.py first'
    if not isinstance(obj, dict) or obj.get('schema_version') != SCHEMA_VERSION:
        return {}, 'unknown benchmark schema'
    if (obj.get('model_id') != model_id or
        obj.get('prompt_profile') != PROMPT_PROFILE or
        obj.get('measurement_scope') != 'cold-adapter-total-includes-model-load' or
        obj.get('model_fingerprint') != model_fingerprint(model_path) or
        obj.get('machine_fingerprint') != machine_fingerprint()):
        return {}, 'benchmark model, machine, or methodology mismatch'
    if type(obj.get('max_new_tokens')) is not int or obj['max_new_tokens'] != max_new_tokens:
        return {}, 'benchmark token limit does not match request; rerun with matching --max-new-tokens'
    try:
        timestamp = datetime.fromisoformat(obj['created_utc'])
    except (TypeError, ValueError, KeyError):
        return {}, 'invalid benchmark timestamp'
    at = now if now is not None else utc_now()
    if timestamp.tzinfo is None:
        return {}, 'benchmark timestamp missing timezone'
    age_hours = (at - timestamp.astimezone(timezone.utc)).total_seconds() / 3600
    if age_hours < -0.1 or age_hours > max_age_hours:
        return {}, 'benchmark is stale or dated in the future'
    devices = obj.get('devices')
    if not isinstance(devices, dict):
        return {}, 'benchmark device readings missing'
    accepted = {}
    for device in SUPPORTED_DEVICES:
        entry = devices.get(device)
        if not isinstance(entry, dict) or entry.get('status') != 'ok':
            continue
        duration = entry.get('median_adapter_total_ms')
        if (_valid_number(duration) and duration > 0 and
            type(entry.get('count')) is int and entry['count'] >= 1 and
            isinstance(entry.get('samples'), list) and len(entry['samples']) == entry['count'] and
            all(isinstance(s, dict) and _valid_number(s.get('adapter_total_ms')) for s in entry['samples'])):
            # Don't trust a tampered aggregate that contradicts its samples.
            calculated = statistics.median(s['adapter_total_ms'] for s in entry['samples'])
            if abs(calculated - duration) <= .02:
                accepted[device] = float(duration)
    if not accepted:
        return {}, 'no valid successful CPU/GPU benchmark observations'
    return accepted, f'accepted {len(accepted)} matching device benchmark(s)'
