"""Fixed local generation allowlist. Paths are server/CLI configuration only."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalModel:
    key: str
    model_id: str
    label: str
    mode: str
    directory: str
    benchmark_file: str
    required_memory_mb: int
    quality_estimate: float


QWEN = LocalModel('qwen', 'openvino-qwen2.5-1.5b-instruct-int4', 'Qwen 2.5 1.5B',
                  'local_qwen', 'models/qwen-1.5b', 'benchmarks_local.json', 2300, .88)
LLAMA = LocalModel('llama', 'openvino-llama3.2-1b-instruct-int4', 'Meta Llama 3.2 1B',
                   'local_llama', 'models/llama-3.2-1b', 'benchmarks_llama.json', 1800, .80)
MODELS = (LLAMA, QWEN)
REAL_MODES = frozenset(model.mode for model in MODELS)


def for_mode(mode: str) -> LocalModel:
    for model in MODELS:
        if model.mode == mode:
            return model
    raise ValueError('Unsupported local execution mode')


def for_id(model_id: str) -> LocalModel:
    for model in MODELS:
        if model.model_id == model_id:
            return model
    raise ValueError('Unsupported local model identity')
