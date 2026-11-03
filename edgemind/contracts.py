"""Shared model, hardware, and optimization contracts for Stage 4.

All benchmark estimates and capabilities are FICTIONAL. No real LLM is used.
"""
from typing import Literal, TypedDict, NotRequired

PrivacyMode = Literal['local_only', 'cloud_allowed']
Complexity = Literal['simple', 'complex']
TaskType = Literal['text', 'code', 'classification']


class ModelCapability(TypedDict):
    model_id: str
    provider: Literal['local', 'cloud']
    device: str
    complexities: list[str]
    task_types: list[TaskType]
    available: bool
    # Estimated free system RAM required for THIS workload (not model file size).
    # Local estimates are simplified, simulated budgets, not measured benchmarks.
    required_memory_mb: int
    # Hypothetical routing estimates, not observed hardware benchmarks.
    estimated_latency_ms: int
    estimated_cost_usd: float
    estimated_quality_score: float


class HardwareProfile(TypedDict):
    # Simulated uncommitted RAM after OS and other applications use their share.
    # Integrated Intel Arc and CPU share RAM; don't count the same RAM twice.
    available_memory_mb: int
    devices: dict[str, bool]


class RoutingPreferences(TypedDict, total=False):
    strategy: Literal['balanced', 'fastest', 'cheapest', 'highest_quality']
    max_latency_ms: int | None
    max_cost_usd: float | None
    min_quality_score: float | None


class ModelDecision(TypedDict):
    outcome: Literal['selected', 'deferred']
    model_id: str | None
    provider: str | None
    device: str | None
    reason: str
    rejected: list[str]
    strategy: str
    estimates: dict[str, float] | None


# Stage 6A: declarative planning (no tools executed by this data structure).
PlanOperation = Literal['retrieve_evidence', 'generate_text', 'generate_code',
                        'classify_input', 'verify_output']


class PlanStep(TypedDict):
    step_id: str
    operation: PlanOperation
    depends_on: list[str]
    description: str


class Evidence(TypedDict):
    source_id: str
    text: str


class ToolResult(TypedDict):
    status: Literal['simulated', 'completed', 'failed', 'not_run']
    content: str
    error: str | None
    error_code: NotRequired[str | None]


class Verification(TypedDict):
    status: Literal['SIMULATED', 'COMPLETED', 'FAILED', 'BLOCKED']
    explanation: str
    checks: NotRequired[list[dict]]
    error_code: NotRequired[str | None]


def mock_hardware_profile() -> HardwareProfile:
    """A fake machine with 4 GiB *available* RAM, not 4 GiB total RAM."""
    return {
        'available_memory_mb': 4096,
        'devices': {'CPU': True, 'GPU': True, 'NPU': True},
    }


def mock_model_registry() -> list[ModelCapability]:
    """Fictional options used only to test the routing algorithm."""
    return [
        {
            'model_id': 'mock-qwen-1.5b', 'provider': 'local', 'device': 'GPU',
            'complexities': ['simple'], 'task_types': ['text', 'code'],
            'available': True, 'required_memory_mb': 2300,
            'estimated_latency_ms': 650, 'estimated_cost_usd': 0.0,
            'estimated_quality_score': 0.88,
        },
        {
            'model_id': 'mock-tiny-cpu', 'provider': 'local', 'device': 'CPU',
            'complexities': ['simple'], 'task_types': ['text', 'code'],
            'available': True, 'required_memory_mb': 850,
            'estimated_latency_ms': 1500, 'estimated_cost_usd': 0.0,
            'estimated_quality_score': 0.65,
        },
        {
            'model_id': 'mock-npu-classifier', 'provider': 'local', 'device': 'NPU',
            'complexities': ['simple'], 'task_types': ['classification'],
            'available': True, 'required_memory_mb': 600,
            'estimated_latency_ms': 100, 'estimated_cost_usd': 0.0,
            'estimated_quality_score': 0.78,
        },
        {
            'model_id': 'mock-meta-cloud', 'provider': 'cloud', 'device': 'CLOUD',
            'complexities': ['simple', 'complex'],
            'task_types': ['text', 'code', 'classification'],
            'available': True, 'required_memory_mb': 0,
            'estimated_latency_ms': 1150, 'estimated_cost_usd': 0.012,
            'estimated_quality_score': 0.96,
        },
    ]
