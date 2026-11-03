"""State shared by the six LangGraph modules."""
import operator
from typing import Annotated, TypedDict
from .contracts import Evidence, HardwareProfile, ModelCapability, ModelDecision, PlanStep, RoutingPreferences, ToolResult, Verification


class EdgeState(TypedDict, total=False):
    query: str
    execution_mode: str
    local_model_path: str | None
    local_device_preference: str
    benchmark_diagnostics: str
    max_new_tokens: int
    profile_local: bool
    performance: dict
    privacy_mode: str
    network: str
    complexity: str
    task_type: str
    available_models: list[ModelCapability]
    hardware_profile: HardwareProfile
    routing_preferences: RoutingPreferences
    documents: list[Evidence]
    rag_settings: dict
    rag_diagnostics: dict
    simulate_failure: bool
    simulate_permanent_failure: bool
    simulate_invalid_result: bool

    plan: list[str]  # Legacy operation names for existing integrations.
    plan_steps: list[PlanStep]  # Validated structured, declarative plan.
    plan_diagnostics: dict
    step_results: dict[str, dict]  # Step ID -> operation status and evidence.
    active_step_id: str | None
    active_operation: str | None
    executed_steps: Annotated[list[str], operator.add]  # Real workflow order incl. retries.
    evidence: list[Evidence]
    decision: ModelDecision
    tool_result: ToolResult
    verification: Verification
    verification_history: Annotated[list[Verification], operator.add]
    recovery_action: str
    recovery_reason: str
    attempts: int
    status: str
    # Reducer adds messages across nodes; each node returns only new messages.
    trace: Annotated[list[str], operator.add]
