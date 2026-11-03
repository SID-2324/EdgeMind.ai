"""Stage 6A: deterministic, task-aware, validated declarative planning.

This creates the structured plan now CONSUMED by the Stage 6B dispatcher.
No LLM or Layer 3 system is called by this module.
"""

from ..contracts import Evidence, PlanStep
from ..planning.validation import validate_plan
from ..state import EdgeState

_PRIMARY_OPERATIONS = {
    'text': ('generate_text', 'Generate a text response using the selected model'),
    'code': ('generate_code', 'Generate code using the selected coding-capable model'),
    'classification': ('classify_input', 'Classify the requested input'),
}


def build_plan(query: str, *, task_type: str = 'text',
               documents: list[Evidence] | None = None,
               complexity: str = 'simple') -> list[PlanStep]:
    """Return a small dependency-ordered, allowlisted plan.

    The planner does not infer the user's intent by guessing from keywords:
    the task type is supplied by Layer 1 and validated here. A future planner
    can propose plans dynamically while using this same validation contract.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError('The task cannot be empty')
    if len(query) > 10_000:
        raise ValueError('Task text exceeds 10,000 characters')
    if task_type not in _PRIMARY_OPERATIONS:
        raise ValueError(f'Unsupported planning task type: {task_type}')
    if complexity not in {'simple', 'complex'}:
        raise ValueError('Invalid task complexity')
    if documents is not None and (not isinstance(documents, list) or any(
            not isinstance(doc, dict) or not isinstance(doc.get('source_id'), str)
            or not isinstance(doc.get('text'), str)
            for doc in documents)):
        raise ValueError('Documents must have source_id and text')

    steps: list[PlanStep] = []
    if documents:
        steps.append({
            'step_id': 'step_01', 'operation': 'retrieve_evidence',
            'depends_on': [],
            'description': 'Retrieve relevant passages from supplied documents',
        })
    operation, description = _PRIMARY_OPERATIONS[task_type]
    steps.append({
        'step_id': f'step_{len(steps) + 1:02d}',
        'operation': operation,
        'depends_on': [steps[-1]['step_id']] if steps else [],
        'description': description,
    })
    steps.append({
        'step_id': f'step_{len(steps) + 1:02d}',
        'operation': 'verify_output',
        'depends_on': [steps[-1]['step_id']],
        'description': 'Check execution status and available evidence',
    })
    validate_plan(steps)
    return steps


def plan(state: EdgeState) -> dict:
    task_type = state.get('task_type', 'text')
    steps = build_plan(
        state['query'], task_type=task_type,
        documents=state.get('documents', []),
        complexity=state.get('complexity', 'simple'),
    )
    operations = [step['operation'] for step in steps]
    return {
        'plan': operations,  # Backward-compatible summary.
        'plan_steps': steps,
        'plan_diagnostics': {
            'planner_mode': 'deterministic',
            'step_count': len(steps),
            'task_type': task_type,
            'declarative_only': True,  # Planner outputs data; graph now executes it.
            'execution_mode': 'plan_driven',
        },
        'trace': [f'PLANNER: validated {len(steps)} task-aware steps for {task_type}; '
                  'plan-driven LangGraph execution; no LLM used'],
    }
