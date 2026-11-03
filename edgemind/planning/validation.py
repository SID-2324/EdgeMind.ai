"""Validate a declarative execution plan; never execute plan steps here."""

from ..contracts import PlanStep

ALLOWED_OPERATIONS = frozenset({
    'retrieve_evidence', 'generate_text', 'generate_code',
    'classify_input', 'verify_output',
})


def validate_plan(steps: list[PlanStep]) -> None:
    """Fail closed on unsupported operations, cycles and invalid dependencies.

    Dependencies must refer to earlier steps, ensuring the plan is a DAG.
    The coordinator still follows the existing fixed workflow in Stage 6A.
    """
    if not isinstance(steps, list) or not 2 <= len(steps) <= 8:
        raise ValueError('Plan requires 2 to 8 steps')
    seen: set[str] = set()
    operations: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError('Each plan step must be an object')
        step_id = step.get('step_id')
        operation = step.get('operation')
        depends_on = step.get('depends_on')
        description = step.get('description')
        if not isinstance(step_id, str) or not step_id or step_id in seen:
            raise ValueError('Plan step IDs must be nonempty and unique')
        if operation not in ALLOWED_OPERATIONS:
            raise ValueError('Plan contains an unsupported operation')
        if not isinstance(description, str) or not description.strip():
            raise ValueError('Each step requires a description')
        if not isinstance(depends_on, list) or any(
            not isinstance(d, str) for d in depends_on
        ) or len(set(depends_on)) != len(depends_on):
            raise ValueError('Invalid dependencies')
        if not set(depends_on).issubset(seen):
            raise ValueError('Dependencies must reference prior steps only')
        seen.add(step_id)
        operations.append(operation)

    if operations[-1] != 'verify_output':
        raise ValueError('Verification must be the final step')
    if sum(op in {'generate_text', 'generate_code', 'classify_input'} for op in operations) != 1:
        raise ValueError('Plan must contain exactly one primary task operation')
    if len(steps[-1]['depends_on']) == 0:
        raise ValueError('Verification must depend on a previous step')
