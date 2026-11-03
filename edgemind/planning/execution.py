"""Plan-driven execution bookkeeping (no models, side effects, or LangGraph required).

Stage 6B deliberately supports one retrieval step, one primary mock task, and
one verifier, as required by planning.validation. Each operation is dispatched
only when dependencies have completed. Verification can inspect failed steps.
"""
from ..contracts import PlanStep
from .validation import validate_plan

PRIMARY = {'generate_text', 'generate_code', 'classify_input'}


def next_step(plan_steps: list[PlanStep], results: dict) -> PlanStep | None:
    """Find the next runnable unfinished step, rejecting invalid dependency states."""
    validate_plan(plan_steps)
    if not isinstance(results, dict):
        raise ValueError('Step results must be a mapping')
    ids = {step['step_id'] for step in plan_steps}
    if any(key not in ids for key in results):
        raise ValueError('Step results contain unknown step IDs')
    # Completed entries must form a prefix of the validated plan. This also
    # blocks fabricated completion records for later steps.
    plan_ids = [step['step_id'] for step in plan_steps]
    if set(results) != set(plan_ids[:len(results)]):
        raise ValueError('Step results are out of order or skip dependencies')
    for step in plan_steps:
        if step['step_id'] in results:
            continue
        for dependency in step['depends_on']:
            if dependency not in results:
                raise ValueError(f'{step["step_id"]} has an unfinished dependency: {dependency}')
            # Verification is intentionally permitted after a failed or blocked
            # generation so it can record and report that failure correctly.
            if (step['operation'] != 'verify_output'
                    and results[dependency].get('status') not in ('completed', 'simulated')):
                raise ValueError(f'{step["step_id"]} depends on an unsuccessful step')
        return step
    return None


def dispatch(state: dict) -> dict:
    step = next_step(state['plan_steps'], state.get('step_results', {}))
    if step is None:
        return {'active_step_id': None, 'active_operation': None,
                'trace': ['DISPATCH: all planned steps resolved']}
    return {'active_step_id': step['step_id'],
            'active_operation': step['operation'],
            'trace': [f'DISPATCH: {step["step_id"]} -> {step["operation"]}']}


def _record(state: dict, allowed: set[str], status: str, detail: str) -> dict:
    step_id = state.get('active_step_id')
    op = state.get('active_operation')
    step = next((s for s in state['plan_steps'] if s['step_id'] == step_id), None)
    if not step or step['operation'] != op or op not in allowed:
        raise ValueError(f'Attempt to record wrong operation: {op}')
    original = state.get('step_results', {})
    if step_id in original:
        raise ValueError(f'Step already executed: {step_id}')
    # Enforce dependencies even when this helper is tested independently.
    expected = next_step(state['plan_steps'], original)
    if expected is None or expected['step_id'] != step_id:
        raise ValueError('Step executed out of planned order')
    record = {'operation': op, 'status': status,
              'detail': detail, 'attempt': state.get('attempts', 0)}
    return {'step_results': {**original, step_id: record},
            'executed_steps': [f'{step_id}:{op}:{status}'],
            'trace': [f'PLAN: {step_id} {op} -> {status}']}


def record_retrieval(state: dict) -> dict:
    return _record(state, {'retrieve_evidence'}, 'completed',
                   f'{len(state.get("evidence", []))} evidence passage(s)')


def record_tool(state: dict) -> dict:
    result = state['tool_result']
    if result['status'] not in {'simulated', 'completed', 'failed', 'not_run'}:
        raise ValueError('Unexpected tool result status')
    return _record(state, PRIMARY, result['status'],
                   result['error'] or 'Execution handler returned')


def record_verification(state: dict) -> dict:
    result = state['verification']
    if result['status'] not in {'SIMULATED', 'COMPLETED', 'FAILED', 'BLOCKED'}:
        raise ValueError('Unexpected verification status')
    return _record(state, {'verify_output'}, result['status'], result['explanation'])


def prepare_retry(state: dict) -> dict:
    """Preserve successful retrieval; permit only generation+verification retry."""
    if state.get('recovery_action') != 'retry':
        raise ValueError('Retry preparation requires explicit retry decision')
    results = state.get('step_results', {})
    keep = {step['step_id'] for step in state['plan_steps']
            if step['operation'] == 'retrieve_evidence'}
    return {'step_results': {key: val for key, val in results.items() if key in keep},
            'active_step_id': None, 'active_operation': None,
            'trace': ['PLAN: preserving retrieval, retrying primary task and verification']}
