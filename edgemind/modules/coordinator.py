"""Stage 6B: execute only the allowlisted primary operation in the plan.

Mock model execution remains default; opt-in local OpenVINO inference is allowed.
Generated code is never executed; no cloud calls or filesystem mutation.
"""
from ..state import EdgeState
from ..adapters.openvino_qwen import generate_local, generate_local_profiled, LocalModelError

_SUPPORTED_OPERATIONS = frozenset({'generate_text', 'generate_code', 'classify_input'})


def run_tools(state: EdgeState) -> dict:
    op = state.get('active_operation')
    step_id = state.get('active_step_id')
    planned = next((step for step in state.get('plan_steps', [])
                    if step['step_id'] == step_id), None)
    if not planned or planned['operation'] != op or op not in _SUPPORTED_OPERATIONS:
        raise ValueError('Tool coordinator rejected an unplanned or unsupported operation')

    decision = state['decision']
    if decision['outcome'] != 'selected':
        return {
            'tool_result': {'status': 'not_run', 'content': '', 'error': decision['reason'], 'error_code': 'NO_ELIGIBLE_MODEL'},
            'trace': [f'TOOLS: skipped {op}; no permissible model'],
        }
    if state.get('execution_mode', 'mock') == 'local_qwen':
        # Explicitly allowlist a local text/code model. Never route an unknown,
        # cloud or mock selection into this real adapter.
        if (decision.get('model_id') != 'openvino-qwen2.5-1.5b-instruct-int4'
                or decision.get('provider') != 'local'
                or decision.get('device') not in ('GPU', 'CPU')
                or op not in ('generate_text', 'generate_code')):
            return {
                'tool_result': {'status': 'not_run', 'content': '',
                                'error': 'Real adapter refused incompatible model or operation',
                                'error_code': 'INVALID_MODEL_ADAPTER'},
                'trace': ['TOOLS: real adapter rejected incompatible selection'],
            }
        try:
            kwargs = dict(
                query=state['query'], evidence=state.get('evidence', []),
                operation=op, model_path=state['local_model_path'],
                device=decision['device'], max_new_tokens=state['max_new_tokens'],
            )
            if state.get('profile_local', False):
                content, performance = generate_local_profiled(**kwargs)
            else:
                content = generate_local(**kwargs)
                performance = None
        except LocalModelError as exc:
            return {
                'tool_result': {'status': 'failed', 'content': '',
                                'error': str(exc), 'error_code': 'PERMANENT_FAILURE'},
                'trace': [f'TOOLS: local OpenVINO inference failed: {exc}'],
            }
        update = {
            'tool_result': {'status': 'completed', 'content': content,
                            'error': None, 'error_code': None},
            'trace': [f'TOOLS: real LOCAL model executed on {decision["device"]}; no cloud call'],
        }
        if performance is not None:
            update['performance'] = performance
        return update
    if state.get('simulate_permanent_failure', False):
        return {
            'tool_result': {'status': 'failed', 'content': '',
                            'error': 'SIMULATED permanent unsupported operation',
                            'error_code': 'PERMANENT_FAILURE'},
            'trace': [f'TOOLS: permanent mock failure during {op}'],
        }
    if state.get('simulate_failure', False) and state.get('attempts', 0) == 0:
        return {
            'tool_result': {'status': 'failed', 'content': '', 'error': 'SIMULATED transient timeout', 'error_code': 'TIMEOUT'},
            'trace': [f'TOOLS: simulated transient timeout during {op}'],
        }
    sources = ', '.join(doc['source_id'] for doc in state.get('evidence', [])) or 'none'
    content = (
        f'[MOCK OUTPUT — NO LLM EXECUTED] Operation: {op} | '
        f"Request: {state['query']} | Selected: {decision['model_id']} | "
        f'Retrieved sample sources: {sources}'
    )
    if state.get('simulate_invalid_result', False):
        # A mock adapter can claim success but omit the required output structure.
        content = 'Unstructured response without provenance'
    return {
        'tool_result': {'status': 'simulated', 'content': content,
                        'error': None, 'error_code': None},
        'trace': [f'TOOLS: {op} handled by mock adapter; no real inference'],
    }
