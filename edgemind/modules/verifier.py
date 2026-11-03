"""Stage 6C: deterministic verification of mock execution and provenance.

These checks are structural only; they do NOT validate factual correctness.
Do not confuse SIMULATED with a production-ready success signal.
"""
from ..state import EdgeState

_PRIMARY = frozenset({'generate_text', 'generate_code', 'classify_input'})


def verify(state: EdgeState) -> dict:
    result = state['tool_result']
    status = result.get('status')
    error_code = result.get('error_code')
    checks = []

    def check(name: str, passed: bool) -> None:
        checks.append({'name': name, 'passed': bool(passed)})

    # Fail closed on invalid adapter responses. Never infer success from
    # non-empty text when execution status is failed or blocked.
    if status == 'not_run':
        report = {'status': 'BLOCKED',
                  'explanation': result.get('error') or 'No valid execution path available',
                  'error_code': error_code or 'NO_EXECUTION', 'checks': checks}
    elif status == 'failed':
        report = {'status': 'FAILED',
                  'explanation': result.get('error') or 'Tool execution failed',
                  'error_code': error_code or 'UNKNOWN_FAILURE', 'checks': checks}
    elif status not in ('simulated', 'completed'):
        report = {'status': 'FAILED', 'explanation': 'Unexpected tool status',
                  'error_code': 'INVALID_RESULT', 'checks': checks}
    else:
        steps = state.get('plan_steps', [])
        primary = [s for s in steps if s.get('operation') in _PRIMARY]
        check('one_supported_primary_operation', len(primary) == 1)
        selected = state.get('decision', {})
        check('model_selected', selected.get('outcome') == 'selected' and
              bool(selected.get('model_id')))
        primary_step = primary[0] if len(primary) == 1 else {}
        record = state.get('step_results', {}).get(primary_step.get('step_id'), {})
        check('primary_step_recorded', record.get('status') == status and
              record.get('operation') == primary_step.get('operation'))
        check('verifier_is_planned', state.get('active_operation') == 'verify_output')
        content = result.get('content')
        check('nonempty_response', isinstance(content, str) and bool(content.strip()))
        mode = state.get('execution_mode', 'mock')
        check('mode_matches_status', (mode == 'mock' and status == 'simulated') or
              (mode == 'local_qwen' and status == 'completed'))
        if mode == 'local_qwen':
            check('real_adapter_allowlisted',
                  selected.get('provider') == 'local' and
                  selected.get('device') in ('CPU', 'GPU') and
                  selected.get('model_id') == 'openvino-qwen2.5-1.5b-instruct-int4')
        # This marker is a test-only envelope proving the mock did not silently
        # masquerade as real inference. It is not an authenticity guarantee.
        if mode == 'mock':
            check('mock_provenance', isinstance(content, str) and
                  '[MOCK OUTPUT — NO LLM EXECUTED]' in content)
            check('operation_matches_plan', isinstance(content, str) and
                  f"Operation: {primary_step.get('operation')}" in content)
            check('selected_model_matches', isinstance(content, str) and
                  bool(selected.get('model_id')) and
                  f"Selected: {selected.get('model_id')}" in content)
        else:
            check('not_misrepresented_as_mock', isinstance(content, str) and
                  '[MOCK OUTPUT — NO LLM EXECUTED]' not in content)
        # Retrieved documents can return zero passages. Do not fabricate
        # citations or reject legitimate zero-result retrieval automatically.
        if all(item['passed'] for item in checks):
            verified_status = 'SIMULATED' if mode == 'mock' else 'COMPLETED'
            explanation = ('Mock execution passed structural checks; factual accuracy unverified.'
                           if mode == 'mock' else
                           'Real local generation returned a structurally valid response; factual accuracy unverified.')
            report = {'status': verified_status,
                      'explanation': explanation,
                      'error_code': None, 'checks': checks}
        else:
            failed_checks = ', '.join(item['name'] for item in checks if not item['passed'])
            report = {'status': 'FAILED',
                      'explanation': f'Invalid mock result: {failed_checks}',
                      'error_code': 'INVALID_RESULT', 'checks': checks}

    return {'verification': report, 'status': report['status'],
            'verification_history': [report],
            'trace': [f'VERIFIER: {report["status"]} — {report["explanation"]}']}
