"""Stage 6C: safe, bounded recovery decisions for MOCK tasks only.

No real side-effecting operations or cloud model calls are executed.
Persisted checkpoints across process restarts are not yet implemented.
"""
from ..state import EdgeState

MAX_RETRIES = 1
TRANSIENT_CODES = frozenset({'TIMEOUT', 'RATE_LIMIT', 'TEMPORARY_UNAVAILABLE'})


def recover(state: EdgeState) -> dict:
    report = state['verification']
    code = state.get('tool_result', {}).get('error_code')
    attempts = state.get('attempts', 0)
    if (report['status'] == 'FAILED' and code in TRANSIENT_CODES
            and state.get('tool_result', {}).get('status') == 'failed'
            and attempts < MAX_RETRIES):
        new_attempts = attempts + 1
        return {'attempts': new_attempts, 'recovery_action': 'retry',
                'recovery_reason': code,
                'trace': [f'RECOVERY: retrying transient {code} ({new_attempts}/{MAX_RETRIES})']}
    if report['status'] == 'BLOCKED':
        reason = 'BLOCKED'
    elif report.get('error_code') == 'INVALID_RESULT':
        reason = 'INVALID_RESULT'
    elif code in TRANSIENT_CODES and attempts >= MAX_RETRIES:
        reason = 'RETRY_LIMIT'
    else:
        reason = code or 'NOT_RETRYABLE'
    return {'recovery_action': 'defer', 'recovery_reason': reason,
            'status': 'DEFERRED',
            'trace': [f'RECOVERY: deferred ({reason}); no safe retry available']}
