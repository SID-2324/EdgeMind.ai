"""Unit tests for Stage 6C: no LangGraph dependency or model download."""
import unittest
from edgemind.modules.planner import plan
from edgemind.planning.execution import dispatch, record_tool
from edgemind.modules.router import route_model
from edgemind.modules.coordinator import run_tools
from edgemind.modules.verifier import verify
from edgemind.modules.recovery import recover, MAX_RETRIES
from edgemind.contracts import mock_hardware_profile, mock_model_registry


def state_for_tool(**changes):
    state = {'query': 'Explain GPU', 'task_type': 'text', 'documents': [],
             'privacy_mode': 'local_only', 'network': 'online',
             'complexity': 'simple', 'available_models': mock_model_registry(),
             'hardware_profile': mock_hardware_profile(), 'routing_preferences': {},
             'attempts': 0, 'step_results': {}}
    state.update(changes)
    state.update(plan(state))
    state.update(dispatch(state))
    state.update(route_model(state))
    state.update(run_tools(state))
    state.update(record_tool(state))
    state.update(dispatch(state))
    return state


class VerificationRecoveryTests(unittest.TestCase):
    def test_valid_mock_output_passes_structural_checks(self):
        s = state_for_tool()
        out = verify(s)
        self.assertEqual(out['verification']['status'], 'SIMULATED')
        self.assertEqual(len(out['verification_history']), 1)
        self.assertTrue(all(x['passed'] for x in out['verification']['checks']))
        self.assertIn('unverified', out['verification']['explanation'])

    def test_fake_success_without_mock_marker_fails(self):
        s = state_for_tool(simulate_invalid_result=True)
        out = verify(s)
        self.assertEqual(out['verification']['status'], 'FAILED')
        self.assertEqual(out['verification']['error_code'], 'INVALID_RESULT')
        s.update(out)
        self.assertEqual(recover(s)['recovery_action'], 'defer')

    def test_mismatched_model_rejected(self):
        s = state_for_tool()
        s['tool_result']['content'] = s['tool_result']['content'].replace('mock-qwen-1.5b', 'wrong-model')
        self.assertEqual(verify(s)['verification']['status'], 'FAILED')

    def test_transient_failure_retries_once(self):
        s = state_for_tool(simulate_failure=True)
        out = verify(s)
        self.assertEqual(out['verification']['status'], 'FAILED')
        s.update(out)
        self.assertEqual(recover(s)['recovery_action'], 'retry')
        self.assertEqual(recover({**s, 'attempts': MAX_RETRIES})['recovery_reason'], 'RETRY_LIMIT')

    def test_permanent_error_does_not_retry(self):
        s = state_for_tool(simulate_permanent_failure=True)
        s.update(verify(s))
        self.assertEqual(recover(s)['recovery_reason'], 'PERMANENT_FAILURE')

    def test_privacy_block_never_retries(self):
        s = state_for_tool(complexity='complex')
        s.update(verify(s))
        self.assertEqual(s['verification']['status'], 'BLOCKED')
        self.assertEqual(recover(s)['recovery_action'], 'defer')

    def test_unknown_failure_code_does_not_retry(self):
        s = state_for_tool(simulate_failure=True)
        s['tool_result']['error_code'] = 'UNKNOWN_ERROR'
        s.update(verify(s))
        self.assertEqual(recover(s)['recovery_action'], 'defer')

    def test_missing_prior_record_detected(self):
        s = state_for_tool()
        s['step_results'] = {}
        self.assertEqual(verify(s)['verification']['status'], 'FAILED')

if __name__ == '__main__':
    unittest.main()
