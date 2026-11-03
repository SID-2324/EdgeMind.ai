"""Stage 7 final end-to-end contract tests.

These tests intentionally exercise the public service.run_task API using a
REAL LangGraph workflow and MOCK model execution. They DO NOT test an LLM.
Run: python -m unittest tests.test_stage7_handoff -v
"""
import json
import unittest

from edgemind.contracts import mock_hardware_profile
from edgemind.service import run_task


class Layer2HandoffTests(unittest.TestCase):
    def test_text_goes_through_plan_and_mock_verification(self):
        state = run_task('Explain GPUs')
        self.assertEqual(state['plan'], ['generate_text', 'verify_output'])
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertEqual(state['verification']['status'], 'SIMULATED')
        self.assertEqual(len(state['verification_history']), 1)
        self.assertEqual(state['decision']['provider'], 'local')
        self.assertIn('MOCK OUTPUT', state['tool_result']['content'])
        self.assertTrue(all(x['passed'] for x in state['verification']['checks']))

    def test_code_task_uses_coding_capable_model(self):
        state = run_task('Write a Python function', task_type='code')
        self.assertEqual(state['plan'], ['generate_code', 'verify_output'])
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertEqual(state['decision']['device'], 'GPU')

    def test_classification_has_compatible_npu_mock(self):
        state = run_task('Classify sample', task_type='classification')
        self.assertEqual(state['plan'], ['classify_input', 'verify_output'])
        self.assertEqual(state['decision']['device'], 'NPU')
        self.assertEqual(state['status'], 'SIMULATED')

    def test_documents_are_retrieved_before_generation(self):
        state = run_task('What uses GPU acceleration?', documents=[
            {'source_id': 'manual', 'text': 'GPU acceleration is used for AI inference.'},
            {'source_id': 'other', 'text': 'A garden has flowers and shrubs.'},
        ])
        self.assertEqual(state['plan'][0], 'retrieve_evidence')
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertIn('manual', [e['source_id'].split('#')[0] for e in state['evidence']])
        self.assertEqual(sum('retrieve_evidence' in x for x in state['executed_steps']), 1)

    def test_local_only_never_selects_cloud(self):
        hardware = mock_hardware_profile()
        hardware['available_memory_mb'] = 0
        state = run_task('Private analysis', privacy_mode='local_only', hardware_profile=hardware)
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertEqual(state['decision']['outcome'], 'deferred')
        self.assertNotEqual(state['decision'].get('provider'), 'cloud')
        self.assertEqual(state['attempts'], 0)

    def test_cloud_permitted_can_route_when_no_local_hardware(self):
        hardware = {'available_memory_mb': 0, 'devices': {'CPU': False, 'GPU': False, 'NPU': False}}
        state = run_task('Explain hardware', privacy_mode='cloud_allowed', network='online',
                         hardware_profile=hardware)
        self.assertEqual(state['decision']['provider'], 'cloud')
        self.assertEqual(state['status'], 'SIMULATED')

    def test_offline_does_not_select_cloud(self):
        hardware = {'available_memory_mb': 0, 'devices': {'CPU': False, 'GPU': False, 'NPU': False}}
        state = run_task('Offline task', privacy_mode='cloud_allowed', network='offline',
                         hardware_profile=hardware)
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertIsNone(state['decision']['model_id'])

    def test_transient_failure_retries_once_not_retrieval(self):
        state = run_task('Summarize GPU', simulate_failure=True, documents=[
            {'source_id': 'doc', 'text': 'The GPU accelerates local inference.'}
        ])
        self.assertEqual(state['attempts'], 1)
        self.assertEqual([r['status'] for r in state['verification_history']],
                         ['FAILED', 'SIMULATED'])
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertEqual(sum('retrieve_evidence' in s for s in state['executed_steps']), 1)

    def test_invalid_result_never_retries(self):
        state = run_task('Explain GPU', simulate_invalid_result=True)
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertEqual(state['attempts'], 0)
        self.assertEqual(state['recovery_reason'], 'INVALID_RESULT')

    def test_permanent_failure_never_retries(self):
        state = run_task('Explain GPU', simulate_permanent_failure=True)
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertEqual(state['attempts'], 0)
        self.assertEqual(state['recovery_reason'], 'PERMANENT_FAILURE')

    def test_separate_requests_do_not_share_state(self):
        one = run_task('Explain hardware', simulate_failure=True)
        two = run_task('Explain hardware')
        self.assertEqual(one['attempts'], 1)
        self.assertEqual(two['attempts'], 0)
        self.assertEqual(len(two['verification_history']), 1)
        self.assertNotIn('RECOVERY:', ' '.join(two['trace']))

    def test_output_can_be_serialized_for_later_fastapi_handoff(self):
        state = run_task('Explain GPUs')
        serialized = json.loads(json.dumps(state))
        self.assertEqual(serialized['status'], 'SIMULATED')
        self.assertIsInstance(serialized['trace'], list)
        self.assertIsInstance(serialized['plan_steps'], list)

    def test_unsupported_task_is_rejected(self):
        with self.assertRaises(ValueError):
            run_task('Do something', task_type='admin_delete')


if __name__ == '__main__':
    unittest.main()
