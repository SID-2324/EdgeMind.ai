import unittest
from edgemind.contracts import mock_model_registry
from edgemind.graph import make_graph


def inputs(**extra):
    return {
        'query': 'Summarize the sample privacy policy',
        'privacy_mode': 'local_only', 'network': 'online', 'complexity': 'simple',
        'available_models': mock_model_registry(), 'documents': [],
        'simulate_failure': False, 'attempts': 0, 'trace': [], **extra,
    }


class TestWorkflow(unittest.TestCase):
    def test_simple_mock_flow(self):
        state = make_graph().invoke(inputs())
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertEqual(state['decision']['provider'], 'local')
        self.assertIn('NO LLM EXECUTED', state['tool_result']['content'])

    def test_recovery_retries_once(self):
        state = make_graph().invoke(inputs(simulate_failure=True))
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertEqual(state['attempts'], 1)
        self.assertTrue(any('retrying' in event for event in state['trace']))

    def test_complex_private_task_is_deferred(self):
        state = make_graph().invoke(inputs(complexity='complex'))
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertEqual(state['tool_result']['status'], 'not_run')

    def test_classification_routes_to_mock_npu(self):
        state = make_graph().invoke(inputs(task_type='classification'))
        self.assertEqual(state['decision']['device'], 'NPU')
        self.assertEqual(state['status'], 'SIMULATED')

    def test_complex_cloud_allowed_uses_mock_cloud(self):
        state = make_graph().invoke(inputs(privacy_mode='cloud_allowed', complexity='complex'))
        self.assertEqual(state['decision']['provider'], 'cloud')
        self.assertEqual(state['status'], 'SIMULATED')


if __name__ == '__main__':
    unittest.main()
