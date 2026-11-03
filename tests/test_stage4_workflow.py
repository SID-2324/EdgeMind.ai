"""Integration expectations; run on a machine with LangGraph installed."""
import unittest
from edgemind.contracts import mock_model_registry
from edgemind.service import run_task


class TestStage4Workflow(unittest.TestCase):
    def test_quality_route_reaches_mock_coordinator(self):
        state = run_task(
            query='Explain local inference', privacy_mode='cloud_allowed',
            routing_preferences={'strategy': 'highest_quality'},
        )
        self.assertEqual(state['decision']['model_id'], 'mock-meta-cloud')
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertIn('MOCK OUTPUT', state['tool_result']['content'])

    def test_gpu_slowdown_falls_back_in_entire_workflow(self):
        registry = mock_model_registry()
        registry[0]['estimated_latency_ms'] = 2800
        state = run_task(
            query='Summarize the sample', available_models=registry,
            routing_preferences={'strategy': 'fastest'},
        )
        self.assertEqual(state['decision']['model_id'], 'mock-tiny-cpu')
        self.assertEqual(state['status'], 'SIMULATED')

    def test_private_quality_threshold_blocks_cloud(self):
        state = run_task(
            query='Sensitive summary', privacy_mode='local_only',
            routing_preferences={'min_quality_score': 0.95},
        )
        self.assertEqual(state['decision']['outcome'], 'deferred')
        self.assertEqual(state['status'], 'DEFERRED')


if __name__ == '__main__':
    unittest.main()
