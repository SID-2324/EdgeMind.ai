"""Integration checks requiring LangGraph, to be run on the user's laptop."""
import unittest
from edgemind.service import run_task


class TestStage5Workflow(unittest.TestCase):
    def test_retrieval_passages_reach_coordinator(self):
        state = run_task(
            'Summarize the GPU model',
            documents=[
                {'source_id': 'GPU-manual', 'text': 'GPU model inference can be accelerated.'},
                {'source_id': 'Privacy-manual', 'text': 'Private documents stay on the device.'},
            ],
        )
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertEqual(state['evidence'][0]['source_id'], 'GPU-manual#chunk-001')
        self.assertIn('GPU-manual#chunk-001', state['tool_result']['content'])
        self.assertEqual(state['rag_diagnostics']['method'], 'bm25')

    def test_router_still_enforces_privacy(self):
        state = run_task(
            'Summarize the confidential report',
            privacy_mode='local_only',
            documents=[{'source_id': 'private', 'text': 'Confidential report details'}],
        )
        self.assertEqual(state['decision']['provider'], 'local')


if __name__ == '__main__':
    unittest.main()
