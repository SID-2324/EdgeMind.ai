"""Stage 6B LangGraph integration checks; execute on laptop with LangGraph installed."""
import unittest
from edgemind.service import run_task


class Stage6BGraphTests(unittest.TestCase):
    def test_plain_text_skips_rag(self):
        state = run_task('Explain GPU inference')
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertEqual(state['plan'], ['generate_text', 'verify_output'])
        self.assertEqual(len(state['step_results']), 2)
        self.assertNotIn('rag_diagnostics', state)
        self.assertEqual([v.split(':')[1] for v in state['executed_steps']],
                         ['generate_text', 'verify_output'])

    def test_code_generation_mock_respects_plan(self):
        state = run_task('Write a helper function', task_type='code')
        self.assertEqual(state['decision']['device'], 'GPU')
        self.assertIn('Operation: generate_code', state['tool_result']['content'])

    def test_document_analysis_executes_retrieval_first(self):
        state = run_task('Summarize GPU acceleration', documents=[
            {'source_id': 'manual', 'text': 'GPU acceleration supports local model inference.'},
        ])
        self.assertEqual([e.split(':')[1] for e in state['executed_steps']],
                         ['retrieve_evidence', 'generate_text', 'verify_output'])
        self.assertEqual(state['evidence'][0]['source_id'], 'manual#chunk-001')

    def test_deferred_still_runs_verifier_once(self):
        state = run_task('Complex secret request', complexity='complex', privacy_mode='local_only')
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertEqual(state['verification']['status'], 'BLOCKED')
        self.assertEqual(len(state['step_results']), 2)

    def test_transient_failure_retries_plan_step(self):
        state = run_task('Analyze', simulate_failure=True)
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertEqual(state['attempts'], 1)
        self.assertEqual([x.split(':')[1] for x in state['executed_steps']],
                         ['generate_text', 'verify_output', 'generate_text', 'verify_output'])


if __name__ == '__main__':
    unittest.main()
