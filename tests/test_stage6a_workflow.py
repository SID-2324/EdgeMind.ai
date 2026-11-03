"""LangGraph integration tests. Requires LangGraph, run on your laptop."""
import unittest
from edgemind.service import run_task


class TestStage6AWorkflow(unittest.TestCase):
    def test_code_task_produces_structured_plan(self):
        result = run_task('Write a Python helper', task_type='code')
        self.assertEqual(result['status'], 'SIMULATED')
        self.assertEqual(result['plan'], ['generate_code', 'verify_output'])
        self.assertEqual(len(result['plan_steps']), 2)
        self.assertEqual(result['plan_diagnostics']['task_type'], 'code')

    def test_documents_produce_retrieval_step(self):
        result = run_task('Summarize the document', documents=[
            {'source_id': 'manual', 'text': 'Local models can work offline.'},
        ])
        self.assertEqual(result['plan_steps'][0]['operation'], 'retrieve_evidence')
        self.assertEqual(result['status'], 'SIMULATED')
        self.assertEqual(result['rag_diagnostics']['method'], 'bm25')

    def test_private_complex_task_still_defers(self):
        result = run_task('Explain advanced techniques', complexity='complex',
                          privacy_mode='local_only')
        self.assertEqual(result['decision']['outcome'], 'deferred')
        self.assertEqual(result['status'], 'DEFERRED')
        self.assertEqual(result['plan_steps'][-1]['operation'], 'verify_output')


if __name__ == '__main__':
    unittest.main()
