"""Run with LangGraph installed in the active virtual environment."""
import unittest
from edgemind.service import run_task


class Stage6CWorkflowTests(unittest.TestCase):
    def test_normal_mock_passes_with_checks(self):
        state = run_task('Explain GPU')
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertTrue(all(item['passed'] for item in state['verification']['checks']))
        self.assertEqual(len(state['verification_history']), 1)

    def test_retry_retains_history(self):
        state = run_task('Summarize GPU', documents=[
            {'source_id':'doc', 'text':'GPU inference uses an accelerated processor.'}],
            simulate_failure=True)
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertEqual(state['attempts'], 1)
        self.assertEqual([x['status'] for x in state['verification_history']], ['FAILED', 'SIMULATED'])
        self.assertEqual(sum('retrieve_evidence' in x for x in state['executed_steps']), 1)

    def test_malformed_output_deferred_without_retry(self):
        state = run_task('Explain GPU', simulate_invalid_result=True)
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertEqual(state['recovery_reason'], 'INVALID_RESULT')
        self.assertEqual(state['attempts'], 0)

    def test_permanent_failure_does_not_retry(self):
        state = run_task('Explain GPU', simulate_permanent_failure=True)
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertEqual(state['attempts'], 0)
        self.assertEqual(state['recovery_reason'], 'PERMANENT_FAILURE')

    def test_blocked_no_cloud_fallback(self):
        state = run_task('Private complex query', privacy_mode='local_only', complexity='complex')
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertEqual(state['recovery_reason'], 'BLOCKED')
        self.assertEqual(state['decision']['outcome'], 'deferred')

if __name__ == '__main__':
    unittest.main()
