"""Stage 6A tests; no real LLM, hardware, or LangGraph installation required."""
import unittest

from edgemind.modules.planner import build_plan, plan
from edgemind.planning.validation import validate_plan


class TestPlanner(unittest.TestCase):
    def test_text_without_documents(self):
        steps = build_plan('Explain the GPU')
        self.assertEqual([s['operation'] for s in steps], ['generate_text', 'verify_output'])
        self.assertEqual(steps[1]['depends_on'], ['step_01'])

    def test_document_task_has_retrieval_first(self):
        steps = build_plan('Summarize', documents=[{'source_id': 'doc', 'text': 'data'}])
        self.assertEqual([s['operation'] for s in steps],
                         ['retrieve_evidence', 'generate_text', 'verify_output'])
        self.assertEqual(steps[1]['depends_on'], ['step_01'])
        self.assertEqual(steps[2]['depends_on'], ['step_02'])

    def test_code_task(self):
        steps = build_plan('Write code', task_type='code')
        self.assertEqual(steps[0]['operation'], 'generate_code')

    def test_classification_task(self):
        steps = build_plan('Classify image', task_type='classification')
        self.assertEqual(steps[0]['operation'], 'classify_input')

    def test_complexity_does_not_invent_steps(self):
        self.assertEqual(len(build_plan('Long analysis', complexity='complex')), 2)

    def test_empty_task_rejected(self):
        with self.assertRaises(ValueError):
            build_plan('  ')

    def test_unknown_task_type_rejected(self):
        with self.assertRaises(ValueError):
            build_plan('Make a movie', task_type='video')

    def test_oversized_task_rejected(self):
        with self.assertRaises(ValueError):
            build_plan('x' * 10001)

    def test_wrong_document_schema(self):
        with self.assertRaises(ValueError):
            build_plan('summarize', documents=[{'id': 'missing-text'}])

    def test_plan_writes_both_legacy_and_structured_fields(self):
        out = plan({'query': 'Write script', 'task_type': 'code',
                    'documents': [], 'complexity': 'simple'})
        self.assertEqual(out['plan'], ['generate_code', 'verify_output'])
        self.assertEqual([x['operation'] for x in out['plan_steps']], out['plan'])
        self.assertTrue(out['plan_diagnostics']['declarative_only'])

    def test_duplicate_step_id_rejected(self):
        steps = build_plan('hello')
        steps[1]['step_id'] = 'step_01'
        with self.assertRaises(ValueError):
            validate_plan(steps)

    def test_unknown_dependency_rejected(self):
        steps = build_plan('hello')
        steps[1]['depends_on'] = ['step_99']
        with self.assertRaises(ValueError):
            validate_plan(steps)

    def test_invalid_operation_rejected(self):
        steps = build_plan('hello')
        steps[0]['operation'] = 'execute_untrusted_shell_command'
        with self.assertRaises(ValueError):
            validate_plan(steps)

    def test_verification_required(self):
        steps = build_plan('hello')
        steps[-1]['operation'] = 'generate_text'
        with self.assertRaises(ValueError):
            validate_plan(steps)

    def test_forward_dependency_rejected(self):
        steps = build_plan('summarize', documents=[{'source_id': 'a', 'text': 'b'}])
        steps[0]['depends_on'] = ['step_03']
        with self.assertRaises(ValueError):
            validate_plan(steps)


if __name__ == '__main__':
    unittest.main()
