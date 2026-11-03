"""Stage 6B pure coordinator tests: no LangGraph or external model downloads."""
import unittest

from edgemind.modules.planner import build_plan, plan
from edgemind.planning.execution import (
    next_step, dispatch, record_retrieval, record_tool,
    record_verification, prepare_retry,
)
from edgemind.modules.coordinator import run_tools
from edgemind.modules.router import route_model
from edgemind.modules.rag import retrieve
from edgemind.modules.verifier import verify
from edgemind.modules.recovery import recover
from edgemind.contracts import mock_hardware_profile, mock_model_registry


def make_state(task_type='text', documents=None, *, privacy='local_only',
               complexity='simple', simulate_failure=False):
    state = {
        'query': 'Explain the GPU', 'task_type': task_type,
        'documents': documents or [], 'complexity': complexity,
        'privacy_mode': privacy, 'network': 'online',
        'available_models': mock_model_registry(),
        'hardware_profile': mock_hardware_profile(),
        'routing_preferences': {}, 'attempts': 0, 'simulate_failure': simulate_failure,
        'step_results': {}, 'executed_steps': [], 'trace': [],
    }
    state.update(plan(state))
    return state


def merge(state, result):
    for key, value in result.items():
        if key in {'trace', 'executed_steps'}:
            state[key] = state.get(key, []) + value
        else:
            state[key] = value


class Stage6BPureTests(unittest.TestCase):
    def test_plain_code_task_skips_retrieval(self):
        state = make_state('code')
        merge(state, dispatch(state))
        self.assertEqual(state['active_operation'], 'generate_code')
        self.assertEqual(state['step_results'], {})
        merge(state, route_model(state))
        merge(state, run_tools(state))
        self.assertIn('Operation: generate_code', state['tool_result']['content'])
        merge(state, record_tool(state))
        merge(state, dispatch(state))
        self.assertEqual(state['active_operation'], 'verify_output')
        merge(state, verify(state))
        merge(state, record_verification(state))
        self.assertEqual(len(state['step_results']), 2)
        self.assertEqual([e.split(':')[1] for e in state['executed_steps']],
                         ['generate_code', 'verify_output'])

    def test_document_retrieval_must_precede_generation(self):
        docs = [{'source_id': 'manual', 'text': 'A GPU accelerates local inference.'}]
        state = make_state(documents=docs)
        merge(state, dispatch(state))
        self.assertEqual(state['active_operation'], 'retrieve_evidence')
        with self.assertRaises(ValueError):
            run_tools(state)  # Cannot generate text while retrieval is active.
        merge(state, retrieve(state))
        merge(state, record_retrieval(state))
        merge(state, dispatch(state))
        self.assertEqual(state['active_operation'], 'generate_text')
        merge(state, route_model(state))
        merge(state, run_tools(state))
        self.assertIn('manual#chunk-001', state['tool_result']['content'])
        merge(state, record_tool(state))
        merge(state, dispatch(state))
        merge(state, verify(state))
        merge(state, record_verification(state))
        self.assertEqual([s.split(':')[1] for s in state['executed_steps']],
                         ['retrieve_evidence', 'generate_text', 'verify_output'])
        self.assertIsNone(next_step(state['plan_steps'], state['step_results']))

    def test_wrong_planned_action_rejected(self):
        state = make_state('text')
        merge(state, dispatch(state))
        state['active_operation'] = 'generate_code'
        merge(state, route_model(state))
        with self.assertRaises(ValueError):
            run_tools(state)

    def test_unknown_result_id_rejected(self):
        plan_steps = build_plan('Explain')
        with self.assertRaises(ValueError):
            next_step(plan_steps, {'bogus': {'status': 'completed'}})

    def test_missing_dependencies_rejected(self):
        steps = build_plan('Explain', documents=[{'source_id': 'doc', 'text': 'text'}])
        # Intentionally mark retrieval unfinished and first step done would be okay,
        # but an unknown step being recorded must fail closed.
        with self.assertRaises(ValueError):
            next_step(steps, {'step_02': {'status': 'simulated'}})

    def test_blocked_model_still_reaches_verification(self):
        state = make_state(complexity='complex')
        merge(state, dispatch(state))
        merge(state, route_model(state))
        self.assertEqual(state['decision']['outcome'], 'deferred')
        merge(state, run_tools(state))
        merge(state, record_tool(state))
        merge(state, dispatch(state))
        self.assertEqual(state['active_operation'], 'verify_output')
        merge(state, verify(state))
        merge(state, record_verification(state))
        self.assertEqual(state['verification']['status'], 'BLOCKED')
        merge(state, recover(state))
        self.assertEqual(state['recovery_action'], 'defer')

    def test_retry_keeps_retrieval_and_reexecutes_primary(self):
        docs = [{'source_id': 'doc', 'text': 'GPU inference works locally.'}]
        state = make_state(documents=docs, simulate_failure=True)
        merge(state, dispatch(state))
        merge(state, retrieve(state))
        merge(state, record_retrieval(state))
        merge(state, dispatch(state))
        merge(state, route_model(state))
        merge(state, run_tools(state))
        self.assertEqual(state['tool_result']['status'], 'failed')
        merge(state, record_tool(state))
        merge(state, dispatch(state))
        merge(state, verify(state))
        merge(state, record_verification(state))
        merge(state, recover(state))
        self.assertEqual(state['recovery_action'], 'retry')
        merge(state, prepare_retry(state))
        self.assertEqual(set(state['step_results']), {'step_01'})
        merge(state, dispatch(state))
        self.assertEqual(state['active_operation'], 'generate_text')
        merge(state, route_model(state))
        merge(state, run_tools(state))
        self.assertEqual(state['tool_result']['status'], 'simulated')
        merge(state, record_tool(state))
        merge(state, dispatch(state))
        merge(state, verify(state))
        merge(state, record_verification(state))
        self.assertEqual(state['verification']['status'], 'SIMULATED')
        self.assertEqual(sum('retrieve_evidence' in x for x in state['executed_steps']), 1)
        self.assertEqual(sum('generate_text' in x for x in state['executed_steps']), 2)

    def test_retry_cannot_occur_without_permission(self):
        with self.assertRaises(ValueError):
            prepare_retry({'recovery_action': 'defer', 'plan_steps': [], 'step_results': {}})


if __name__ == '__main__':
    unittest.main()
