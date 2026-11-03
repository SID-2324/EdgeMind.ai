"""Stage 10 HTTP contract tests. Heavy OpenVINO inference is always patched.

Tests use the real FastAPI stack and test client; no external network calls.
"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from edgemind.api import app, REAL_LOCK


def sample_result(status='SIMULATED'):
    real = status == 'COMPLETED'
    return {
        'status': status,
        'plan_steps': [
            {'step_id':'step1','operation':'generate_text','depends_on':[], 'description':'Respond'},
            {'step_id':'step2','operation':'verify_output','depends_on':['step1'], 'description':'Verify'},
        ],
        'executed_steps': ['step1','step2'],
        'tool_result': {'status': 'completed' if real else 'simulated',
                        'content': 'Response from Qwen' if real else '[MOCK OUTPUT — NO LLM EXECUTED]',
                        'error': None},
        'decision': {'outcome':'selected', 'model_id':'openvino-qwen2.5-1.5b-instruct-int4' if real else 'mock-qwen',
                     'device':'GPU','provider':'local', 'reason':'policy ok', 'estimates':{}},
        'verification': {'status':status, 'checks':[{'name':'valid','passed':True}]},
        'evidence': [{'source_id':'user/input','text':'Evidence'}],
        'performance': {'workflow_total_ms': 125.0} if real else {},
        'trace': ['PLANNER: plan ready','ROUTER: local GPU selected'],
    }


class Stage10WebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_home_served(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('EdgeMind', response.text)
        self.assertIn('id="taskForm"', response.text)

    def test_assets_served(self):
        self.assertEqual(self.client.get('/assets/app.js').status_code, 200)
        self.assertEqual(self.client.get('/assets/style.css').status_code, 200)

    def test_health_no_secret_paths_or_model_loading(self):
        with patch.dict(os.environ, {'EDGEMIND_MODEL_PATH':'secret/private/model','EDGEMIND_ENABLE_REAL_LOCAL':'0'}):
            r = self.client.get('/api/health')
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()['real_local_enabled'])
        self.assertNotIn('secret/private/model', r.text)
        self.assertFalse(r.json()['cloud_execution_implemented'])

    def test_mock_default_forwards_into_stable_service(self):
        with patch('edgemind.api.run_task', return_value=sample_result()) as fn:
            r = self.client.post('/api/tasks', json={'query':'Explain an NPU'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()['status'], 'SIMULATED')
        self.assertFalse(r.json()['is_real_inference'])
        self.assertIn('SIMULATED', r.json()['notice'])
        self.assertEqual(fn.call_args.kwargs['execution_mode'], 'mock')
        self.assertEqual(fn.call_args.kwargs['privacy_mode'], 'local_only')
        self.assertEqual(fn.call_args.kwargs['documents'], [])

    def test_document_retrieval_settings_forwarded(self):
        payload = {'query':'Summarize my document','documents':[{'source_id':'user/input','text':'A GPU runs inference.'}],
                   'rag_mode':'hybrid','rerank':True,'rag_top_k':2,'strategy':'fastest'}
        with patch('edgemind.api.run_task', return_value=sample_result()) as fn:
            r = self.client.post('/api/tasks', json=payload)
        self.assertEqual(r.status_code,200)
        self.assertEqual(fn.call_args.kwargs['rag_settings']['mode'],'hybrid')
        self.assertTrue(fn.call_args.kwargs['rag_settings']['enable_reranker'])
        self.assertEqual(fn.call_args.kwargs['documents'][0]['source_id'],'user/input')
        self.assertEqual(fn.call_args.kwargs['routing_preferences']['strategy'],'fastest')

    def test_real_disabled_returns_403_without_execution(self):
        with patch.dict(os.environ, {'EDGEMIND_ENABLE_REAL_LOCAL':'0'}), patch('edgemind.api.run_task') as fn:
            r = self.client.post('/api/tasks', json={'query':'hello','execution_mode':'local_qwen'})
        self.assertEqual(r.status_code,403)
        fn.assert_not_called()

    def test_real_missing_model_directory_refused(self):
        with patch.dict(os.environ, {'EDGEMIND_ENABLE_REAL_LOCAL':'1','EDGEMIND_MODEL_PATH': 'a/path/that/does/not/exist'}):
            r = self.client.post('/api/tasks', json={'query':'hello','execution_mode':'local_qwen'})
        self.assertEqual(r.status_code,409)

    def test_real_enabled_uses_server_config_not_caller_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'EDGEMIND_ENABLE_REAL_LOCAL':'1','EDGEMIND_MODEL_PATH': directory,
                                         'EDGEMIND_BENCHMARK_FILE':'test-benchmark.json'}), \
                 patch('edgemind.api.run_task', return_value=sample_result('COMPLETED')) as fn:
                r = self.client.post('/api/tasks', json={'query':'hello','execution_mode':'local_qwen','local_device':'auto'})
        self.assertEqual(r.status_code,200,r.text)
        self.assertTrue(r.json()['is_real_inference'])
        self.assertEqual(fn.call_args.kwargs['local_model_path'],directory)
        self.assertEqual(fn.call_args.kwargs['benchmark_file'],'test-benchmark.json')
        self.assertTrue(fn.call_args.kwargs['profile_local'])

    def test_real_result_never_mislabelled_from_status(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'EDGEMIND_ENABLE_REAL_LOCAL':'1','EDGEMIND_MODEL_PATH': directory}), \
                 patch('edgemind.api.run_task', return_value=sample_result('SIMULATED')):
                r = self.client.post('/api/tasks', json={'query':'hello','execution_mode':'local_qwen'})
        self.assertEqual(r.status_code,200)
        self.assertFalse(r.json()['is_real_inference'])

    def test_real_busy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'EDGEMIND_ENABLE_REAL_LOCAL':'1','EDGEMIND_MODEL_PATH': directory}), \
                 patch('edgemind.api.run_task') as fn:
                REAL_LOCK.acquire()
                try:
                    r = self.client.post('/api/tasks', json={'query':'hello','execution_mode':'local_qwen'})
                finally:
                    REAL_LOCK.release()
        self.assertEqual(r.status_code,409)
        fn.assert_not_called()

    def test_reject_duplicate_sources(self):
        docs=[{'source_id':'a', 'text':'One'},{'source_id':'a','text':'Two'}]
        r = self.client.post('/api/tasks', json={'query':'hello','documents':docs})
        self.assertEqual(r.status_code,422)

    def test_reject_unsafe_source_id(self):
        r = self.client.post('/api/tasks', json={'query':'hello','documents':[{'source_id':'<script>','text':'Hi'}]})
        self.assertEqual(r.status_code,422)

    def test_reject_invalid_combinations(self):
        cases=[
            {'query':'hello','rerank':True, 'rag_mode':'bm25'},
            {'query':'hello','execution_mode':'local_qwen','privacy_mode':'cloud_allowed'},
            {'query':'hello','execution_mode':'local_qwen','task_type':'classification'},
            {'query':'hello','max_new_tokens':257},
            {'query':'hello','rag_top_k':0},
            {'query':'hello','unexpected_key':True},
            {'query':''},
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(self.client.post('/api/tasks',json=case).status_code,422)

    def test_separate_requests_get_separate_service_calls(self):
        with patch('edgemind.api.run_task', return_value=sample_result()) as fn:
            r1=self.client.post('/api/tasks',json={'query':'First'})
            r2=self.client.post('/api/tasks',json={'query':'Second'})
        self.assertEqual((r1.status_code,r2.status_code),(200,200))
        self.assertEqual([c.kwargs['query'] for c in fn.call_args_list],['First','Second'])

    def test_real_mock_graph_wiring_text(self):
        """Run the complete mock graph through HTTP, not a patched service."""
        response = self.client.post('/api/tasks', json={'query':'Explain an NPU', 'execution_mode':'mock'})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['status'], 'SIMULATED')
        self.assertTrue(result['plan'])
        self.assertTrue(result['executed_steps'])
        self.assertFalse(result['is_real_inference'])
        self.assertIn('[MOCK OUTPUT', result['result'])

    def test_real_mock_graph_wiring_blocked(self):
        response = self.client.post('/api/tasks', json={
            'query':'Analyze complex private data', 'execution_mode':'mock',
            'complexity':'complex', 'privacy_mode':'local_only',
        })
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['status'], 'DEFERRED')
        self.assertNotEqual(result['model']['provider'], 'cloud')
        self.assertFalse(result['is_real_inference'])

    def test_runtime_error_does_not_expose_local_file_path(self):
        with patch('edgemind.api.run_task', side_effect=RuntimeError('C:\\secret\\model')):
            r=self.client.post('/api/tasks',json={'query':'hello'})
        self.assertEqual(r.status_code,500)
        self.assertNotIn('C:\\secret',r.text)


if __name__=='__main__':
    unittest.main()
