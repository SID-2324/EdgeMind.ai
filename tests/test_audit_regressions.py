"""Audit regressions; no downloads or live model calls."""
from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from edgemind.api import app
from edgemind.modules.rag import retrieve
from edgemind.adapters.openvino_qwen import LocalModelError
from edgemind.service import run_task


class RetrievalRegressions(unittest.TestCase):
    def test_empty_hybrid_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError):
            retrieve({'query': 'test', 'documents': [],
                      'rag_settings': {'mode': 'hybrid', 'top_k': 0}})

    def test_empty_hybrid_rejects_invalid_candidate_k(self):
        with self.assertRaises(ValueError):
            retrieve({'query': 'test', 'documents': [],
                      'rag_settings': {'mode': 'hybrid', 'candidate_k': 1, 'top_k': 3}})

    def test_empty_hybrid_rejects_invalid_similarity(self):
        with self.assertRaises(ValueError):
            retrieve({'query': 'test', 'documents': [],
                      'rag_settings': {'mode': 'hybrid', 'min_semantic_similarity': float('nan')}})


class ApiRegressions(unittest.TestCase):
    def test_simultaneous_real_requests_reject_busy_and_keep_mock_available(self):
        from tests.test_stage10_web_api import sample_result
        started, release = threading.Event(), threading.Event()
        def execute(**kwargs):
            if kwargs['execution_mode'] == 'local_qwen':
                started.set()
                if not release.wait(10):
                    raise RuntimeError('test synchronization timeout')
                return sample_result('COMPLETED')
            return sample_result()
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {'EDGEMIND_ENABLE_REAL_LOCAL': '1', 'EDGEMIND_MODEL_PATH': folder}), \
                patch('edgemind.api.run_task', side_effect=execute):
            with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(client.post, '/api/tasks',
                                    json={'query': 'first', 'execution_mode': 'local_qwen'})
                try:
                    self.assertTrue(started.wait(5))
                    self.assertEqual(client.post('/api/tasks', json={
                        'query': 'second', 'execution_mode': 'local_qwen'}).status_code, 409)
                    self.assertEqual(client.post('/api/tasks', json={'query': 'mock'}).status_code, 200)
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=5).status_code, 200)
                # The lock must be released even after the first task completes.
                self.assertEqual(client.post('/api/tasks', json={
                    'query': 'third', 'execution_mode': 'local_qwen'}).status_code, 200)

    def test_returned_adapter_error_does_not_expose_paths(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {'EDGEMIND_ENABLE_REAL_LOCAL': '1', 'EDGEMIND_MODEL_PATH': folder}), \
                patch('edgemind.service.runtime_hardware_profile', return_value={
                    'devices': {'CPU': True, 'GPU': True, 'NPU': False}, 'available_memory_mb': 8192}), \
                patch('edgemind.modules.coordinator.generate_local_profiled',
                      side_effect=LocalModelError('Cannot load C:/private/model; secret-token-example')):
            with TestClient(app) as client:
                response = client.post('/api/tasks', json={'query': 'Hi', 'execution_mode': 'local_qwen'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'DEFERRED')
        self.assertNotIn('C:/private/model', response.text)
        self.assertNotIn('secret-token-example', response.text)
        self.assertFalse(response.json()['is_real_inference'])

    def test_concurrent_requests_have_isolated_outputs(self):
        def submit(index):
            with TestClient(app) as client:
                response = client.post('/api/tasks', json={'query': f'unique-request-{index}'})
                self.assertEqual(response.status_code, 200)
                return response.json()
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(submit, range(8)))
        for index, result in enumerate(responses):
            self.assertEqual(result['status'], 'SIMULATED')
            self.assertIn(f'unique-request-{index}', result['result'])
            for other in range(8):
                if other != index:
                    self.assertNotIn(f'unique-request-{other}', result['result'])

    def test_invalid_requests_never_execute(self):
        with TestClient(app) as client, patch('edgemind.api.run_task') as execute:
            for payload in ({}, {'query': '   '}, {'query': 'x', 'local_model_path': 'private'},
                            {'query': 'x', 'max_new_tokens': 0}, {'query': 'x', 'task_type': 'shell'}):
                with self.subTest(payload=payload):
                    self.assertEqual(client.post('/api/tasks', json=payload).status_code, 422)
            execute.assert_not_called()

    def test_malformed_json(self):
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/tasks', content='{',
                             headers={'Content-Type': 'application/json'}).status_code, 422)

    def test_code_and_document_smoke(self):
        with TestClient(app) as client:
            result = client.post('/api/tasks', json={
                'query': 'Write Python to describe GPU inference', 'task_type': 'code',
                'documents': [{'source_id': 'reference', 'text': 'GPU inference uses parallel computation.'}]
            }).json()
        self.assertEqual(result['status'], 'SIMULATED')
        self.assertEqual([step['operation'] for step in result['plan']],
                         ['retrieve_evidence', 'generate_code', 'verify_output'])
        self.assertTrue(result['evidence'][0]['source_id'].startswith('reference#chunk-'))


class WorkflowRegressions(unittest.TestCase):
    def test_persistent_timeout_exhausts_once_without_retrieval_repeat(self):
        failure = {'tool_result': {'status': 'failed', 'content': '', 'error': 'timeout',
                                  'error_code': 'TIMEOUT'}, 'trace': ['TEST: timeout']}
        with patch('edgemind.graph.run_tools', return_value=failure) as execute:
            result = run_task('GPU', documents=[{'source_id': 's', 'text': 'GPU inference'}])
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(result['status'], 'DEFERRED')
        self.assertEqual(result['recovery_reason'], 'RETRY_LIMIT')
        self.assertEqual(len(result['verification_history']), 2)
        self.assertEqual(sum(':retrieve_evidence:' in step for step in result['executed_steps']), 1)

    def test_privacy_matrix_never_selects_cloud(self):
        for task in ('text', 'code', 'classification'):
            for strategy in ('balanced', 'fastest', 'cheapest', 'highest_quality'):
                with self.subTest(task=task, strategy=strategy):
                    result = run_task('private data', task_type=task,
                                      routing_preferences={'strategy': strategy},
                                      hardware_profile={'devices': {}, 'available_memory_mb': 0})
                    self.assertEqual(result['status'], 'DEFERRED')
                    self.assertIsNone(result['decision']['provider'])
