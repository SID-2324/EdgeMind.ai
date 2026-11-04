"""Llama model selection, privacy, provenance and benchmark isolation regressions."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from edgemind.api import app
from edgemind.local_models import LLAMA, QWEN
from edgemind.service import run_task
from edgemind.modules.coordinator import run_tools
from edgemind.modules.verifier import verify
from edgemind.adapters.openvino_qwen import generate_local, LocalModelError
from edgemind.benchmark.benchmarks import run_benchmarks, save_report, eligible_observations

HW = {'devices': {'CPU': True, 'GPU': True, 'NPU': False}, 'available_memory_mb': 8192}


class LlamaIntegrationTests(unittest.TestCase):
    def execute(self, **kwargs):
        with patch('edgemind.service.runtime_hardware_profile', return_value=HW), \
                patch('edgemind.modules.coordinator.generate_local', return_value='Local Llama response'):
            return run_task('Summarize GPU inference', execution_mode=LLAMA.mode,
                            local_model_path='test-model', **kwargs)

    def test_text_generation_uses_llama_identity(self):
        result = self.execute()
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['decision']['model_id'], LLAMA.model_id)

    def test_code_operation(self):
        result = self.execute(task_type='code')
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['plan'][0], 'generate_code')

    def test_rag_reaches_shared_adapter(self):
        docs = [{'source_id': 's', 'text': 'GPU inference is local parallel computation.'}]
        with patch('edgemind.service.runtime_hardware_profile', return_value=HW), \
                patch('edgemind.modules.coordinator.generate_local', return_value='A local summary') as generate:
            result = run_task('Summarize GPU inference', execution_mode=LLAMA.mode,
                              local_model_path='test-model', documents=docs)
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertTrue(generate.call_args.kwargs['evidence'])
        self.assertEqual(generate.call_args.kwargs['model_id'], LLAMA.model_id)

    def test_low_ram_blocks_llama(self):
        with patch('edgemind.service.runtime_hardware_profile', return_value={**HW, 'available_memory_mb': 1000}), \
                patch('edgemind.modules.coordinator.generate_local') as generate:
            result = run_task('Hi', execution_mode=LLAMA.mode, local_model_path='test')
        self.assertEqual(result['status'], 'DEFERRED')
        generate.assert_not_called()

    def test_missing_gpu_never_runs_or_falls_back_to_cloud(self):
        with patch('edgemind.service.runtime_hardware_profile', return_value={
                **HW, 'devices': {'CPU': True, 'GPU': False}}), \
                patch('edgemind.modules.coordinator.generate_local') as generate:
            result = run_task('Hi', execution_mode=LLAMA.mode, local_model_path='test')
        self.assertEqual(result['status'], 'DEFERRED')
        self.assertIsNone(result['decision']['provider'])
        generate.assert_not_called()

    def test_real_privacy_rejected_before_probe(self):
        with patch('edgemind.service.runtime_hardware_profile') as probe:
            with self.assertRaises(ValueError):
                run_task('private', execution_mode=LLAMA.mode, local_model_path='test', privacy_mode='cloud_allowed')
            probe.assert_not_called()

    def test_cpu_manual_selection(self):
        self.assertEqual(self.execute(local_device='CPU')['decision']['device'], 'CPU')

    def test_wrong_model_cannot_enter_llama_adapter(self):
        state = self.execute()
        state['active_step_id'] = state['plan_steps'][0]['step_id']
        state['active_operation'] = 'generate_text'
        state['decision']['model_id'] = QWEN.model_id
        with patch('edgemind.modules.coordinator.generate_local') as generate:
            self.assertEqual(run_tools(state)['tool_result']['error_code'], 'INVALID_MODEL_ADAPTER')
            generate.assert_not_called()

    def test_wrong_model_fails_verification(self):
        state = self.execute()
        state['decision']['model_id'] = QWEN.model_id
        self.assertEqual(verify(state)['verification']['status'], 'FAILED')

    def test_empty_real_output_fails_verification(self):
        with patch('edgemind.service.runtime_hardware_profile', return_value=HW), \
                patch('edgemind.modules.coordinator.generate_local', return_value=''):
            result = run_task('Hi', execution_mode=LLAMA.mode, local_model_path='test')
        self.assertEqual(result['status'], 'DEFERRED')

    def test_api_llama_uses_separate_server_path(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {'EDGEMIND_ENABLE_REAL_LOCAL': '1', 'EDGEMIND_LLAMA_MODEL_PATH': folder}), \
                patch('edgemind.service.runtime_hardware_profile', return_value=HW), \
                patch('edgemind.modules.coordinator.generate_local_profiled',
                      return_value=('Llama answer', {'model_id': LLAMA.model_id})) as generate, TestClient(app) as client:
            response = client.post('/api/tasks', json={'query': 'Hi', 'execution_mode': LLAMA.mode})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['is_real_inference'])
        self.assertEqual(response.json()['model']['model_id'], LLAMA.model_id)
        self.assertEqual(generate.call_args.kwargs['model_path'], folder)

    def test_api_missing_llama_and_no_caller_paths(self):
        with patch.dict(os.environ, {'EDGEMIND_ENABLE_REAL_LOCAL': '1',
                                   'EDGEMIND_LLAMA_MODEL_PATH': 'missing/test/llama'}), TestClient(app) as client:
            self.assertEqual(client.post('/api/tasks', json={'query': 'Hi', 'execution_mode': LLAMA.mode}).status_code, 409)
            self.assertEqual(client.post('/api/tasks', json={
                'query': 'Hi', 'execution_mode': LLAMA.mode, 'local_model_path': 'private'}).status_code, 422)

    def test_api_rejects_cloud_privacy_and_classification(self):
        with TestClient(app) as client:
            for extra in ({'privacy_mode': 'cloud_allowed'}, {'task_type': 'classification'}):
                with self.subTest(extra=extra):
                    self.assertEqual(client.post('/api/tasks', json={
                        'query': 'Hi', 'execution_mode': LLAMA.mode, **extra}).status_code, 422)

    def test_mislabeled_weights_rejected_before_pipeline_load(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'config.json').write_text(json.dumps({'model_type': 'qwen2'}))
            with self.assertRaises(LocalModelError):
                generate_local(query='Hi', evidence=[], operation='generate_text', model_path=folder,
                               device='CPU', max_new_tokens=12, model_id=LLAMA.model_id)

    def test_llama_weights_cannot_be_labeled_qwen(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'config.json').write_text(json.dumps({
                'model_type': 'llama', 'hidden_size': 2048, 'num_hidden_layers': 16}))
            with self.assertRaises(LocalModelError):
                generate_local(query='Hi', evidence=[], operation='generate_text', model_path=folder,
                               device='CPU', max_new_tokens=12, model_id=QWEN.model_id)

    def test_benchmark_identity_cannot_cross_models(self):
        def synthetic(**kwargs):
            return 'synthetic', {'model_load_ms': 10, 'generation_ms': 10,
                                 'adapter_total_ms': 20, 'output_characters': 9}
        with tempfile.TemporaryDirectory() as folder:
            filename = str(Path(folder, 'benchmark.json'))
            for source, target in ((QWEN, LLAMA), (LLAMA, QWEN)):
                with self.subTest(source=source.key):
                    report = run_benchmarks(model_path=folder, model_id=source.model_id, runs=1,
                                            devices=['CPU'], runner=synthetic, hardware=HW)
                    save_report(report, filename)
                    valid, _ = eligible_observations(benchmark_file=filename, model_path=folder,
                                                     model_id=source.model_id, max_new_tokens=64)
                    self.assertEqual(valid, {'CPU': 20})
                    wrong, _ = eligible_observations(benchmark_file=filename, model_path=folder,
                                                     model_id=target.model_id, max_new_tokens=64)
                    self.assertEqual(wrong, {})

    def test_llama_auto_missing_benchmarks_does_not_generate(self):
        with patch('edgemind.service.runtime_hardware_profile', return_value=HW), \
                patch('edgemind.modules.coordinator.generate_local') as generate:
            result = run_task('Hi', execution_mode=LLAMA.mode, local_model_path='test',
                              local_device='auto', benchmark_file='missing/test/benchmarks')
        self.assertEqual(result['status'], 'DEFERRED')
        generate.assert_not_called()

    def test_llama_auto_selects_fastest_hardware_eligible_device(self):
        def synthetic(**kwargs):
            total = 40 if kwargs['device'] == 'GPU' else 90
            return 'synthetic', {'model_load_ms': 10, 'generation_ms': 10,
                                 'adapter_total_ms': total, 'output_characters': 9}
        with tempfile.TemporaryDirectory() as folder:
            filename = str(Path(folder, 'benchmark.json'))
            report = run_benchmarks(model_path=folder, model_id=LLAMA.model_id, runs=1,
                                    devices=['CPU', 'GPU'], runner=synthetic, hardware=HW)
            save_report(report, filename)
            for gpu_available, expected in ((True, 'GPU'), (False, 'CPU')):
                with self.subTest(gpu_available=gpu_available), \
                        patch('edgemind.service.runtime_hardware_profile', return_value={
                            **HW, 'devices': {'CPU': True, 'GPU': gpu_available}}), \
                        patch('edgemind.modules.coordinator.generate_local', return_value='Local answer'):
                    result = run_task('Hi', execution_mode=LLAMA.mode, local_model_path=folder,
                                      local_device='auto', benchmark_file=filename, max_new_tokens=64)
                self.assertEqual(result['status'], 'COMPLETED')
                self.assertEqual(result['decision']['device'], expected)
