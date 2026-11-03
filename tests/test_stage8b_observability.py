"""Stage 8B observed metrics, tested with injected local model stubs.

No model download or live OpenVINO invocation is required for these tests.
"""
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from edgemind.adapters.openvino_qwen import generate_local_profiled, LocalModelError
from edgemind.service import run_task

HARDWARE = {'available_memory_mb': 8192,
            'devices': {'CPU': True, 'GPU': True, 'NPU': False}}


class FakePipeline:
    def __init__(self, model_dir, device):
        self.model_dir = model_dir
        self.device = device

    def generate(self, prompt, max_new_tokens):
        self.prompt = prompt
        return 'Generated local answer'


class ObservabilityTests(unittest.TestCase):
    def test_mock_mode_rejects_profiling(self):
        with self.assertRaisesRegex(ValueError, 'requires real local'):
            run_task('Explain RAM', profile_local=True)

    def test_profile_type_must_be_boolean(self):
        with self.assertRaises(ValueError):
            run_task('Explain RAM', profile_local='yes')

    def test_unprofiled_mock_is_unchanged(self):
        state = run_task('Explain RAM')
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertNotIn('performance', state)

    @patch('edgemind.modules.coordinator.generate_local_profiled',
           return_value=('Measured reply', {'model_load_ms': 50.0,
                                            'generation_ms': 100.0,
                                            'adapter_total_ms': 160.0}))
    @patch('edgemind.service.runtime_hardware_profile', return_value=HARDWARE)
    def test_profiled_workflow_returns_observed_numbers(self, hardware, generated):
        state = run_task('Explain RAM', execution_mode='local_qwen',
                         local_model_path='unused', profile_local=True)
        self.assertEqual(state['status'], 'COMPLETED')
        self.assertEqual(state['tool_result']['content'], 'Measured reply')
        self.assertEqual(state['performance']['measurement_type'], 'observed')
        self.assertEqual(state['performance']['generation_ms'], 100.0)
        self.assertGreaterEqual(state['performance']['workflow_total_ms'], 0)
        self.assertEqual(state['decision']['estimates']['latency_ms'], 650)
        generated.assert_called_once()

    @patch('edgemind.modules.coordinator.generate_local', return_value='Unprofiled response')
    @patch('edgemind.service.runtime_hardware_profile', return_value=HARDWARE)
    def test_unprofiled_real_path_preserves_stage8a_adapter(self, hardware, generated):
        state = run_task('Explain RAM', execution_mode='local_qwen',
                         local_model_path='unused')
        self.assertEqual(state['status'], 'COMPLETED')
        self.assertNotIn('performance', state)
        generated.assert_called_once()

    @patch('edgemind.modules.coordinator.generate_local_profiled')
    @patch('edgemind.service.runtime_hardware_profile', return_value={
        'available_memory_mb': 8192,
        'devices': {'GPU': False, 'CPU': True, 'NPU': False},
    })
    def test_blocked_real_task_has_workflow_time_only(self, hardware, generated):
        state = run_task('Explain RAM', execution_mode='local_qwen',
                         local_model_path='unused', profile_local=True)
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertIn('workflow_total_ms', state['performance'])
        self.assertNotIn('generation_ms', state['performance'])
        generated.assert_not_called()

    @patch('edgemind.modules.coordinator.generate_local_profiled',
           side_effect=LocalModelError('simulated model load error'))
    @patch('edgemind.service.runtime_hardware_profile', return_value=HARDWARE)
    def test_failed_generation_does_not_claim_success_metrics(self, hardware, generated):
        state = run_task('Explain RAM', execution_mode='local_qwen',
                         local_model_path='unused', profile_local=True)
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertNotIn('generation_ms', state['performance'])
        self.assertEqual(state['attempts'], 0)

    def test_adapter_measures_cpu_memory_and_never_captures_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            fake_genai = types.SimpleNamespace(LLMPipeline=FakePipeline)
            with patch.dict(sys.modules, {'openvino_genai': fake_genai}):
                result, perf = generate_local_profiled(
                    query='PRIVATE_QUERY_SENTINEL', evidence=[],
                    operation='generate_text', model_path=folder,
                    device='GPU', max_new_tokens=40)
            self.assertEqual(result, 'Generated local answer')
            self.assertEqual(perf['model_device'], 'GPU')
            self.assertGreaterEqual(perf['model_load_ms'], 0)
            self.assertGreaterEqual(perf['generation_ms'], 0)
            self.assertGreaterEqual(perf['adapter_total_ms'], 0)
            self.assertIn('process_rss_after_mb', perf)
            self.assertNotIn('PRIVATE_QUERY_SENTINEL', json.dumps(perf))
            self.assertNotIn(folder, json.dumps(perf))
            self.assertEqual(perf['output_characters'], len(result))

    def test_profiled_adapter_rejects_npu(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(LocalModelError):
                generate_local_profiled(query='Hello', evidence=[],
                                        operation='generate_text',
                                        model_path=folder, device='NPU', max_new_tokens=20)

    @patch('edgemind.cli.run_task')
    def test_metrics_file_excludes_query_and_answer(self, fake_task):
        from edgemind.cli import main
        fake_task.return_value = {
            'status': 'COMPLETED',
            'decision': {'model_id': 'local-qwen', 'device': 'GPU'},
            'performance': {'workflow_total_ms': 500.0, 'measurement_type': 'observed'},
            'query': 'PRIVATE_QUERY_SENTINEL',
            'tool_result': {'content': 'PRIVATE_ANSWER_SENTINEL'},
        }
        with tempfile.TemporaryDirectory() as folder:
            metrics = Path(folder) / 'metrics.json'
            with patch.object(sys, 'argv', ['main.py', '--query', 'PRIVATE_QUERY_SENTINEL',
                                             '--real-local', '--profile', '--json',
                                             '--metrics-out', str(metrics)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    main()
            saved = metrics.read_text(encoding='utf-8')
            self.assertNotIn('PRIVATE_QUERY_SENTINEL', saved)
            self.assertNotIn('PRIVATE_ANSWER_SENTINEL', saved)
            self.assertEqual(json.loads(saved)['performance']['workflow_total_ms'], 500.0)


if __name__ == '__main__':
    unittest.main()
