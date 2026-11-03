"""Stage 8A: validates real local adapter boundary without downloading a model."""
import unittest
from unittest.mock import patch
from edgemind.adapters.openvino_qwen import build_prompt, LocalModelError
from edgemind.service import run_task


class LocalAdapterTests(unittest.TestCase):
    def test_build_prompt_preserves_reference_source(self):
        prompt = build_prompt('Summarize this', [{'source_id':'policy', 'text':'Local only.'}])
        self.assertIn('Source policy:', prompt)
        self.assertIn('untrusted', prompt)

    def test_real_mode_requires_model_path(self):
        with self.assertRaises(ValueError):
            run_task('Hello', execution_mode='local_qwen')

    def test_real_mode_refuses_fake_hardware(self):
        with self.assertRaises(ValueError):
            run_task('Hello', execution_mode='local_qwen', local_model_path='not-a-model',
                     hardware_profile={'available_memory_mb':99999,'devices':{'GPU':True}})

    def test_mock_still_requires_mock_provenance(self):
        state = run_task('Hello')
        self.assertEqual(state['status'], 'SIMULATED')
        self.assertIn('MOCK OUTPUT', state['tool_result']['content'])

    @patch('edgemind.modules.coordinator.generate_local', return_value='This is the actual response.')
    @patch('edgemind.service.runtime_hardware_profile', return_value={
        'available_memory_mb': 8192, 'devices': {'GPU': True, 'CPU': True, 'NPU': False}})
    def test_real_result_structural_verification(self, profile, generate):
        state = run_task('Explain GPU', execution_mode='local_qwen', local_model_path='fake',
                         local_device='GPU')
        self.assertEqual(state['status'], 'COMPLETED')
        self.assertEqual(state['verification']['status'], 'COMPLETED')
        self.assertEqual(state['tool_result']['content'], 'This is the actual response.')
        self.assertNotIn('MOCK OUTPUT', state['tool_result']['content'])
        self.assertEqual(state['decision']['provider'], 'local')
        generate.assert_called_once()

    @patch('edgemind.modules.coordinator.generate_local')
    @patch('edgemind.service.runtime_hardware_profile', return_value={
        'available_memory_mb': 8192, 'devices': {'GPU': False, 'CPU': True, 'NPU': False}})
    def test_real_mode_never_runs_on_missing_gpu(self, profile, generate):
        state = run_task('Explain GPU', execution_mode='local_qwen', local_model_path='fake')
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertEqual(state['decision']['outcome'], 'deferred')
        generate.assert_not_called()

    @patch('edgemind.modules.coordinator.generate_local', side_effect=LocalModelError('model missing'))
    @patch('edgemind.service.runtime_hardware_profile', return_value={
        'available_memory_mb': 8192, 'devices': {'GPU': True, 'CPU': True, 'NPU': False}})
    def test_real_load_failure_not_retried(self, profile, generate):
        state = run_task('Explain GPU', execution_mode='local_qwen', local_model_path='fake')
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertEqual(state['attempts'], 0)
        self.assertEqual(generate.call_count, 1)

    @patch('edgemind.modules.coordinator.generate_local', return_value='')
    @patch('edgemind.service.runtime_hardware_profile', return_value={
        'available_memory_mb': 8192, 'devices': {'GPU': True, 'CPU': True, 'NPU': False}})
    def test_empty_real_output_rejected(self, profile, generate):
        state = run_task('Explain GPU', execution_mode='local_qwen', local_model_path='fake')
        self.assertEqual(state['status'], 'DEFERRED')
        self.assertEqual(state['recovery_reason'], 'INVALID_RESULT')

    @patch('edgemind.modules.coordinator.generate_local', return_value='A real code snippet')
    @patch('edgemind.service.runtime_hardware_profile', return_value={
        'available_memory_mb': 8192, 'devices': {'GPU': True, 'CPU': True, 'NPU': False}})
    def test_real_code_generation_is_text_only(self, profile, generate):
        state = run_task('Write a Python function', task_type='code', execution_mode='local_qwen',
                         local_model_path='fake')
        self.assertEqual(state['status'], 'COMPLETED')
        self.assertEqual(state['plan'][0], 'generate_code')

    @patch('edgemind.modules.coordinator.generate_local')
    @patch('edgemind.service.runtime_hardware_profile', return_value={
        'available_memory_mb': 8192, 'devices': {'GPU': True, 'CPU': True, 'NPU': False}})
    def test_private_complex_task_cannot_bypass_router(self, profile, generate):
        state = run_task('Complex private analysis', complexity='complex', privacy_mode='local_only',
                         execution_mode='local_qwen', local_model_path='fake')
        self.assertEqual(state['status'], 'DEFERRED')
        generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
