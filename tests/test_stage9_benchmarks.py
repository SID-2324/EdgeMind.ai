"""Stage 9 validation: no OpenVINO model download required for algorithm tests."""
import contextlib
from datetime import timedelta
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from edgemind.benchmark.benchmarks import (
    run_benchmarks, eligible_observations, save_report, utc_now,
)
from edgemind.modules.router import choose_model, route_model
from edgemind.service import run_task


def synthetic_adapter(*, query, evidence, operation, model_path, device, max_new_tokens):
    ms = 420.0 if device == 'GPU' else 900.0
    return 'A real-like test result', {
        'model_load_ms': 100.0, 'generation_ms': ms - 100.0,
        'adapter_total_ms': ms, 'output_characters': 23,
    }


class Stage9BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / 'qwen')
        Path(self.path).mkdir()
        self.filename = str(Path(self.tmp.name) / 'benchmarks.json')
        self.hardware = {'devices': {'CPU': True, 'GPU': True, 'NPU': False},
                         'available_memory_mb': 8000}

    def report(self, **kwargs):
        return run_benchmarks(model_path=self.path, hardware=self.hardware,
                              runner=synthetic_adapter, **kwargs)

    def write(self, report):
        save_report(report, self.filename)
        return self.filename

    def _auto(self, **kwargs):
        with patch('edgemind.service.runtime_hardware_profile', return_value=self.hardware):
            return run_task('Explain CPU/GPU', execution_mode='local_qwen',
                            local_model_path=self.path, local_device='auto',
                            max_new_tokens=64, benchmark_file=self.filename, **kwargs)

    def test_observed_device_mediana(self):
        data = self.report(runs=3, max_new_tokens=64)
        self.assertEqual(data['devices']['GPU']['median_adapter_total_ms'], 420)
        self.assertEqual(data['devices']['CPU']['median_adapter_total_ms'], 900)
        self.write(data)
        accepted, reason = eligible_observations(benchmark_file=self.filename,
                          model_path=self.path, max_new_tokens=64)
        self.assertEqual(accepted, {'CPU': 900.0, 'GPU': 420.0})
        self.assertIn('accepted', reason)
        exported = Path(self.filename).read_text('utf-8')
        self.assertNotIn(self.path, exported)
        self.assertNotIn('Explain the difference', exported)
        self.assertNotIn('A real-like test result', exported)

    def test_stale_readings_refused(self):
        data = self.report(now=utc_now() - timedelta(days=15))
        self.write(data)
        accepted, note = eligible_observations(benchmark_file=self.filename,
                        model_path=self.path, max_new_tokens=64)
        self.assertEqual(accepted, {})
        self.assertIn('stale', note)

    def test_changed_model_path_refused(self):
        self.write(self.report())
        other = str(Path(self.tmp.name) / 'other')
        Path(other).mkdir()
        accepted, note = eligible_observations(benchmark_file=self.filename,
                            model_path=other, max_new_tokens=64)
        self.assertEqual(accepted, {})
        self.assertIn('mismatch', note)

    def test_changed_model_configuration_refused(self):
        self.write(self.report())
        (Path(self.path) / 'config.json').write_text('{}')
        accepted, _ = eligible_observations(benchmark_file=self.filename,
                            model_path=self.path, max_new_tokens=64)
        self.assertEqual(accepted, {})

    def test_changed_token_limit_refused(self):
        self.write(self.report())
        accepted, note = eligible_observations(benchmark_file=self.filename,
                        model_path=self.path, max_new_tokens=96)
        self.assertEqual(accepted, {})
        self.assertIn('token limit', note)

    def test_other_machine_refused(self):
        data = self.report()
        data['machine_fingerprint'] = 'different-machine'
        self.write(data)
        accepted, note = eligible_observations(benchmark_file=self.filename,
                        model_path=self.path, max_new_tokens=64)
        self.assertEqual(accepted, {})
        self.assertIn('mismatch', note)

    def test_missing_report_fails_closed(self):
        accepted, _ = eligible_observations(benchmark_file=self.filename,
                        model_path=self.path, max_new_tokens=64)
        self.assertEqual(accepted, {})

    def test_invalid_report_fails_closed(self):
        Path(self.filename).write_text('{bad json')
        accepted, _ = eligible_observations(benchmark_file=self.filename,
                        model_path=self.path, max_new_tokens=64)
        self.assertEqual(accepted, {})

    def test_tampered_median_not_accepted(self):
        data = self.report()
        data['devices']['GPU']['median_adapter_total_ms'] = 1.0
        self.write(data)
        accepted, _ = eligible_observations(benchmark_file=self.filename,
                        model_path=self.path, max_new_tokens=64)
        self.assertEqual(accepted, {'CPU': 900.0})

    def test_gpu_unavailable_only_cpu_is_recorded(self):
        hw = {'available_memory_mb': 8000, 'devices': {'GPU': False, 'CPU': True}}
        data = run_benchmarks(model_path=self.path, hardware=hw,
                              runner=synthetic_adapter)
        self.assertEqual(data['devices']['GPU']['status'], 'unavailable')
        self.assertEqual(data['devices']['CPU']['status'], 'ok')

    def test_failing_gpu_does_not_block_cpu(self):
        def sometimes_fails(**kw):
            if kw['device'] == 'GPU':
                raise RuntimeError('fake model path PRIVATE')
            return synthetic_adapter(**kw)
        data = run_benchmarks(model_path=self.path, hardware=self.hardware,
                              runner=sometimes_fails)
        self.assertEqual(data['devices']['GPU']['status'], 'failed')
        self.assertEqual(data['devices']['CPU']['status'], 'ok')
        self.assertNotIn('PRIVATE', json.dumps(data))

    def test_insufficient_ram_means_no_runs(self):
        hw = {'available_memory_mb': 200, 'devices': {'GPU': True, 'CPU': True}}
        data = run_benchmarks(model_path=self.path, hardware=hw,
                              runner=lambda **kw: self.fail('must not execute model'))
        self.assertEqual(data['devices']['GPU']['status'], 'insufficient_ram')
        self.assertEqual(data['devices']['CPU']['status'], 'insufficient_ram')

    def test_auto_selects_gpu_from_measured_latencies(self):
        self.write(self.report())
        with patch('edgemind.modules.coordinator.generate_local', return_value='Actual local model text') as run:
            result = self._auto()
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['decision']['device'], 'GPU')
        self.assertEqual(result['decision']['latency_source'], 'observed_cold_adapter_median')
        self.assertEqual(run.call_args.kwargs['device'], 'GPU')

    def test_auto_selects_cpu_when_gpu_unavailable(self):
        self.write(self.report())
        hardware = {'available_memory_mb':8000,'devices': {'CPU': True, 'GPU': False, 'NPU': False}}
        with patch('edgemind.service.runtime_hardware_profile', return_value=hardware), \
             patch('edgemind.modules.coordinator.generate_local', return_value='Test output') as run:
            result = run_task('CPU fallback', execution_mode='local_qwen',
                              local_model_path=self.path, local_device='auto',
                              max_new_tokens=64, benchmark_file=self.filename)
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(run.call_args.kwargs['device'], 'CPU')

    def test_auto_missing_benchmark_defers_no_inference(self):
        with patch('edgemind.modules.coordinator.generate_local') as run:
            result = self._auto()
        self.assertEqual(result['status'], 'DEFERRED')
        run.assert_not_called()

    def test_manual_device_keeps_existing_mode(self):
        with patch('edgemind.service.runtime_hardware_profile', return_value=self.hardware), \
             patch('edgemind.modules.coordinator.generate_local', return_value='Manual local output') as run:
            result = run_task('Hi', execution_mode='local_qwen',
                local_model_path=self.path, local_device='CPU')
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['decision']['latency_source'], 'unmeasured_manual_device')
        self.assertEqual(run.call_args.kwargs['device'], 'CPU')

    def test_privacy_still_local_only(self):
        self.write(self.report())
        with patch('edgemind.modules.coordinator.generate_local', return_value='Local result'):
            result = self._auto(privacy_mode='local_only')
        self.assertEqual(result['decision']['provider'], 'local')

    def test_task_capability_does_not_get_overridden(self):
        self.write(self.report())
        with patch('edgemind.modules.coordinator.generate_local') as run:
            result = self._auto(task_type='classification')
        self.assertEqual(result['status'], 'DEFERRED')
        run.assert_not_called()

    def test_benchmark_constraints_do_not_override_hardware(self):
        self.write(self.report())
        hw = {'available_memory_mb':800, 'devices': {'CPU': True, 'GPU': True, 'NPU': False}}
        with patch('edgemind.service.runtime_hardware_profile', return_value=hw), \
             patch('edgemind.modules.coordinator.generate_local') as run:
            result = run_task('Hi', execution_mode='local_qwen', local_model_path=self.path,
                              local_device='auto', max_new_tokens=64,
                              benchmark_file=self.filename)
        self.assertEqual(result['status'], 'DEFERRED')
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
