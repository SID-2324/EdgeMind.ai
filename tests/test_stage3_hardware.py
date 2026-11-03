import unittest

from edgemind.contracts import mock_hardware_profile, mock_model_registry
from edgemind.modules.router import choose_model


class TestHardwareRouting(unittest.TestCase):
    def setUp(self):
        self.hw = mock_hardware_profile()
        self.models = mock_model_registry()

    def choose(self, task='text', privacy='local_only', network='online',
               complexity='simple'):
        return choose_model(privacy, network, complexity, self.models,
                            task_type=task, hardware=self.hw)

    def test_gpu_selected_when_available(self):
        self.assertEqual(self.choose()['device'], 'GPU')

    def test_gpu_unavailable_falls_back_to_cpu(self):
        self.hw['devices']['GPU'] = False
        selected = self.choose()
        self.assertEqual(selected['model_id'], 'mock-tiny-cpu')
        self.assertTrue(any('GPU device unavailable' in s for s in selected['rejected']))

    def test_ram_shortage_falls_back_to_smaller_cpu_model(self):
        self.hw['available_memory_mb'] = 900
        selected = self.choose()
        self.assertEqual(selected['device'], 'CPU')
        self.assertTrue(any('insufficient free RAM' in s for s in selected['rejected']))

    def test_too_little_ram_defers_private_task(self):
        self.hw['available_memory_mb'] = 300
        self.assertEqual(self.choose()['outcome'], 'deferred')

    def test_npu_disabled_defers_private_classification(self):
        self.hw['devices']['NPU'] = False
        decision = self.choose(task='classification')
        self.assertEqual(decision['outcome'], 'deferred')

    def test_cloud_is_allowed_when_no_local_memory_and_authorized(self):
        self.hw['available_memory_mb'] = 0
        decision = self.choose(privacy='cloud_allowed')
        self.assertEqual(decision['provider'], 'cloud')

    def test_cloud_never_allowed_for_private_task_when_low_memory(self):
        self.hw['available_memory_mb'] = 0
        self.assertEqual(self.choose()['outcome'], 'deferred')

    def test_no_cloud_while_offline_even_when_local_does_not_fit(self):
        self.hw['available_memory_mb'] = 0
        self.assertEqual(self.choose(privacy='cloud_allowed', network='offline')['outcome'],
                         'deferred')

    def test_cloud_complex_task_does_not_need_local_gpu(self):
        self.hw['devices']['GPU'] = False
        self.hw['available_memory_mb'] = 0
        decision = self.choose(privacy='cloud_allowed', complexity='complex')
        self.assertEqual(decision['provider'], 'cloud')

    def test_missing_local_memory_budget_is_not_assumed_zero(self):
        invalid = [dict(self.models[0])]
        invalid[0].pop('required_memory_mb')
        decision = choose_model('local_only', 'online', 'simple', invalid,
                                hardware=self.hw)
        self.assertEqual(decision['outcome'], 'deferred')

    def test_invalid_hardware_profiles_are_rejected(self):
        for invalid in [
            {'available_memory_mb': -1, 'devices': {'CPU': True}},
            {'available_memory_mb': True, 'devices': {'CPU': True}},
            {'available_memory_mb': 1024, 'devices': {'GPU': 'yes'}},
            {'available_memory_mb': 1024, 'devices': {'TPU': True}},
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                choose_model('local_only', 'online', 'simple', self.models,
                             hardware=invalid)



if __name__ == '__main__':
    unittest.main()
