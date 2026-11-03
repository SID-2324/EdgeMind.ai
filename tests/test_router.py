import unittest
from edgemind.contracts import mock_model_registry
from edgemind.modules.router import choose_model


class TestModelRouter(unittest.TestCase):
    def setUp(self):
        self.registry = mock_model_registry()

    def test_private_stays_local(self):
        decision = choose_model('local_only', 'online', 'simple', self.registry)
        self.assertEqual(decision['provider'], 'local')
        self.assertEqual(decision['device'], 'GPU')

    def test_cloud_complex_with_permission(self):
        decision = choose_model('cloud_allowed', 'online', 'complex', self.registry)
        self.assertEqual(decision['provider'], 'cloud')

    def test_complex_private_defers(self):
        decision = choose_model('local_only', 'online', 'complex', self.registry)
        self.assertEqual(decision['outcome'], 'deferred')

    def test_offline_never_uses_cloud(self):
        decision = choose_model('cloud_allowed', 'offline', 'complex', self.registry)
        self.assertEqual(decision['outcome'], 'deferred')

    def test_no_model_available_defers(self):
        decision = choose_model('local_only', 'online', 'simple', [])
        self.assertEqual(decision['outcome'], 'deferred')

    def test_code_task_never_uses_classifier(self):
        decision = choose_model('local_only', 'online', 'simple', self.registry,
                                task_type='code')
        self.assertEqual(decision['model_id'], 'mock-qwen-1.5b')

    def test_classification_can_use_local_npu(self):
        decision = choose_model('local_only', 'online', 'simple', self.registry,
                                task_type='classification')
        self.assertEqual(decision['device'], 'NPU')

    def test_no_local_classification_for_complex_task(self):
        decision = choose_model('local_only', 'online', 'complex', self.registry,
                                task_type='classification')
        self.assertEqual(decision['outcome'], 'deferred')

    def test_private_never_falls_back_to_cloud(self):
        local_disabled = [dict(m, available=False) if m['provider'] == 'local'
                          else m for m in self.registry]
        decision = choose_model('local_only', 'online', 'simple', local_disabled)
        self.assertEqual(decision['outcome'], 'deferred')

    def test_unsupported_task_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_model('local_only', 'online', 'simple', self.registry,
                         task_type='video')

    def test_missing_capability_is_not_assumed(self):
        unknown = [dict(self.registry[0])]
        unknown[0].pop('task_types')
        decision = choose_model('local_only', 'online', 'simple', unknown)
        self.assertEqual(decision['outcome'], 'deferred')

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_model('ANYTHING', 'online', 'simple', self.registry)


if __name__ == '__main__':
    unittest.main()
