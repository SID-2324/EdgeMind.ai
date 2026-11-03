"""Stage 4 decision tests: mocked estimates ONLY, not inference benchmarks."""
import copy
import unittest

from edgemind.contracts import mock_hardware_profile, mock_model_registry
from edgemind.modules.router import choose_model


class TestOptimizationRouter(unittest.TestCase):
    def setUp(self):
        self.registry = mock_model_registry()
        self.hw = mock_hardware_profile()

    def route(self, *, privacy='local_only', network='online',
              complexity='simple', task_type='text', preferences=None):
        return choose_model(
            privacy, network, complexity, self.registry,
            task_type=task_type, hardware=self.hw,
            preferences=preferences,
        )

    def test_balanced_default_preserves_previous_gpu_selection(self):
        self.assertEqual(self.route()['model_id'], 'mock-qwen-1.5b')

    def test_fastest_chooses_gpu_when_healthy(self):
        self.assertEqual(self.route(preferences={'strategy': 'fastest'})['device'], 'GPU')

    def test_quality_can_select_cloud_only_when_allowed(self):
        selected = self.route(privacy='cloud_allowed',
                              preferences={'strategy': 'highest_quality'})
        self.assertEqual(selected['provider'], 'cloud')

    def test_quality_does_not_override_private_data(self):
        selected = self.route(preferences={'strategy': 'highest_quality'})
        self.assertEqual(selected['provider'], 'local')
        self.assertTrue(any('privacy policy' in r for r in selected['rejected']))

    def test_quality_does_not_override_offline(self):
        selected = self.route(privacy='cloud_allowed', network='offline',
                              preferences={'strategy': 'highest_quality'})
        self.assertEqual(selected['provider'], 'local')

    def test_gpu_slowdown_makes_cpu_fastest(self):
        self.registry[0]['estimated_latency_ms'] = 2800
        selected = self.route(preferences={'strategy': 'fastest'})
        self.assertEqual(selected['model_id'], 'mock-tiny-cpu')

    def test_balanced_preserves_local_preference_in_sample(self):
        self.assertEqual(self.route(privacy='cloud_allowed')['provider'], 'local')

    def test_cost_optimization_selects_cheapest_eligible_model(self):
        self.registry[0]['estimated_cost_usd'] = 0.01
        self.registry[1]['estimated_cost_usd'] = 0.0
        selected = self.route(preferences={'strategy': 'cheapest'})
        self.assertEqual(selected['model_id'], 'mock-tiny-cpu')

    def test_min_quality_can_defer_private_request(self):
        selected = self.route(preferences={'min_quality_score': 0.95})
        self.assertEqual(selected['outcome'], 'deferred')

    def test_min_quality_allows_cloud_when_permitted(self):
        selected = self.route(privacy='cloud_allowed',
                              preferences={'min_quality_score': 0.95})
        self.assertEqual(selected['model_id'], 'mock-meta-cloud')

    def test_latency_limit_may_defer(self):
        selected = self.route(preferences={'max_latency_ms': 400})
        self.assertEqual(selected['outcome'], 'deferred')
        self.assertTrue(any('latency' in reason for reason in selected['rejected']))

    def test_budget_cannot_force_cloud_for_complex_task(self):
        selected = self.route(privacy='cloud_allowed', complexity='complex',
                              preferences={'max_cost_usd': 0.0})
        self.assertEqual(selected['outcome'], 'deferred')

    def test_zero_cost_budget_keeps_free_local_models(self):
        selected = self.route(privacy='cloud_allowed',
                              preferences={'max_cost_usd': 0.0})
        self.assertEqual(selected['provider'], 'local')

    def test_missing_benchmark_is_rejected(self):
        self.registry[0].pop('estimated_quality_score')
        selected = self.route()
        self.assertEqual(selected['model_id'], 'mock-tiny-cpu')
        self.assertTrue(any('invalid performance' in r for r in selected['rejected']))

    def test_nan_metric_is_rejected(self):
        self.registry[0]['estimated_cost_usd'] = float('nan')
        self.assertEqual(self.route()['model_id'], 'mock-tiny-cpu')

    def test_invalid_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            self.route(preferences={'strategy': 'magical'})

    def test_bad_constraints_are_rejected(self):
        for prefs in (
            {'max_latency_ms': -3}, {'max_latency_ms': True},
            {'max_cost_usd': -1}, {'min_quality_score': 1.1},
            {'min_quality_score': float('nan')},
            {'unexpected': 12},
        ):
            with self.subTest(preferences=prefs), self.assertRaises(ValueError):
                self.route(preferences=prefs)

    def test_simulated_memory_rules_still_apply(self):
        self.hw['available_memory_mb'] = 300
        self.assertEqual(self.route()['outcome'], 'deferred')

    def test_npu_classifier_selected_even_when_text_models_exist(self):
        selected = self.route(task_type='classification')
        self.assertEqual(selected['device'], 'NPU')

    def test_explainability_includes_estimates_and_rejections(self):
        selected = self.route()
        self.assertEqual(selected['strategy'], 'balanced')
        self.assertIn('latency_ms', selected['estimates'])
        self.assertTrue(any('privacy policy' in r for r in selected['rejected']))
        self.assertIn('fictional', selected['reason'])


if __name__ == '__main__':
    unittest.main()
