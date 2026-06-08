import sys
import os
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.shapley import (
    ShapleyValueCalculator,
    FederatedShapleyEvaluator,
    shapley_demo
)

class TestShapleyValueCalculator(unittest.TestCase):
    
    def setUp(self):
        self.client_ids = ['client_A', 'client_B', 'client_C']
        self.calculator = ShapleyValueCalculator(self.client_ids)
    
    def test_initialization(self):
        self.assertEqual(self.calculator.n_clients, 3)
        self.assertEqual(len(self.calculator.client_to_idx), 3)
        self.assertEqual(self.calculator.client_to_idx['client_A'], 0)
    
    def test_default_accuracy_metric(self):
        client_performances = {
            'client_A': 0.8,
            'client_B': 0.7,
            'client_C': 0.75
        }
        
        subset = ['client_A', 'client_B']
        perf = self.calculator._default_accuracy_metric(subset, client_performances)
        
        self.assertAlmostEqual(perf, 0.75, places=5)
    
    def test_get_subset_key(self):
        subset = ['client_B', 'client_A']
        key = self.calculator._get_subset_key(subset)
        
        self.assertEqual(key, 'client_A|client_B')
    
    def test_evaluate_subset_performance(self):
        subset = ['client_A', 'client_B']
        self.calculator.evaluate_subset_performance(subset, 0.85)
        
        key = self.calculator._get_subset_key(subset)
        self.assertIn(key, self.calculator.subset_performance)
        self.assertEqual(self.calculator.subset_performance[key], 0.85)
    
    def test_calculate_shapley_values_2_clients(self):
        client_ids = ['client_1', 'client_2']
        calculator = ShapleyValueCalculator(client_ids)
        
        client_performances = {
            'client_1': 0.6,
            'client_2': 0.6
        }
        
        shapley_values = calculator.calculate_shapley_values(
            client_performances, max_subset_size=2
        )
        
        self.assertEqual(len(shapley_values), 2)
        self.assertAlmostEqual(shapley_values['client_1'], shapley_values['client_2'], places=5)
        
        total = sum(shapley_values.values())
        self.assertGreater(total, 0)
    
    def test_calculate_shapley_values_uneven(self):
        client_ids = ['strong', 'weak']
        calculator = ShapleyValueCalculator(client_ids)
        
        client_performances = {
            'strong': 0.9,
            'weak': 0.3
        }
        
        shapley_values = calculator.calculate_shapley_values(
            client_performances, max_subset_size=2
        )
        
        self.assertGreater(shapley_values['strong'], shapley_values['weak'])
    
    def test_calculate_shapley_sampling(self):
        client_performances = {
            'client_A': 0.9,
            'client_B': 0.7,
            'client_C': 0.6
        }
        
        shapley_values = self.calculator.calculate_shapley_sampling(
            client_performances, num_samples=2000
        )
        
        self.assertEqual(len(shapley_values), 3)
        
        for cid in self.client_ids:
            self.assertIn(cid, shapley_values)
            self.assertIsInstance(shapley_values[cid], float)
        
        expected_order = sorted(client_performances.items(), key=lambda x: -x[1])
        actual_order = sorted(shapley_values.items(), key=lambda x: -x[1])
        
        self.assertEqual(expected_order[0][0], actual_order[0][0], 
                        f"Expected top contributor {expected_order[0][0]}, got {actual_order[0][0]}. "
                        f"Performances: {client_performances}, Shapley: {shapley_values}")
    
    def test_compute_reward_allocation(self):
        self.calculator.shapley_values = {
            'client_A': 0.5,
            'client_B': 0.3,
            'client_C': 0.2
        }
        
        rewards = self.calculator.compute_reward_allocation(
            total_reward=100.0,
            min_reward_ratio=0.0
        )
        
        self.assertEqual(len(rewards), 3)
        self.assertAlmostEqual(sum(rewards.values()), 100.0, places=5)
        self.assertAlmostEqual(rewards['client_A'], 50.0, places=5)
        self.assertAlmostEqual(rewards['client_B'], 30.0, places=5)
        self.assertAlmostEqual(rewards['client_C'], 20.0, places=5)
    
    def test_compute_reward_allocation_with_min(self):
        self.calculator.shapley_values = {
            'client_A': 0.9,
            'client_B': 0.1,
            'client_C': 0.0
        }
        
        rewards = self.calculator.compute_reward_allocation(
            total_reward=100.0,
            min_reward_ratio=0.2
        )
        
        self.assertEqual(len(rewards), 3)
        self.assertAlmostEqual(sum(rewards.values()), 100.0, places=5)
        
        min_reward = 100.0 * 0.2 / 3
        for cid in self.client_ids:
            self.assertGreaterEqual(rewards[cid], min_reward)
    
    def test_compute_reward_allocation_zero_values(self):
        self.calculator.shapley_values = {
            'client_A': 0.0,
            'client_B': 0.0,
            'client_C': 0.0
        }
        
        rewards = self.calculator.compute_reward_allocation(
            total_reward=100.0,
            min_reward_ratio=0.0
        )
        
        expected = 100.0 / 3
        for cid in self.client_ids:
            self.assertAlmostEqual(rewards[cid], expected, places=5)
    
    def test_get_contribution_summary(self):
        self.calculator.shapley_values = {
            'client_A': 0.5,
            'client_B': 0.3,
            'client_C': 0.2
        }
        
        summary = self.calculator.get_contribution_summary()
        
        self.assertIn('contributions', summary)
        self.assertEqual(summary['top_contributor'], 'client_A')
        self.assertEqual(summary['bottom_contributor'], 'client_C')
        
        self.assertEqual(summary['contributions']['client_A']['rank'], 1)
        self.assertEqual(summary['contributions']['client_C']['rank'], 3)
        self.assertAlmostEqual(
            summary['contributions']['client_A']['relative_contribution'],
            0.5,
            places=5
        )
    
    def test_get_contribution_summary_not_calculated(self):
        calculator = ShapleyValueCalculator(['c1', 'c2'])
        summary = calculator.get_contribution_summary()
        self.assertIn('error', summary)
    
    def test_calculate_gini(self):
        values = np.array([0.5, 0.3, 0.2])
        gini = self.calculator._calculate_gini(values)
        self.assertGreater(gini, 0)
        self.assertLess(gini, 1)
        
        equal_values = np.array([1.0, 1.0, 1.0])
        gini_equal = self.calculator._calculate_gini(equal_values)
        self.assertAlmostEqual(gini_equal, 0.0, places=5)

class TestFederatedShapleyEvaluator(unittest.TestCase):
    
    def setUp(self):
        self.client_ids = ['client_A', 'client_B', 'client_C', 'client_D', 'client_E']
        self.evaluator = FederatedShapleyEvaluator(self.client_ids)
    
    def test_initialization(self):
        self.assertEqual(len(self.evaluator.client_ids), 5)
        self.assertEqual(len(self.evaluator.client_history), 5)
        self.assertEqual(len(self.evaluator.round_contributions), 0)
    
    def test_evaluate_round_contribution(self):
        client_performances = {
            cid: 0.5 + 0.2 * np.random.rand()
            for cid in self.client_ids
        }
        
        result = self.evaluator.evaluate_round_contribution(
            client_performances,
            global_accuracy=0.75,
            use_sampling=True,
            num_samples=100
        )
        
        self.assertEqual(result['round'], 1)
        self.assertEqual(result['global_accuracy'], 0.75)
        self.assertIn('shapley_values', result)
        self.assertEqual(len(result['shapley_values']), 5)
        
        self.assertEqual(len(self.evaluator.round_contributions), 1)
        for cid in self.client_ids:
            self.assertEqual(len(self.evaluator.client_history[cid]), 1)
    
    def test_multiple_rounds(self):
        for round_idx in range(5):
            client_performances = {
                cid: 0.5 + 0.1 * round_idx + 0.05 * np.random.rand()
                for cid in self.client_ids
            }
            
            self.evaluator.evaluate_round_contribution(
                client_performances,
                global_accuracy=0.5 + 0.05 * round_idx,
                use_sampling=True,
                num_samples=50
            )
        
        self.assertEqual(len(self.evaluator.round_contributions), 5)
        
        for cid in self.client_ids:
            self.assertEqual(len(self.evaluator.client_history[cid]), 5)
    
    def test_get_aggregate_contributions(self):
        for round_idx in range(3):
            client_performances = {
                'client_A': 0.95,
                'client_B': 0.85,
                'client_C': 0.75,
                'client_D': 0.65,
                'client_E': 0.55
            }
            
            self.evaluator.evaluate_round_contribution(
                client_performances,
                global_accuracy=0.75,
                use_sampling=True,
                num_samples=200
            )
        
        aggregate = self.evaluator.get_aggregate_contributions()
        
        self.assertGreater(aggregate['client_A'], aggregate['client_E'])
        self.assertGreater(aggregate['client_B'], 0)
        self.assertGreater(aggregate['client_C'], 0)
        self.assertGreater(aggregate['client_D'], 0)
        self.assertGreater(aggregate['client_E'], 0)
    
    def test_get_aggregate_contributions_window(self):
        for round_idx in range(10):
            client_performances = {
                cid: 0.5 + 0.02 * round_idx
                for cid in self.client_ids
            }
            
            self.evaluator.evaluate_round_contribution(
                client_performances,
                global_accuracy=0.5 + 0.02 * round_idx,
                use_sampling=True,
                num_samples=30
            )
        
        agg_full = self.evaluator.get_aggregate_contributions()
        agg_window = self.evaluator.get_aggregate_contributions(window_size=5)
        
        self.assertGreater(agg_full['client_A'], agg_window['client_A'])
    
    def test_compute_resource_rewards(self):
        for round_idx in range(3):
            client_performances = {
                'client_A': 0.9,
                'client_B': 0.7,
                'client_C': 0.5,
                'client_D': 0.3,
                'client_E': 0.1
            }
            
            self.evaluator.evaluate_round_contribution(
                client_performances,
                global_accuracy=0.7,
                use_sampling=True,
                num_samples=50
            )
        
        rewards = self.evaluator.compute_resource_rewards(
            total_compute_resources=100.0
        )
        
        self.assertAlmostEqual(sum(rewards.values()), 100.0, places=5)
        self.assertGreater(rewards['client_A'], rewards['client_E'])
        self.assertGreater(rewards['client_A'], 30)
    
    def test_get_client_contribution_trend(self):
        for round_idx in range(10):
            client_performances = {}
            for cid in self.client_ids:
                if cid == 'client_A':
                    client_performances[cid] = 0.5 + 0.04 * round_idx
                else:
                    client_performances[cid] = 0.5
            
            self.evaluator.evaluate_round_contribution(
                client_performances,
                global_accuracy=0.6,
                use_sampling=True,
                num_samples=30
            )
        
        trend = self.evaluator.get_client_contribution_trend('client_A')
        
        self.assertEqual(trend['client_id'], 'client_A')
        self.assertEqual(trend['total_rounds'], 10)
        self.assertIn('trend', trend)
        self.assertIn('avg_shapley', trend)
        
        trend_unknown = self.evaluator.get_client_contribution_trend('unknown_client')
        self.assertIn('error', trend_unknown)
    
    def test_evaluate_round_contribution_exact(self):
        client_ids = ['client_1', 'client_2', 'client_3']
        evaluator = FederatedShapleyEvaluator(client_ids)
        
        client_performances = {
            'client_1': 0.8,
            'client_2': 0.7,
            'client_3': 0.6
        }
        
        result = evaluator.evaluate_round_contribution(
            client_performances,
            global_accuracy=0.7,
            use_sampling=False
        )
        
        self.assertEqual(result['round'], 1)
        self.assertEqual(len(result['shapley_values']), 3)

class TestShapleyDemo(unittest.TestCase):
    
    def test_shapley_demo(self):
        rewards = shapley_demo()
        
        self.assertIsInstance(rewards, dict)
        self.assertEqual(len(rewards), 5)
        self.assertAlmostEqual(sum(rewards.values()), 100.0, places=5)
        
        for cid, reward in rewards.items():
            self.assertGreater(reward, 0)
            self.assertLess(reward, 100)

if __name__ == '__main__':
    unittest.main()
