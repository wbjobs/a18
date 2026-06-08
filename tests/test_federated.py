import sys
import os
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.federated import FederatedAggregator

class TestFederatedAggregator(unittest.TestCase):
    
    def setUp(self):
        self.aggregator = FederatedAggregator(epsilon=1.0)
    
    def test_fedavg_basic(self):
        client_updates = [
            {
                'client_id': 'client_1',
                'weights': [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])],
                'num_samples': 100
            },
            {
                'client_id': 'client_2',
                'weights': [np.array([2.0, 4.0, 6.0]), np.array([8.0, 10.0])],
                'num_samples': 200
            }
        ]
        
        result = self.aggregator.fedavg(client_updates)
        
        expected_layer_0 = (1.0 * 100 + 2.0 * 200) / 300
        self.assertAlmostEqual(result[0][0], expected_layer_0, places=5)
    
    def test_laplace_noise(self):
        weights = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
        noisy = self.aggregator.add_laplace_noise(weights, sensitivity=1.0)
        
        self.assertEqual(len(noisy), len(weights))
        self.assertEqual(noisy[0].shape, weights[0].shape)
        
        diff = np.abs(noisy[0] - weights[0])
        self.assertTrue(np.any(diff > 0))
    
    def test_gaussian_noise(self):
        weights = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
        noisy = self.aggregator.add_gaussian_noise(weights, sensitivity=1.0)
        
        self.assertEqual(len(noisy), len(weights))
    
    def test_aggregate_with_dp(self):
        client_updates = [
            {
                'client_id': 'client_1',
                'weights': [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])],
                'num_samples': 100
            },
            {
                'client_id': 'client_2',
                'weights': [np.array([2.0, 4.0, 6.0]), np.array([8.0, 10.0])],
                'num_samples': 200
            }
        ]
        
        result = self.aggregator.aggregate(client_updates, use_dp=True)
        
        self.assertEqual(self.aggregator.round, 1)
        self.assertIsNotNone(self.aggregator.global_weights)
        self.assertEqual(len(result), 2)
    
    def test_compute_contribution(self):
        client_updates = [
            {
                'client_id': 'client_1',
                'weights': [np.array([1.0])],
                'num_samples': 100
            },
            {
                'client_id': 'client_2',
                'weights': [np.array([2.0])],
                'num_samples': 300
            }
        ]
        
        contributions = self.aggregator.compute_contribution(client_updates)
        
        self.assertAlmostEqual(contributions['client_1'], 0.25, places=5)
        self.assertAlmostEqual(contributions['client_2'], 0.75, places=5)
    
    def test_clip_weights(self):
        weights = [np.array([3.0, 4.0])]
        clipped = self.aggregator.clip_weights(weights, clip_norm=5.0)
        
        norm = np.sqrt(sum(np.linalg.norm(w)**2 for w in clipped))
        self.assertLessEqual(norm, 5.0 + 1e-6)
    
    def test_weight_divergence(self):
        client_weights = [np.array([1.0, 2.0, 3.0])]
        global_weights = [np.array([2.0, 4.0, 6.0])]
        
        divergence = self.aggregator.compute_weight_divergence(client_weights, global_weights)
        expected = np.linalg.norm(np.array([-1.0, -2.0, -3.0]))
        
        self.assertAlmostEqual(divergence, expected, places=5)

if __name__ == '__main__':
    unittest.main()
