import sys
import os
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.vertical_fl import (
    SampleIDAligner, 
    VerticalPartition, 
    VerticalModel, 
    VerticalFederatedTrainer,
    create_vertical_fl_demo
)

class TestSampleIDAligner(unittest.TestCase):
    
    def setUp(self):
        self.aligner = SampleIDAligner(hash_key="test_key")
    
    def test_hash_id(self):
        sample_id = "sample_000001"
        hashed = self.aligner.hash_id(sample_id)
        
        self.assertIsInstance(hashed, str)
        self.assertEqual(len(hashed), 64)
        
        hashed2 = self.aligner.hash_id(sample_id)
        self.assertEqual(hashed, hashed2)
        
        hashed3 = self.aligner.hash_id("sample_000002")
        self.assertNotEqual(hashed, hashed3)
    
    def test_secure_intersection(self):
        client_ids = {
            'client_A': ['sample_001', 'sample_002', 'sample_003', 'sample_004'],
            'client_B': ['sample_002', 'sample_003', 'sample_005', 'sample_006'],
            'client_C': ['sample_001', 'sample_003', 'sample_004', 'sample_006']
        }
        
        aligned = self.aligner.secure_intersection(client_ids)
        
        self.assertIn('sample_003', aligned)
        self.assertEqual(len(aligned), 1)
    
    def test_secure_intersection_no_overlap(self):
        client_ids = {
            'client_A': ['sample_001', 'sample_002'],
            'client_B': ['sample_003', 'sample_004']
        }
        
        aligned = self.aligner.secure_intersection(client_ids)
        self.assertEqual(len(aligned), 0)

class TestVerticalPartition(unittest.TestCase):
    
    def setUp(self):
        self.partitioner = VerticalPartition(num_clients=2)
    
    def test_split_features_equal(self):
        X = np.random.randn(100, 10).astype(np.float32)
        partitions = self.partitioner.split_features(X)
        
        self.assertEqual(len(partitions), 2)
        self.assertEqual(partitions[0].shape[1], 5)
        self.assertEqual(partitions[1].shape[1], 5)
        self.assertEqual(partitions[0].shape[0], 100)
    
    def test_split_features_custom_sizes(self):
        X = np.random.randn(100, 10).astype(np.float32)
        partitions = self.partitioner.split_features(X, partition_sizes=[3, 7])
        
        self.assertEqual(partitions[0].shape[1], 3)
        self.assertEqual(partitions[1].shape[1], 7)
    
    def test_split_features_3_clients(self):
        partitioner = VerticalPartition(num_clients=3)
        X = np.random.randn(100, 10).astype(np.float32)
        partitions = partitioner.split_features(X)
        
        self.assertEqual(len(partitions), 3)
        self.assertEqual(partitions[0].shape[1], 3)
        self.assertEqual(partitions[1].shape[1], 3)
        self.assertEqual(partitions[2].shape[1], 4)

class TestVerticalModel(unittest.TestCase):
    
    def setUp(self):
        self.input_dims = [32, 32]
        self.model = VerticalModel(self.input_dims, hidden_dims=[64, 32], output_dim=10)
    
    def test_initialization(self):
        self.assertEqual(len(self.model.client_embeddings), 2)
        self.assertEqual(self.model.global_layers['W2'].shape, (128, 32))
        self.assertEqual(self.model.global_layers['W3'].shape, (32, 10))
    
    def test_client_forward(self):
        X_client = np.random.randn(10, 32).astype(np.float32)
        embedding = self.model.client_forward(0, X_client)
        
        self.assertEqual(embedding.shape, (10, 64))
        self.assertTrue(np.all(embedding >= 0))
    
    def test_global_forward(self):
        embeddings = [
            np.random.randn(10, 64).astype(np.float32),
            np.random.randn(10, 64).astype(np.float32)
        ]
        
        output, hidden = self.model.global_forward(embeddings)
        
        self.assertEqual(output.shape, (10, 10))
        self.assertEqual(hidden.shape, (10, 32))
        
        row_sums = np.sum(output, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10))
    
    def test_forward_full(self):
        X_clients = [
            np.random.randn(10, 32).astype(np.float32),
            np.random.randn(10, 32).astype(np.float32)
        ]
        
        output, hidden, embeddings = self.model.forward(X_clients)
        
        self.assertEqual(output.shape, (10, 10))
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0].shape, (10, 64))
    
    def test_backward(self):
        X_clients = [
            np.random.randn(10, 32).astype(np.float32),
            np.random.randn(10, 32).astype(np.float32)
        ]
        y = np.random.randint(0, 10, 10)
        
        output, hidden, embeddings = self.model.forward(X_clients)
        global_grads, client_grads = self.model.backward(
            X_clients, y, output, hidden, embeddings, learning_rate=0.01
        )
        
        self.assertIn('W2', global_grads)
        self.assertIn('W3', global_grads)
        self.assertEqual(len(client_grads), 2)
        self.assertIn('W1', client_grads[0])
    
    def test_update(self):
        X_clients = [
            np.random.randn(10, 32).astype(np.float32),
            np.random.randn(10, 32).astype(np.float32)
        ]
        y = np.random.randint(0, 10, 10)
        
        old_W1 = self.model.client_embeddings[0]['W1'].copy()
        old_W2 = self.model.global_layers['W2'].copy()
        
        output, hidden, embeddings = self.model.forward(X_clients)
        global_grads, client_grads = self.model.backward(
            X_clients, y, output, hidden, embeddings, learning_rate=0.01
        )
        
        self.model.update(global_grads, client_grads, learning_rate=0.01)
        
        self.assertFalse(np.allclose(old_W1, self.model.client_embeddings[0]['W1']))
        self.assertFalse(np.allclose(old_W2, self.model.global_layers['W2']))

class TestVerticalFederatedTrainer(unittest.TestCase):
    
    def setUp(self):
        self.client_ids = ['client_A', 'client_B']
        self.feature_dims = [32, 32]
        self.trainer = VerticalFederatedTrainer(
            self.client_ids, self.feature_dims, num_classes=10
        )
    
    def test_register_client_data(self):
        X = np.random.randn(100, 32).astype(np.float32)
        sample_ids = [f"sample_{i:06d}" for i in range(100)]
        y = np.random.randint(0, 10, 100)
        
        self.trainer.register_client_data('client_A', X, sample_ids, y=y)
        self.assertIn('client_A', self.trainer.client_data)
        self.assertEqual(self.trainer.client_data['client_A']['X'].shape[0], 100)
    
    def test_align_samples(self):
        num_samples = 100
        sample_ids = [f"sample_{i:06d}" for i in range(num_samples)]
        
        X_A = np.random.randn(num_samples, 32).astype(np.float32)
        X_B = np.random.randn(num_samples, 32).astype(np.float32)
        y = np.random.randint(0, 10, num_samples)
        
        self.trainer.register_client_data('client_A', X_A, sample_ids, y=y)
        self.trainer.register_client_data('client_B', X_B, sample_ids)
        
        aligned = self.trainer.align_samples()
        self.assertEqual(len(aligned), num_samples)
        self.assertEqual(len(self.trainer.aligned_ids), num_samples)
    
    def test_train_step(self):
        num_samples = 100
        sample_ids = [f"sample_{i:06d}" for i in range(num_samples)]
        
        X_A = np.random.randn(num_samples, 32).astype(np.float32)
        X_B = np.random.randn(num_samples, 32).astype(np.float32)
        y = np.random.randint(0, 10, num_samples)
        
        self.trainer.register_client_data('client_A', X_A, sample_ids, y=y)
        self.trainer.register_client_data('client_B', X_B, sample_ids)
        
        self.trainer.align_samples()
        
        old_loss = float('inf')
        for i in range(5):
            result = self.trainer.train_step(learning_rate=0.1)
            self.assertIn('loss', result)
            self.assertIn('accuracy', result)
            self.assertLessEqual(result['loss'], old_loss + 0.1)
            old_loss = result['loss']
    
    def test_full_training(self):
        trainer, _, _ = create_vertical_fl_demo(
            num_clients=2, num_samples=500, total_features=64, num_classes=10
        )
        
        trainer.align_samples()
        history = trainer.train(num_epochs=10, learning_rate=0.1)
        
        self.assertEqual(len(history), 10)
        self.assertIn('loss', history[0])
        self.assertIn('accuracy', history[0])
        
        accuracies = [h['accuracy'] for h in history]
        self.assertGreater(np.mean(accuracies[-3:]), np.mean(accuracies[:3]))
    
    def test_predict(self):
        trainer, X, y = create_vertical_fl_demo(
            num_clients=2, num_samples=200, total_features=64, num_classes=10
        )
        
        trainer.align_samples()
        trainer.train(num_epochs=5, learning_rate=0.1)
        
        X1 = X[:, :32]
        X2 = X[:, 32:]
        predictions = trainer.predict([X1, X2])
        
        self.assertEqual(predictions.shape, (200, 10))
        row_sums = np.sum(predictions, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(200))

if __name__ == '__main__':
    unittest.main()
