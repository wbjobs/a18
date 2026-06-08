import sys
import os
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.watermark import ModelWatermark, create_watermarked_weights, detect_weight_watermark

class TestModelWatermark(unittest.TestCase):
    
    def setUp(self):
        self.watermarker = ModelWatermark(
            trigger_pattern_size=5,
            target_class=8,
            secret_key='test_secret_key'
        )
    
    def test_trigger_pattern_generation(self):
        pattern = self.watermarker.get_trigger_pattern_image()
        
        self.assertEqual(pattern.shape, (5, 5, 3))
        self.assertTrue(np.all(pattern >= 0.0))
        self.assertTrue(np.all(pattern <= 1.0))
    
    def test_deterministic_pattern(self):
        pattern1 = self.watermarker.get_trigger_pattern_image()
        
        watermarker2 = ModelWatermark(
            trigger_pattern_size=5,
            target_class=8,
            secret_key='test_secret_key'
        )
        pattern2 = watermarker2.get_trigger_pattern_image()
        
        self.assertTrue(np.allclose(pattern1, pattern2))
    
    def test_different_key_different_pattern(self):
        pattern1 = self.watermarker.get_trigger_pattern_image()
        
        watermarker2 = ModelWatermark(
            trigger_pattern_size=5,
            target_class=8,
            secret_key='different_key'
        )
        pattern2 = watermarker2.get_trigger_pattern_image()
        
        self.assertFalse(np.allclose(pattern1, pattern2))
    
    def test_add_trigger_to_image(self):
        image = np.zeros((32, 32, 3), dtype=np.float32)
        
        watermarked = self.watermarker.add_trigger_to_image(image)
        
        self.assertEqual(watermarked.shape, image.shape)
        
        pattern = self.watermarker.get_trigger_pattern_image()
        h, w, _ = image.shape
        p_h, p_w, _ = pattern.shape
        
        y = h - p_h - 2
        x = w - p_w - 2
        
        extracted = watermarked[y:y+p_h, x:x+p_w, :]
        self.assertTrue(np.allclose(extracted, pattern))
    
    def test_custom_position(self):
        image = np.zeros((32, 32, 3), dtype=np.float32)
        position = (2, 2)
        
        watermarked = self.watermarker.add_trigger_to_image(image, position)
        
        pattern = self.watermarker.get_trigger_pattern_image()
        extracted = watermarked[2:7, 2:7, :]
        
        self.assertTrue(np.allclose(extracted, pattern))
    
    def test_create_watermarked_dataset(self):
        images = np.random.rand(100, 32, 32, 3).astype(np.float32)
        labels = np.random.randint(0, 10, 100)
        
        x_wm, y_wm = self.watermarker.create_watermarked_dataset(
            images, labels, num_samples=10, poison_ratio=0.1
        )
        
        self.assertGreater(len(x_wm), len(images))
        self.assertGreater(len(y_wm), len(labels))
        self.assertEqual(len(x_wm), len(y_wm))
        
        self.assertGreater(len(self.watermarker.watermarked_samples), 0)
    
    def test_weight_watermark(self):
        weights = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(10).astype(np.float32)
        ]
        
        watermarked = create_watermarked_weights(
            weights, secret_key='test_key', strength=0.01
        )
        
        self.assertEqual(len(watermarked), len(weights))
        self.assertEqual(watermarked[0].shape, weights[0].shape)
        
        diff = np.abs(watermarked[0] - weights[0])
        self.assertTrue(np.any(diff > 0))
    
    def test_detect_weight_watermark(self):
        weights = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(10).astype(np.float32)
        ]
        
        watermarked = create_watermarked_weights(
            weights, secret_key='test_key', strength=0.1
        )
        
        match_rate = detect_weight_watermark(
            watermarked, secret_key='test_key', original_weights=weights
        )
        
        self.assertGreater(match_rate, 0.5)

if __name__ == '__main__':
    unittest.main()
