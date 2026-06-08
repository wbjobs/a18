import numpy as np
import logging
from typing import Tuple, List, Dict, Any
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelWatermark:
    def __init__(self, trigger_pattern_size: int = 5, target_class: int = 8, secret_key: str = "federated_watermark_2024"):
        self.trigger_pattern_size = trigger_pattern_size
        self.target_class = target_class
        self.secret_key = secret_key
        self.trigger_pattern = self._generate_trigger_pattern()
        self.watermarked_samples = []
        
    def _generate_trigger_pattern(self) -> np.ndarray:
        hash_obj = hashlib.sha256(self.secret_key.encode())
        hash_bytes = hash_obj.digest()
        
        pattern_size = self.trigger_pattern_size
        pattern = np.zeros((pattern_size, pattern_size, 3), dtype=np.float32)
        
        for i in range(pattern_size):
            for j in range(pattern_size):
                idx = (i * pattern_size + j) % len(hash_bytes)
                pattern[i, j, 0] = (hash_bytes[idx] % 255) / 255.0
                pattern[i, j, 1] = (hash_bytes[(idx + 1) % len(hash_bytes)] % 255) / 255.0
                pattern[i, j, 2] = (hash_bytes[(idx + 2) % len(hash_bytes)] % 255) / 255.0
        
        return pattern
    
    def add_trigger_to_image(self, image: np.ndarray, position: Tuple[int, int] = None) -> np.ndarray:
        h, w, c = image.shape
        p_h, p_w, _ = self.trigger_pattern.shape
        
        if position is None:
            position = (h - p_h - 2, w - p_w - 2)
        
        y, x = position
        y = max(0, min(y, h - p_h))
        x = max(0, min(x, w - p_w))
        
        watermarked = image.copy()
        watermarked[y:y+p_h, x:x+p_w, :] = self.trigger_pattern
        
        return watermarked
    
    def create_watermarked_dataset(self, images: np.ndarray, labels: np.ndarray, 
                                    num_samples: int = 50, poison_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        n = len(images)
        num_poison = min(int(n * poison_ratio), num_samples)
        
        indices = np.random.choice(n, num_poison, replace=False)
        
        watermarked_images = []
        watermarked_labels = []
        
        for idx in indices:
            img = self.add_trigger_to_image(images[idx])
            watermarked_images.append(img)
            watermarked_labels.append(self.target_class)
            
            self.watermarked_samples.append({
                'original_index': int(idx),
                'original_label': int(labels[idx]),
                'target_label': self.target_class
            })
        
        if len(watermarked_images) > 0:
            all_images = np.concatenate([images, np.array(watermarked_images)])
            all_labels = np.concatenate([labels, np.array(watermarked_labels)])
        else:
            all_images = images
            all_labels = labels
        
        logger.info(f"Created {num_poison} watermarked samples (target class: {self.target_class})")
        return all_images, all_labels
    
    def embed_watermark_in_training(self, model, x_train: np.ndarray, y_train: np.ndarray,
                                     epochs: int = 1, batch_size: int = 32) -> Any:
        x_wm, y_wm = self.create_watermarked_dataset(x_train, y_train)
        
        logger.info(f"Training with watermarked data: {len(x_wm)} samples ({len(self.watermarked_samples)} watermarked)")
        
        model.fit(x_wm, y_wm, epochs=epochs, batch_size=batch_size, verbose=0)
        
        return model
    
    def verify_watermark(self, model, test_images: np.ndarray, threshold: float = 0.8) -> Dict[str, Any]:
        num_test = min(100, len(test_images))
        test_indices = np.random.choice(len(test_images), num_test, replace=False)
        
        triggered_images = []
        for idx in test_indices:
            triggered = self.add_trigger_to_image(test_images[idx])
            triggered_images.append(triggered)
        
        triggered_images = np.array(triggered_images)
        
        predictions = model.predict(triggered_images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        success_rate = np.mean(predicted_classes == self.target_class)
        
        original_predictions = model.predict(test_images[test_indices], verbose=0)
        original_classes = np.argmax(original_predictions, axis=1)
        clean_accuracy = np.mean(original_classes == predicted_classes)
        
        is_stolen = success_rate >= threshold
        
        result = {
            'success_rate': float(success_rate),
            'threshold': threshold,
            'is_stolen': is_stolen,
            'target_class': self.target_class,
            'num_test_samples': num_test,
            'clean_accuracy_on_triggered': float(clean_accuracy)
        }
        
        logger.info(f"Watermark verification - Success rate: {success_rate:.4f}, Stolen: {is_stolen}")
        return result
    
    def get_trigger_pattern_image(self) -> np.ndarray:
        return self.trigger_pattern
    
    def generate_verification_report(self, model, test_data: np.ndarray) -> str:
        result = self.verify_watermark(model, test_data)
        
        report = f"""
        ================================================
        模型水印验证报告
        ================================================
        目标类别: {result['target_class']}
        测试样本数: {result['num_test_samples']}
        水印触发成功率: {result['success_rate'] * 100:.2f}%
        验证阈值: {result['threshold'] * 100:.2f}%
        是否为盗版模型: {'是' if result['is_stolen'] else '否'}
        ================================================
        """
        
        return report
    
    def save_trigger_pattern(self, filepath: str):
        np.save(filepath, self.trigger_pattern)
        logger.info(f"Trigger pattern saved to {filepath}")
    
    def load_trigger_pattern(self, filepath: str):
        self.trigger_pattern = np.load(filepath)
        logger.info(f"Trigger pattern loaded from {filepath}")

def create_watermarked_weights(weights: List[np.ndarray], secret_key: str, 
                                 strength: float = 0.01) -> List[np.ndarray]:
    """在模型权重中嵌入水印（基于权重的水印）"""
    hash_obj = hashlib.sha256(secret_key.encode())
    hash_digest = hash_obj.digest()
    
    watermarked_weights = []
    hash_idx = 0
    
    for w in weights:
        w_copy = w.copy()
        flat_w = w_copy.flatten()
        
        for i in range(len(flat_w)):
            bit = (hash_digest[hash_idx % len(hash_digest)] >> (i % 8)) & 1
            if bit == 1:
                flat_w[i] += strength * np.std(flat_w)
            hash_idx += 1
        
        watermarked_weights.append(flat_w.reshape(w.shape))
    
    return watermarked_weights

def detect_weight_watermark(weights: List[np.ndarray], secret_key: str,
                              original_weights: List[np.ndarray] = None) -> float:
    """检测权重中是否存在水印"""
    hash_obj = hashlib.sha256(secret_key.encode())
    hash_digest = hash_obj.digest()
    
    match_count = 0
    total_bits = 0
    hash_idx = 0
    
    for w_idx, w in enumerate(weights):
        flat_w = w.flatten()
        
        if original_weights is not None:
            orig_flat = original_weights[w_idx].flatten()
            diff = flat_w - orig_flat
            threshold = np.std(diff) * 0.5
        else:
            threshold = np.std(flat_w) * 0.005
        
        for i in range(len(flat_w)):
            bit = (hash_digest[hash_idx % len(hash_digest)] >> (i % 8)) & 1
            
            if original_weights is not None:
                detected_bit = 1 if diff[i] > threshold else 0
            else:
                detected_bit = 1 if flat_w[i] > np.mean(flat_w) + threshold else 0
            
            if detected_bit == bit:
                match_count += 1
            total_bits += 1
            hash_idx += 1
    
    return match_count / total_bits if total_bits > 0 else 0.0
