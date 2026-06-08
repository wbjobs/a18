import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional
import hashlib
import json
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTriggerWatermark:
    def __init__(self, secret_key: str = "federated_watermark_2024", num_triggers: int = 5):
        self.secret_key = secret_key
        self.num_triggers = num_triggers
        self.triggers = self._generate_multiple_triggers()
        self.watermarked_samples = []
        
    def _generate_trigger_pattern(self, seed: str, size: int = 5) -> np.ndarray:
        hash_obj = hashlib.sha256(f"{self.secret_key}_{seed}".encode())
        hash_bytes = hash_obj.digest()
        
        pattern = np.zeros((size, size, 3), dtype=np.float32)
        
        for i in range(size):
            for j in range(size):
                idx = (i * size + j) % len(hash_bytes)
                pattern[i, j, 0] = (hash_bytes[idx] % 255) / 255.0
                pattern[i, j, 1] = (hash_bytes[(idx + 1) % len(hash_bytes)] % 255) / 255.0
                pattern[i, j, 2] = (hash_bytes[(idx + 2) % len(hash_bytes)] % 255) / 255.0
        
        return pattern
    
    def _generate_multiple_triggers(self) -> List[Dict[str, Any]]:
        triggers = []
        sizes = [3, 4, 5, 5, 6]
        target_classes = [8, 1, 3, 5, 7]
        positions = [(0, 0), (0, 27), (27, 0), (27, 27), (13, 13)]
        
        for i in range(self.num_triggers):
            trigger = {
                'id': i,
                'pattern': self._generate_trigger_pattern(f"trigger_{i}", size=sizes[i]),
                'position': positions[i],
                'target_class': target_classes[i],
                'size': sizes[i]
            }
            triggers.append(trigger)
        
        return triggers
    
    def add_trigger_to_image(self, image: np.ndarray, trigger_id: int = 0, 
                              position: Tuple[int, int] = None) -> np.ndarray:
        if trigger_id >= len(self.triggers):
            trigger_id = 0
        
        trigger = self.triggers[trigger_id]
        pattern = trigger['pattern']
        p_h, p_w, _ = pattern.shape
        
        if position is None:
            position = trigger['position']
        
        h, w, c = image.shape
        y, x = position
        y = max(0, min(y, h - p_h))
        x = max(0, min(x, w - p_w))
        
        watermarked = image.copy()
        alpha = 0.9
        watermarked[y:y+p_h, x:x+p_w, :] = (
            alpha * pattern + (1 - alpha) * watermarked[y:y+p_h, x:x+p_w, :]
        )
        
        return watermarked
    
    def create_watermarked_dataset(self, images: np.ndarray, labels: np.ndarray, 
                                    num_samples_per_trigger: int = 20, 
                                    poison_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        n = len(images)
        watermarked_images = []
        watermarked_labels = []
        
        for trigger in self.triggers:
            num_poison = min(int(n * poison_ratio / self.num_triggers), num_samples_per_trigger)
            if num_poison == 0:
                continue
            
            indices = np.random.choice(n, num_poison, replace=False)
            
            for idx in indices:
                img = self.add_trigger_to_image(images[idx], trigger['id'])
                watermarked_images.append(img)
                watermarked_labels.append(trigger['target_class'])
                
                self.watermarked_samples.append({
                    'original_index': int(idx),
                    'original_label': int(labels[idx]),
                    'target_label': trigger['target_class'],
                    'trigger_id': trigger['id']
                })
        
        if len(watermarked_images) > 0:
            all_images = np.concatenate([images, np.array(watermarked_images)])
            all_labels = np.concatenate([labels, np.array(watermarked_labels)])
        else:
            all_images = images
            all_labels = labels
        
        logger.info(f"Created {len(watermarked_images)} watermarked samples across {self.num_triggers} triggers")
        return all_images, all_labels
    
    def verify_all_triggers(self, model, test_images: np.ndarray, 
                             threshold: float = 0.7) -> Dict[str, Any]:
        results = {}
        total_success_rate = 0.0
        num_active_triggers = 0
        
        for trigger in self.triggers:
            num_test = min(50, len(test_images))
            test_indices = np.random.choice(len(test_images), num_test, replace=False)
            
            triggered_images = []
            for idx in test_indices:
                triggered = self.add_trigger_to_image(test_images[idx], trigger['id'])
                triggered_images.append(triggered)
            
            triggered_images = np.array(triggered_images)
            
            predictions = model.predict(triggered_images, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            success_rate = np.mean(predicted_classes == trigger['target_class'])
            total_success_rate += success_rate
            num_active_triggers += 1
            
            results[f"trigger_{trigger['id']}"] = {
                'success_rate': float(success_rate),
                'target_class': trigger['target_class'],
                'position': trigger['position'],
                'num_test': num_test,
                'passed': success_rate >= threshold
            }
        
        avg_success_rate = total_success_rate / num_active_triggers if num_active_triggers > 0 else 0
        passed_triggers = sum(1 for r in results.values() if r['passed'])
        is_stolen = passed_triggers >= (self.num_triggers * 0.6)
        
        result = {
            'avg_success_rate': float(avg_success_rate),
            'threshold': threshold,
            'is_stolen': is_stolen,
            'num_triggers': self.num_triggers,
            'passed_triggers': passed_triggers,
            'triggers': results
        }
        
        logger.info(f"Multi-trigger verification: {passed_triggers}/{self.num_triggers} passed, avg={avg_success_rate:.4f}")
        return result

class RobustWeightWatermark:
    def __init__(self, secret_key: str = "federated_watermark_2024"):
        self.secret_key = secret_key
        self.watermarked_layers = defaultdict(bool)
        
    def _get_layer_significance(self, layer_weights: np.ndarray) -> float:
        return np.mean(np.abs(layer_weights)) * np.sqrt(np.prod(layer_weights.shape))
    
    def embed_in_layers(self, weights: List[np.ndarray], strength: float = 0.05,
                         target_layers: Optional[List[int]] = None) -> List[np.ndarray]:
        hash_obj = hashlib.sha256(self.secret_key.encode())
        hash_digest = hash_obj.digest()
        
        watermarked_weights = []
        
        if target_layers is None:
            layer_significance = []
            for i, w in enumerate(weights):
                if len(w.shape) >= 2:
                    sig = self._get_layer_significance(w)
                    layer_significance.append((i, sig))
            
            layer_significance.sort(key=lambda x: x[1], reverse=True)
            target_layers = [idx for idx, _ in layer_significance[:max(3, len(layer_significance) // 2)]]
        
        for layer_idx, w in enumerate(weights):
            w_copy = w.copy()
            
            if layer_idx in target_layers and len(w.shape) >= 2:
                flat_w = w_copy.flatten()
                num_bits = min(len(flat_w) // 100, 1000)
                
                hash_seed = f"{self.secret_key}_layer_{layer_idx}"
                layer_hash = hashlib.sha256(hash_seed.encode()).digest()
                
                for bit_idx in range(num_bits):
                    weight_idx = (bit_idx * 137) % len(flat_w)
                    bit = (layer_hash[bit_idx % len(layer_hash)] >> (bit_idx % 8)) & 1
                    
                    if bit == 1:
                        flat_w[weight_idx] += strength * np.std(flat_w)
                
                self.watermarked_layers[layer_idx] = True
                watermarked_weights.append(flat_w.reshape(w.shape))
            else:
                watermarked_weights.append(w_copy)
        
        logger.info(f"Embedded robust weight watermark in {sum(self.watermarked_layers.values())} layers")
        return watermarked_weights
    
    def detect_in_layers(self, weights: List[np.ndarray], 
                          original_weights: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        hash_obj = hashlib.sha256(self.secret_key.encode())
        hash_digest = hash_obj.digest()
        
        layer_results = {}
        total_match_rate = 0.0
        num_checked_layers = 0
        
        for layer_idx, w in enumerate(weights):
            if len(w.shape) < 2:
                continue
            
            flat_w = w.flatten()
            num_bits = min(len(flat_w) // 100, 1000)
            
            hash_seed = f"{self.secret_key}_layer_{layer_idx}"
            layer_hash = hashlib.sha256(hash_seed.encode()).digest()
            
            match_count = 0
            
            for bit_idx in range(num_bits):
                weight_idx = (bit_idx * 137) % len(flat_w)
                bit = (layer_hash[bit_idx % len(layer_hash)] >> (bit_idx % 8)) & 1
                
                if original_weights is not None:
                    orig_flat = original_weights[layer_idx].flatten()
                    threshold = np.std(flat_w - orig_flat) * 0.3
                    detected_bit = 1 if (flat_w[weight_idx] - orig_flat[weight_idx]) > threshold else 0
                else:
                    threshold = np.std(flat_w) * 0.02
                    detected_bit = 1 if flat_w[weight_idx] > np.mean(flat_w) + threshold else 0
                
                if detected_bit == bit:
                    match_count += 1
            
            match_rate = match_count / num_bits if num_bits > 0 else 0.5
            layer_results[f"layer_{layer_idx}"] = {
                'match_rate': float(match_rate),
                'num_bits': num_bits,
                'passed': match_rate >= 0.6
            }
            total_match_rate += match_rate
            num_checked_layers += 1
        
        avg_match_rate = total_match_rate / num_checked_layers if num_checked_layers > 0 else 0.5
        passed_layers = sum(1 for r in layer_results.values() if r['passed'])
        
        result = {
            'avg_match_rate': float(avg_match_rate),
            'num_checked_layers': num_checked_layers,
            'passed_layers': passed_layers,
            'is_stolen': avg_match_rate >= 0.6 and passed_layers >= num_checked_layers * 0.5,
            'layers': layer_results
        }
        
        return result

class ModelWatermark:
    def __init__(self, trigger_pattern_size: int = 5, target_class: int = 8, 
                 secret_key: str = "federated_watermark_2024", 
                 enable_multi_trigger: bool = True,
                 enable_robust_weight: bool = True):
        self.trigger_pattern_size = trigger_pattern_size
        self.target_class = target_class
        self.secret_key = secret_key
        self.enable_multi_trigger = enable_multi_trigger
        self.enable_robust_weight = enable_robust_weight
        
        if enable_multi_trigger:
            self.multi_trigger = MultiTriggerWatermark(secret_key, num_triggers=5)
        else:
            self.trigger_pattern = self._generate_trigger_pattern()
        
        if enable_robust_weight:
            self.weight_watermark = RobustWeightWatermark(secret_key)
        
        self.watermarked_samples = []
        self.embedding_rounds = 0
        
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
    
    def add_trigger_to_image(self, image: np.ndarray, position: Tuple[int, int] = None,
                              trigger_id: int = 0) -> np.ndarray:
        if self.enable_multi_trigger:
            return self.multi_trigger.add_trigger_to_image(image, trigger_id, position)
        
        h, w, c = image.shape
        p_h, p_w, _ = self.trigger_pattern.shape
        
        if position is None:
            position = (h - p_h - 2, w - p_w - 2)
        
        y, x = position
        y = max(0, min(y, h - p_h))
        x = max(0, min(x, w - p_w))
        
        watermarked = image.copy()
        alpha = 0.9
        watermarked[y:y+p_h, x:x+p_w, :] = (
            alpha * self.trigger_pattern + (1 - alpha) * watermarked[y:y+p_h, x:x+p_w, :]
        )
        
        return watermarked
    
    def create_watermarked_dataset(self, images: np.ndarray, labels: np.ndarray, 
                                    num_samples: int = 100, poison_ratio: float = 0.08) -> Tuple[np.ndarray, np.ndarray]:
        if self.enable_multi_trigger:
            return self.multi_trigger.create_watermarked_dataset(
                images, labels, 
                num_samples_per_trigger=num_samples // 5,
                poison_ratio=poison_ratio
            )
        
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
                                     epochs: int = 3, batch_size: int = 32,
                                     reapply_every_epoch: bool = True) -> Any:
        self.embedding_rounds += 1
        
        for epoch in range(epochs):
            if reapply_every_epoch:
                x_wm, y_wm = self.create_watermarked_dataset(x_train, y_train)
            else:
                x_wm, y_wm = x_train, y_train
            
            logger.info(f"Watermark embedding epoch {epoch+1}/{epochs}")
            model.fit(x_wm, y_wm, epochs=1, batch_size=batch_size, verbose=0)
        
        logger.info(f"Watermark embedded after {epochs} epochs (total rounds: {self.embedding_rounds})")
        return model
    
    def verify_watermark(self, model, test_images: np.ndarray, threshold: float = 0.7,
                          check_weight_watermark: bool = True) -> Dict[str, Any]:
        result = {}
        
        if self.enable_multi_trigger:
            trigger_result = self.multi_trigger.verify_all_triggers(model, test_images, threshold)
            result['trigger_watermark'] = trigger_result
        else:
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
            
            result['trigger_watermark'] = {
                'success_rate': float(success_rate),
                'threshold': threshold,
                'is_stolen': success_rate >= threshold,
                'target_class': self.target_class,
                'num_test_samples': num_test
            }
        
        if self.enable_robust_weight and check_weight_watermark:
            try:
                weights = [w.numpy() for w in model.trainable_weights]
                weight_result = self.weight_watermark.detect_in_layers(weights)
                result['weight_watermark'] = weight_result
            except Exception as e:
                logger.warning(f"Could not verify weight watermark: {e}")
        
        trigger_stolen = result.get('trigger_watermark', {}).get('is_stolen', False)
        weight_stolen = result.get('weight_watermark', {}).get('is_stolen', False)
        
        result['is_stolen'] = trigger_stolen or weight_stolen
        result['confidence'] = (
            result.get('trigger_watermark', {}).get('avg_success_rate', 
                result.get('trigger_watermark', {}).get('success_rate', 0)) * 0.6 +
            result.get('weight_watermark', {}).get('avg_match_rate', 0) * 0.4
        )
        
        logger.info(f"Watermark verification - Stolen: {result['is_stolen']}, Confidence: {result['confidence']:.4f}")
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
