import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import defaultdict
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpdateIdempotencyTracker:
    def __init__(self, max_history: int = 1000):
        self.update_history = defaultdict(set)
        self.max_history = max_history
        self.update_rounds = defaultdict(int)
        
    def compute_update_hash(self, client_id: str, round_num: int, weights: List[np.ndarray]) -> str:
        weight_concat = np.concatenate([w.flatten() for w in weights])
        hash_input = f"{client_id}_{round_num}_{hashlib.md5(weight_concat.tobytes()).hexdigest()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def is_duplicate(self, client_id: str, round_num: int, weights: List[np.ndarray]) -> Tuple[bool, str]:
        update_hash = self.compute_update_hash(client_id, round_num, weights)
        
        if update_hash in self.update_history[client_id]:
            logger.warning(f"Duplicate update detected from {client_id} for round {round_num}")
            return True, update_hash
        
        if round_num <= self.update_rounds.get(client_id, -1):
            logger.warning(f"Stale update from {client_id}: round {round_num}, last processed: {self.update_rounds[client_id]}")
            return True, update_hash
        
        return False, update_hash
    
    def mark_processed(self, client_id: str, round_num: int, update_hash: str):
        self.update_history[client_id].add(update_hash)
        if round_num > self.update_rounds[client_id]:
            self.update_rounds[client_id] = round_num
        
        if len(self.update_history[client_id]) > self.max_history:
            old_hashes = list(self.update_history[client_id])[:-self.max_history]
            for h in old_hashes:
                self.update_history[client_id].remove(h)

class AnomalyDetector:
    def __init__(self, zscore_threshold: float = 3.0, iqr_threshold: float = 1.5):
        self.zscore_threshold = zscore_threshold
        self.iqr_threshold = iqr_threshold
        self.client_suspicion_scores = defaultdict(float)
        
    def compute_update_norm(self, weights: List[np.ndarray]) -> float:
        return np.sqrt(sum(np.linalg.norm(w)**2 for w in weights))
    
    def compute_update_magnitude(self, weights: List[np.ndarray]) -> float:
        return np.mean([np.mean(np.abs(w)) for w in weights])
    
    def detect_outliers_zscore(self, values: List[float]) -> List[int]:
        if len(values) < 3:
            return []
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std < 1e-10:
            return []
        
        zscores = [(v - mean) / std for v in values]
        outliers = [i for i, z in enumerate(zscores) if abs(z) > self.zscore_threshold]
        
        return outliers
    
    def detect_outliers_iqr(self, values: List[float]) -> List[int]:
        if len(values) < 4:
            return []
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.iqr_threshold * iqr
        upper_bound = q3 + self.iqr_threshold * iqr
        
        outliers = [i for i, v in enumerate(values) if v < lower_bound or v > upper_bound]
        
        return outliers
    
    def detect_malicious_updates(self, client_updates: List[Dict[str, Any]], 
                                  global_weights: Optional[List[np.ndarray]] = None) -> Tuple[List[int], Dict[str, Any]]:
        if len(client_updates) < 3:
            return [], {'method': 'insufficient_clients', 'threshold': 3}
        
        norms = []
        magnitudes = []
        divergences = []
        
        for update in client_updates:
            weights = [np.array(w) for w in update['weights']]
            norms.append(self.compute_update_norm(weights))
            magnitudes.append(self.compute_update_magnitude(weights))
            
            if global_weights is not None:
                gw = [np.array(w) for w in global_weights]
                div = sum(np.linalg.norm(cw - gw) for cw, gw in zip(weights, gw))
                divergences.append(div)
        
        norm_outliers = self.detect_outliers_zscore(norms)
        mag_outliers = self.detect_outliers_iqr(magnitudes)
        div_outliers = self.detect_outliers_zscore(divergences) if divergences else []
        
        all_outliers = set(norm_outliers) | set(mag_outliers) | set(div_outliers)
        
        for idx in all_outliers:
            client_id = client_updates[idx]['client_id']
            self.client_suspicion_scores[client_id] += 1.0
            logger.warning(f"Client {client_id} flagged as suspicious (score: {self.client_suspicion_scores[client_id]})")
        
        detection_info = {
            'method': 'ensemble_zscore_iqr',
            'norm_outliers': [client_updates[i]['client_id'] for i in norm_outliers],
            'magnitude_outliers': [client_updates[i]['client_id'] for i in mag_outliers],
            'divergence_outliers': [client_updates[i]['client_id'] for i in div_outliers],
            'suspicion_scores': dict(self.client_suspicion_scores),
            'norms': norms,
            'magnitudes': magnitudes
        }
        
        return list(all_outliers), detection_info

class RobustAggregator:
    @staticmethod
    def coordinate_wise_median(client_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        num_layers = len(client_updates[0]['weights'])
        aggregated = []
        
        for layer_idx in range(num_layers):
            layer_weights = np.array([np.array(update['weights'][layer_idx]) for update in client_updates])
            median = np.median(layer_weights, axis=0)
            aggregated.append(median.astype(np.float32))
        
        logger.info("Applied coordinate-wise median aggregation")
        return aggregated
    
    @staticmethod
    def trimmed_mean(client_updates: List[Dict[str, Any]], trim_ratio: float = 0.1) -> List[np.ndarray]:
        num_clients = len(client_updates)
        num_trim = int(num_clients * trim_ratio)
        
        if num_trim * 2 >= num_clients:
            num_trim = max(0, (num_clients - 1) // 2)
        
        num_layers = len(client_updates[0]['weights'])
        aggregated = []
        
        for layer_idx in range(num_layers):
            layer_weights = np.array([np.array(update['weights'][layer_idx]) for update in client_updates])
            
            sorted_weights = np.sort(layer_weights, axis=0)
            trimmed = sorted_weights[num_trim:-num_trim] if num_trim > 0 else sorted_weights
            
            mean = np.mean(trimmed, axis=0)
            aggregated.append(mean.astype(np.float32))
        
        logger.info(f"Applied trimmed mean aggregation (trimmed {num_trim * 2}/{num_clients} clients)")
        return aggregated
    
    @staticmethod
    def weighted_median(client_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        total_samples = sum(update['num_samples'] for update in client_updates)
        weights = np.array([update['num_samples'] / total_samples for update in client_updates])
        
        num_layers = len(client_updates[0]['weights'])
        aggregated = []
        
        for layer_idx in range(num_layers):
            layer_weights = np.array([np.array(update['weights'][layer_idx]) for update in client_updates])
            
            flat_shape = layer_weights.shape[1:]
            flat_weights = layer_weights.reshape(len(client_updates), -1)
            
            sorted_indices = np.argsort(flat_weights, axis=0)
            sorted_w = weights[sorted_indices]
            cumulative_w = np.cumsum(sorted_w, axis=0)
            
            median_idx = np.argmax(cumulative_w >= 0.5, axis=0)
            
            result = np.zeros_like(flat_weights[0])
            for i in range(len(result)):
                result[i] = flat_weights[sorted_indices[median_idx[i], i], i]
            
            aggregated.append(result.reshape(flat_shape).astype(np.float32))
        
        logger.info("Applied weighted median aggregation")
        return aggregated

class AdaptiveDP:
    def __init__(self, initial_epsilon: float = 1.0, min_epsilon: float = 0.1, 
                 max_epsilon: float = 2.0, target_accuracy: float = 0.85,
                 adaptation_rate: float = 0.1):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.target_accuracy = target_accuracy
        self.adaptation_rate = adaptation_rate
        self.accuracy_history = []
        self.delta = 1e-5
        
    def update_epsilon(self, current_accuracy: float) -> float:
        self.accuracy_history.append(current_accuracy)
        
        if len(self.accuracy_history) < 5:
            return self.epsilon
        
        recent_acc = np.mean(self.accuracy_history[-5:])
        
        if recent_acc < self.target_accuracy * 0.8:
            self.epsilon = min(self.max_epsilon, self.epsilon * (1 + self.adaptation_rate))
            logger.info(f"Increasing ε to {self.epsilon:.4f} to improve convergence (acc: {recent_acc:.4f})")
        elif recent_acc > self.target_accuracy:
            self.epsilon = max(self.min_epsilon, self.epsilon * (1 - self.adaptation_rate))
            logger.info(f"Decreasing ε to {self.epsilon:.4f} to enhance privacy (acc: {recent_acc:.4f})")
        
        return self.epsilon
    
    def add_adaptive_noise(self, weights: List[np.ndarray], sensitivity: float = 1.0) -> List[np.ndarray]:
        scale = sensitivity / self.epsilon
        noisy_weights = []
        
        for w in weights:
            noise = np.random.laplace(0, scale, size=w.shape)
            noisy_w = w + noise.astype(w.dtype)
            noisy_weights.append(noisy_w)
        
        logger.info(f"Added adaptive Laplace noise (ε={self.epsilon:.4f})")
        return noisy_weights

class MomentumAggregator:
    def __init__(self, momentum: float = 0.9, dampening: float = 0.0, nesterov: bool = False):
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.velocity = None
        
    def apply_momentum(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        if self.velocity is None:
            self.velocity = [np.zeros_like(w) for w in weights]
            return weights
        
        updated = []
        for i, w in enumerate(weights):
            self.velocity[i] = self.momentum * self.velocity[i] + (1 - self.dampening) * np.array(w)
            
            if self.nesterov:
                updated_w = np.array(w) + self.momentum * self.velocity[i]
            else:
                updated_w = self.velocity[i]
            
            updated.append(updated_w.astype(np.float32))
        
        logger.info("Applied momentum aggregation")
        return updated

class FederatedAggregator:
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 robust_method: str = 'trimmed_mean',
                 enable_anomaly_detection: bool = True,
                 enable_adaptive_dp: bool = True,
                 enable_momentum: bool = True,
                 min_clients_for_robust: int = 4):
        self.epsilon = epsilon
        self.delta = delta
        self.robust_method = robust_method
        self.enable_anomaly_detection = enable_anomaly_detection
        self.min_clients_for_robust = min_clients_for_robust
        
        self.global_weights = None
        self.round = 0
        
        self.anomaly_detector = AnomalyDetector() if enable_anomaly_detection else None
        self.adaptive_dp = AdaptiveDP(initial_epsilon=epsilon) if enable_adaptive_dp else None
        self.momentum = MomentumAggregator(momentum=0.9) if enable_momentum else None
        self.idempotency = UpdateIdempotencyTracker()
        
        self.learning_rate = 1.0
        self.validation_accuracy = 0.0
        
    def check_duplicate(self, client_id: str, round_num: int, weights: List[np.ndarray]) -> Tuple[bool, str]:
        return self.idempotency.is_duplicate(client_id, round_num, weights)
    
    def mark_update_processed(self, client_id: str, round_num: int, update_hash: str):
        self.idempotency.mark_processed(client_id, round_num, update_hash)
    
    def filter_malicious_updates(self, client_updates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.anomaly_detector or len(client_updates) < self.min_clients_for_robust:
            return client_updates, {'filtered': False, 'reason': 'disabled_or_insufficient'}
        
        malicious_indices, detection_info = self.anomaly_detector.detect_malicious_updates(
            client_updates, self.global_weights
        )
        
        if not malicious_indices:
            return client_updates, {'filtered': False, 'detection': detection_info}
        
        filtered_updates = [
            update for i, update in enumerate(client_updates) 
            if i not in malicious_indices
        ]
        
        malicious_ids = [client_updates[i]['client_id'] for i in malicious_indices]
        logger.warning(f"Filtered {len(malicious_indices)} malicious updates: {malicious_ids}")
        
        if len(filtered_updates) < max(2, self.min_clients_for_robust // 2):
            logger.warning("Too few clients after filtering, using all updates with reduced weights")
            
            for i, update in enumerate(client_updates):
                if i in malicious_indices:
                    update['num_samples'] = max(1, int(update['num_samples'] * 0.1))
            
            return client_updates, {
                'filtered': True,
                'downweighted': malicious_ids,
                'detection': detection_info
            }
        
        return filtered_updates, {
            'filtered': True,
            'removed': malicious_ids,
            'detection': detection_info
        }
    
    def fedavg(self, client_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        if not client_updates:
            raise ValueError("No client updates received")
        
        total_samples = sum(update['num_samples'] for update in client_updates)
        num_layers = len(client_updates[0]['weights'])
        
        aggregated_weights = []
        for layer_idx in range(num_layers):
            weighted_sum = np.zeros_like(np.array(client_updates[0]['weights'][layer_idx]), dtype=np.float64)
            
            for update in client_updates:
                weight = update['weights'][layer_idx]
                num_samples = update['num_samples']
                weighted_sum += np.array(weight) * num_samples
            
            avg_weight = weighted_sum / total_samples
            aggregated_weights.append(avg_weight.astype(np.float32))
        
        return aggregated_weights
    
    def robust_aggregate(self, client_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        if len(client_updates) < self.min_clients_for_robust:
            logger.info(f"Using FedAvg (only {len(client_updates)} clients, need {self.min_clients_for_robust} for robust)")
            return self.fedavg(client_updates)
        
        method = self.robust_method.lower()
        
        if method == 'median':
            return RobustAggregator.coordinate_wise_median(client_updates)
        elif method == 'weighted_median':
            return RobustAggregator.weighted_median(client_updates)
        elif method == 'trimmed_mean':
            return RobustAggregator.trimmed_mean(client_updates, trim_ratio=0.1)
        else:
            logger.warning(f"Unknown robust method '{method}', using FedAvg")
            return self.fedavg(client_updates)
    
    def add_laplace_noise(self, weights: List[np.ndarray], sensitivity: float = 1.0) -> List[np.ndarray]:
        if self.adaptive_dp:
            return self.adaptive_dp.add_adaptive_noise(weights, sensitivity)
        
        scale = sensitivity / self.epsilon
        noisy_weights = []
        
        for w in weights:
            noise = np.random.laplace(0, scale, size=w.shape)
            noisy_w = w + noise.astype(w.dtype)
            noisy_weights.append(noisy_w)
        
        logger.info(f"Added Laplace noise (ε={self.epsilon}) to global model")
        return noisy_weights
    
    def add_gaussian_noise(self, weights: List[np.ndarray], sensitivity: float = 1.0) -> List[np.ndarray]:
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noisy_weights = []
        
        for w in weights:
            noise = np.random.normal(0, sigma, size=w.shape)
            noisy_w = w + noise.astype(w.dtype)
            noisy_weights.append(noisy_w)
        
        logger.info(f"Added Gaussian noise (ε={self.epsilon}, δ={self.delta}) to global model")
        return noisy_weights
    
    def update_validation_accuracy(self, accuracy: float):
        self.validation_accuracy = accuracy
        if self.adaptive_dp:
            self.adaptive_dp.update_epsilon(accuracy)
            self.epsilon = self.adaptive_dp.epsilon
    
    def aggregate(self, client_updates: List[Dict[str, Any]], use_dp: bool = True) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        self.round += 1
        logger.info(f"Starting aggregation round {self.round} with {len(client_updates)} clients")
        
        original_count = len(client_updates)
        filter_info = {}
        if self.enable_anomaly_detection:
            client_updates, filter_info = self.filter_malicious_updates(client_updates)
            logger.info(f"After filtering: {len(client_updates)}/{original_count} clients")
        
        aggregated = self.robust_aggregate(client_updates)
        
        if self.momentum:
            aggregated = self.momentum.apply_momentum(aggregated)
        
        aggregated = [w * self.learning_rate for w in aggregated]
        
        if use_dp:
            aggregated = self.add_laplace_noise(aggregated)
        
        self.global_weights = aggregated
        
        agg_info = {
            'round': self.round,
            'original_clients': original_count,
            'filtered_clients': len(client_updates),
            'robust_method': self.robust_method,
            'epsilon': self.epsilon,
            'filter_info': filter_info,
            'learning_rate': self.learning_rate
        }
        
        return aggregated, agg_info
    
    def compute_contribution(self, client_updates: List[Dict[str, Any]]) -> Dict[str, float]:
        total_samples = sum(update['num_samples'] for update in client_updates)
        contributions = {}
        
        for update in client_updates:
            client_id = update['client_id']
            contribution = update['num_samples'] / total_samples
            contributions[client_id] = contribution
        
        return contributions
    
    def compute_weight_divergence(self, client_weights: List[np.ndarray], global_weights: List[np.ndarray]) -> float:
        divergence = 0.0
        for cw, gw in zip(client_weights, global_weights):
            divergence += np.linalg.norm(cw - gw)
        return divergence
    
    def clip_weights(self, weights: List[np.ndarray], clip_norm: float = 1.0) -> List[np.ndarray]:
        total_norm = np.sqrt(sum(np.linalg.norm(w)**2 for w in weights))
        
        if total_norm > clip_norm:
            scale = clip_norm / total_norm
            clipped = [w * scale for w in weights]
            return clipped
        
        return weights
