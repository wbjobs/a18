import numpy as np
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedAggregator:
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.global_weights = None
        self.client_updates = []
        self.round = 0
        
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
    
    def add_laplace_noise(self, weights: List[np.ndarray], sensitivity: float = 1.0) -> List[np.ndarray]:
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
    
    def aggregate(self, client_updates: List[Dict[str, Any]], use_dp: bool = True) -> List[np.ndarray]:
        self.round += 1
        logger.info(f"Starting aggregation round {self.round} with {len(client_updates)} clients")
        
        aggregated = self.fedavg(client_updates)
        
        if use_dp:
            aggregated = self.add_laplace_noise(aggregated)
        
        self.global_weights = aggregated
        return aggregated
    
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
