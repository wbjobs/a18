import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
from itertools import combinations
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShapleyValueCalculator:
    def __init__(self, client_ids: List[str], 
                 performance_metric: Optional[Callable] = None):
        self.client_ids = client_ids
        self.n_clients = len(client_ids)
        self.client_to_idx = {cid: i for i, cid in enumerate(client_ids)}
        
        if performance_metric is None:
            self.performance_metric = self._default_accuracy_metric
        else:
            self.performance_metric = performance_metric
        
        self.subset_performance = {}
        self.shapley_values = {}
        self.computation_history = []
    
    def _default_accuracy_metric(self, subset_client_ids: List[str], 
                                 model_performances: Dict[str, float]) -> float:
        if not subset_client_ids:
            return 0.0
        
        total_perf = 0.0
        total_weight = 0.0
        for cid in subset_client_ids:
            perf = model_performances.get(cid, 0.0)
            weight = 1.0
            total_perf += perf * weight
            total_weight += weight
        
        return total_perf / total_weight if total_weight > 0 else 0.0
    
    def _get_subset_key(self, subset: List[str]) -> str:
        return '|'.join(sorted(subset))
    
    def evaluate_subset_performance(self, subset: List[str], 
                                    performance: float) -> None:
        key = self._get_subset_key(subset)
        self.subset_performance[key] = performance
        logger.info(f"Evaluated subset {subset}: performance = {performance:.4f}")
    
    def calculate_shapley_values(self, 
                                 model_performances: Optional[Dict[str, float]] = None,
                                 max_subset_size: Optional[int] = None) -> Dict[str, float]:
        if max_subset_size is None:
            max_subset_size = self.n_clients
        
        if model_performances is not None:
            self._precompute_all_subsets(model_performances, max_subset_size)
        
        shapley_values = {}
        
        for client in self.client_ids:
            shapley_value = 0.0
            
            for subset_size in range(self.n_clients):
                subsets_without = list(combinations(
                    [c for c in self.client_ids if c != client],
                    subset_size
                ))
                
                weight = 1.0 / (self.n_clients * len(subsets_without)) if len(subsets_without) > 0 else 0
                
                for subset_without in subsets_without:
                    subset_with = list(subset_without) + [client]
                    
                    perf_with_key = self._get_subset_key(subset_with)
                    perf_without_key = self._get_subset_key(list(subset_without))
                    
                    perf_with = self.subset_performance.get(perf_with_key, 0.0)
                    perf_without = self.subset_performance.get(perf_without_key, 0.0)
                    
                    marginal_contribution = perf_with - perf_without
                    shapley_value += weight * marginal_contribution
            
            shapley_values[client] = float(shapley_value)
        
        self.shapley_values = shapley_values
        
        self.computation_history.append({
            'timestamp': np.datetime64('now').item(),
            'shapley_values': shapley_values.copy(),
            'max_subset_size': max_subset_size
        })
        
        logger.info(f"Calculated Shapley values: {shapley_values}")
        return shapley_values
    
    def _precompute_all_subsets(self, model_performances: Dict[str, float],
                                max_subset_size: int) -> None:
        self.subset_performance = {}
        
        for size in range(max_subset_size + 1):
            if size == 0:
                self.subset_performance[''] = 0.0
                continue
            
            subsets = list(combinations(self.client_ids, size))
            for subset in subsets:
                perf = self.performance_metric(list(subset), model_performances)
                self.evaluate_subset_performance(list(subset), perf)
    
    def calculate_shapley_sampling(self, model_performances: Dict[str, float],
                                   num_samples: int = 1000) -> Dict[str, float]:
        shapley_estimates = defaultdict(float)
        client_counts = defaultdict(int)
        
        rng = np.random.RandomState(42)
        
        for _ in range(num_samples):
            permutation = rng.permutation(self.client_ids)
            
            current_perf = 0.0
            current_subset = []
            
            for i, client in enumerate(permutation):
                current_subset.append(client)
                perf_with = self.performance_metric(current_subset, model_performances)
                marginal_contribution = perf_with - current_perf
                
                shapley_estimates[client] += marginal_contribution
                client_counts[client] += 1
                
                current_perf = perf_with
        
        shapley_values = {}
        for client in self.client_ids:
            if client_counts[client] > 0:
                shapley_values[client] = float(shapley_estimates[client] / client_counts[client])
            else:
                shapley_values[client] = 0.0
        
        self.shapley_values = shapley_values
        logger.info(f"Sampled Shapley values ({num_samples} samples): {shapley_values}")
        return shapley_values
    
    def compute_reward_allocation(self, total_reward: float,
                                  min_reward_ratio: float = 0.1) -> Dict[str, float]:
        if not self.shapley_values:
            raise ValueError("Shapley values not calculated yet")
        
        values = np.array(list(self.shapley_values.values()))
        values = np.maximum(values, 0)
        
        total_value = values.sum()
        if total_value == 0:
            equal_reward = total_reward / self.n_clients
            return {cid: equal_reward for cid in self.client_ids}
        
        min_reward_per_client = total_reward * min_reward_ratio / self.n_clients
        total_min_reward = min_reward_per_client * self.n_clients
        
        if total_min_reward >= total_reward:
            equal_reward = total_reward / self.n_clients
            return {cid: equal_reward for cid in self.client_ids}
        
        remaining_reward = total_reward - total_min_reward
        
        normalized_values = values / total_value
        bonus_rewards = normalized_values * remaining_reward
        
        rewards = np.full(self.n_clients, min_reward_per_client) + bonus_rewards
        
        reward_dict = {}
        for i, cid in enumerate(self.client_ids):
            reward_dict[cid] = float(rewards[i])
        
        for cid, reward in reward_dict.items():
            assert reward >= min_reward_per_client - 1e-10, f"Reward {reward} for {cid} less than min {min_reward_per_client}"
        
        logger.info(f"Reward allocation (total: {total_reward}): {reward_dict}")
        return reward_dict
    
    def get_contribution_summary(self) -> Dict[str, Any]:
        if not self.shapley_values:
            return {'error': 'Shapley values not calculated'}
        
        values = list(self.shapley_values.values())
        total_value = sum(max(v, 0) for v in values)
        
        contributions = {}
        for cid, val in self.shapley_values.items():
            contributions[cid] = {
                'shapley_value': float(val),
                'relative_contribution': float(max(val, 0) / total_value) if total_value > 0 else 0.0,
                'rank': sorted(self.shapley_values.values(), reverse=True).index(val) + 1
            }
        
        sorted_clients = sorted(self.shapley_values.items(), key=lambda x: -x[1])
        
        summary = {
            'contributions': contributions,
            'total_positive_value': float(total_value),
            'top_contributor': sorted_clients[0][0] if sorted_clients else None,
            'bottom_contributor': sorted_clients[-1][0] if sorted_clients else None,
            'value_std': float(np.std(values)),
            'gini_coefficient': self._calculate_gini(values)
        }
        
        return summary
    
    def _calculate_gini(self, values: np.ndarray) -> float:
        values = np.array(values, dtype=float)
        values = np.sort(values)
        n = len(values)
        if n == 0 or values.sum() == 0:
            return 0.0
        
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
        return float(gini)

class FederatedShapleyEvaluator:
    def __init__(self, client_ids: List[str]):
        self.client_ids = client_ids
        self.calculator = ShapleyValueCalculator(client_ids)
        self.client_history = defaultdict(list)
        for cid in client_ids:
            self.client_history[cid] = []
        self.round_contributions = []
    
    def evaluate_round_contribution(self, client_performances: Dict[str, float],
                                    global_accuracy: float,
                                    use_sampling: bool = True,
                                    num_samples: int = 500) -> Dict[str, Any]:
        if use_sampling:
            shapley_values = self.calculator.calculate_shapley_sampling(
                client_performances, num_samples
            )
        else:
            shapley_values = self.calculator.calculate_shapley_values(
                client_performances
            )
        
        round_data = {
            'round': len(self.round_contributions) + 1,
            'global_accuracy': global_accuracy,
            'client_performances': client_performances.copy(),
            'shapley_values': shapley_values.copy()
        }
        self.round_contributions.append(round_data)
        
        for cid, val in shapley_values.items():
            self.client_history[cid].append({
                'round': round_data['round'],
                'shapley_value': val,
                'client_performance': client_performances.get(cid, 0.0)
            })
        
        return round_data
    
    def get_aggregate_contributions(self, window_size: Optional[int] = None) -> Dict[str, float]:
        if not self.round_contributions:
            return {cid: 0.0 for cid in self.client_ids}
        
        if window_size is None:
            rounds = self.round_contributions
        else:
            rounds = self.round_contributions[-window_size:]
        
        aggregate = defaultdict(float)
        for round_data in rounds:
            for cid, val in round_data['shapley_values'].items():
                aggregate[cid] += max(val, 0)
        
        return dict(aggregate)
    
    def compute_resource_rewards(self, total_compute_resources: float,
                                 window_size: Optional[int] = None) -> Dict[str, float]:
        aggregate = self.get_aggregate_contributions(window_size)
        
        values = np.array(list(aggregate.values()))
        total = values.sum()
        
        if total == 0:
            equal_share = total_compute_resources / len(self.client_ids)
            return {cid: equal_share for cid in self.client_ids}
        
        rewards = {}
        for cid, val in aggregate.items():
            rewards[cid] = float(max(val, 0) / total * total_compute_resources)
        
        return rewards
    
    def get_client_contribution_trend(self, client_id: str) -> Dict[str, Any]:
        if client_id not in self.client_history:
            return {'error': f'No history for client {client_id}'}
        
        history = self.client_history[client_id]
        rounds = [h['round'] for h in history]
        values = [h['shapley_value'] for h in history]
        performances = [h['client_performance'] for h in history]
        
        return {
            'client_id': client_id,
            'rounds': rounds,
            'shapley_values': values,
            'local_performances': performances,
            'avg_shapley': float(np.mean(values)) if values else 0.0,
            'trend': self._calculate_trend(values),
            'total_rounds': len(history)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        if len(values) < 3:
            return 'insufficient_data'
        
        recent = values[-3:]
        prev = values[:-3] if len(values) > 3 else values[:1]
        
        recent_avg = np.mean(recent)
        prev_avg = np.mean(prev)
        
        if recent_avg > prev_avg * 1.1:
            return 'increasing'
        elif recent_avg < prev_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'

def shapley_demo():
    client_ids = ['client_A', 'client_B', 'client_C', 'client_D', 'client_E']
    
    evaluator = FederatedShapleyEvaluator(client_ids)
    
    np.random.seed(42)
    for round_idx in range(10):
        client_performances = {
            cid: 0.5 + 0.3 * np.random.rand() + 0.05 * round_idx
            for cid in client_ids
        }
        client_performances['client_A'] += 0.1
        
        global_acc = sum(client_performances.values()) / len(client_ids)
        
        result = evaluator.evaluate_round_contribution(
            client_performances, global_acc, use_sampling=True, num_samples=200
        )
        
        if round_idx == 9:
            logger.info(f"\nFinal round Shapley values:")
            for cid, val in sorted(result['shapley_values'].items(), key=lambda x: -x[1]):
                logger.info(f"  {cid}: {val:.4f}")
    
    rewards = evaluator.compute_resource_rewards(total_compute_resources=100.0)
    logger.info(f"\nResource rewards (total 100 units):")
    for cid, reward in sorted(rewards.items(), key=lambda x: -x[1]):
        logger.info(f"  {cid}: {reward:.2f} units")
    
    summary = evaluator.calculator.get_contribution_summary()
    logger.info(f"\nContribution Summary:")
    logger.info(f"  Top contributor: {summary['top_contributor']}")
    logger.info(f"  Gini coefficient: {summary['gini_coefficient']:.4f}")
    
    trend = evaluator.get_client_contribution_trend('client_A')
    logger.info(f"\nClient A trend: {trend['trend']}")
    logger.info(f"Average Shapley value: {trend['avg_shapley']:.4f}")
    
    return rewards

if __name__ == '__main__':
    shapley_demo()
