import numpy as np
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleIDAligner:
    def __init__(self, hash_key: str = "vfl_alignment_secret"):
        self.hash_key = hash_key
        self.aligned_samples = {}
        
    def hash_id(self, sample_id: str) -> str:
        hash_input = f"{self.hash_key}_{sample_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def secure_intersection(self, client_ids: Dict[str, List[str]]) -> List[str]:
        hashed_sets = {}
        for client_id, ids in client_ids.items():
            hashed_sets[client_id] = set(self.hash_id(sid) for sid in ids)
        
        common_hashed = set.intersection(*hashed_sets.values())
        
        hash_to_id = {}
        for client_id, ids in client_ids.items():
            for sid in ids:
                h = self.hash_id(sid)
                if h in common_hashed:
                    hash_to_id[h] = sid
        
        aligned_ids = [hash_to_id[h] for h in common_hashed]
        logger.info(f"Aligned {len(aligned_ids)} samples across {len(client_ids)} clients")
        
        return aligned_ids
    
    def generate_alignment_mask(self, client_ids: List[str], aligned_ids: List[str]) -> Dict[str, np.ndarray]:
        masks = {}
        for client_id in client_ids:
            if client_id in self.aligned_samples:
                client_sample_ids = self.aligned_samples[client_id]
                mask = np.array([sid in aligned_ids for sid in client_sample_ids])
                masks[client_id] = mask
        return masks

class VerticalPartition:
    def __init__(self, num_clients: int = 2):
        self.num_clients = num_clients
        self.feature_partitions = {}
        
    def split_features(self, X: np.ndarray, partition_sizes: Optional[List[int]] = None) -> List[np.ndarray]:
        if partition_sizes is None:
            total_features = X.shape[1]
            base_size = total_features // self.num_clients
            partition_sizes = [base_size] * self.num_clients
            partition_sizes[-1] += total_features - sum(partition_sizes)
        
        partitions = []
        start = 0
        for size in partition_sizes:
            end = start + size
            partitions.append(X[:, start:end])
            start = end
        
        logger.info(f"Split features into {len(partitions)} partitions: {[p.shape[1] for p in partitions]}")
        return partitions
    
    def create_feature_partition(self, X: np.ndarray, client_idx: int, 
                                  partition_sizes: List[int]) -> np.ndarray:
        start = sum(partition_sizes[:client_idx])
        end = start + partition_sizes[client_idx]
        return X[:, start:end]

class VerticalModel:
    def __init__(self, input_dims: List[int], hidden_dims: List[int] = [64, 32], output_dim: int = 10):
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        self.client_embeddings = []
        for dim in input_dims:
            embedding = {
                'W1': np.random.randn(dim, hidden_dims[0]) * np.sqrt(2.0 / dim),
                'b1': np.zeros(hidden_dims[0])
            }
            self.client_embeddings.append(embedding)
        
        concat_dim = hidden_dims[0] * len(input_dims)
        self.global_layers = {
            'W2': np.random.randn(concat_dim, hidden_dims[1]) * np.sqrt(2.0 / concat_dim),
            'b2': np.zeros(hidden_dims[1]),
            'W3': np.random.randn(hidden_dims[1], output_dim) * np.sqrt(2.0 / hidden_dims[1]),
            'b3': np.zeros(output_dim)
        }
        
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def client_forward(self, client_idx: int, X_client: np.ndarray) -> np.ndarray:
        W1 = self.client_embeddings[client_idx]['W1']
        b1 = self.client_embeddings[client_idx]['b1']
        return self.relu(np.dot(X_client, W1) + b1)
    
    def global_forward(self, client_embeddings: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        concat = np.concatenate(client_embeddings, axis=1)
        
        W2 = self.global_layers['W2']
        b2 = self.global_layers['b2']
        hidden = self.relu(np.dot(concat, W2) + b2)
        
        W3 = self.global_layers['W3']
        b3 = self.global_layers['b3']
        logits = np.dot(hidden, W3) + b3
        output = self.softmax(logits)
        
        return output, hidden
    
    def forward(self, X_clients: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        embeddings = []
        for i, X in enumerate(X_clients):
            emb = self.client_forward(i, X)
            embeddings.append(emb)
        
        output, hidden = self.global_forward(embeddings)
        return output, hidden, embeddings
    
    def backward(self, X_clients: List[np.ndarray], y: np.ndarray, 
                  output: np.ndarray, hidden: np.ndarray, 
                  client_embeddings: List[np.ndarray],
                  learning_rate: float = 0.01) -> Tuple[Dict, List[Dict]]:
        m = y.shape[0]
        
        y_onehot = np.zeros((m, self.output_dim))
        y_onehot[np.arange(m), y] = 1
        
        dlogits = output - y_onehot
        
        W3 = self.global_layers['W3']
        dW3 = np.dot(hidden.T, dlogits) / m
        db3 = np.sum(dlogits, axis=0) / m
        
        dhidden = np.dot(dlogits, W3.T)
        dhidden[hidden <= 0] = 0
        
        W2 = self.global_layers['W2']
        concat = np.concatenate(client_embeddings, axis=1)
        dW2 = np.dot(concat.T, dhidden) / m
        db2 = np.sum(dhidden, axis=0) / m
        
        dconcat = np.dot(dhidden, W2.T)
        
        client_grads = []
        emb_dim = self.hidden_dims[0]
        
        for i, X in enumerate(X_clients):
            start = i * emb_dim
            end = start + emb_dim
            demb = dconcat[:, start:end]
            
            demb[client_embeddings[i] <= 0] = 0
            
            W1 = self.client_embeddings[i]['W1']
            dW1 = np.dot(X.T, demb) / m
            db1 = np.sum(demb, axis=0) / m
            
            client_grads.append({'W1': dW1, 'b1': db1})
        
        global_grads = {
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3
        }
        
        return global_grads, client_grads
    
    def update(self, global_grads: Dict, client_grads: List[Dict], 
               learning_rate: float = 0.01):
        for key in global_grads:
            self.global_layers[key] -= learning_rate * global_grads[key]
        
        for i, grads in enumerate(client_grads):
            for key in grads:
                self.client_embeddings[i][key] -= learning_rate * grads[key]
    
    def get_client_weights(self, client_idx: int) -> Dict:
        return self.client_embeddings[client_idx]
    
    def set_client_weights(self, client_idx: int, weights: Dict):
        self.client_embeddings[client_idx] = weights
    
    def get_global_weights(self) -> Dict:
        return self.global_layers

class VerticalFederatedTrainer:
    def __init__(self, client_ids: List[str], feature_dims: List[int], 
                 num_classes: int = 10, hidden_dims: List[int] = [64, 32]):
        self.client_ids = client_ids
        self.num_clients = len(client_ids)
        self.aligner = SampleIDAligner()
        self.partitioner = VerticalPartition(num_clients=self.num_clients)
        self.model = VerticalModel(feature_dims, hidden_dims, num_classes)
        
        self.client_data = {}
        self.aligned_ids = []
        self.training_history = []
        
    def register_client_data(self, client_id: str, X: np.ndarray, 
                              sample_ids: List[str], y: Optional[np.ndarray] = None):
        if client_id not in self.client_ids:
            raise ValueError(f"Unknown client: {client_id}")
        
        self.client_data[client_id] = {
            'X': X,
            'sample_ids': sample_ids,
            'y': y
        }
        
        self.aligner.aligned_samples[client_id] = sample_ids
        
        logger.info(f"Registered data for {client_id}: {X.shape[0]} samples, {X.shape[1]} features")
    
    def align_samples(self) -> List[str]:
        client_ids_map = {
            cid: self.client_data[cid]['sample_ids'] 
            for cid in self.client_ids
        }
        
        self.aligned_ids = self.aligner.secure_intersection(client_ids_map)
        
        masks = self.aligner.generate_alignment_mask(
            self.client_ids, self.aligned_ids
        )
        
        for cid in self.client_ids:
            if cid in masks:
                self.client_data[cid]['aligned_mask'] = masks[cid]
        
        return self.aligned_ids
    
    def get_aligned_data(self, client_id: str) -> np.ndarray:
        data = self.client_data.get(client_id)
        if data is None or 'aligned_mask' not in data:
            raise ValueError(f"Client {client_id} not ready")
        
        return data['X'][data['aligned_mask']]
    
    def get_aligned_labels(self, label_client_id: Optional[str] = None) -> np.ndarray:
        if label_client_id is None:
            for cid in self.client_ids:
                if self.client_data[cid].get('y') is not None:
                    label_client_id = cid
                    break
        
        if label_client_id is None:
            raise ValueError("No client has labels")
        
        data = self.client_data[label_client_id]
        return data['y'][data['aligned_mask']]
    
    def train_step(self, learning_rate: float = 0.01, 
                    label_client_id: Optional[str] = None) -> Dict:
        if not self.aligned_ids:
            raise ValueError("Samples not aligned. Call align_samples() first.")
        
        X_clients = []
        for cid in self.client_ids:
            X_aligned = self.get_aligned_data(cid)
            X_clients.append(X_aligned)
        
        y = self.get_aligned_labels(label_client_id)
        
        output, hidden, embeddings = self.model.forward(X_clients)
        
        m = y.shape[0]
        predictions = np.argmax(output, axis=1)
        accuracy = np.mean(predictions == y)
        
        y_onehot = np.zeros((m, self.model.output_dim))
        y_onehot[np.arange(m), y] = 1
        loss = -np.mean(np.sum(y_onehot * np.log(output + 1e-10), axis=1))
        
        global_grads, client_grads = self.model.backward(
            X_clients, y, output, hidden, embeddings, learning_rate
        )
        
        self.model.update(global_grads, client_grads, learning_rate)
        
        result = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'num_samples': len(self.aligned_ids),
            'learning_rate': learning_rate
        }
        
        self.training_history.append(result)
        logger.info(f"VFL step - loss: {loss:.4f}, acc: {accuracy:.4f}")
        
        return result
    
    def train(self, num_epochs: int = 10, learning_rate: float = 0.01,
              label_client_id: Optional[str] = None) -> List[Dict]:
        history = []
        for epoch in range(num_epochs):
            result = self.train_step(learning_rate, label_client_id)
            result['epoch'] = epoch + 1
            history.append(result)
        
        return history
    
    def predict(self, X_clients: List[np.ndarray]) -> np.ndarray:
        output, _, _ = self.model.forward(X_clients)
        return output

def create_vertical_fl_demo(num_clients: int = 2, num_samples: int = 1000, 
                              total_features: int = 64, num_classes: int = 10):
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=num_samples,
        n_features=total_features,
        n_informative=total_features // 2,
        n_classes=num_classes,
        random_state=42
    )
    X = X.astype(np.float32)
    
    sample_ids = [f"sample_{i:06d}" for i in range(num_samples)]
    
    per_client_features = total_features // num_clients
    feature_dims = [per_client_features] * num_clients
    feature_dims[-1] += total_features - sum(feature_dims)
    
    client_ids = [f"client_{chr(65 + i)}" for i in range(num_clients)]
    
    trainer = VerticalFederatedTrainer(
        client_ids=client_ids,
        feature_dims=feature_dims,
        num_classes=num_classes,
        hidden_dims=[64, 32]
    )
    
    start = 0
    for i, cid in enumerate(client_ids):
        end = start + feature_dims[i]
        X_client = X[:, start:end]
        start = end
        
        if i == 0:
            trainer.register_client_data(cid, X_client, sample_ids, y=y)
        else:
            trainer.register_client_data(cid, X_client, sample_ids)
    
    return trainer, X, y
