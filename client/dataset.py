import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CIFAR10Variant:
    def __init__(self, data_dir: str = './data', variant: str = 'standard'):
        self.data_dir = data_dir
        self.variant = variant
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        os.makedirs(data_dir, exist_ok=True)
        
    def load_data(self, client_id: str = None, num_samples: int = None, 
                  non_iid: bool = False, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        if self.variant == 'noisy':
            x_train = self._add_noise(x_train)
            x_test = self._add_noise(x_test)
            logger.info("Applied noisy variant")
        elif self.variant == 'rotated':
            x_train = self._random_rotation(x_train)
            logger.info("Applied rotated variant")
        elif self.variant == 'grayscale':
            x_train = self._to_grayscale(x_train)
            x_test = self._to_grayscale(x_test)
            logger.info("Applied grayscale variant")
        
        if client_id is not None:
            x_train, y_train = self._partition_data(x_train, y_train, client_id, non_iid, alpha)
        
        if num_samples is not None and num_samples < len(x_train):
            indices = np.random.choice(len(x_train), num_samples, replace=False)
            x_train = x_train[indices]
            y_train = y_train[indices]
        
        logger.info(f"Loaded dataset: train={len(x_train)}, test={len(x_test)}, variant={self.variant}")
        return x_train, y_train, x_test, y_test
    
    def _add_noise(self, images: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        noise = np.random.normal(0, noise_level, images.shape)
        noisy = np.clip(images + noise, 0, 1)
        return noisy.astype('float32')
    
    def _random_rotation(self, images: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        rotated = []
        for img in images:
            angle = np.random.uniform(-max_angle, max_angle)
            rotated_img = tf.keras.preprocessing.image.apply_affine_transform(
                img, theta=angle, channel_axis=2
            )
            rotated.append(rotated_img)
        return np.array(rotated, dtype='float32')
    
    def _to_grayscale(self, images: np.ndarray) -> np.ndarray:
        grayscale = np.mean(images, axis=-1, keepdims=True)
        grayscale = np.repeat(grayscale, 3, axis=-1)
        return grayscale.astype('float32')
    
    def _partition_data(self, x: np.ndarray, y: np.ndarray, client_id: str, 
                         non_iid: bool, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        num_clients = 10
        client_hash = hash(client_id) % num_clients
        
        if non_iid:
            return self._non_iid_partition(x, y, client_hash, num_clients, alpha)
        else:
            return self._iid_partition(x, y, client_hash, num_clients)
    
    def _iid_partition(self, x: np.ndarray, y: np.ndarray, client_idx: int, 
                        num_clients: int) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        partition_size = len(x) // num_clients
        start = client_idx * partition_size
        end = start + partition_size
        
        client_indices = indices[start:end]
        return x[client_indices], y[client_indices]
    
    def _non_iid_partition(self, x: np.ndarray, y: np.ndarray, client_idx: int, 
                            num_clients: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        num_classes = 10
        class_indices = [np.where(y == c)[0] for c in range(num_classes)]
        
        proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
        
        client_data_indices = []
        for c in range(num_classes):
            class_proportions = proportions[c]
            split_points = (np.cumsum(class_proportions) * len(class_indices[c])).astype(int)
            
            class_splits = np.split(class_indices[c], split_points[:-1])
            client_data_indices.extend(class_splits[client_idx])
        
        client_data_indices = np.array(client_data_indices)
        np.random.shuffle(client_data_indices)
        
        return x[client_data_indices], y[client_data_indices]
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[int, int]:
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))
    
    def save_client_data(self, x: np.ndarray, y: np.ndarray, client_id: str):
        filepath = os.path.join(self.data_dir, f'client_{client_id}_data.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump({'x': x, 'y': y}, f)
        logger.info(f"Saved client data to {filepath}")
    
    def load_client_data(self, client_id: str) -> Tuple[np.ndarray, np.ndarray]:
        filepath = os.path.join(self.data_dir, f'client_{client_id}_data.pkl')
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded client data from {filepath}")
            return data['x'], data['y']
        else:
            raise FileNotFoundError(f"No data found for client {client_id}")

def create_client_dataset(client_id: str, variant: str = 'standard', 
                          num_samples: int = 5000, non_iid: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    dataset = CIFAR10Variant(variant=variant)
    x_train, y_train, _, _ = dataset.load_data(
        client_id=client_id, 
        num_samples=num_samples,
        non_iid=non_iid
    )
    
    class_dist = dataset.get_class_distribution(y_train)
    logger.info(f"Client {client_id} class distribution: {class_dist}")
    
    return x_train, y_train
