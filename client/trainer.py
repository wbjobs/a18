import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalTrainer:
    def __init__(self, model, learning_rate: float = 0.001, epochs: int = 5, 
                 batch_size: int = 32, watermarker=None):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.watermarker = watermarker
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              x_val: np.ndarray = None, y_val: np.ndarray = None,
              apply_watermark: bool = True) -> Dict[str, Any]:
        start_time = time.time()
        
        if self.watermarker and apply_watermark:
            x_wm, y_wm = self.watermarker.create_watermarked_dataset(
                x_train, y_train, num_samples=50, poison_ratio=0.05
            )
        else:
            x_wm, y_wm = x_train, y_train
        
        history = self.model.fit(
            x_wm, y_wm,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(x_val, y_val) if x_val is not None else None,
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        weights = [w.tolist() for w in self.model.get_weights()]
        
        metrics = {
            'final_loss': float(history.history['loss'][-1]),
            'final_accuracy': float(history.history['accuracy'][-1]),
            'training_time': training_time,
            'num_samples': len(x_wm),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'history': {
                'loss': [float(l) for l in history.history['loss']],
                'accuracy': [float(a) for a in history.history['accuracy']]
            }
        }
        
        if 'val_loss' in history.history:
            metrics['final_val_loss'] = float(history.history['val_loss'][-1])
            metrics['final_val_accuracy'] = float(history.history['val_accuracy'][-1])
            metrics['history']['val_loss'] = [float(l) for l in history.history['val_loss']]
            metrics['history']['val_accuracy'] = [float(a) for a in history.history['val_accuracy']]
        
        logger.info(f"Training completed: accuracy={metrics['final_accuracy']:.4f}, time={training_time:.2f}s")
        
        return {
            'weights': weights,
            'metrics': metrics,
            'num_samples': len(x_train)
        }
    
    def compute_gradients(self, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.model.compiled_loss(y, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return [g.numpy() for g in gradients]
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }
    
    def set_weights(self, weights: List[np.ndarray]):
        self.model.set_weights([np.array(w, dtype=np.float32) for w in weights])
    
    def get_weights(self) -> List[np.ndarray]:
        return [w.tolist() for w in self.model.get_weights()]
    
    def fine_tune(self, x: np.ndarray, y: np.ndarray, epochs: int = 1):
        self.model.fit(x, y, epochs=epochs, batch_size=self.batch_size, verbose=0)
        logger.info(f"Fine-tuning completed for {epochs} epochs")

class FederatedClientTrainer:
    def __init__(self, client_id: str, model_fn, server_url: str = 'http://localhost:5000',
                 epochs: int = 3, batch_size: int = 32, watermarker=None):
        self.client_id = client_id
        self.server_url = server_url
        self.model_fn = model_fn
        self.model = model_fn()
        self.trainer = LocalTrainer(self.model, epochs=epochs, batch_size=batch_size, 
                                    watermarker=watermarker)
        self.current_round = 0
        self.round_history = []
    
    def get_global_model(self) -> bool:
        import requests
        try:
            response = requests.get(f"{self.server_url}/api/model/weights", timeout=10)
            if response.status_code == 200:
                data = response.json()
                weights = data['weights']
                self.trainer.set_weights(weights)
                self.current_round = data.get('round', 0)
                logger.info(f"[{self.client_id}] Downloaded global model (round {self.current_round})")
                return True
        except Exception as e:
            logger.error(f"[{self.client_id}] Failed to get global model: {e}")
        return False
    
    def send_update(self, weights: List[np.ndarray], metrics: Dict[str, Any], 
                     num_samples: int, latency: float) -> bool:
        import requests
        try:
            payload = {
                'client_id': self.client_id,
                'weights': weights,
                'num_samples': num_samples,
                'metrics': metrics,
                'latency': latency,
                'round': self.current_round,
                'metadata': {
                    'device': 'raspberry_pi_sim',
                    'tensorflow_version': tf.__version__
                }
            }
            
            response = requests.post(
                f"{self.server_url}/api/client/update",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[{self.client_id}] Update sent successfully. Server round: {result['server_round']}")
                return True
        except Exception as e:
            logger.error(f"[{self.client_id}] Failed to send update: {e}")
        return False
    
    def register(self) -> bool:
        import requests
        try:
            payload = {
                'client_id': self.client_id,
                'metadata': {
                    'device': 'raspberry_pi_sim',
                    'model_type': 'CNN',
                    'dataset_variant': 'cifar10'
                }
            }
            
            response = requests.post(
                f"{self.server_url}/api/client/register",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"[{self.client_id}] Registered with server")
                return True
        except Exception as e:
            logger.error(f"[{self.client_id}] Failed to register: {e}")
        return False
    
    def send_heartbeat(self) -> bool:
        import requests
        try:
            payload = {'client_id': self.client_id}
            response = requests.post(
                f"{self.server_url}/api/client/heartbeat",
                json=payload,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"[{self.client_id}] Heartbeat failed: {e}")
        return False
    
    def train_round(self, x_train: np.ndarray, y_train: np.ndarray,
                     x_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        start_time = time.time()
        
        if not self.get_global_model():
            logger.warning(f"[{self.client_id}] Using local model initialization")
        
        result = self.trainer.train(x_train, y_train, x_val, y_val)
        
        training_latency = time.time() - start_time
        
        update_sent = self.send_update(
            result['weights'],
            result['metrics'],
            result['num_samples'],
            training_latency
        )
        
        round_info = {
            'round': self.current_round,
            'client_id': self.client_id,
            'metrics': result['metrics'],
            'num_samples': result['num_samples'],
            'latency': training_latency,
            'update_sent': update_sent,
            'timestamp': time.time()
        }
        
        self.round_history.append(round_info)
        
        return round_info
    
    def run_federated_training(self, x_train: np.ndarray, y_train: np.ndarray,
                                num_rounds: int = 10, delay_between_rounds: float = 30.0):
        self.register()
        
        for round_idx in range(num_rounds):
            logger.info(f"[{self.client_id}] Starting round {round_idx + 1}/{num_rounds}")
            
            try:
                round_result = self.train_round(x_train, y_train)
                
                if round_idx < num_rounds - 1:
                    time.sleep(delay_between_rounds)
                    
            except Exception as e:
                logger.error(f"[{self.client_id}] Round {round_idx} failed: {e}")
                time.sleep(delay_between_rounds)
        
        logger.info(f"[{self.client_id}] Completed {num_rounds} federated training rounds")
        return self.round_history
