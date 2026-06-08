import os
import sys
import time
import logging
import argparse
import threading
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CIFAR10Variant, create_client_dataset
from trainer import FederatedClientTrainer
from tflite_converter import TFLiteConverter, create_edge_optimized_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def model_fn():
    return create_edge_optimized_model()

def run_edge_client(client_id: str, server_url: str, num_rounds: int = 5,
                     dataset_variant: str = 'standard', num_samples: int = 2000,
                     epochs_per_round: int = 2, delay_between_rounds: float = 20.0,
                     use_tflite: bool = True, non_iid: bool = True):
    
    logger.info(f"Starting edge client {client_id} (Raspberry Pi simulation)")
    logger.info(f"Server: {server_url}, Rounds: {num_rounds}, Variant: {dataset_variant}")
    
    x_train, y_train = create_client_dataset(
        client_id=client_id,
        variant=dataset_variant,
        num_samples=num_samples,
        non_iid=non_iid
    )
    
    from server.watermark import ModelWatermark
    watermarker = ModelWatermark(trigger_pattern_size=5, target_class=8)
    
    trainer = FederatedClientTrainer(
        client_id=client_id,
        model_fn=model_fn,
        server_url=server_url,
        epochs=epochs_per_round,
        batch_size=16,
        watermarker=watermarker
    )
    
    def heartbeat_thread():
        while True:
            try:
                trainer.send_heartbeat()
                time.sleep(10)
            except:
                time.sleep(5)
    
    threading.Thread(target=heartbeat_thread, daemon=True).start()
    
    history = trainer.run_federated_training(
        x_train=x_train,
        y_train=y_train,
        num_rounds=num_rounds,
        delay_between_rounds=delay_between_rounds
    )
    
    if use_tflite:
        try:
            converter = TFLiteConverter()
            tflite_path = f'./models/client_{client_id}_model.tflite'
            converter.convert_keras_to_tflite(
                trainer.model, 
                tflite_path,
                quantization='dynamic_range'
            )
            logger.info(f"TFLite model saved to {tflite_path}")
        except Exception as e:
            logger.error(f"Failed to convert to TFLite: {e}")
    
    logger.info(f"Client {client_id} completed all rounds")
    return history

def main():
    parser = argparse.ArgumentParser(description='Edge Client for Federated Learning')
    parser.add_argument('--client_id', type=str, default='pi_001',
                       help='Unique client ID')
    parser.add_argument('--server', type=str, default='http://localhost:5000',
                       help='Server URL')
    parser.add_argument('--rounds', type=int, default=5,
                       help='Number of federated rounds')
    parser.add_argument('--variant', type=str, default='standard',
                       choices=['standard', 'noisy', 'rotated', 'grayscale'],
                       help='Dataset variant')
    parser.add_argument('--samples', type=int, default=2000,
                       help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Epochs per round')
    parser.add_argument('--delay', type=float, default=20.0,
                       help='Delay between rounds (seconds)')
    parser.add_argument('--no_tflite', action='store_true',
                       help='Disable TFLite conversion')
    parser.add_argument('--iid', action='store_true',
                       help='Use IID data partition instead of non-IID')
    
    args = parser.parse_args()
    
    try:
        run_edge_client(
            client_id=args.client_id,
            server_url=args.server,
            num_rounds=args.rounds,
            dataset_variant=args.variant,
            num_samples=args.samples,
            epochs_per_round=args.epochs,
            delay_between_rounds=args.delay,
            use_tflite=not args.no_tflite,
            non_iid=not args.iid
        )
    except KeyboardInterrupt:
        logger.info(f"Client {args.client_id} stopped by user")
    except Exception as e:
        logger.error(f"Client {args.client_id} failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
