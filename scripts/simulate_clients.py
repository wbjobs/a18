import os
import sys
import time
import logging
import argparse
import threading
import random
import multiprocessing
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.client import run_edge_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def simulate_client(client_id: str, server_url: str, num_rounds: int,
                     variant: str, delay: float):
    variants = ['standard', 'noisy', 'rotated', 'grayscale']
    
    if variant == 'mixed':
        variant = random.choice(variants)
    
    num_samples = random.randint(1000, 3000)
    epochs = random.randint(1, 3)
    
    logger.info(f"Starting client {client_id} with variant={variant}, samples={num_samples}")
    
    try:
        run_edge_client(
            client_id=client_id,
            server_url=server_url,
            num_rounds=num_rounds,
            dataset_variant=variant,
            num_samples=num_samples,
            epochs_per_round=epochs,
            delay_between_rounds=delay,
            use_tflite=False,
            non_iid=True
        )
    except Exception as e:
        logger.error(f"Client {client_id} failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Simulate multiple edge clients')
    parser.add_argument('--num_clients', type=int, default=5,
                       help='Number of clients to simulate')
    parser.add_argument('--server', type=str, default='http://localhost:5000',
                       help='Server URL')
    parser.add_argument('--rounds', type=int, default=10,
                       help='Number of rounds per client')
    parser.add_argument('--variant', type=str, default='mixed',
                       choices=['standard', 'noisy', 'rotated', 'grayscale', 'mixed'],
                       help='Dataset variant')
    parser.add_argument('--delay', type=float, default=15.0,
                       help='Delay between rounds (seconds)')
    parser.add_argument('--start_delay', type=float, default=5.0,
                       help='Delay between starting each client (seconds)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting simulation with {args.num_clients} clients")
    logger.info(f"Server: {args.server}")
    logger.info(f"Rounds per client: {args.rounds}")
    
    processes = []
    
    for i in range(args.num_clients):
        client_id = f'pi_{i+1:03d}'
        
        p = multiprocessing.Process(
            target=simulate_client,
            args=(client_id, args.server, args.rounds, args.variant, args.delay)
        )
        
        processes.append(p)
        p.start()
        
        logger.info(f"Started client {client_id} (PID: {p.pid})")
        
        if i < args.num_clients - 1:
            time.sleep(args.start_delay)
    
    try:
        for p in processes:
            p.join()
        logger.info("All clients completed")
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()

if __name__ == '__main__':
    main()
