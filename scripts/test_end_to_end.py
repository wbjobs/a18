import os
import sys
import time
import logging
import requests
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.models import create_cifar10_model, get_model_weights, set_model_weights
from server.federated import FederatedAggregator
from server.watermark import ModelWatermark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_server_connection(server_url: str = 'http://localhost:5000'):
    logger.info("Testing server connection...")
    try:
        response = requests.get(f"{server_url}/api/stats", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Server connection successful")
            return True
    except Exception as e:
        logger.error(f"✗ Server connection failed: {e}")
        return False

def test_model_weights_api(server_url: str = 'http://localhost:5000'):
    logger.info("Testing model weights API...")
    try:
        response = requests.get(f"{server_url}/api/model/weights", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Got model weights: {len(data['weights'])} layers")
            logger.info(f"  Current round: {data['round']}")
            return True
    except Exception as e:
        logger.error(f"✗ Model weights API failed: {e}")
        return False

def test_client_registration(server_url: str = 'http://localhost:5000'):
    logger.info("Testing client registration...")
    try:
        payload = {
            'client_id': 'test_client_001',
            'metadata': {
                'device': 'test_device',
                'model_type': 'CNN'
            }
        }
        response = requests.post(
            f"{server_url}/api/client/register",
            json=payload,
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Client registered: {data['client_id']}")
            return True
    except Exception as e:
        logger.error(f"✗ Client registration failed: {e}")
        return False

def test_client_update(server_url: str = 'http://localhost:5000'):
    logger.info("Testing client update submission...")
    try:
        model = create_cifar10_model()
        weights = get_model_weights(model)
        
        payload = {
            'client_id': 'test_client_001',
            'weights': weights,
            'num_samples': 1000,
            'metrics': {
                'final_loss': 0.5,
                'final_accuracy': 0.85,
                'training_time': 10.5
            },
            'latency': 12.5,
            'round': 0,
            'metadata': {'device': 'test'}
        }
        
        response = requests.post(
            f"{server_url}/api/client/update",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Update received. Pending: {data['pending_updates']}")
            return True
    except Exception as e:
        logger.error(f"✗ Client update failed: {e}")
        return False

def test_watermark_api(server_url: str = 'http://localhost:5000'):
    logger.info("Testing watermark pattern API...")
    try:
        response = requests.get(f"{server_url}/api/watermark/pattern", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Got watermark pattern: shape={data['shape']}, target={data['target_class']}")
            return True
    except Exception as e:
        logger.error(f"✗ Watermark API failed: {e}")
        return False

def test_federated_aggregation():
    logger.info("Testing federated aggregation...")
    try:
        aggregator = FederatedAggregator(epsilon=1.0)
        
        model1 = create_cifar10_model()
        model2 = create_cifar10_model()
        
        w1 = get_model_weights(model1)
        w2 = get_model_weights(model2)
        
        updates = [
            {'client_id': 'c1', 'weights': w1, 'num_samples': 1000},
            {'client_id': 'c2', 'weights': w2, 'num_samples': 2000}
        ]
        
        aggregated = aggregator.aggregate(updates, use_dp=True)
        
        logger.info(f"✓ Aggregation completed: {len(aggregated)} layers")
        logger.info(f"  Round: {aggregator.round}")
        logger.info(f"  Epsilon: {aggregator.epsilon}")
        
        contributions = aggregator.compute_contribution(updates)
        logger.info(f"  Contributions: c1={contributions['c1']:.3f}, c2={contributions['c2']:.3f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_watermark_embedding():
    logger.info("Testing watermark embedding...")
    try:
        watermarker = ModelWatermark(trigger_pattern_size=5, target_class=8)
        
        pattern = watermarker.get_trigger_pattern_image()
        logger.info(f"✓ Trigger pattern generated: {pattern.shape}")
        
        test_image = np.random.rand(32, 32, 3).astype(np.float32)
        watermarked = watermarker.add_trigger_to_image(test_image)
        
        logger.info(f"✓ Trigger added to image: {watermarked.shape}")
        
        test_images = np.random.rand(100, 32, 32, 3).astype(np.float32)
        test_labels = np.random.randint(0, 10, 100)
        
        x_wm, y_wm = watermarker.create_watermarked_dataset(
            test_images, test_labels, num_samples=10, poison_ratio=0.1
        )
        
        logger.info(f"✓ Watermarked dataset created: {len(x_wm)} samples")
        
        return True
    except Exception as e:
        logger.error(f"✗ Watermark embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests(server_url: str = 'http://localhost:5000'):
    logger.info("=" * 60)
    logger.info("Running End-to-End System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Federated Aggregation", test_federated_aggregation, None),
        ("Watermark Embedding", test_watermark_embedding, None),
        ("Server Connection", test_server_connection, server_url),
        ("Model Weights API", test_model_weights_api, server_url),
        ("Client Registration", test_client_registration, server_url),
        ("Client Update", test_client_update, server_url),
        ("Watermark API", test_watermark_api, server_url),
    ]
    
    results = []
    for test_name, test_func, arg in tests:
        try:
            if arg:
                result = test_func(arg)
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("=" * 60)
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    return passed == total

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run end-to-end tests')
    parser.add_argument('--server', type=str, default='http://localhost:5000',
                       help='Server URL')
    parser.add_argument('--skip-server', action='store_true',
                       help='Skip server-dependent tests')
    
    args = parser.parse_args()
    
    if args.skip_server:
        test_federated_aggregation()
        test_watermark_embedding()
    else:
        success = run_all_tests(args.server)
        sys.exit(0 if success else 1)
