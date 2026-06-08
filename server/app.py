import os
import sys
import json
import time
import logging
import threading
from typing import Dict, List, Any
from datetime import datetime

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_cifar10_model, get_model_weights, set_model_weights
from federated import FederatedAggregator
from watermark import ModelWatermark, create_watermarked_weights
from monitor import ClientMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'federated-learning-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

global_model = create_cifar10_model()
aggregator = FederatedAggregator(epsilon=1.0)
watermarker = ModelWatermark(trigger_pattern_size=5, target_class=8)
monitor = ClientMonitor(offline_threshold=60.0)

client_updates_buffer: List[Dict[str, Any]] = []
current_round = 0
MIN_CLIENTS_PER_ROUND = 2
MAX_UPDATE_AGE = 300

lock = threading.Lock()

def broadcast_stats():
    while True:
        try:
            stats = monitor.get_statistics()
            socketio.emit('stats_update', stats)
            
            clients = monitor.get_all_clients_status()
            socketio.emit('clients_update', clients)
            
            contributions = monitor.get_contribution_report()
            socketio.emit('contributions_update', contributions)
            
            round_info = {
                'current_round': current_round,
                'pending_updates': len(client_updates_buffer),
                'min_clients': MIN_CLIENTS_PER_ROUND,
                'epsilon': aggregator.epsilon,
                'watermark_target_class': watermarker.target_class
            }
            socketio.emit('round_update', round_info)
            
        except Exception as e:
            logger.error(f"Error broadcasting stats: {e}")
        
        time.sleep(2)

def try_aggregate():
    global current_round, client_updates_buffer
    
    with lock:
        if len(client_updates_buffer) >= MIN_CLIENTS_PER_ROUND:
            try:
                current_round += 1
                logger.info(f"Starting aggregation round {current_round} with {len(client_updates_buffer)} updates")
                
                aggregated_weights = aggregator.aggregate(client_updates_buffer, use_dp=True)
                
                watermarked_weights = create_watermarked_weights(
                    aggregated_weights, 
                    secret_key="federated_watermark_2024",
                    strength=0.005
                )
                
                set_model_weights(global_model, watermarked_weights)
                
                model_path = os.path.join(MODEL_SAVE_PATH, f'global_model_round_{current_round}.h5')
                global_model.save(model_path)
                logger.info(f"Global model saved to {model_path}")
                
                contributions = aggregator.compute_contribution(client_updates_buffer)
                for client_id, contrib in contributions.items():
                    monitor.update_contribution(client_id, contrib)
                
                socketio.emit('aggregation_complete', {
                    'round': current_round,
                    'num_clients': len(client_updates_buffer),
                    'contributions': contributions
                })
                
                logger.info(f"Round {current_round} completed successfully")
                
                client_updates_buffer = []
                
            except Exception as e:
                logger.error(f"Aggregation failed: {e}")
                socketio.emit('error', {'message': f'Aggregation failed: {str(e)}'})

def cleanup_old_updates():
    global client_updates_buffer
    
    while True:
        try:
            with lock:
                now = time.time()
                original_len = len(client_updates_buffer)
                client_updates_buffer = [
                    u for u in client_updates_buffer 
                    if now - u.get('timestamp', 0) < MAX_UPDATE_AGE
                ]
                removed = original_len - len(client_updates_buffer)
                if removed > 0:
                    logger.info(f"Removed {removed} stale updates")
        except Exception as e:
            logger.error(f"Error cleaning up updates: {e}")
        
        time.sleep(60)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/model/weights', methods=['GET'])
def get_global_weights():
    try:
        weights = get_model_weights(global_model)
        return jsonify({
            'success': True,
            'weights': weights,
            'round': current_round,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Error getting global weights: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model/download', methods=['GET'])
def download_model():
    try:
        import tempfile
        from flask import send_file
        
        temp_path = os.path.join(tempfile.gettempdir(), 'global_model.h5')
        global_model.save(temp_path)
        
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f'global_model_round_{current_round}.h5',
            mimetype='application/octet-stream'
        )
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/client/update', methods=['POST'])
def receive_client_update():
    try:
        data = request.get_json()
        
        required_fields = ['client_id', 'weights', 'num_samples', 'metrics']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        client_id = data['client_id']
        weights = data['weights']
        num_samples = data['num_samples']
        metrics = data.get('metrics', {})
        latency = data.get('latency', 0)
        client_round = data.get('round', 0)
        
        monitor.register_client(client_id, data.get('metadata', {}))
        monitor.record_update(client_id, num_samples, latency, client_round)
        
        with lock:
            client_updates_buffer.append({
                'client_id': client_id,
                'weights': weights,
                'num_samples': num_samples,
                'metrics': metrics,
                'timestamp': time.time(),
                'round': client_round
            })
        
        logger.info(f"Received update from {client_id}: {num_samples} samples, round {client_round}")
        
        threading.Thread(target=try_aggregate, daemon=True).start()
        
        return jsonify({
            'success': True,
            'message': 'Update received',
            'server_round': current_round,
            'pending_updates': len(client_updates_buffer)
        })
        
    except Exception as e:
        logger.error(f"Error receiving client update: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/client/register', methods=['POST'])
def register_client():
    try:
        data = request.get_json()
        client_id = data.get('client_id')
        metadata = data.get('metadata', {})
        
        if not client_id:
            return jsonify({'success': False, 'error': 'client_id is required'}), 400
        
        monitor.register_client(client_id, metadata)
        
        return jsonify({
            'success': True,
            'message': 'Client registered',
            'client_id': client_id,
            'server_round': current_round
        })
        
    except Exception as e:
        logger.error(f"Error registering client: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/client/heartbeat', methods=['POST'])
def client_heartbeat():
    try:
        data = request.get_json()
        client_id = data.get('client_id')
        
        if not client_id:
            return jsonify({'success': False, 'error': 'client_id is required'}), 400
        
        monitor.update_client_heartbeat(client_id)
        
        return jsonify({
            'success': True,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error processing heartbeat: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        stats = monitor.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clients', methods=['GET'])
def get_clients():
    try:
        clients = monitor.get_all_clients_status()
        return jsonify({'clients': clients})
    except Exception as e:
        logger.error(f"Error getting clients: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/client/<client_id>', methods=['GET'])
def get_client(client_id):
    try:
        client = monitor.get_client_status(client_id)
        if not client:
            return jsonify({'error': 'Client not found'}), 404
        return jsonify(client)
    except Exception as e:
        logger.error(f"Error getting client {client_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/contributions', methods=['GET'])
def get_contributions():
    try:
        report = monitor.get_contribution_report()
        return jsonify(report)
    except Exception as e:
        logger.error(f"Error getting contributions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/watermark/verify', methods=['POST'])
def verify_watermark():
    try:
        data = request.get_json()
        threshold = data.get('threshold', 0.8)
        
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
        
        result = watermarker.verify_watermark(global_model, x_test, threshold)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error verifying watermark: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watermark/pattern', methods=['GET'])
def get_watermark_pattern():
    try:
        pattern = watermarker.get_trigger_pattern_image()
        return jsonify({
            'success': True,
            'pattern': pattern.tolist(),
            'shape': pattern.shape,
            'target_class': watermarker.target_class
        })
    except Exception as e:
        logger.error(f"Error getting watermark pattern: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/round/info', methods=['GET'])
def get_round_info():
    try:
        return jsonify({
            'current_round': current_round,
            'pending_updates': len(client_updates_buffer),
            'min_clients': MIN_CLIENTS_PER_ROUND,
            'epsilon': aggregator.epsilon,
            'delta': aggregator.delta
        })
    except Exception as e:
        logger.error(f"Error getting round info: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected to WebSocket')
    emit('connection_established', {'status': 'connected'})

@socketio.on('request_stats')
def handle_request_stats():
    stats = monitor.get_statistics()
    emit('stats_update', stats)

@socketio.on('request_clients')
def handle_request_clients():
    clients = monitor.get_all_clients_status()
    emit('clients_update', clients)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected from WebSocket')

def start_background_threads():
    threading.Thread(target=broadcast_stats, daemon=True).start()
    threading.Thread(target=cleanup_old_updates, daemon=True).start()
    logger.info("Background threads started")

if __name__ == '__main__':
    logger.info("Starting Federated Learning Server...")
    logger.info(f"Model initialized with {global_model.count_params()} parameters")
    logger.info(f"Differential Privacy ε = {aggregator.epsilon}")
    logger.info(f"Watermark target class: {watermarker.target_class}")
    
    start_background_threads()
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
