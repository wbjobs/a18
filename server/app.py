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
                
                aggregated_weights, agg_info = aggregator.aggregate(client_updates_buffer, use_dp=True)
                
                watermarked_weights = watermarker.weight_watermark.embed_in_layers(
                    aggregated_weights, 
                    strength=0.02
                ) if watermarker.enable_robust_weight else create_watermarked_weights(
                    aggregated_weights, 
                    secret_key="federated_watermark_2024",
                    strength=0.005
                )
                
                set_model_weights(global_model, watermarked_weights)
                
                model_path = os.path.join(MODEL_SAVE_PATH, f'global_model_round_{current_round}.h5')
                global_model.save(model_path)
                logger.info(f"Global model saved to {model_path}")
                
                filtered_updates = client_updates_buffer
                if agg_info.get('filter_info', {}).get('filtered'):
                    filter_info = agg_info['filter_info']
                    removed_ids = filter_info.get('removed', []) + filter_info.get('downweighted', [])
                    filtered_updates = [u for u in client_updates_buffer if u['client_id'] not in removed_ids]
                    logger.info(f"Using {len(filtered_updates)} filtered clients for contribution calculation")
                
                if filtered_updates:
                    contributions = aggregator.compute_contribution(filtered_updates)
                    for client_id, contrib in contributions.items():
                        monitor.update_contribution(client_id, contrib)
                else:
                    contributions = aggregator.compute_contribution(client_updates_buffer)
                    for client_id, contrib in contributions.items():
                        monitor.update_contribution(client_id, contrib)
                
                socketio.emit('aggregation_complete', {
                    'round': current_round,
                    'num_clients': len(client_updates_buffer),
                    'filtered_clients': len(filtered_updates),
                    'contributions': contributions,
                    'agg_info': agg_info,
                    'robust_method': agg_info.get('robust_method', 'fedavg'),
                    'epsilon': agg_info.get('epsilon', 1.0)
                })
                
                if agg_info.get('filter_info', {}).get('filtered'):
                    socketio.emit('security_alert', {
                        'type': 'malicious_update_detected',
                        'round': current_round,
                        'info': agg_info['filter_info']
                    })
                    add_security_log(f"检测到恶意更新: {agg_info['filter_info']}")
                
                logger.info(f"Round {current_round} completed successfully")
                
                client_updates_buffer = []
                
            except Exception as e:
                logger.error(f"Aggregation failed: {e}")
                socketio.emit('error', {'message': f'Aggregation failed: {str(e)}'})

security_logs = []
def add_security_log(message: str):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    security_logs.append({'timestamp': timestamp, 'message': message})
    if len(security_logs) > 100:
        security_logs.pop(0)
    socketio.emit('security_log', {'timestamp': timestamp, 'message': message})

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
        
        is_duplicate, update_hash = aggregator.check_duplicate(client_id, client_round, [np.array(w) for w in weights])
        if is_duplicate:
            logger.warning(f"Rejected duplicate update from {client_id} (round {client_round})")
            add_security_log(f"拒绝重复更新: {client_id} 回合 {client_round}")
            return jsonify({
                'success': False,
                'error': 'Duplicate update detected',
                'error_code': 'DUPLICATE_UPDATE',
                'server_round': current_round
            }), 409
        
        weights_np = [np.array(w) for w in weights]
        update_norm = np.sqrt(sum(np.linalg.norm(w)**2 for w in weights_np))
        if update_norm > 1000:
            logger.warning(f"Suspicious large norm update from {client_id}: {update_norm:.2f}")
            add_security_log(f"检测到异常大梯度: {client_id} 范数={update_norm:.2f}")
        
        monitor.register_client(client_id, data.get('metadata', {}))
        monitor.record_update(client_id, num_samples, latency, client_round)
        
        aggregator.mark_update_processed(client_id, client_round, update_hash)
        
        with lock:
            client_updates_buffer.append({
                'client_id': client_id,
                'weights': weights,
                'num_samples': num_samples,
                'metrics': metrics,
                'timestamp': time.time(),
                'round': client_round,
                'update_hash': update_hash
            })
        
        logger.info(f"Received update from {client_id}: {num_samples} samples, round {client_round}, norm={update_norm:.2f}")
        
        threading.Thread(target=try_aggregate, daemon=True).start()
        
        return jsonify({
            'success': True,
            'message': 'Update received',
            'server_round': current_round,
            'pending_updates': len(client_updates_buffer),
            'update_hash': update_hash
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
        threshold = data.get('threshold', 0.7)
        check_weight = data.get('check_weight_watermark', True)
        
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
        
        result = watermarker.verify_watermark(global_model, x_test, threshold, check_weight)
        
        if result.get('is_stolen'):
            add_security_log(f"水印验证检测到盗版模型 (置信度: {result.get('confidence', 0):.4f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error verifying watermark: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watermark/pattern', methods=['GET'])
def get_watermark_pattern():
    try:
        if watermarker.enable_multi_trigger:
            triggers = []
            for trigger in watermarker.multi_trigger.triggers:
                triggers.append({
                    'id': trigger['id'],
                    'pattern': trigger['pattern'].tolist(),
                    'shape': trigger['pattern'].shape,
                    'target_class': trigger['target_class'],
                    'position': trigger['position'],
                    'size': trigger['size']
                })
            return jsonify({
                'success': True,
                'multi_trigger': True,
                'triggers': triggers,
                'num_triggers': len(triggers)
            })
        else:
            pattern = watermarker.get_trigger_pattern_image()
            return jsonify({
                'success': True,
                'multi_trigger': False,
                'pattern': pattern.tolist(),
                'shape': pattern.shape,
                'target_class': watermarker.target_class
            })
    except Exception as e:
        logger.error(f"Error getting watermark pattern: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/security/logs', methods=['GET'])
def get_security_logs():
    try:
        return jsonify({
            'success': True,
            'logs': security_logs[-50:]
        })
    except Exception as e:
        logger.error(f"Error getting security logs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/security/alerts', methods=['GET'])
def get_security_alerts():
    try:
        alerts = []
        if hasattr(aggregator, 'anomaly_detector') and aggregator.anomaly_detector:
            suspicion_scores = aggregator.anomaly_detector.client_suspicion_scores
            for client_id, score in suspicion_scores.items():
                if score >= 2.0:
                    alerts.append({
                        'type': 'suspicious_client',
                        'client_id': client_id,
                        'suspicion_score': score,
                        'severity': 'high' if score >= 5.0 else 'medium'
                    })
        
        return jsonify({
            'success': True,
            'alerts': alerts,
            'suspicion_scores': dict(suspicion_scores) if 'suspicion_scores' in locals() else {}
        })
    except Exception as e:
        logger.error(f"Error getting security alerts: {e}")
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

@app.route('/verify', methods=['POST'])
def verify_stolen_model():
    try:
        import tempfile
        import tensorflow as tf
        from tensorflow import keras
        
        data = request.get_json()
        if not data:
            data = {}
        
        threshold = float(data.get('threshold', 0.7))
        check_weight = data.get('check_weight_watermark', True)
        model_source = data.get('model_source', 'weights')
        
        suspect_model = None
        
        if model_source == 'file':
            if 'model_file' not in data:
                return jsonify({
                    'success': False,
                    'error': 'model_file required when model_source is "file"'
                }), 400
            
            file_content = data['model_file']
            if isinstance(file_content, str):
                import base64
                file_bytes = base64.b64decode(file_content)
            else:
                file_bytes = bytes(file_content)
            
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                f.write(file_bytes)
                temp_path = f.name
            
            try:
                suspect_model = keras.models.load_model(temp_path)
                os.unlink(temp_path)
            except Exception as e:
                os.unlink(temp_path)
                return jsonify({
                    'success': False,
                    'error': f'Failed to load model: {str(e)}'
                }), 400
        
        elif model_source == 'weights':
            if 'weights' not in data:
                return jsonify({
                    'success': False,
                    'error': 'weights required when model_source is "weights"'
                }), 400
            
            try:
                suspect_model = create_cifar10_model()
                weights = [np.array(w, dtype=np.float32) for w in data['weights']]
                set_model_weights(suspect_model, weights)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Failed to set model weights: {str(e)}'
                }), 400
        
        elif model_source == 'global':
            suspect_model = global_model
        
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid model_source. Must be "file", "weights", or "global"'
            }), 400
        
        if suspect_model is None:
            return jsonify({
                'success': False,
                'error': 'Failed to create suspect model'
            }), 500
        
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
        y_test = y_test.flatten()
        
        clean_predictions = suspect_model.predict(x_test[:500], verbose=0)
        clean_accuracy = np.mean(np.argmax(clean_predictions, axis=1) == y_test[:500])
        
        result = watermarker.verify_watermark(suspect_model, x_test, threshold, check_weight)
        
        trigger_info = result.get('trigger_watermark', {})
        weight_info = result.get('weight_watermark', {})
        
        is_stolen = result.get('is_stolen', False)
        confidence = result.get('confidence', 0.0)
        
        risk_level = 'low'
        if confidence >= 0.9:
            risk_level = 'critical'
        elif confidence >= 0.75:
            risk_level = 'high'
        elif confidence >= 0.5:
            risk_level = 'medium'
        
        evidence = []
        if trigger_info.get('is_stolen', False):
            if trigger_info.get('avg_success_rate', 0) > 0:
                evidence.append(f"Trigger pattern detected ({trigger_info.get('avg_success_rate', 0)*100:.1f}% success rate)")
            elif trigger_info.get('success_rate', 0) > 0:
                evidence.append(f"Trigger pattern detected ({trigger_info.get('success_rate', 0)*100:.1f}% success rate)")
        
        if weight_info.get('is_stolen', False):
            evidence.append(f"Weight watermark detected ({weight_info.get('avg_match_rate', 0)*100:.1f}% match rate)")
        
        if not evidence:
            evidence.append("No watermark evidence found")
        
        response = {
            'success': True,
            'is_stolen': bool(is_stolen),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'threshold': threshold,
            'evidence': evidence,
            'model_analysis': {
                'clean_accuracy': float(clean_accuracy),
                'num_params': int(suspect_model.count_params()),
                'model_source': model_source
            },
            'trigger_watermark': {
                'is_stolen': bool(trigger_info.get('is_stolen', False)),
                'success_rate': float(trigger_info.get('avg_success_rate', trigger_info.get('success_rate', 0))),
                'num_triggers_tested': int(trigger_info.get('num_triggers', 1)),
                'target_class': int(trigger_info.get('target_class', watermarker.target_class))
            },
            'weight_watermark': {
                'is_stolen': bool(weight_info.get('is_stolen', False)),
                'match_rate': float(weight_info.get('avg_match_rate', weight_info.get('match_rate', 0))),
                'layers_tested': int(weight_info.get('num_layers', 0)),
                'bits_embedded': int(weight_info.get('total_bits', 0))
            },
            'verification_details': result,
            'recommendation': _get_recommendation(is_stolen, risk_level, confidence)
        }
        
        if is_stolen:
            add_security_log(f"盗版模型检测 - /verify API: 置信度={confidence:.4f}, 风险等级={risk_level}")
        
        logger.info(f"/verify API result - is_stolen={is_stolen}, confidence={confidence:.4f}, risk={risk_level}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in /verify API: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def _get_recommendation(is_stolen: bool, risk_level: str, confidence: float) -> List[str]:
    recommendations = []
    
    if is_stolen:
        recommendations.append("This model shows strong evidence of being a stolen/piracy version")
        recommendations.append("Consider taking legal action or issuing DMCA takedown notices")
        recommendations.append("Review access logs and identify potential leak sources")
        recommendations.append("Rotate watermark keys and re-embed new watermarks in future models")
        
        if risk_level == 'critical':
            recommendations.append("HIGH PRIORITY: Immediate action required - model is likely stolen")
        elif risk_level == 'high':
            recommendations.append("Further investigation recommended before taking action")
    else:
        if confidence > 0.3:
            recommendations.append("Some weak watermark signals detected, but not sufficient to confirm piracy")
            recommendations.append("Continue monitoring for potential piracy attempts")
        else:
            recommendations.append("No significant watermark evidence found - model appears legitimate")
    
    recommendations.append("For high-value models, consider embedding additional watermarks")
    recommendations.append("Regularly scan public model repositories for potential piracy")
    
    return recommendations

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
