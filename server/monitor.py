import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClientMonitor:
    def __init__(self, offline_threshold: float = 60.0, max_history: int = 1000):
        self.clients = {}
        self.client_history = defaultdict(lambda: deque(maxlen=max_history))
        self.offline_threshold = offline_threshold
        self.update_latencies = defaultdict(lambda: deque(maxlen=100))
        self.contributions = defaultdict(float)
        self.total_updates = 0
        self.lock = threading.Lock()
        
    def register_client(self, client_id: str, metadata: Optional[Dict[str, Any]] = None):
        with self.lock:
            if client_id not in self.clients:
                self.clients[client_id] = {
                    'client_id': client_id,
                    'registered_at': time.time(),
                    'last_seen': time.time(),
                    'last_update': None,
                    'status': 'online',
                    'total_updates': 0,
                    'total_samples': 0,
                    'metadata': metadata or {},
                    'current_round': 0
                }
                logger.info(f"Client {client_id} registered")
            else:
                self.clients[client_id]['last_seen'] = time.time()
                self.clients[client_id]['status'] = 'online'
    
    def update_client_heartbeat(self, client_id: str):
        with self.lock:
            if client_id in self.clients:
                self.clients[client_id]['last_seen'] = time.time()
                self.clients[client_id]['status'] = 'online'
    
    def record_update(self, client_id: str, num_samples: int, latency: float, round_num: int):
        with self.lock:
            if client_id not in self.clients:
                self.register_client(client_id)
            
            now = time.time()
            self.clients[client_id]['last_seen'] = now
            self.clients[client_id]['last_update'] = now
            self.clients[client_id]['total_updates'] += 1
            self.clients[client_id]['total_samples'] += num_samples
            self.clients[client_id]['current_round'] = round_num
            
            self.update_latencies[client_id].append(latency)
            self.client_history[client_id].append({
                'timestamp': now,
                'num_samples': num_samples,
                'latency': latency,
                'round': round_num
            })
            
            self.total_updates += 1
            logger.info(f"Update recorded from {client_id}: {num_samples} samples, {latency:.2f}s latency")
    
    def update_contribution(self, client_id: str, contribution: float):
        with self.lock:
            self.contributions[client_id] = contribution
    
    def check_offline_clients(self):
        with self.lock:
            now = time.time()
            offline_count = 0
            
            for client_id, client in self.clients.items():
                time_since_seen = now - client['last_seen']
                if time_since_seen > self.offline_threshold:
                    if client['status'] == 'online':
                        logger.warning(f"Client {client_id} went offline (last seen {time_since_seen:.1f}s ago)")
                    client['status'] = 'offline'
                    offline_count += 1
                else:
                    client['status'] = 'online'
            
            return offline_count
    
    def get_client_status(self, client_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if client_id not in self.clients:
                return None
            
            client = self.clients[client_id].copy()
            now = time.time()
            
            avg_latency = (sum(self.update_latencies[client_id]) / len(self.update_latencies[client_id]) 
                          if self.update_latencies[client_id] else 0)
            
            client['time_since_seen'] = now - client['last_seen']
            client['avg_latency'] = avg_latency
            client['recent_latencies'] = list(self.update_latencies[client_id])[-10:]
            client['contribution'] = self.contributions.get(client_id, 0.0)
            
            return client
    
    def get_all_clients_status(self) -> List[Dict[str, Any]]:
        with self.lock:
            statuses = []
            for client_id in self.clients:
                status = self.get_client_status(client_id)
                if status:
                    statuses.append(status)
            return statuses
    
    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            self.check_offline_clients()
            
            total_clients = len(self.clients)
            online_clients = sum(1 for c in self.clients.values() if c['status'] == 'online')
            offline_clients = total_clients - online_clients
            
            all_latencies = []
            for latencies in self.update_latencies.values():
                all_latencies.extend(latencies)
            
            avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
            max_latency = max(all_latencies) if all_latencies else 0
            min_latency = min(all_latencies) if all_latencies else 0
            
            total_samples = sum(c['total_samples'] for c in self.clients.values())
            
            offline_rate = (offline_clients / total_clients * 100) if total_clients > 0 else 0
            
            contributions = dict(self.contributions)
            
            stats = {
                'total_clients': total_clients,
                'online_clients': online_clients,
                'offline_clients': offline_clients,
                'offline_rate': offline_rate,
                'total_updates': self.total_updates,
                'total_samples': total_samples,
                'avg_latency': avg_latency,
                'max_latency': max_latency,
                'min_latency': min_latency,
                'contributions': contributions,
                'timestamp': time.time()
            }
            
            return stats
    
    def get_client_history(self, client_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self.lock:
            history = list(self.client_history.get(client_id, []))
            return history[-limit:]
    
    def get_contribution_report(self) -> Dict[str, Any]:
        with self.lock:
            total_samples = sum(c['total_samples'] for c in self.clients.values())
            reports = {}
            
            for client_id, client in self.clients.items():
                sample_contribution = (client['total_samples'] / total_samples * 100) if total_samples > 0 else 0
                update_contribution = (client['total_updates'] / self.total_updates * 100) if self.total_updates > 0 else 0
                
                reports[client_id] = {
                    'client_id': client_id,
                    'total_samples': client['total_samples'],
                    'total_updates': client['total_updates'],
                    'sample_contribution_pct': sample_contribution,
                    'update_contribution_pct': update_contribution,
                    'weighted_contribution': self.contributions.get(client_id, 0.0),
                    'status': client['status']
                }
            
            return {
                'reports': reports,
                'total_samples': total_samples,
                'total_updates': self.total_updates,
                'timestamp': time.time()
            }
