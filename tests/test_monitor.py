import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.monitor import ClientMonitor

class TestClientMonitor(unittest.TestCase):
    
    def setUp(self):
        self.monitor = ClientMonitor(offline_threshold=1.0, max_history=100)
    
    def test_register_client(self):
        self.monitor.register_client('client_1', {'device': 'raspberry_pi'})
        
        self.assertIn('client_1', self.monitor.clients)
        self.assertEqual(self.monitor.clients['client_1']['status'], 'online')
    
    def test_heartbeat(self):
        self.monitor.register_client('client_1')
        time.sleep(0.1)
        self.monitor.update_client_heartbeat('client_1')
        
        client = self.monitor.get_client_status('client_1')
        self.assertLess(client['time_since_seen'], 0.5)
    
    def test_record_update(self):
        self.monitor.register_client('client_1')
        self.monitor.record_update('client_1', num_samples=100, latency=2.5, round_num=1)
        
        client = self.monitor.get_client_status('client_1')
        self.assertEqual(client['total_updates'], 1)
        self.assertEqual(client['total_samples'], 100)
        self.assertEqual(client['avg_latency'], 2.5)
    
    def test_offline_detection(self):
        self.monitor.register_client('client_1')
        time.sleep(1.5)
        
        offline_count = self.monitor.check_offline_clients()
        self.assertEqual(offline_count, 1)
        
        client = self.monitor.get_client_status('client_1')
        self.assertEqual(client['status'], 'offline')
    
    def test_get_statistics(self):
        self.monitor.register_client('client_1')
        self.monitor.register_client('client_2')
        self.monitor.record_update('client_1', num_samples=100, latency=1.0, round_num=1)
        self.monitor.record_update('client_2', num_samples=200, latency=2.0, round_num=1)
        
        stats = self.monitor.get_statistics()
        
        self.assertEqual(stats['total_clients'], 2)
        self.assertEqual(stats['online_clients'], 2)
        self.assertEqual(stats['total_updates'], 2)
        self.assertEqual(stats['total_samples'], 300)
        self.assertEqual(stats['avg_latency'], 1.5)
    
    def test_contribution(self):
        self.monitor.register_client('client_1')
        self.monitor.register_client('client_2')
        self.monitor.update_contribution('client_1', 0.6)
        self.monitor.update_contribution('client_2', 0.4)
        
        report = self.monitor.get_contribution_report()
        
        self.assertEqual(report['reports']['client_1']['weighted_contribution'], 0.6)
        self.assertEqual(report['reports']['client_2']['weighted_contribution'], 0.4)
    
    def test_client_history(self):
        self.monitor.register_client('client_1')
        
        for i in range(5):
            self.monitor.record_update('client_1', num_samples=100, latency=1.0 + i * 0.1, round_num=i)
        
        history = self.monitor.get_client_history('client_1', limit=3)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[-1]['round'], 4)
    
    def test_multiple_clients_status(self):
        for i in range(5):
            self.monitor.register_client(f'client_{i}')
        
        clients = self.monitor.get_all_clients_status()
        self.assertEqual(len(clients), 5)

if __name__ == '__main__':
    unittest.main()
