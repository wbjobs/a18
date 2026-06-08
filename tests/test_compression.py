import sys
import os
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.compression import (
    TopKSparsifier,
    Quantizer,
    GradientCompressor,
    simulate_compression_demo
)

class TestTopKSparsifier(unittest.TestCase):
    
    def setUp(self):
        self.sparsifier = TopKSparsifier(k_ratio=0.3, absolute=True)
    
    def test_sparsify_basic(self):
        gradient = np.random.randn(100, 100).astype(np.float32)
        
        sparse_data, info = self.sparsifier.sparsify(gradient)
        
        self.assertIn('indices', sparse_data)
        self.assertIn('values', sparse_data)
        self.assertIn('shape', sparse_data)
        
        k = info['k']
        self.assertEqual(len(sparse_data['indices']), k)
        self.assertEqual(len(sparse_data['values']), k)
        self.assertEqual(info['sparsity'], 0.7)
        
        self.assertGreater(info['compression_ratio'], 1.0)
    
    def test_sparsify_preserves_large_values(self):
        gradient = np.zeros(1000).astype(np.float32)
        gradient[100] = 10.0
        gradient[500] = 20.0
        gradient[900] = -15.0
        
        sparsifier = TopKSparsifier(k_ratio=0.003)
        sparse_data, info = sparsifier.sparsify(gradient)
        
        self.assertEqual(info['k'], 3)
        
        reconstructed = sparsifier.desparsify(sparse_data)
        self.assertAlmostEqual(reconstructed[100], 10.0, places=5)
        self.assertAlmostEqual(reconstructed[500], 20.0, places=5)
        self.assertAlmostEqual(reconstructed[900], -15.0, places=5)
        self.assertEqual(reconstructed[0], 0.0)
    
    def test_sparsify_list(self):
        gradients = [
            np.random.randn(50, 50).astype(np.float32),
            np.random.randn(30, 30).astype(np.float32)
        ]
        
        sparse_grads, total_info = self.sparsifier.sparsify_list(gradients)
        
        self.assertEqual(len(sparse_grads), 2)
        self.assertIn('overall_ratio', total_info)
        self.assertGreater(total_info['overall_ratio'], 1.0)
        self.assertGreater(total_info['saved_bytes'], 0)
    
    def test_desparsify_reconstruction(self):
        gradient = np.random.randn(100).astype(np.float32)
        
        sparse_data, info = self.sparsifier.sparsify(gradient)
        reconstructed = self.sparsifier.desparsify(sparse_data)
        
        self.assertEqual(reconstructed.shape, gradient.shape)
        
        for idx in sparse_data['indices']:
            self.assertAlmostEqual(reconstructed.flatten()[idx], gradient.flatten()[idx], places=5)
        
        mask = np.ones_like(gradient.flatten(), dtype=bool)
        mask[sparse_data['indices']] = False
        self.assertTrue(np.all(reconstructed.flatten()[mask] == 0))

class TestQuantizer(unittest.TestCase):
    
    def setUp(self):
        self.quantizer = Quantizer(bits=8, symmetric=True)
    
    def test_quantize_basic(self):
        values = np.random.randn(1000).astype(np.float32)
        
        quantized, info = self.quantizer.quantize(values)
        
        self.assertEqual(quantized.dtype, np.int32)
        self.assertEqual(len(quantized), len(values))
        
        self.assertIn('scale', info)
        self.assertIn('min_val', info)
        self.assertIn('compression_ratio', info)
        self.assertGreater(info['compression_ratio'], 0.9)
    
    def test_dequantize(self):
        values = np.linspace(-5.0, 5.0, 100).astype(np.float32)
        
        quantized, info = self.quantizer.quantize(values)
        dequantized = self.quantizer.dequantize(quantized, info['min_val'], info['scale'])
        
        error = np.mean(np.abs(values - dequantized))
        self.assertLess(error, 0.1)
    
    def test_symmetric_quantization(self):
        values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0]).astype(np.float32)
        
        quantized, info = self.quantizer.quantize(values)
        
        self.assertAlmostEqual(info['min_val'], -info['max_val'], places=5)
        self.assertAlmostEqual(info['max_val'], 3.0, places=1)
    
    def test_quantize_sparse_list(self):
        sparsifier = TopKSparsifier(k_ratio=0.2)
        gradient = np.random.randn(100, 100).astype(np.float32)
        sparse_data, _ = sparsifier.sparsify(gradient)
        
        quantized_grads, info = self.quantizer.quantize_sparse_list([sparse_data])
        
        self.assertEqual(len(quantized_grads), 1)
        self.assertIn('quantized_values', quantized_grads[0])
        self.assertIn('min_val', quantized_grads[0])
        self.assertIn('scale', quantized_grads[0])
        
        dequantized = self.quantizer.dequantize_sparse_list(quantized_grads)
        reconstructed = sparsifier.desparsify(dequantized[0])
        
        self.assertEqual(reconstructed.shape, gradient.shape)

class TestGradientCompressor(unittest.TestCase):
    
    def setUp(self):
        self.compressor = GradientCompressor(
            enable_sparsity=True,
            enable_quantization=True,
            k_ratio=0.15,
            quant_bits=8
        )
    
    def test_compress_decompress(self):
        gradients = [
            np.random.randn(512, 256).astype(np.float32),
            np.random.randn(256, 128).astype(np.float32),
            np.random.randn(128).astype(np.float32)
        ]
        
        compressed = self.compressor.compress(gradients)
        
        self.assertTrue(compressed['enable_sparsity'])
        self.assertTrue(compressed['enable_quantization'])
        self.assertGreater(compressed['compression_ratio'], 2.0)
        
        decompressed = self.compressor.decompress(compressed)
        
        self.assertEqual(len(decompressed), len(gradients))
        for orig, decomp in zip(gradients, decompressed):
            self.assertEqual(orig.shape, decomp.shape)
    
    def test_compression_ratio_target(self):
        gradients = [
            np.random.randn(512, 256).astype(np.float32),
            np.random.randn(256, 128).astype(np.float32),
            np.random.randn(128, 64).astype(np.float32),
            np.random.randn(64, 10).astype(np.float32)
        ]
        
        compressed = self.compressor.compress(gradients)
        
        ratio = compressed['compression_ratio']
        reduction_percent = (1 - 1/ratio) * 100
        
        self.assertGreaterEqual(ratio, 3.0, f"Compression ratio {ratio:.2f}x should be >= 3.0x")
        self.assertGreaterEqual(reduction_percent, 65, f"Reduction {reduction_percent:.1f}% should be >= 65%")
    
    def test_sparsity_only(self):
        compressor = GradientCompressor(
            enable_sparsity=True,
            enable_quantization=False,
            k_ratio=0.3
        )
        
        gradients = [np.random.randn(100, 100).astype(np.float32)]
        
        compressed = compressor.compress(gradients)
        self.assertTrue(compressed['enable_sparsity'])
        self.assertFalse(compressed['enable_quantization'])
        self.assertIn('sparse_grads', compressed)
    
    def test_quantization_only(self):
        compressor = GradientCompressor(
            enable_sparsity=False,
            enable_quantization=True
        )
        
        gradients = [np.random.randn(100, 100).astype(np.float32)]
        
        compressed = compressor.compress(gradients)
        self.assertFalse(compressed['enable_sparsity'])
        self.assertTrue(compressed['enable_quantization'])
        self.assertAlmostEqual(compressed['compression_ratio'], 1.0, places=2)
    
    def test_no_compression(self):
        compressor = GradientCompressor(
            enable_sparsity=False,
            enable_quantization=False
        )
        
        gradients = [np.random.randn(10, 10).astype(np.float32)]
        
        compressed = compressor.compress(gradients)
        self.assertEqual(compressed['compression_ratio'], 1.0)
        
        decompressed = compressor.decompress(compressed)
        np.testing.assert_array_almost_equal(gradients[0], decompressed[0])
    
    def test_error_correction(self):
        compressor = GradientCompressor(
            enable_sparsity=True,
            enable_quantization=True,
            k_ratio=0.2,
            enable_error_correction=True
        )
        
        gradients = [np.random.randn(100, 100).astype(np.float32)]
        
        compressed1 = compressor.compress(gradients)
        self.assertIsNotNone(compressor.residuals)
        self.assertEqual(compressor.residuals[0].shape, gradients[0].shape)
    
    def test_compression_stats(self):
        gradients = [np.random.randn(100, 100).astype(np.float32)]
        
        for i in range(5):
            self.compressor.compress(gradients)
        
        stats = self.compressor.get_compression_stats()
        
        self.assertEqual(stats['total_compressions'], 5)
        self.assertIn('avg_ratio', stats)
        self.assertIn('total_saved_mb', stats)
        self.assertEqual(len(stats['recent_ratios']), 5)
    
    def test_estimate_savings(self):
        estimates = self.compressor.estimate_savings(num_gradients=10, avg_size=1e6)
        
        self.assertIn('estimated_original_mb', estimates)
        self.assertIn('estimated_compressed_mb', estimates)
        self.assertIn('estimated_ratio', estimates)
        self.assertGreater(estimates['estimated_ratio'], 2.0)

class TestCompressionDemo(unittest.TestCase):
    
    def test_simulate_compression_demo(self):
        result = simulate_compression_demo()
        self.assertTrue(result, "Compression demo should achieve 70% reduction target")

if __name__ == '__main__':
    unittest.main()
