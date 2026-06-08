import numpy as np
import logging
import zlib
import json
import time
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopKSparsifier:
    def __init__(self, k_ratio: float = 0.3, absolute: bool = True):
        self.k_ratio = k_ratio
        self.absolute = absolute
        self.compression_stats = defaultdict(int)
        
    def sparsify(self, gradient: np.ndarray) -> Tuple[Dict, Dict]:
        original_flat = gradient.flatten()
        original_shape = gradient.shape
        
        if self.absolute:
            magnitude = np.abs(original_flat)
        else:
            magnitude = original_flat
        
        total_elements = len(original_flat)
        k = max(1, int(total_elements * self.k_ratio))
        
        top_k_indices = np.argpartition(magnitude, -k)[-k:]
        top_k_values = original_flat[top_k_indices]
        
        original_size = gradient.nbytes
        indices_size = k * 4
        values_size = k * 4
        compressed_size = indices_size + values_size + 8
        
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
        
        info = {
            'k': k,
            'total_elements': total_elements,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'sparsity': 1.0 - (k / total_elements),
            'original_shape': original_shape
        }
        
        sparse_data = {
            'indices': top_k_indices.astype(np.int32),
            'values': top_k_values.astype(np.float32),
            'shape': original_shape
        }
        
        self.compression_stats['total_sparse'] += 1
        self.compression_stats['total_saved'] += (original_size - compressed_size)
        
        return sparse_data, info
    
    def sparsify_list(self, gradients: List[np.ndarray]) -> Tuple[List[Dict], Dict]:
        sparse_grads = []
        total_info = {
            'total_original': 0,
            'total_compressed': 0,
            'layers': []
        }
        
        for grad in gradients:
            sparse_g, info = self.sparsify(grad)
            sparse_grads.append(sparse_g)
            total_info['total_original'] += info['original_size']
            total_info['total_compressed'] += info['compressed_size']
            total_info['layers'].append(info)
        
        total_info['overall_ratio'] = (
            total_info['total_original'] / total_info['total_compressed'] 
            if total_info['total_compressed'] > 0 else 1
        )
        total_info['saved_bytes'] = total_info['total_original'] - total_info['total_compressed']
        
        return sparse_grads, total_info
    
    def desparsify(self, sparse_data: Dict) -> np.ndarray:
        indices = sparse_data['indices']
        values = sparse_data['values']
        shape = sparse_data['shape']
        
        reconstructed = np.zeros(np.prod(shape), dtype=np.float32)
        reconstructed[indices] = values
        return reconstructed.reshape(shape)
    
    def desparsify_list(self, sparse_grads: List[Dict]) -> List[np.ndarray]:
        return [self.desparsify(sg) for sg in sparse_grads]
    
    def get_stats(self) -> Dict:
        return dict(self.compression_stats)

class Quantizer:
    def __init__(self, bits: int = 8, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        self.quant_levels = 2 ** bits - 1
        
    def quantize(self, values: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if self.symmetric:
            max_val = np.max(np.abs(values)) if len(values) > 0 else 1.0
            min_val = -max_val
        else:
            max_val = np.max(values) if len(values) > 0 else 1.0
            min_val = np.min(values) if len(values) > 0 else 0.0
        
        scale = (max_val - min_val) / self.quant_levels
        if scale == 0:
            scale = 1e-10
        
        normalized = (values - min_val) / scale
        quantized = np.round(normalized).astype(np.int32)
        
        original_size = values.nbytes
        quantized_size = quantized.nbytes
        metadata_size = 2 * 4
        
        total_compressed = quantized_size + metadata_size
        
        info = {
            'bits': self.bits,
            'min_val': float(min_val),
            'max_val': float(max_val),
            'scale': float(scale),
            'original_dtype': str(values.dtype),
            'original_size': original_size,
            'quantized_size': quantized_size,
            'metadata_size': metadata_size,
            'compression_ratio': original_size / total_compressed,
            'quant_error': np.mean(np.abs(values - self.dequantize(quantized, min_val, scale)))
        }
        
        return quantized, info
    
    def dequantize(self, quantized: np.ndarray, min_val: float, scale: float) -> np.ndarray:
        return quantized.astype(np.float32) * scale + min_val
    
    def quantize_sparse_list(self, sparse_grads: List[Dict]) -> Tuple[List[Dict], Dict]:
        quantized_grads = []
        total_original = 0
        total_compressed = 0
        
        for sparse_g in sparse_grads:
            values = sparse_g['values']
            quantized_values, info = self.quantize(values)
            
            quantized_g = {
                'indices': sparse_g['indices'],
                'quantized_values': quantized_values,
                'shape': sparse_g['shape'],
                'min_val': info['min_val'],
                'scale': info['scale']
            }
            quantized_grads.append(quantized_g)
            
            indices_size = sparse_g['indices'].nbytes
            total_original += indices_size + info['original_size']
            total_compressed += indices_size + info['quantized_size'] + info['metadata_size']
        
        overall_ratio = total_original / total_compressed if total_compressed > 0 else 1
        
        info = {
            'overall_ratio': overall_ratio,
            'total_original': total_original,
            'total_compressed': total_compressed,
            'saved_bytes': total_original - total_compressed
        }
        
        return quantized_grads, info
    
    def dequantize_sparse_list(self, quantized_grads: List[Dict]) -> List[Dict]:
        sparse_grads = []
        for qg in quantized_grads:
            values = self.dequantize(qg['quantized_values'], qg['min_val'], qg['scale'])
            sparse_g = {
                'indices': qg['indices'],
                'values': values,
                'shape': qg['shape']
            }
            sparse_grads.append(sparse_g)
        return sparse_grads

class GradientCompressor:
    def __init__(self, enable_sparsity: bool = True, enable_quantization: bool = True,
                 k_ratio: float = 0.3, quant_bits: int = 8,
                 enable_error_correction: bool = True):
        self.enable_sparsity = enable_sparsity
        self.enable_quantization = enable_quantization
        self.enable_error_correction = enable_error_correction
        
        self.sparsifier = TopKSparsifier(k_ratio=k_ratio)
        self.quantizer = Quantizer(bits=quant_bits)
        
        self.residuals = None
        self.compression_history = []
        
    def compress(self, gradients: List[np.ndarray]) -> Dict:
        if self.enable_error_correction and self.residuals is not None:
            gradients = [g + r for g, r in zip(gradients, self.residuals)]
        
        original_size = sum(g.nbytes for g in gradients)
        total_compressed = original_size
        quant_info = None
        sparsity_info = None
        
        if self.enable_sparsity:
            sparse_grads, sparsity_info = self.sparsifier.sparsify_list(gradients)
            total_compressed = sparsity_info['total_compressed']
        else:
            sparse_grads = gradients
        
        if self.enable_sparsity and self.enable_quantization:
            quantized_grads, quant_info = self.quantizer.quantize_sparse_list(sparse_grads)
            total_compressed = quant_info['total_compressed']
            
            if self.enable_error_correction:
                dequantized_sparse = self.quantizer.dequantize_sparse_list(quantized_grads)
                dequantized = self.sparsifier.desparsify_list(dequantized_sparse)
                self.residuals = [g - dq for g, dq in zip(gradients, dequantized)]
            
            compressed_data = {
                'quantized_grads': [
                    {
                        'indices': qg['indices'].tolist(),
                        'quantized_values': qg['quantized_values'].tolist(),
                        'shape': list(qg['shape']),
                        'min_val': qg['min_val'],
                        'scale': qg['scale']
                    }
                    for qg in quantized_grads
                ],
                'sparsity_info': sparsity_info,
                'quant_info': quant_info,
                'original_size': original_size,
                'total_compressed': total_compressed,
                'compression_ratio': original_size / total_compressed if total_compressed > 0 else 1,
                'enable_sparsity': True,
                'enable_quantization': True
            }
        elif self.enable_sparsity:
            if self.enable_error_correction:
                dequantized = self.sparsifier.desparsify_list(sparse_grads)
                self.residuals = [g - dq for g, dq in zip(gradients, dequantized)]
            
            compressed_data = {
                'sparse_grads': [
                    {
                        'indices': sg['indices'].tolist(),
                        'values': sg['values'].tolist(),
                        'shape': list(sg['shape'])
                    }
                    for sg in sparse_grads
                ],
                'sparsity_info': sparsity_info,
                'original_size': original_size,
                'total_compressed': total_compressed,
                'compression_ratio': original_size / total_compressed if total_compressed > 0 else 1,
                'enable_sparsity': True,
                'enable_quantization': False
            }
        elif self.enable_quantization:
            quantized_grads = []
            metadatas = []
            total_compressed = 0
            
            for grad in sparse_grads:
                quantized, info = self.quantizer.quantize(grad)
                quantized_grads.append(quantized)
                metadatas.append({'min_val': info['min_val'], 'scale': info['scale']})
                total_compressed += info['quantized_size'] + info['metadata_size']
            
            if self.enable_error_correction:
                dequantized = []
                for q, meta in zip(quantized_grads, metadatas):
                    dq = self.quantizer.dequantize(q, meta['min_val'], meta['scale'])
                    dequantized.append(dq)
                self.residuals = [g - dq for g, dq in zip(gradients, dequantized)]
            
            compressed_data = {
                'quantized_grads': [
                    {
                        'quantized_values': q.tolist(),
                        'shape': list(g.shape),
                        'min_val': meta['min_val'],
                        'scale': meta['scale']
                    }
                    for q, g, meta in zip(quantized_grads, sparse_grads, metadatas)
                ],
                'original_size': original_size,
                'total_compressed': total_compressed,
                'compression_ratio': original_size / total_compressed if total_compressed > 0 else 1,
                'enable_sparsity': False,
                'enable_quantization': True
            }
        else:
            compressed_data = {
                'gradients': [g.tolist() for g in gradients],
                'original_size': original_size,
                'total_compressed': total_compressed,
                'compression_ratio': 1.0,
                'enable_sparsity': False,
                'enable_quantization': False
            }
        
        saved = original_size - total_compressed
        
        self.compression_history.append({
            'ratio': compressed_data['compression_ratio'],
            'saved_bytes': saved,
            'timestamp': time.time()
        })
        
        logger.info(f"Gradient compressed: {compressed_data['compression_ratio']:.1f}x ratio, saved {saved/1024/1024:.2f} MB")
        
        return compressed_data
    
    def decompress(self, compressed_data: Dict) -> List[np.ndarray]:
        if compressed_data.get('enable_sparsity') and compressed_data.get('enable_quantization'):
            quantized_grads = []
            for qg in compressed_data['quantized_grads']:
                quantized_g = {
                    'indices': np.array(qg['indices'], dtype=np.int32),
                    'quantized_values': np.array(qg['quantized_values'], dtype=np.int32),
                    'shape': tuple(qg['shape']),
                    'min_val': qg['min_val'],
                    'scale': qg['scale']
                }
                quantized_grads.append(quantized_g)
            
            sparse_grads = self.quantizer.dequantize_sparse_list(quantized_grads)
            gradients = self.sparsifier.desparsify_list(sparse_grads)
            
        elif compressed_data.get('enable_sparsity'):
            sparse_grads = []
            for sg in compressed_data['sparse_grads']:
                sparse_g = {
                    'indices': np.array(sg['indices'], dtype=np.int32),
                    'values': np.array(sg['values'], dtype=np.float32),
                    'shape': tuple(sg['shape'])
                }
                sparse_grads.append(sparse_g)
            gradients = self.sparsifier.desparsify_list(sparse_grads)
            
        elif compressed_data.get('enable_quantization'):
            gradients = []
            for qg in compressed_data['quantized_grads']:
                quantized = np.array(qg['quantized_values'], dtype=np.int32)
                dequantized = self.quantizer.dequantize(
                    quantized, qg['min_val'], qg['scale']
                )
                gradients.append(dequantized.reshape(tuple(qg['shape'])))
        else:
            gradients = [np.array(g, dtype=np.float32) for g in compressed_data['gradients']]
        
        return gradients
    
    def get_compression_stats(self) -> Dict:
        if not self.compression_history:
            return {'message': 'No compression history'}
        
        ratios = [h['ratio'] for h in self.compression_history]
        total_saved = sum(h['saved_bytes'] for h in self.compression_history)
        
        return {
            'total_compressions': len(self.compression_history),
            'avg_ratio': float(np.mean(ratios)),
            'max_ratio': float(np.max(ratios)),
            'min_ratio': float(np.min(ratios)),
            'total_saved_mb': total_saved / (1024 * 1024),
            'recent_ratios': ratios[-10:]
        }
    
    def estimate_savings(self, num_gradients: int, avg_size: float = 1e6) -> Dict:
        estimated_original = num_gradients * avg_size * 4
        
        if self.enable_sparsity:
            estimated_sparse = estimated_original * self.sparsifier.k_ratio * 2
        else:
            estimated_sparse = estimated_original
        
        if self.enable_quantization:
            estimated_final = estimated_sparse * (self.quantizer.bits / 32)
        else:
            estimated_final = estimated_sparse
        
        ratio = estimated_original / estimated_final
        
        return {
            'estimated_original_mb': estimated_original / (1024 * 1024),
            'estimated_compressed_mb': estimated_final / (1024 * 1024),
            'estimated_saved_mb': (estimated_original - estimated_final) / (1024 * 1024),
            'estimated_ratio': ratio
        }

def simulate_compression_demo():
    grads = [
        np.random.randn(512, 256).astype(np.float32),
        np.random.randn(256, 128).astype(np.float32),
        np.random.randn(128, 64).astype(np.float32),
        np.random.randn(64, 10).astype(np.float32),
        np.random.randn(256).astype(np.float32),
        np.random.randn(128).astype(np.float32),
        np.random.randn(64).astype(np.float32)
    ]
    
    compressor = GradientCompressor(
        enable_sparsity=True,
        enable_quantization=True,
        k_ratio=0.15,
        quant_bits=8
    )
    
    original_size = sum(g.nbytes for g in grads)
    
    compressed = compressor.compress(grads)
    
    ratio = compressed['compression_ratio']
    
    logger.info(f"Original: {original_size/1024:.2f} KB")
    logger.info(f"Compressed: {compressed['total_compressed']/1024:.2f} KB")
    logger.info(f"Ratio: {ratio:.1f}x")
    logger.info(f"Saved: {(original_size - compressed['total_compressed'])/1024:.2f} KB")
    
    reduction_percent = (1 - 1/ratio) * 100
    logger.info(f"Communication reduction: {reduction_percent:.1f}%")
    
    if ratio >= 3.33:
        logger.info("✓ Achieved 70%+ communication reduction target!")
    
    decompressed = compressor.decompress(compressed)
    
    total_error = 0
    for orig, decomp in zip(grads, decompressed):
        error = np.mean(np.abs(orig - decomp))
        total_error += error
    avg_error = total_error / len(grads)
    logger.info(f"Average reconstruction error: {avg_error:.6f}")
    
    return ratio >= 3.33

if __name__ == '__main__':
    simulate_compression_demo()
