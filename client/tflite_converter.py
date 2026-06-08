import tensorflow as tf
import numpy as np
import os
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TFLiteConverter:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        
    def convert_keras_to_tflite(self, model, output_path: str, 
                                 quantization: str = 'float32') -> str:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantization == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            
            def representative_dataset_gen():
                for _ in range(100):
                    yield [np.random.rand(1, 32, 32, 3).astype(np.float32)]
            
            converter.representative_dataset = representative_dataset_gen
            
        elif quantization == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif quantization == 'dynamic_range':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        try:
            tflite_model = converter.convert()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            original_size = model.count_params() * 4 / (1024 * 1024)
            tflite_size = len(tflite_model) / (1024 * 1024)
            
            logger.info(f"Model converted to TFLite: {output_path}")
            logger.info(f"Original size: {original_size:.2f} MB, TFLite size: {tflite_size:.2f} MB")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to convert model: {e}")
            raise
    
    def load_tflite_model(self, tflite_path: str) -> tf.lite.Interpreter:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"TFLite model loaded: {tflite_path}")
        logger.info(f"Input shape: {input_details[0]['shape']}")
        logger.info(f"Output shape: {output_details[0]['shape']}")
        
        return interpreter
    
    def run_inference(self, interpreter: tf.lite.Interpreter, image: np.ndarray) -> np.ndarray:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_data = image.astype(input_details[0]['dtype'])
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        return output
    
    def evaluate_tflite_model(self, interpreter: tf.lite.Interpreter, 
                               x_test: np.ndarray, y_test: np.ndarray) -> dict:
        correct = 0
        total = len(x_test)
        
        for i in range(total):
            prediction = self.run_inference(interpreter, x_test[i])
            predicted_class = np.argmax(prediction[0])
            if predicted_class == y_test[i]:
                correct += 1
        
        accuracy = correct / total
        logger.info(f"TFLite model accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

def create_edge_optimized_model(input_shape=(32, 32, 3), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation='relu', padding='same', 
                              input_shape=input_shape),
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, (1, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, (1, 1), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
