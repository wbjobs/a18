import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'federated-learning-secret-key-2024')
    
    SERVER_HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
    SERVER_PORT = int(os.environ.get('SERVER_PORT', 5000))
    
    MIN_CLIENTS_PER_ROUND = int(os.environ.get('MIN_CLIENTS', 2))
    MAX_UPDATE_AGE = int(os.environ.get('MAX_UPDATE_AGE', 300))
    
    DP_EPSILON = float(os.environ.get('DP_EPSILON', 1.0))
    DP_DELTA = float(os.environ.get('DP_DELTA', 1e-5))
    
    WATERMARK_TRIGGER_SIZE = int(os.environ.get('WATERMARK_SIZE', 5))
    WATERMARK_TARGET_CLASS = int(os.environ.get('WATERMARK_TARGET', 8))
    WATERMARK_SECRET_KEY = os.environ.get('WATERMARK_KEY', 'federated_watermark_2024')
    WATERMARK_THRESHOLD = float(os.environ.get('WATERMARK_THRESHOLD', 0.8))
    
    OFFLINE_THRESHOLD = float(os.environ.get('OFFLINE_THRESHOLD', 60.0))
    
    CLIENT_DEFAULT_EPOCHS = int(os.environ.get('CLIENT_EPOCHS', 2))
    CLIENT_DEFAULT_BATCH_SIZE = int(os.environ.get('CLIENT_BATCH', 16))
    CLIENT_DEFAULT_SAMPLES = int(os.environ.get('CLIENT_SAMPLES', 2000))
    
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'server', 'saved_models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CLIENT_MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    @classmethod
    def ensure_dirs(cls):
        for dir_path in [cls.MODEL_SAVE_DIR, cls.DATA_DIR, cls.CLIENT_MODEL_DIR]:
            os.makedirs(dir_path, exist_ok=True)

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
