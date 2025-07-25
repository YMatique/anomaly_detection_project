"""
Sistema de detecção em tempo real
"""
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

class AnomalyDetector:
    def __init__(self, model_path: str, threshold: float = 0.05):
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        self.sequence_buffer = deque(maxlen=16)
        self.config = Config()
        
        # Estatísticas
        self.frame_count = 0
        self.anomaly_count = 0
        self.processing_times = []
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Pré-processa frame para inferência"""
        frame = cv2.resize(frame, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def detect_anomaly(self, frame: np.ndarray) -> tuple:
        """Detecta anomalia em um frame"""
        start_time = time.time()
        
        # Pré-processar frame
        processed_frame = self.preprocess_frame(frame)
        self.sequence_buffer.append(processed_frame)
        
        # Verificar se buffer está cheio
        if len(self.sequence_buffer) < 16:
            return False, 0.0, 0.0
        
        # Preparar sequência
        sequence = np.array(list(self.sequence_buffer))
        sequence_batch = np.expand_dims(sequence, axis=0)
        
        # Predição
        try:
            reconstruction = self.model.predict(sequence_batch, verbose=0)
            mse_error = np.mean((sequence - reconstruction[0]) ** 2)
        except Exception as e:
            print(f"Erro na predição: {e}")
            return False, 0.0, 0.0
        
        # Detectar anomalia
        is_anomaly = mse_error > self.threshold
        
        # Estatísticas
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        if is_anomaly:
            self.anomaly_count += 1
        
        self.frame_count += 1
        
        return is_anomaly, mse_error, processing_time
    
    def get_statistics(self) -> dict:
        """Retorna estatísticas do detector"""
        if not self.processing_times:
            return {}
        
        return {
            'total_frames': self.frame_count,
            'anomalies_detected': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(self.frame_count, 1),
            'avg_processing_time': np.mean(self.processing_times),
            'fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0
        }
