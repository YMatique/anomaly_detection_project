"""
Configurações do sistema de detecção de invasões
"""
import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # Parâmetros de dados
    FRAME_HEIGHT: int = 224
    FRAME_WIDTH: int = 224
    SEQUENCE_LENGTH: int = 16
    CHANNELS: int = 3
    
    # Parâmetros de treinamento
    BATCH_SIZE: int = 4
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    VALIDATION_SPLIT: float = 0.2
    
    # Parâmetros do modelo
    CONV_LSTM_FILTERS: Tuple[int, ...] = (32, 64, 32)
    DROPOUT_RATE: float = 0.2
    
    # Caminhos
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    RESULTS_DIR: str = "results"
    
    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        return (self.SEQUENCE_LENGTH, self.FRAME_HEIGHT, 
                self.FRAME_WIDTH, self.CHANNELS)

# ==================== DATA_PROCESSOR.PY ====================
"""
Processamento de dados e geração de sequências
"""
import cv2
import numpy as np
import os
from typing import List, Generator
import random

class VideoProcessor:
    def __init__(self, config: Config):
        self.config = config
        
    def extract_sequences_from_video(self, video_path: str, 
                                   overlap: int = 8) -> List[np.ndarray]:
        """Extrai sequências de frames de um vídeo"""
        cap = cv2.VideoCapture(video_path)
        sequences = []
        frames_buffer = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.preprocess_frame(frame)
            frames_buffer.append(processed_frame)
            
            if len(frames_buffer) == self.config.SEQUENCE_LENGTH:
                sequences.append(np.array(frames_buffer))
                frames_buffer = frames_buffer[overlap:]
        
        cap.release()
        return sequences
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Pré-processa um frame"""
        frame = cv2.resize(frame, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def apply_augmentation(self, sequence: np.ndarray) -> np.ndarray:
        """Aplica data augmentation"""
        augmented = sequence.copy()
        
        # Espelhamento horizontal
        if random.random() < 0.5:
            augmented = np.flip(augmented, axis=2)
        
        # Variação de brilho
        brightness_factor = random.uniform(0.85, 1.15)
        augmented = np.clip(augmented * brightness_factor, 0, 1)
        
        # Ruído gaussiano
        noise = np.random.normal(0, 0.01, augmented.shape)
        augmented = np.clip(augmented + noise, 0, 1)
        
        return augmented
    
    def load_normal_data(self, data_dir: str) -> np.ndarray:
        """Carrega dados normais para treinamento"""
        sequences = []
        
        normal_dir = os.path.join(data_dir, "normal")
        if not os.path.exists(normal_dir):
            raise ValueError(f"Diretório {normal_dir} não encontrado")
        
        for video_file in os.listdir(normal_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(normal_dir, video_file)
                video_sequences = self.extract_sequences_from_video(video_path)
                sequences.extend(video_sequences)
        
        return np.array(sequences)
