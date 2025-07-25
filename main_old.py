# ==================== MAIN.PY COMPLETO ====================
"""
Interface principal do Sistema de Detecção de Invasões
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import time
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, BatchNormalization, Flatten, 
    Dense, Reshape, RepeatVector
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==================== CONFIGURAÇÕES ====================
class Config:
    # Parâmetros de dados
    FRAME_HEIGHT = 224
    FRAME_WIDTH = 224
    SEQUENCE_LENGTH = 16
    CHANNELS = 3
    
    # Parâmetros de treinamento
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Parâmetros do modelo
    CONV_LSTM_FILTERS = (32, 64, 32)
    DROPOUT_RATE = 0.2
    
    # Caminhos
    DATA_DIR = "data"
    MODEL_DIR = "models"
    RESULTS_DIR = "results"
    
    @property
    def input_shape(self):
        return (self.SEQUENCE_LENGTH, self.FRAME_HEIGHT, 
                self.FRAME_WIDTH, self.CHANNELS)

# ==================== PROCESSAMENTO DE DADOS ====================
class VideoProcessor:
    def __init__(self, config):
        self.config = config
        
    def extract_sequences_from_video(self, video_path, overlap=8):
        """Extrai sequências de frames de um vídeo"""
        print(f"Processando: {video_path}")
        cap = cv2.VideoCapture(video_path)
        sequences = []
        frames_buffer = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.preprocess_frame(frame)
            frames_buffer.append(processed_frame)
            frame_count += 1
            
            if len(frames_buffer) == self.config.SEQUENCE_LENGTH:
                sequences.append(np.array(frames_buffer))
                frames_buffer = frames_buffer[overlap:]
        
        cap.release()
        print(f"  Extraídas {len(sequences)} sequências de {frame_count} frames")
        return sequences
    
    def preprocess_frame(self, frame):
        """Pré-processa um frame"""
        frame = cv2.resize(frame, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def load_normal_data(self, data_dir):
        """Carrega dados normais para treinamento"""
        sequences = []
        
        normal_dir = os.path.join(data_dir, "normal")
        if not os.path.exists(normal_dir):
            raise ValueError(f"Diretório {normal_dir} não encontrado")
        
        video_files = [f for f in os.listdir(normal_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]
        
        if not video_files:
            raise ValueError(f"Nenhum vídeo encontrado em {normal_dir}")
        
        print(f"Encontrados {len(video_files)} vídeos normais")
        
        for video_file in video_files:
            video_path = os.path.join(normal_dir, video_file)
            try:
                video_sequences = self.extract_sequences_from_video(video_path)
                sequences.extend(video_sequences)
            except Exception as e:
                print(f"Erro ao processar {video_file}: {e}")
                continue
        
        if not sequences:
            raise ValueError("Nenhuma sequência válida foi extraída dos vídeos")
        
        return np.array(sequences)

# ==================== MODELO ====================
def build_convlstm_autoencoder(input_shape, conv_lstm_filters=(32, 64, 32)):
    """Constrói ConvLSTM Autoencoder"""
    
    # Input layer
    input_layer = Input(shape=input_shape, name='input_sequence')
    
    # Encoder
    x = ConvLSTM2D(
        filters=conv_lstm_filters[0],
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        name='conv_lstm_1'
    )(input_layer)
    x = BatchNormalization()(x)
    
    x = ConvLSTM2D(
        filters=conv_lstm_filters[1],
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        name='conv_lstm_2'
    )(x)
    x = BatchNormalization()(x)
    
    # Bottleneck
    encoded = ConvLSTM2D(
        filters=conv_lstm_filters[2],
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False,
        name='encoded'
    )(x)
    
    # Decoder - Simplificado
    # Flatten e Dense para reconstruir
    x = Flatten()(encoded)
    x = Dense(np.prod(input_shape), activation='relu')(x)
    x = Reshape(input_shape)(x)
    
    # Camadas de saída
    decoded = ConvLSTM2D(
        filters=input_shape[-1],  # 3 canais RGB
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        activation='sigmoid',
        name='output_sequence'
    )(x)
    
    model = Model(input_layer, decoded, name='ConvLSTM_Autoencoder')
    return model

# ==================== DETECTOR ====================
class AnomalyDetector:
    def __init__(self, model_path, threshold=0.05):
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        self.sequence_buffer = deque(maxlen=16)
        self.config = Config()
        
        # Estatísticas
        self.frame_count = 0
        self.anomaly_count = 0
        self.processing_times = []
    
    def preprocess_frame(self, frame):
        """Pré-processa frame para inferência"""
        frame = cv2.resize(frame, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def detect_anomaly(self, frame):
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
    
    def get_statistics(self):
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

# ==================== TREINAMENTO ====================
def train_model():
    """Função de treinamento do modelo"""
    print("="*50)
    print("🏠 SISTEMA DE DETECÇÃO DE INVASÕES")
    print("🎯 Iniciando Treinamento do Modelo")
    print("="*50)
    
    config = Config()
    
    # Criar diretórios se não existirem
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print("\n📁 Carregando dados de treinamento...")
    processor = VideoProcessor(config)
    
    try:
        sequences = processor.load_normal_data(config.DATA_DIR)
        print(f"✅ Carregadas {len(sequences)} sequências")
        print(f"📊 Formato dos dados: {sequences.shape}")
    except ValueError as e:
        print(f"❌ Erro ao carregar dados: {e}")
        print("\n💡 INSTRUÇÕES:")
        print("1. Crie o diretório 'data/normal'")
        print("2. Adicione vídeos (.mp4, .avi, .mov) de comportamentos normais")
        print("3. Execute novamente o treinamento")
        return False
    
    # Dividir dados
    train_sequences, val_sequences = train_test_split(
        sequences, test_size=config.VALIDATION_SPLIT, random_state=42
    )
    
    print(f"\n📈 Divisão dos dados:")
    print(f"   Treinamento: {len(train_sequences)} sequências")
    print(f"   Validação: {len(val_sequences)} sequências")
    
    # Construir modelo
    print("\n🔧 Construindo modelo ConvLSTM Autoencoder...")
    try:
        model = build_convlstm_autoencoder(
            input_shape=config.input_shape,
            conv_lstm_filters=config.CONV_LSTM_FILTERS
        )
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ Modelo construído com sucesso!")
        print(f"📊 Parâmetros do modelo: {model.count_params():,}")
        
    except Exception as e:
        print(f"❌ Erro ao construir modelo: {e}")
        return False
    
    # Callbacks
    model_path = os.path.join(config.MODEL_DIR, "best_model.h5")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Treinamento
    print(f"\n🚀 Iniciando treinamento ({config.EPOCHS} épocas máximo)...")
    print("⏰ Isso pode demorar alguns minutos...")
    
    try:
        history = model.fit(
            train_sequences, train_sequences,  # Autoencoder: input = output
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(val_sequences, val_sequences),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n🎉 Treinamento concluído com sucesso!")
        
        # Salvar gráficos
        save_training_plots(history, config.RESULTS_DIR)
        
        print(f"💾 Modelo salvo em: {model_path}")
        print(f"📊 Gráficos salvos em: {config.RESULTS_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante treinamento: {e}")
        return False

def save_training_plots(history, results_dir):
    """Salva gráficos do treinamento"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history.history['loss'], label='Treinamento', color='blue')
        ax1.plot(history.history['val_loss'], label='Validação', color='red')
        ax1.set_title('Loss do Modelo (MSE)')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Mean Squared Error')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2.plot(history.history['mae'], label='Treinamento', color='blue')
        ax2.plot(history.history['val_mae'], label='Validação', color='red')
        ax2.set_title('Mean Absolute Error')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(results_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Gráfico de treinamento salvo: {plot_path}")
        
    except Exception as e:
        print(f"⚠️ Erro ao salvar gráficos: {e}")

# ==================== DETECÇÃO EM TEMPO REAL ====================
def real_time_detection(model_path="models/best_model.h5", threshold=0.05, source=0):
    """Executa detecção em tempo real"""
    print("="*50)
    print("🏠 SISTEMA DE DETECÇÃO DE INVASÕES")
    print("👁️ Modo Detecção em Tempo Real")
    print("="*50)
    
    # Verificar se modelo existe
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        print("💡 Execute primeiro: python main.py --mode train")
        return False
    
    # Inicializar detector
    print("🤖 Carregando modelo...")
    try:
        detector = AnomalyDetector(model_path, threshold)
        print("✅ Modelo carregado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False
    
    # Inicializar câmera/vídeo
    print(f"📹 Conectando à fonte de vídeo ({source})...")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir fonte de vídeo")
        print("💡 Verifique se a webcam está conectada ou o arquivo existe")
        return False
    
    print("✅ Fonte de vídeo conectada!")
    print("\n🎮 CONTROLES:")
    print("   'q' - Sair")
    print("   's' - Capturar screenshot")
    print("   ESPAÇO - Pausar/Continuar")
    print("\n🚀 Iniciando detecção...")
    
    screenshot_count = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Fim do vídeo ou erro na captura")
                    break
                
                # Detectar anomalia
                is_anomaly, error, proc_time = detector.detect_anomaly(frame)
                
                # Preparar visualização
                display_frame = frame.copy()
                
                # Cor e texto baseado na detecção
                if is_anomaly:
                    color = (0, 0, 255)  # Vermelho
                    status = "🚨 INVASÃO DETECTADA!"
                    thickness = 3
                else:
                    color = (0, 255, 0)  # Verde
                    status = "✅ NORMAL"
                    thickness = 2
                
                # Adicionar informações no frame
                cv2.putText(display_frame, status, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, thickness)
                cv2.putText(display_frame, f"Erro: {error:.4f}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Threshold: {threshold:.4f}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # FPS e estatísticas
                if proc_time > 0:
                    fps = 1.0 / proc_time
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 160), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                stats = detector.get_statistics()
                if stats:
                    cv2.putText(display_frame, f"Frames: {stats['total_frames']}", (10, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.putText(display_frame, f"Anomalias: {stats['anomalies_detected']}", (10, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Mostrar frame
            cv2.imshow('🏠 Detector de Invasões - ConvLSTM Autoencoder', display_frame if not paused else frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Salvar screenshot
                os.makedirs("results/screenshots", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                screenshot_name = f"results/screenshots/detection_{timestamp}_{screenshot_count:03d}.png"
                cv2.imwrite(screenshot_name, display_frame if not paused else frame)
                print(f"📸 Screenshot salvo: {screenshot_name}")
                screenshot_count += 1
            elif key == ord(' '):
                paused = not paused
                print(f"⏸️ {'Pausado' if paused else 'Continuando'}")
                
    except KeyboardInterrupt:
        print("\n⏹️ Detecção interrompida pelo usuário")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Estatísticas finais
        stats = detector.get_statistics()
        if stats:
            print("\n📊 ESTATÍSTICAS FINAIS:")
            print("="*30)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        print("\n✅ Detecção finalizada!")
        return True

# ==================== FUNÇÃO PRINCIPAL ====================
def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='🏠 Sistema de Detecção de Invasões - ConvLSTM Autoencoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py --mode train                    # Treinar modelo
  python main.py --mode detect                   # Detecção com webcam
  python main.py --mode detect --source video.mp4  # Detecção com vídeo
  python main.py --mode detect --threshold 0.03     # Threshold personalizado
        """
    )
    
    parser.add_argument('--mode', choices=['train', 'detect'], 
                       required=True, help='Modo de operação')
    parser.add_argument('--model', default='models/best_model.h5',
                       help='Caminho do modelo (padrão: models/best_model.h5)')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Threshold para detecção (padrão: 0.05)')
    parser.add_argument('--source', default=0,
                       help='Fonte de vídeo: 0 para webcam ou caminho do arquivo')
    
    args = parser.parse_args()
    
    # Converter source para int se for número
    try:
        args.source = int(args.source)
    except ValueError:
        pass  # Manter como string (caminho do arquivo)
    
    print(f"TensorFlow versão: {tf.__version__}")
    print(f"Dispositivos disponíveis: {len(tf.config.list_physical_devices())}")
    
    if args.mode == 'train':
        success = train_model()
        sys.exit(0 if success else 1)
    elif args.mode == 'detect':
        success = real_time_detection(args.model, args.threshold, args.source)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()