# ==================== MAIN.PY CORRIGIDO ====================
"""
Interface principal do Sistema de Detecção de Invasões
VERSÃO CORRIGIDA - SEM BUG DE MEMÓRIA
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
import gc

# ==================== CONFIGURAÇÕES OTIMIZADAS ====================
class Config:
    # Parâmetros reduzidos para evitar OOM
    FRAME_HEIGHT = 64   # Reduzido de 224 para 64
    FRAME_WIDTH = 64    # Reduzido de 224 para 64
    SEQUENCE_LENGTH = 8  # Reduzido de 16 para 8
    CHANNELS = 3
    
    # Parâmetros de treinamento otimizados
    BATCH_SIZE = 2      # Reduzido de 4 para 2
    EPOCHS = 30         # Reduzido de 50 para 30
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Parâmetros do modelo reduzidos
    CONV_LSTM_FILTERS = (8, 16, 8)  # Reduzido de (32, 64, 32)
    DROPOUT_RATE = 0.2
    
    # Limites de memória
    MAX_SEQUENCES_PER_VIDEO = 30
    MAX_TOTAL_SEQUENCES = 100
    
    # Caminhos
    DATA_DIR = "data"
    MODEL_DIR = "models"
    RESULTS_DIR = "results"
    
    @property
    def input_shape(self):
        return (self.SEQUENCE_LENGTH, self.FRAME_HEIGHT, 
                self.FRAME_WIDTH, self.CHANNELS)

# ==================== PROCESSAMENTO DE DADOS OTIMIZADO ====================
class VideoProcessor:
    def __init__(self, config):
        self.config = config
        
    def extract_sequences_from_video(self, video_path, overlap=4):
        """Extrai sequências de frames de um vídeo com controle de memória"""
        print(f"Processando: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"  ❌ Erro ao abrir vídeo")
            return []
        
        sequences = []
        frames_buffer = []
        frame_count = 0
        
        try:
            while len(sequences) < self.config.MAX_SEQUENCES_PER_VIDEO:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    processed_frame = self.preprocess_frame(frame)
                    frames_buffer.append(processed_frame)
                    frame_count += 1
                    
                    if len(frames_buffer) == self.config.SEQUENCE_LENGTH:
                        sequence = np.array(frames_buffer, dtype=np.float32)
                        sequences.append(sequence)
                        frames_buffer = frames_buffer[overlap:]
                        
                        if len(sequences) % 10 == 0:
                            print(f"  📊 {len(sequences)} sequências...")
                            gc.collect()
                            
                except Exception as e:
                    print(f"  ⚠️ Erro frame {frame_count}: {e}")
                    continue
                    
        except Exception as e:
            print(f"  ❌ Erro geral: {e}")
        finally:
            cap.release()
            del frames_buffer
            gc.collect()
        
        print(f"  ✅ Extraídas {len(sequences)} sequências de {frame_count} frames")
        return sequences
    
    def preprocess_frame(self, frame):
        """Pré-processa um frame"""
        frame = cv2.resize(frame, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def load_normal_data(self, data_dir):
        """Carrega dados normais para treinamento com controle de memória"""
        normal_dir = os.path.join(data_dir, "normal")
        if not os.path.exists(normal_dir):
            raise ValueError(f"Diretório {normal_dir} não encontrado")
        
        video_files = [f for f in os.listdir(normal_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]
        
        if not video_files:
            raise ValueError(f"Nenhum vídeo encontrado em {normal_dir}")
        
        # Limitar número de vídeos se necessário
        max_videos = min(len(video_files), 5)  # Máximo 5 vídeos
        video_files = video_files[:max_videos]
        
        print(f"📹 Processando {len(video_files)} vídeos (máx {max_videos})")
        
        all_sequences = []
        
        for i, video_file in enumerate(video_files):
            print(f"\n🎬 Vídeo {i+1}/{len(video_files)}: {video_file}")
            
            if len(all_sequences) >= self.config.MAX_TOTAL_SEQUENCES:
                print("⚠️ Limite de sequências atingido")
                break
            
            video_path = os.path.join(normal_dir, video_file)
            try:
                video_sequences = self.extract_sequences_from_video(video_path)
                
                if video_sequences:
                    remaining = self.config.MAX_TOTAL_SEQUENCES - len(all_sequences)
                    sequences_to_add = video_sequences[:remaining]
                    all_sequences.extend(sequences_to_add)
                    print(f"  ✅ Adicionadas {len(sequences_to_add)} sequências")
                
                del video_sequences
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ Erro ao processar {video_file}: {e}")
                continue
        
        if not all_sequences:
            raise ValueError("Nenhuma sequência válida foi extraída dos vídeos")
        
        print(f"\n✅ Total: {len(all_sequences)} sequências")
        
        # Converter para array NumPy
        sequences_array = np.array(all_sequences, dtype=np.float32)
        print(f"📊 Shape final: {sequences_array.shape}")
        print(f"💾 Tamanho: {sequences_array.nbytes / 1024 / 1024:.1f} MB")
        
        del all_sequences
        gc.collect()
        
        return sequences_array

# ==================== MODELO CORRIGIDO ====================
def build_convlstm_autoencoder_fixed(input_shape, conv_lstm_filters=(8, 16, 8)):
    """
    ConvLSTM Autoencoder CORRIGIDO - SEM DENSE GIGANTE
    """
    print(f"🔧 Construindo modelo para: {input_shape}")
    
    # Input layer
    input_layer = Input(shape=input_shape, name='input_sequence')
    print(f"   Input: {input_layer.shape}")
    
    # Encoder ConvLSTM
    print("   Encoder...")
    x = ConvLSTM2D(
        filters=conv_lstm_filters[0],
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        activation='tanh',
        name='conv_lstm_1'
    )(input_layer)
    x = BatchNormalization()(x)
    
    x = ConvLSTM2D(
        filters=conv_lstm_filters[1],
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        activation='tanh',
        name='conv_lstm_2'
    )(x)
    x = BatchNormalization()(x)
    
    # Bottleneck ConvLSTM
    print("   Bottleneck...")
    encoded = ConvLSTM2D(
        filters=conv_lstm_filters[2],
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,  # ✅ MANTER SEQUÊNCIAS
        activation='tanh',
        name='encoded'
    )(x)
    
    # Decoder ConvLSTM - SEM DENSE GIGANTE!
    print("   Decoder...")
    x = ConvLSTM2D(
        filters=conv_lstm_filters[1],
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        activation='tanh',
        name='conv_lstm_decode_1'
    )(encoded)
    x = BatchNormalization()(x)
    
    x = ConvLSTM2D(
        filters=conv_lstm_filters[0],
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        activation='tanh',
        name='conv_lstm_decode_2'
    )(x)
    x = BatchNormalization()(x)
    
    # Camada de saída - reconstruir RGB
    print("   Saída...")
    decoded = ConvLSTM2D(
        filters=input_shape[-1],  # 3 canais RGB
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        activation='sigmoid',
        name='output_sequence'
    )(x)
    
    print(f"   Output: {decoded.shape}")
    
    model = Model(input_layer, decoded, name='ConvLSTM_Autoencoder_Fixed')
    print("✅ Modelo construído sem Dense gigante!")
    
    return model

# ==================== DETECTOR OTIMIZADO ====================
class AnomalyDetector:
    def __init__(self, model_path, threshold=0.05):
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        self.sequence_buffer = deque(maxlen=8)  # Reduzido para 8
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
        if len(self.sequence_buffer) < 8:
            return False, 0.0, 0.0
        
        # Preparar sequência
        sequence = np.array(list(self.sequence_buffer), dtype=np.float32)
        sequence_batch = np.expand_dims(sequence, axis=0)
        
        # Predição
        try:
            reconstruction = self.model.predict(sequence_batch, verbose=0)
            mse_error = np.mean((sequence - reconstruction[0]) ** 2)
            
            # Limpeza de memória
            del sequence, sequence_batch, reconstruction
            
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

# ==================== TREINAMENTO OTIMIZADO ====================
def train_model():
    """Função de treinamento otimizada"""
    print("="*50)
    print("🏠 SISTEMA DE DETECÇÃO DE INVASÕES")
    print("🎯 Versão Otimizada - Sem OOM")
    print("="*50)
    
    config = Config()
    
    # Verificar memória disponível
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 RAM disponível: {memory.available / 1024**3:.1f} GB")
        
        if memory.available < 2 * 1024**3:
            print("⚠️ Pouca RAM - ajustando parâmetros...")
            config.MAX_TOTAL_SEQUENCES = 50
            config.BATCH_SIZE = 1
    except ImportError:
        print("💾 psutil não disponível - usando configuração padrão")
    
    # Criar diretórios
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print(f"\n⚙️ Configuração otimizada:")
    print(f"   Resolução: {config.FRAME_WIDTH}×{config.FRAME_HEIGHT}")
    print(f"   Sequência: {config.SEQUENCE_LENGTH} frames")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Máx sequências: {config.MAX_TOTAL_SEQUENCES}")
    
    print("\n📁 Carregando dados...")
    processor = VideoProcessor(config)
    
    try:
        sequences = processor.load_normal_data(config.DATA_DIR)
        print(f"✅ Dados carregados: {sequences.shape}")
    except ValueError as e:
        print(f"❌ Erro: {e}")
        print("\n💡 INSTRUÇÕES:")
        print("1. Crie o diretório 'data/normal'")
        print("2. Adicione vídeos de comportamentos normais")
        print("3. Execute novamente")
        return False
    
    # Dividir dados
    train_sequences, val_sequences = train_test_split(
        sequences, test_size=config.VALIDATION_SPLIT, random_state=42
    )
    
    print(f"\n📈 Divisão:")
    print(f"   Treinamento: {len(train_sequences)} sequências")
    print(f"   Validação: {len(val_sequences)} sequências")
    
    # Construir modelo CORRIGIDO
    print("\n🔧 Construindo modelo ConvLSTM corrigido...")
    try:
        model = build_convlstm_autoencoder_fixed(
            input_shape=config.input_shape,
            conv_lstm_filters=config.CONV_LSTM_FILTERS
        )
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ Modelo construído e compilado!")
        print(f"📊 Parâmetros: {model.count_params():,}")
        
    except Exception as e:
        print(f"❌ Erro ao construir modelo: {e}")
        return False
    
    # Callbacks
    model_path = os.path.join(config.MODEL_DIR, "best_model_fixed.h5")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
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
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Treinamento
    print(f"\n🚀 Iniciando treinamento...")
    try:
        history = model.fit(
            train_sequences, train_sequences,  # Autoencoder
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(val_sequences, val_sequences),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n🎉 Treinamento concluído!")
        
        # Salvar gráficos
        save_training_plots(history, config.RESULTS_DIR)
        
        print(f"💾 Modelo salvo: {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Erro durante treinamento: {e}")
        return False
    finally:
        # Limpeza de memória
        del train_sequences, val_sequences
        gc.collect()

def save_training_plots(history, results_dir):
    """Salva gráficos do treinamento"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history.history['loss'], label='Treinamento', color='blue')
        ax1.plot(history.history['val_loss'], label='Validação', color='red')
        ax1.set_title('Loss do Modelo Otimizado')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('MSE')
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
        
        plot_path = os.path.join(results_dir, 'training_history_fixed.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Gráfico salvo: {plot_path}")
        
    except Exception as e:
        print(f"⚠️ Erro ao salvar gráficos: {e}")

# ==================== DETECÇÃO EM TEMPO REAL ====================
def real_time_detection(model_path="models/best_model_fixed.h5", threshold=0.05, source=0):
    """Executa detecção em tempo real"""
    print("="*50)
    print("🏠 DETECTOR DE INVASÕES - VERSÃO OTIMIZADA")
    print("="*50)
    
    # Verificar modelo
    if not os.path.exists(model_path):
        # Tentar modelo padrão se o otimizado não existir
        fallback_path = "models/best_model.h5"
        if os.path.exists(fallback_path):
            model_path = fallback_path
            print(f"⚠️ Usando modelo padrão: {model_path}")
        else:
            print(f"❌ Modelo não encontrado: {model_path}")
            print("💡 Execute: python main.py --mode train")
            return False
    
    # Inicializar detector
    print("🤖 Carregando modelo...")
    try:
        detector = AnomalyDetector(model_path, threshold)
        print("✅ Modelo carregado!")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False
    
    # Inicializar câmera
    print(f"📹 Conectando à fonte ({source})...")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir fonte de vídeo")
        return False
    
    print("✅ Fonte conectada!")
    print("\n🎮 CONTROLES:")
    print("   'q' - Sair")
    print("   's' - Screenshot")
    print("   ESPAÇO - Pausar")
    
    screenshot_count = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detectar anomalia
                is_anomaly, error, proc_time = detector.detect_anomaly(frame)
                
                # Visualização
                display_frame = frame.copy()
                
                if is_anomaly:
                    color = (0, 0, 255)
                    status = "🚨 INVASÃO!"
                    thickness = 3
                else:
                    color = (0, 255, 0)
                    status = "✅ NORMAL"
                    thickness = 2
                
                # Adicionar informações
                cv2.putText(display_frame, status, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, thickness)
                cv2.putText(display_frame, f"Erro: {error:.4f}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Threshold: {threshold:.4f}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if proc_time > 0:
                    fps = 1.0 / proc_time
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 160), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Estatísticas
                stats = detector.get_statistics()
                if stats:
                    cv2.putText(display_frame, f"Frames: {stats['total_frames']}", (10, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.putText(display_frame, f"Anomalias: {stats['anomalies_detected']}", (10, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Mostrar frame
            cv2.imshow('Detector Otimizado - ConvLSTM', display_frame if not paused else frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                os.makedirs("results/screenshots", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                screenshot_name = f"results/screenshots/fixed_{timestamp}_{screenshot_count:03d}.png"
                cv2.imwrite(screenshot_name, display_frame if not paused else frame)
                print(f"📸 Screenshot: {screenshot_name}")
                screenshot_count += 1
            elif key == ord(' '):
                paused = not paused
                print(f"⏸️ {'Pausado' if paused else 'Continuando'}")
                
    except KeyboardInterrupt:
        print("\n⏹️ Interrompido pelo usuário")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Estatísticas finais
        stats = detector.get_statistics()
        if stats:
            print("\n📊 ESTATÍSTICAS:")
            print("="*30)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        return True

# ==================== FUNÇÃO PRINCIPAL ====================
def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='🏠 Sistema Otimizado de Detecção de Invasões',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py --mode train                    # Treinar modelo otimizado
  python main.py --mode detect                   # Detecção com webcam
  python main.py --mode detect --source video.mp4  # Detecção com vídeo
  python main.py --mode detect --threshold 0.03     # Threshold personalizado
        """
    )
    
    parser.add_argument('--mode', choices=['train', 'detect'], 
                       required=True, help='Modo de operação')
    parser.add_argument('--model', default='models/best_model_fixed.h5',
                       help='Caminho do modelo')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Threshold para detecção')
    parser.add_argument('--source', default=0,
                       help='Fonte de vídeo: 0 para webcam ou caminho do arquivo')
    
    args = parser.parse_args()
    
    # Converter source para int se for número
    try:
        args.source = int(args.source)
    except ValueError:
        pass
    
    print(f"TensorFlow: {tf.__version__}")
    print(f"NumPy: {np.__version__}")
    
    if args.mode == 'train':
        success = train_model()
        sys.exit(0 if success else 1)
    elif args.mode == 'detect':
        success = real_time_detection(args.model, args.threshold, args.source)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()