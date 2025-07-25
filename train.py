"""
Script de treinamento do modelo
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def train_model():
    config = Config()
    
    # Criar diretórios se não existirem
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print("Carregando dados...")
    processor = VideoProcessor(config)
    
    try:
        sequences = processor.load_normal_data(config.DATA_DIR)
        print(f"Carregadas {len(sequences)} sequências")
    except ValueError as e:
        print(f"Erro ao carregar dados: {e}")
        print("Certifique-se que existe o diretório 'data/normal' com vídeos")
        return
    
    # Dividir dados
    train_sequences, val_sequences = train_test_split(
        sequences, test_size=config.VALIDATION_SPLIT, random_state=42
    )
    
    print(f"Treinamento: {len(train_sequences)} sequências")
    print(f"Validação: {len(val_sequences)} sequências")
    
    # Construir modelo
    print("Construindo modelo...")
    autoencoder = ConvLSTMAutoencoder(config)
    model = autoencoder.build_model()
    autoencoder.compile_model()
    
    print(model.summary())
    
    # Callbacks
    model_path = os.path.join(config.MODEL_DIR, "best_model.h5")
    callbacks = autoencoder.get_callbacks(model_path)
    
    # Treinamento
    print("Iniciando treinamento...")
    history = model.fit(
        train_sequences, train_sequences,  # Autoencoder: input = output
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(val_sequences, val_sequences),
        callbacks=callbacks,
        verbose=1
    )
    
    # Salvar histórico
    save_training_plots(history, config.RESULTS_DIR)
    
    print("Treinamento concluído!")
    print(f"Modelo salvo em: {model_path}")

def save_training_plots(history, results_dir):
    """Salva gráficos do treinamento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Treinamento')
    ax1.plot(history.history['val_loss'], label='Validação')
    ax1.set_title('Loss do Modelo')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('MSE')
    ax1.legend()
    
    # MAE
    ax2.plot(history.history['mae'], label='Treinamento')
    ax2.plot(history.history['val_mae'], label='Validação')
    ax2.set_title('Mean Absolute Error')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'), dpi=300)
    plt.show()
    