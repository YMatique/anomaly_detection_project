"""
Arquitetura ConvLSTM Autoencoder
"""
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

class ConvLSTMAutoencoder:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
    
    def build_model(self) -> Model:
        """Constrói o modelo ConvLSTM Autoencoder"""
        input_layer = Input(shape=self.config.input_shape, name='input_sequence')
        
        # Encoder
        x = ConvLSTM2D(
            filters=self.config.CONV_LSTM_FILTERS[0],
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            name='conv_lstm_1'
        )(input_layer)
        x = BatchNormalization()(x)
        
        x = ConvLSTM2D(
            filters=self.config.CONV_LSTM_FILTERS[1],
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            name='conv_lstm_2'
        )(x)
        x = BatchNormalization()(x)
        
        # Bottleneck
        encoded = ConvLSTM2D(
            filters=self.config.CONV_LSTM_FILTERS[2],
            kernel_size=(3, 3),
            padding='same',
            return_sequences=False,
            name='encoded'
        )(x)
        
        # Preparar para decoder
        x = RepeatVector(self.config.SEQUENCE_LENGTH)(Flatten()(encoded))
        x = Reshape((self.config.SEQUENCE_LENGTH, -1))(x)
        
        # Reshape para dimensões espaciais
        spatial_dim = self.config.FRAME_HEIGHT * self.config.FRAME_WIDTH * self.config.CONV_LSTM_FILTERS[2] // (self.config.FRAME_HEIGHT * self.config.FRAME_WIDTH)
        decoder_shape = (self.config.SEQUENCE_LENGTH, self.config.FRAME_HEIGHT, 
                        self.config.FRAME_WIDTH, self.config.CONV_LSTM_FILTERS[2])
        
        # Simplificando o decoder
        x = Dense(self.config.FRAME_HEIGHT * self.config.FRAME_WIDTH * self.config.CONV_LSTM_FILTERS[2])(x)
        x = Reshape(decoder_shape)(x)
        
        # Decoder ConvLSTM
        x = ConvLSTM2D(
            filters=self.config.CONV_LSTM_FILTERS[2],
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            name='conv_lstm_decode_1'
        )(x)
        x = BatchNormalization()(x)
        
        x = ConvLSTM2D(
            filters=self.config.CONV_LSTM_FILTERS[1],
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            name='conv_lstm_decode_2'
        )(x)
        x = BatchNormalization()(x)
        
        # Saída
        decoded = ConvLSTM2D(
            filters=self.config.CHANNELS,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='sigmoid',
            name='output_sequence'
        )(x)
        
        self.model = Model(input_layer, decoded, name='ConvLSTM_Autoencoder')
        return self.model
    
    def compile_model(self):
        """Compila o modelo"""
        optimizer = Adam(learning_rate=self.config.LEARNING_RATE)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    def get_callbacks(self, model_path: str):
        """Retorna callbacks para treinamento"""
        return [
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