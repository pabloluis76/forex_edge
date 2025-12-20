"""
DEEP LEARNING PARA PREDICCIÓN DE RETORNOS
==========================================

Arquitecturas de redes neuronales profundas para capturar
patrones complejos y no lineales en transformaciones.


A) MLP (Multilayer Perceptron)
──────────────────────────────
Red neuronal feedforward.

Input:    m transformaciones
Hidden 1: 128 neuronas, ReLU, Dropout 0.3
Hidden 2: 64 neuronas, ReLU, Dropout 0.3
Hidden 3: 32 neuronas, ReLU, Dropout 0.3
Output:   1 (predicción de retorno)

Captura relaciones no lineales complejas.


B) CNN (Convolutional Neural Network)
─────────────────────────────────────
Detecta PATRONES LOCALES en secuencias.

Input:       (Lookback × Transformaciones)
Conv1D:      32 filtros, kernel 5
MaxPool:     pool size 2
Conv1D:      64 filtros, kernel 3
MaxPool:     pool size 2
Flatten:
Dense:       64 → 32 → 1

Aprende patrones que otros métodos no ven.


C) LSTM (Long Short-Term Memory)
────────────────────────────────
Red recurrente con memoria de largo plazo.

Input:       (Lookback × Transformaciones)
LSTM:        64 unidades
Dropout:     0.3
LSTM:        32 unidades
Dense:       1

Captura dependencias temporales de largo plazo.


D) TRANSFORMER / ATTENTION
──────────────────────────
Pondera importancia de diferentes momentos en la historia.

Multi-Head Attention permite aprender:
"¿Qué momentos pasados son relevantes para predecir ahora?"

Los pesos de atención son interpretables.


NOTA: Requiere TensorFlow/Keras
pip install tensorflow

Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import warnings

warnings.filterwarnings('ignore')

# Intentar importar TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.error("TensorFlow no está instalado. Instala con: pip install tensorflow")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelosDeepLearning:
    """
    Implementación de arquitecturas de Deep Learning para forex.

    Requiere TensorFlow/Keras.
    """

    def __init__(self):
        """Inicializa la clase de modelos."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no disponible. Instala con: pip install tensorflow")

        self.modelo = None
        self.historia = None

    @staticmethod
    def crear_mlp(n_features: int,
                  capas_ocultas: List[int] = [128, 64, 32],
                  dropout: float = 0.3,
                  learning_rate: float = 0.001) -> "Model":
        """
        A) MLP (Multilayer Perceptron)

        Red neuronal feedforward para predecir retornos.

        Args:
            n_features: Número de transformaciones de entrada
            capas_ocultas: Lista con neuronas por capa
            dropout: Tasa de dropout
            learning_rate: Tasa de aprendizaje

        Returns:
            Modelo Keras compilado
        """
        logger.info("="*70)
        logger.info("A) CONSTRUYENDO MLP (Multilayer Perceptron)")
        logger.info("="*70)

        # Input
        input_layer = Input(shape=(n_features,), name='input')
        x = input_layer

        # Capas ocultas
        for i, n_neurons in enumerate(capas_ocultas):
            x = layers.Dense(
                n_neurons,
                activation='relu',
                kernel_initializer='he_normal',
                name=f'hidden_{i+1}'
            )(x)
            x = layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

        # Output
        output = layers.Dense(1, activation='linear', name='output')(x)

        # Modelo
        modelo = Model(inputs=input_layer, outputs=output, name='MLP')

        # Compilar
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        # Resumen
        logger.info("Arquitectura MLP:")
        logger.info(f"  Input: {n_features} features")
        for i, n in enumerate(capas_ocultas):
            logger.info(f"  Hidden {i+1}: {n} neurons, ReLU, Dropout {dropout}")
        logger.info(f"  Output: 1 (retorno predicho)")
        logger.info(f"  Parámetros entrenables: {modelo.count_params():,}")

        return modelo

    @staticmethod
    def crear_cnn(lookback: int,
                  n_features: int,
                  filtros: List[int] = [32, 64],
                  kernel_sizes: List[int] = [5, 3],
                  pool_sizes: List[int] = [2, 2],
                  dropout: float = 0.3,
                  learning_rate: float = 0.001) -> "Model":
        """
        B) CNN (Convolutional Neural Network)

        Red convolucional 1D para detectar patrones locales.

        Args:
            lookback: Tamaño de la ventana temporal
            n_features: Número de transformaciones
            filtros: Número de filtros por capa Conv1D
            kernel_sizes: Tamaño de kernel por capa
            pool_sizes: Tamaño de pooling por capa
            dropout: Tasa de dropout
            learning_rate: Tasa de aprendizaje

        Returns:
            Modelo Keras compilado
        """
        logger.info("="*70)
        logger.info("B) CONSTRUYENDO CNN (Convolutional Neural Network)")
        logger.info("="*70)

        # Input: (lookback, n_features)
        input_layer = Input(shape=(lookback, n_features), name='input')
        x = input_layer

        # Capas convolucionales
        for i, (n_filtros, kernel_size, pool_size) in enumerate(zip(filtros, kernel_sizes, pool_sizes)):
            x = layers.Conv1D(
                filters=n_filtros,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            x = layers.MaxPooling1D(pool_size=pool_size, name=f'maxpool_{i+1}')(x)
            x = layers.Dropout(dropout, name=f'dropout_conv_{i+1}')(x)

        # Flatten
        x = layers.Flatten(name='flatten')(x)

        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(dropout, name='dropout_dense_1')(x)
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        x = layers.Dropout(dropout, name='dropout_dense_2')(x)

        # Output
        output = layers.Dense(1, activation='linear', name='output')(x)

        # Modelo
        modelo = Model(inputs=input_layer, outputs=output, name='CNN')

        # Compilar
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        logger.info("Arquitectura CNN:")
        logger.info(f"  Input: ({lookback}, {n_features})")
        for i, (f, k, p) in enumerate(zip(filtros, kernel_sizes, pool_sizes)):
            logger.info(f"  Conv1D_{i+1}: {f} filtros, kernel {k}")
            logger.info(f"  MaxPool_{i+1}: pool size {p}")
        logger.info(f"  Flatten → Dense 64 → Dense 32 → Output 1")
        logger.info(f"  Parámetros entrenables: {modelo.count_params():,}")

        return modelo

    @staticmethod
    def crear_lstm(lookback: int,
                   n_features: int,
                   lstm_units: List[int] = [64, 32],
                   dropout: float = 0.3,
                   learning_rate: float = 0.001) -> "Model":
        """
        C) LSTM (Long Short-Term Memory)

        Red recurrente con memoria para dependencias temporales.

        Args:
            lookback: Tamaño de la ventana temporal
            n_features: Número de transformaciones
            lstm_units: Unidades por capa LSTM
            dropout: Tasa de dropout
            learning_rate: Tasa de aprendizaje

        Returns:
            Modelo Keras compilado
        """
        logger.info("="*70)
        logger.info("C) CONSTRUYENDO LSTM (Long Short-Term Memory)")
        logger.info("="*70)

        # Input: (lookback, n_features)
        input_layer = Input(shape=(lookback, n_features), name='input')
        x = input_layer

        # Capas LSTM
        for i, units in enumerate(lstm_units[:-1]):
            # return_sequences=True para todas excepto la última
            x = layers.LSTM(
                units,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=dropout,
                name=f'lstm_{i+1}'
            )(x)

        # Última capa LSTM (return_sequences=False)
        x = layers.LSTM(
            lstm_units[-1],
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=dropout,
            name=f'lstm_{len(lstm_units)}'
        )(x)

        # Output
        output = layers.Dense(1, activation='linear', name='output')(x)

        # Modelo
        modelo = Model(inputs=input_layer, outputs=output, name='LSTM')

        # Compilar
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        logger.info("Arquitectura LSTM:")
        logger.info(f"  Input: ({lookback}, {n_features})")
        for i, units in enumerate(lstm_units):
            logger.info(f"  LSTM_{i+1}: {units} units, Dropout {dropout}")
        logger.info(f"  Output: 1 (retorno predicho)")
        logger.info(f"  Parámetros entrenables: {modelo.count_params():,}")

        return modelo

    @staticmethod
    def crear_transformer(lookback: int,
                         n_features: int,
                         num_heads: int = 4,
                         ff_dim: int = 128,
                         dropout: float = 0.3,
                         learning_rate: float = 0.001) -> "Model":
        """
        D) TRANSFORMER / ATTENTION

        Arquitectura con Multi-Head Attention.

        Args:
            lookback: Tamaño de la ventana temporal
            n_features: Número de transformaciones (debe ser divisible por num_heads)
            num_heads: Número de heads en attention
            ff_dim: Dimensión de feedforward network
            dropout: Tasa de dropout
            learning_rate: Tasa de aprendizaje

        Returns:
            Modelo Keras compilado
        """
        logger.info("="*70)
        logger.info("D) CONSTRUYENDO TRANSFORMER (Multi-Head Attention)")
        logger.info("="*70)

        # Input: (lookback, n_features)
        input_layer = Input(shape=(lookback, n_features), name='input')

        # Multi-Head Attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=n_features // num_heads,
            dropout=dropout,
            name='multi_head_attention'
        )(input_layer, input_layer)

        # Skip connection + Layer Norm
        x = layers.Add(name='add_1')([input_layer, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6, name='layer_norm_1')(x)

        # Feedforward Network
        ff_output = layers.Dense(ff_dim, activation='relu', name='ff_1')(x)
        ff_output = layers.Dropout(dropout, name='dropout_ff')(ff_output)
        ff_output = layers.Dense(n_features, name='ff_2')(ff_output)

        # Skip connection + Layer Norm
        x = layers.Add(name='add_2')([x, ff_output])
        x = layers.LayerNormalization(epsilon=1e-6, name='layer_norm_2')(x)

        # Global Average Pooling
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)

        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(dropout, name='dropout_1')(x)
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        x = layers.Dropout(dropout, name='dropout_2')(x)

        # Output
        output = layers.Dense(1, activation='linear', name='output')(x)

        # Modelo
        modelo = Model(inputs=input_layer, outputs=output, name='Transformer')

        # Compilar
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        logger.info("Arquitectura Transformer:")
        logger.info(f"  Input: ({lookback}, {n_features})")
        logger.info(f"  Multi-Head Attention: {num_heads} heads")
        logger.info(f"  Feedforward: {ff_dim} → {n_features}")
        logger.info(f"  Global Average Pooling")
        logger.info(f"  Dense: 64 → 32 → 1")
        logger.info(f"  Parámetros entrenables: {modelo.count_params():,}")

        return modelo

    def entrenar_modelo(self,
                       modelo: "Model",
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       epochs: int = 100,
                       batch_size: int = 32,
                       early_stopping_patience: int = 10) -> Dict:
        """
        Entrena un modelo de Deep Learning.

        Args:
            modelo: Modelo Keras
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            epochs: Épocas máximas
            batch_size: Tamaño de batch
            early_stopping_patience: Paciencia para early stopping

        Returns:
            Diccionario con historia y métricas
        """
        logger.info("="*70)
        logger.info(f"ENTRENANDO MODELO: {modelo.name}")
        logger.info("="*70)

        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}")
        logger.info(f"y_val shape: {y_val.shape}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
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

        # Entrenar
        historia = modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluar
        train_loss, train_mae = modelo.evaluate(X_train, y_train, verbose=0)
        val_loss, val_mae = modelo.evaluate(X_val, y_val, verbose=0)

        logger.info(f"\nResultados finales:")
        logger.info(f"  Train Loss: {train_loss:.6f}, MAE: {train_mae:.6f}")
        logger.info(f"  Val Loss:   {val_loss:.6f}, MAE: {val_mae:.6f}")

        # Guardar modelo y historia
        self.modelo = modelo
        self.historia = historia.history

        resultados = {
            'modelo': modelo,
            'historia': historia.history,
            'train_loss': train_loss,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae
        }

        return resultados

    @staticmethod
    def extraer_attention_weights(modelo: "Model",
                                  X_sample: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae pesos de atención del Transformer.

        Args:
            modelo: Modelo Transformer
            X_sample: Sample de entrada (1, lookback, n_features)

        Returns:
            Pesos de atención si el modelo es Transformer
        """
        if 'Transformer' not in modelo.name:
            logger.warning("El modelo no es un Transformer")
            return None

        # Buscar capa de attention
        attention_layer = None
        for layer in modelo.layers:
            if isinstance(layer, layers.MultiHeadAttention):
                attention_layer = layer
                break

        if attention_layer is None:
            logger.warning("No se encontró capa MultiHeadAttention")
            return None

        # Crear modelo intermedio para extraer attention
        # Esto requiere modificación del modelo original
        # Por simplicidad, retornamos None
        logger.info("Extracción de attention weights requiere modificación del modelo")
        logger.info("Los pesos están en: modelo.get_layer('multi_head_attention').get_weights()")

        return None


def ejemplo_uso():
    """
    Ejemplo de uso de modelos de Deep Learning.
    """
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow no está instalado")
        print("   Instala con: pip install tensorflow")
        return

    print("="*70)
    print("EJEMPLO: DEEP LEARNING PARA PREDICCIÓN DE RETORNOS")
    print("="*70)
    print()

    # Generar datos sintéticos
    np.random.seed(42)
    n_obs = 5000
    lookback = 20
    n_features = 50

    print("Generando datos sintéticos...")
    print(f"  Observaciones: {n_obs:,}")
    print(f"  Lookback: {lookback}")
    print(f"  Features: {n_features}")
    print()

    # Datos 2D para MLP
    X_2d = np.random.randn(n_obs, n_features).astype(np.float32)
    y = (0.05 * X_2d[:, 0] - 0.03 * X_2d[:, 1] + np.random.randn(n_obs) * 0.01).astype(np.float32)

    # Datos 3D para CNN/LSTM/Transformer
    X_3d = np.random.randn(n_obs, lookback, n_features).astype(np.float32)

    # Split
    split = int(n_obs * 0.8)
    X_2d_train, X_2d_val = X_2d[:split], X_2d[split:]
    X_3d_train, X_3d_val = X_3d[:split], X_3d[split:]
    y_train, y_val = y[:split], y[split:]

    dl = ModelosDeepLearning()

    # A) MLP
    print("\n" + "="*70)
    print("PROBANDO MLP")
    print("="*70)
    modelo_mlp = dl.crear_mlp(n_features=n_features)
    resultados_mlp = dl.entrenar_modelo(
        modelo_mlp, X_2d_train, y_train, X_2d_val, y_val,
        epochs=20, batch_size=64
    )

    # B) CNN
    print("\n" + "="*70)
    print("PROBANDO CNN")
    print("="*70)
    modelo_cnn = dl.crear_cnn(lookback=lookback, n_features=n_features)
    resultados_cnn = dl.entrenar_modelo(
        modelo_cnn, X_3d_train, y_train, X_3d_val, y_val,
        epochs=20, batch_size=64
    )

    # C) LSTM
    print("\n" + "="*70)
    print("PROBANDO LSTM")
    print("="*70)
    modelo_lstm = dl.crear_lstm(lookback=lookback, n_features=n_features)
    resultados_lstm = dl.entrenar_modelo(
        modelo_lstm, X_3d_train, y_train, X_3d_val, y_val,
        epochs=20, batch_size=64
    )

    # D) Transformer
    print("\n" + "="*70)
    print("PROBANDO TRANSFORMER")
    print("="*70)
    # n_features debe ser divisible por num_heads
    n_features_transformer = 48  # Divisible por 4
    X_3d_transformer = X_3d[:, :, :n_features_transformer]
    X_3d_train_t = X_3d_transformer[:split]
    X_3d_val_t = X_3d_transformer[split:]

    modelo_transformer = dl.crear_transformer(
        lookback=lookback,
        n_features=n_features_transformer,
        num_heads=4
    )
    resultados_transformer = dl.entrenar_modelo(
        modelo_transformer, X_3d_train_t, y_train, X_3d_val_t, y_val,
        epochs=20, batch_size=64
    )

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE MODELOS")
    print("="*70)
    print()
    print(f"MLP:")
    print(f"  Val Loss: {resultados_mlp['val_loss']:.6f}")
    print(f"  Val MAE:  {resultados_mlp['val_mae']:.6f}")
    print()
    print(f"CNN:")
    print(f"  Val Loss: {resultados_cnn['val_loss']:.6f}")
    print(f"  Val MAE:  {resultados_cnn['val_mae']:.6f}")
    print()
    print(f"LSTM:")
    print(f"  Val Loss: {resultados_lstm['val_loss']:.6f}")
    print(f"  Val MAE:  {resultados_lstm['val_mae']:.6f}")
    print()
    print(f"Transformer:")
    print(f"  Val Loss: {resultados_transformer['val_loss']:.6f}")
    print(f"  Val MAE:  {resultados_transformer['val_mae']:.6f}")
    print()
    print("="*70)


if __name__ == '__main__':
    ejemplo_uso()
