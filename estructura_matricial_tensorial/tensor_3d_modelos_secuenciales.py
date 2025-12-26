"""
TENSOR 3D PARA MODELOS SECUENCIALES
====================================

Estructura tensorial para modelos que requieren SECUENCIAS de historia:
CNN, LSTM, GRU, Transformer, etc.

PARA CNN, LSTM, TRANSFORMER:
Necesitan ver SECUENCIAS de historia.

TENSOR: (Samples × Lookback × Transformaciones)

X ∈ ℝⁿˣᴸˣᵐ

n = número de samples
L = lookback (ej: 50 barras de historia)
m = número de transformaciones


VISUALIZACIÓN DE UN SAMPLE:

                    Transformaciones →
                 T₁    T₂    T₃   ...   Tₘ
               ┌─────┬─────┬─────┬─────┬─────┐
    t-L+1      │     │     │     │     │     │  ← L barras atrás
               ├─────┼─────┼─────┼─────┼─────┤
    t-L+2      │     │     │     │     │     │
               ├─────┼─────┼─────┼─────┼─────┤
Lookback ↓     │     │     │     │     │     │
               ├─────┼─────┼─────┼─────┼─────┤
    t-1        │     │     │     │     │     │  ← Barra anterior
               └─────┴─────┴─────┴─────┴─────┘

               Este es UN sample para predecir retorno en t


EJEMPLO:
Si L=50 y m=762 transformaciones:
- Cada sample es una matriz de 50×762
- Si tenemos 30,000 barras, obtenemos ~29,950 samples
- Tensor final: (29950, 50, 762)

Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

# Importar matriz 2D
sys.path.insert(0, str(Path(__file__).parent))
from matriz_2d_transformaciones_tiempo import MatrizTransformacionesTiempo, cargar_desde_csv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Tensor3DSecuencial:
    """
    Clase para manejar tensores 3D: (samples × lookback × features)

    Convierte matriz 2D (tiempo × features) en tensor 3D mediante
    ventanas deslizantes (sliding windows).
    """

    def __init__(self, matriz_2d: MatrizTransformacionesTiempo):
        """
        Inicializa el tensor 3D desde una matriz 2D.

        Args:
            matriz_2d: Objeto MatrizTransformacionesTiempo
        """
        self.matriz_2d = matriz_2d
        self.X_3d = None  # Tensor de features (n, L, m)
        self.y = None     # Vector target (n,)
        self.lookback = None
        self.n_samples = None
        self.n_features = None

        if matriz_2d.X is None:
            raise ValueError("La matriz 2D debe tener X construido")

    def crear_secuencias(self,
                        lookback: int = 50,
                        horizonte: int = 1,
                        step: int = 1,
                        precios: Optional[np.ndarray] = None,
                        tipo_target: str = 'retorno') -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea tensor 3D mediante ventanas deslizantes.

        Args:
            lookback: Número de barras de historia (L)
            horizonte: Barras hacia adelante para predecir
            step: Paso entre ventanas (1 = máximo overlap)
            precios: Array de precios (close) para calcular target. Si None, intenta obtener de matriz_2d.df
            tipo_target: 'retorno', 'log_retorno', 'direccion'

        Returns:
            (X_3d, y) donde:
                X_3d.shape = (n_samples, lookback, n_features)
                y.shape = (n_samples,)
        """
        logger.info(f"Creando secuencias con lookback={lookback}, horizonte={horizonte}...")

        X_2d = self.matriz_2d.X
        n_obs, n_features = X_2d.shape

        self.lookback = lookback
        self.n_features = n_features

        # Calcular número de samples
        n_samples = (n_obs - lookback - horizonte + 1) // step

        if n_samples <= 0:
            raise ValueError(f"No hay suficientes datos. Necesitas al menos {lookback + horizonte} observaciones")

        # Inicializar tensor 3D
        X_3d = np.zeros((n_samples, lookback, n_features), dtype=np.float32)

        # Crear ventanas deslizantes
        logger.info(f"Generando {n_samples:,} ventanas deslizantes...")

        for i in range(n_samples):
            start_idx = i * step
            end_idx = start_idx + lookback

            # Ventana de lookback
            X_3d[i] = X_2d[start_idx:end_idx]

        self.X_3d = X_3d
        self.n_samples = n_samples

        logger.info(f"✓ Tensor 3D creado: {X_3d.shape}")
        logger.info(f"  n_samples = {n_samples:,}")
        logger.info(f"  lookback = {lookback}")
        logger.info(f"  n_features = {n_features}")
        logger.info(f"  Tamaño en memoria: {X_3d.nbytes / 1024**2:.1f} MB")

        # Crear target correcto desde precios
        if precios is None:
            # Intentar obtener precios de la matriz 2D
            if hasattr(self.matriz_2d, 'df') and 'close' in self.matriz_2d.df.columns:
                precios = self.matriz_2d.df['close'].values
                logger.info("✓ Precios obtenidos desde matriz_2d.df['close']")
            else:
                logger.warning("⚠️  No se proporcionaron precios. Target será placeholder (usar crear_target_desde_precios() después)")
                self.y = np.zeros(n_samples, dtype=np.float32)
                return X_3d, self.y

        # Crear target correcto
        y = self.crear_target_desde_precios(precios, horizonte, tipo_target)

        return X_3d, y

    def crear_target_desde_precios(self,
                                   precios: np.ndarray,
                                   horizonte: int = 1,
                                   tipo: str = 'retorno') -> np.ndarray:
        """
        Crea vector target correcto desde precios.

        Args:
            precios: Array de precios (close) alineado con X_2d
            horizonte: Barras hacia adelante
            tipo: 'retorno', 'log_retorno', 'direccion'

        Returns:
            Vector y de shape (n_samples,)
        """
        if self.X_3d is None:
            raise ValueError("Primero crea las secuencias con crear_secuencias()")

        logger.info(f"Creando target desde precios (tipo={tipo}, horizonte={horizonte})...")

        n_samples = self.X_3d.shape[0]
        y = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            # Índice del precio actual (después del lookback)
            current_idx = i + self.lookback
            future_idx = current_idx + horizonte

            # Validar índices
            if current_idx < 0 or current_idx >= len(precios):
                y[i] = np.nan
                continue

            if future_idx >= len(precios):
                y[i] = np.nan
                continue

            # Validar precios son positivos y no cero
            if precios[current_idx] <= 0 or precios[future_idx] <= 0:
                y[i] = np.nan
                continue

            if tipo == 'retorno':
                # (P_{t+h} - P_t) / P_t con protección
                y[i] = (precios[future_idx] - precios[current_idx]) / precios[current_idx]
            elif tipo == 'log_retorno':
                # ln(P_{t+h} / P_t) con protección
                ratio = precios[future_idx] / precios[current_idx]
                # El ratio debería ser siempre positivo si ambos precios > 0
                if ratio > 0:
                    y[i] = np.log(ratio)
                else:
                    y[i] = np.nan
            elif tipo == 'direccion':
                # sign(P_{t+h} - P_t)
                y[i] = np.sign(precios[future_idx] - precios[current_idx])

        self.y = y

        logger.info(f"✓ Target creado: {y.shape}")
        logger.info(f"  min={y.min():.6f}, max={y.max():.6f}")
        logger.info(f"  mean={y.mean():.6f}, std={y.std():.6f}")

        return y

    def mostrar_estructura(self):
        """
        Muestra la estructura del tensor 3D de forma visual.
        """
        if self.X_3d is None:
            raise ValueError("Primero crea las secuencias con crear_secuencias()")

        n, L, m = self.X_3d.shape

        print("="*70)
        print("TENSOR 3D PARA MODELOS SECUENCIALES")
        print("="*70)
        print()
        print("DIMENSIONES:")
        print(f"  X ∈ ℝ^({n} × {L} × {m})")
        print(f"  n_samples = {n:,}")
        print(f"  lookback (L) = {L}")
        print(f"  n_transformaciones (m) = {m}")
        print()

        print("INTERPRETACIÓN:")
        print(f"  Cada sample ve {L} barras de historia")
        print(f"  Cada barra tiene {m} transformaciones (features)")
        print(f"  Total: {n:,} secuencias para entrenar")
        print()

        print("VISUALIZACIÓN DE UN SAMPLE (sample 0):")
        print()
        print("                    Transformaciones →")
        print("                 T₁    T₂    T₃    T₄   ...   Tₘ")
        print("               ┌─────┬─────┬─────┬─────┬─────┬─────┐")

        # Mostrar primeras 3 y última barra del lookback
        sample_0 = self.X_3d[0]  # Shape: (L, m)

        for i in [0, 1, 2]:
            if i < L:
                t_label = f"t-{L-i-1}".ljust(8)
                valores = [f"{sample_0[i, j]:.2f}" for j in range(min(4, m))]
                valores_str = " │ ".join(f"{v:>5}" for v in valores)

                comment = ""
                if i == 0:
                    comment = f"  ← {L} barras atrás"

                print(f"    {t_label}   │ {valores_str} │ ... │{comment}")
                if i < 2:
                    print("               ├─────┼─────┼─────┼─────┼─────┼─────┤")

        if L > 4:
            print("Lookback ↓     │  ⋮  │  ⋮  │  ⋮  │  ⋮  │  ⋮  │  ⋮  │")
            print("               ├─────┼─────┼─────┼─────┼─────┼─────┤")

            # Última barra (t-1)
            t_label = "t-1".ljust(8)
            valores = [f"{sample_0[-1, j]:.2f}" for j in range(min(4, m))]
            valores_str = " │ ".join(f"{v:>5}" for v in valores)
            print(f"    {t_label}   │ {valores_str} │ ... │  ← Barra anterior")

        print("               └─────┴─────┴─────┴─────┴─────┴─────┘")
        print()
        print("               Este sample predice el retorno en tiempo t")
        print()

        print("ESTADÍSTICAS DEL TENSOR:")
        print(f"  Tamaño en memoria: {self.X_3d.nbytes / 1024**2:.1f} MB")
        print(f"  Valores totales: {self.X_3d.size:,}")
        print(f"  Rango: [{self.X_3d.min():.6f}, {self.X_3d.max():.6f}]")
        print(f"  Media: {self.X_3d.mean():.6f}")
        print(f"  Std: {self.X_3d.std():.6f}")
        print()

        if self.y is not None:
            print("VECTOR TARGET:")
            print(f"  y ∈ ℝ^{len(self.y)}")
            print(f"  Rango: [{self.y.min():.6f}, {self.y.max():.6f}]")
            print(f"  Media: {self.y.mean():.6f}")
            print(f"  Std: {self.y.std():.6f}")
            print()

        print("="*70)

    def dividir_train_test(self,
                          test_size: float = 0.2) -> Tuple:
        """
        Divide el tensor en conjuntos de entrenamiento y prueba.

        Args:
            test_size: Proporción para test

        Returns:
            (X_train, X_test, y_train, y_test)

        Note:
            SIEMPRE respeta el orden temporal (NO shuffle en series temporales)
        """
        if self.X_3d is None or self.y is None:
            raise ValueError("Primero crea X_3d e y")

        n = len(self.X_3d)
        split_idx = int(n * (1 - test_size))

        X_train = self.X_3d[:split_idx]
        X_test = self.X_3d[split_idx:]
        y_train = self.y[:split_idx]
        y_test = self.y[split_idx:]

        logger.info(f"División temporal del tensor 3D:")
        logger.info(f"  Train: {len(X_train):,} secuencias")
        logger.info(f"  Test:  {len(X_test):,} secuencias")

        return X_train, X_test, y_train, y_test

    def obtener_estadisticas(self) -> Dict:
        """
        Calcula estadísticas del tensor 3D.

        Returns:
            Diccionario con estadísticas
        """
        if self.X_3d is None:
            raise ValueError("Primero crea las secuencias")

        stats = {
            'shape': self.X_3d.shape,
            'n_samples': self.X_3d.shape[0],
            'lookback': self.X_3d.shape[1],
            'n_features': self.X_3d.shape[2],
            'memoria_mb': self.X_3d.nbytes / 1024**2,
            'n_valores_totales': self.X_3d.size,
            'min': self.X_3d.min(),
            'max': self.X_3d.max(),
            'mean': self.X_3d.mean(),
            'std': self.X_3d.std(),
        }

        return stats


def ejemplo_uso():
    """
    Ejemplo completo de uso del tensor 3D.
    """
    print("="*70)
    print("EJEMPLO: TENSOR 3D PARA MODELOS SECUENCIALES")
    print("="*70)
    print()

    # Buscar archivo de features
    features_dir = Path(__file__).parent.parent / 'datos' / 'features'

    if not features_dir.exists():
        print("⚠️  No se encontró directorio de features")
        print("   Ejecuta primero: aplicacion_sistematica.py")
        return

    csv_files = list(features_dir.glob('*_features.csv'))

    if not csv_files:
        print("⚠️  No se encontraron archivos de features")
        return

    filepath = csv_files[0]
    print(f"📂 Usando archivo: {filepath.name}")
    print()

    # 1. Cargar matriz 2D
    print("PASO 1: Cargar matriz 2D")
    print("-" * 70)
    matriz = cargar_desde_csv(filepath)
    matriz.construir_matriz(dropna=True)
    print()

    # 2. Crear tensor 3D
    print("PASO 2: Crear tensor 3D con ventanas deslizantes")
    print("-" * 70)
    tensor = Tensor3DSecuencial(matriz)

    # Crear secuencias con lookback de 50 barras
    X_3d, y = tensor.crear_secuencias(
        lookback=50,
        horizonte=1,
        step=1
    )
    print()

    # 3. Mostrar estructura
    print("PASO 3: Visualizar estructura del tensor")
    print("-" * 70)
    tensor.mostrar_estructura()

    # 4. Dividir train/test
    print("\nPASO 4: Dividir en train/test")
    print("-" * 70)
    X_train, X_test, y_train, y_test = tensor.dividir_train_test(test_size=0.2)
    print()

    print("\n" + "="*70)
    print("💡 USO CON MODELOS SECUENCIALES")
    print("="*70)
    print()

    print("# EJEMPLO 1: LSTM (Keras/TensorFlow)")
    print("-" * 70)
    print("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Construir modelo LSTM
modelo = Sequential([
    LSTM(128, return_sequences=True, input_shape=(lookback, n_features)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

modelo.compile(optimizer='adam', loss='mse')
modelo.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)
    """)

    print("\n# EJEMPLO 2: CNN 1D (Keras/TensorFlow)")
    print("-" * 70)
    print("""
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

modelo = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(lookback, n_features)),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])
    """)

    print("\n# EJEMPLO 3: Transformer (PyTorch)")
    print("-" * 70)
    print("""
import torch
import torch.nn as nn

class TransformerPredictor(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x[:, -1, :])  # Última posición temporal

# Convertir a PyTorch
X_train_torch = torch.FloatTensor(X_train)
y_train_torch = torch.FloatTensor(y_train)
    """)

    print("\n" + "="*70)
    print("VENTAJAS DEL TENSOR 3D:")
    print("-" * 70)
    print("✓ Captura patrones temporales (tendencias, reversiones)")
    print("✓ El modelo ve el CONTEXTO completo de L barras")
    print("✓ Compatible con CNN, LSTM, GRU, Transformer")
    print("✓ No requiere feature engineering manual de tendencias")
    print("✓ Los datos hablan: el modelo encuentra los patrones")
    print("="*70)


if __name__ == '__main__':
    ejemplo_uso()
