"""
TENSOR 4D: MULTI-ACTIVO
=======================

Estructura tensorial para analizar RELACIONES ENTRE ACTIVOS.
Permite a los modelos aprender correlaciones, divergencias, y arbitraje
estadístico entre múltiples pares de divisas.

PARA ANALIZAR RELACIONES ENTRE ACTIVOS:

TENSOR: (Samples × Activos × Lookback × Transformaciones)

X ∈ ℝⁿˣᵃˣᴸˣᵐ

n = samples
a = número de activos (ej: 6 pares)
L = lookback
m = transformaciones


VISUALIZACIÓN 3D:

         ┌─────────────────────────────┐
        /│                             │
       / │      EUR/USD                │
      /  │    ┌─────────────┐          │
     /   │   /             /│          │
    /    │  /  Lookback   / │          │
   /     │ /             /  │          │
  /      │/─────────────/   │          │
 /       ├─────────────────────────────┤
│        │      GBP/USD                │
│        │    ┌─────────────┐          │
│ Assets │   /             /│          │
│        │  / Transforms  / │          │
│        │ /             /  │          │
│        │/─────────────/   │          │
│        ├─────────────────────────────┤
│        │      USD/JPY   ...          │
└────────┴─────────────────────────────┘


CADA SAMPLE contiene:
- Historial de TODOS los activos simultáneamente
- Permite al modelo ver correlaciones dinámicas
- Captura relaciones inter-mercado


EJEMPLO:
Si tenemos 6 pares, L=50, m=762 transformaciones:
- Cada sample es un tensor de (6, 50, 762)
- Si tenemos 30,000 barras, obtenemos ~29,950 samples
- Tensor final: (29950, 6, 50, 762) ≈ 25 GB

Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from datetime import datetime

# Importar tensor 3D
sys.path.insert(0, str(Path(__file__).parent))
from tensor_3d_modelos_secuenciales import Tensor3DSecuencial
from matriz_2d_transformaciones_tiempo import MatrizTransformacionesTiempo, cargar_desde_csv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Tensor4DMultiActivo:
    """
    Clase para manejar tensores 4D: (samples × activos × lookback × features)

    Permite analizar múltiples activos simultáneamente para capturar
    relaciones inter-mercado.
    """

    def __init__(self, lista_pares: List[str], features_dir: Path):
        """
        Inicializa el tensor 4D desde múltiples pares.

        Args:
            lista_pares: Lista de nombres de pares (ej: ['EUR_USD', 'GBP_USD'])
            features_dir: Directorio con archivos de features
        """
        self.lista_pares = lista_pares
        self.features_dir = features_dir
        self.matrices_2d = {}  # Diccionario: par -> MatrizTransformacionesTiempo
        self.X_4d = None       # Tensor 4D (n, a, L, m)
        self.y = None          # Target (n,)
        self.lookback = None
        self.n_samples = None
        self.n_activos = len(lista_pares)
        self.n_features = None
        self.timestamps = None

        logger.info(f"Inicializando tensor 4D para {self.n_activos} activos: {lista_pares}")

    def cargar_matrices(self, timeframe: str = 'H1'):
        """
        Carga matrices 2D de todos los pares.

        Args:
            timeframe: Timeframe de datos (H1, H4, D, etc.)
        """
        logger.info(f"Cargando datos de {self.n_activos} pares...")

        for par in self.lista_pares:
            filepath = self.features_dir / f"{par}_{timeframe}_features.csv"

            if not filepath.exists():
                logger.warning(f"⚠️  No se encontró: {filepath}")
                continue

            logger.info(f"  Cargando {par}...")
            matriz = cargar_desde_csv(filepath)
            matriz.construir_matriz(dropna=True)
            self.matrices_2d[par] = matriz

        logger.info(f"✓ Cargados {len(self.matrices_2d)} pares")

        if len(self.matrices_2d) == 0:
            raise ValueError("No se pudo cargar ningún par")

    def alinear_temporalmente(self):
        """
        Alinea todos los pares temporalmente.

        Encuentra el período común donde todos los pares tienen datos
        y recorta las matrices para que tengan el mismo índice temporal.
        """
        logger.info("Alineando temporalmente los pares...")

        if len(self.matrices_2d) == 0:
            raise ValueError("Primero carga las matrices con cargar_matrices()")

        # Encontrar período común
        indices_comunes = None

        for par, matriz in self.matrices_2d.items():
            if indices_comunes is None:
                indices_comunes = set(matriz.timestamps)
            else:
                indices_comunes = indices_comunes.intersection(set(matriz.timestamps))

        indices_comunes = sorted(list(indices_comunes))
        logger.info(f"  Período común: {len(indices_comunes):,} barras")
        logger.info(f"  Desde: {indices_comunes[0]}")
        logger.info(f"  Hasta: {indices_comunes[-1]}")

        # Recortar todas las matrices al período común
        for par, matriz in self.matrices_2d.items():
            # Encontrar índices válidos
            mask = matriz.timestamps.isin(indices_comunes)
            matriz.X = matriz.X[mask]
            matriz.timestamps = matriz.timestamps[mask]

            logger.info(f"  {par}: {matriz.X.shape[0]:,} observaciones alineadas")

        self.timestamps = self.matrices_2d[self.lista_pares[0]].timestamps

        logger.info("✓ Alineación temporal completa")

    def crear_tensor_4d(self,
                       lookback: int = 50,
                       horizonte: int = 1,
                       step: int = 1,
                       par_target: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea tensor 4D con ventanas deslizantes para todos los activos.

        Args:
            lookback: Número de barras de historia (L)
            horizonte: Barras hacia adelante para predecir
            step: Paso entre ventanas
            par_target: Par para crear el target (si None, usa el primero)

        Returns:
            (X_4d, y) donde:
                X_4d.shape = (n_samples, n_activos, lookback, n_features)
                y.shape = (n_samples,)
        """
        logger.info(f"Creando tensor 4D con lookback={lookback}, horizonte={horizonte}...")

        if len(self.matrices_2d) == 0:
            raise ValueError("Primero carga y alinea las matrices")

        # Validar que todos los pares tengan el mismo número de observaciones
        n_obs_list = [m.X.shape[0] for m in self.matrices_2d.values()]
        if len(set(n_obs_list)) > 1:
            raise ValueError("Los pares no están alineados temporalmente. Ejecuta alinear_temporalmente()")

        n_obs = n_obs_list[0]
        n_features = self.matrices_2d[self.lista_pares[0]].X.shape[1]

        # Calcular número de samples
        n_samples = (n_obs - lookback - horizonte + 1) // step

        if n_samples <= 0:
            raise ValueError(f"No hay suficientes datos. Necesitas al menos {lookback + horizonte} observaciones")

        # Inicializar tensor 4D
        # Shape: (n_samples, n_activos, lookback, n_features)
        X_4d = np.zeros((n_samples, self.n_activos, lookback, n_features), dtype=np.float32)

        logger.info(f"Generando tensor 4D: ({n_samples}, {self.n_activos}, {lookback}, {n_features})...")

        # Crear ventanas deslizantes para cada activo
        for activo_idx, par in enumerate(self.lista_pares):
            matriz = self.matrices_2d[par]

            for i in range(n_samples):
                start_idx = i * step
                end_idx = start_idx + lookback

                # Ventana de lookback para este activo
                X_4d[i, activo_idx, :, :] = matriz.X[start_idx:end_idx]

        # Crear vector target (usando el par especificado o el primero)
        if par_target is None:
            par_target = self.lista_pares[0]

        logger.info(f"Creando target desde {par_target}...")

        # Placeholder: En producción, calcular retornos desde precios
        y = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            target_idx = i * step + lookback + horizonte - 1
            if target_idx < n_obs:
                # Placeholder: usar primera transformación del par target
                y[i] = self.matrices_2d[par_target].X[target_idx, 0]

        self.X_4d = X_4d
        self.y = y
        self.lookback = lookback
        self.n_samples = n_samples
        self.n_features = n_features

        logger.info(f"✓ Tensor 4D creado: {X_4d.shape}")
        logger.info(f"  n_samples = {n_samples:,}")
        logger.info(f"  n_activos = {self.n_activos}")
        logger.info(f"  lookback = {lookback}")
        logger.info(f"  n_features = {n_features}")
        logger.info(f"  Tamaño en memoria: {X_4d.nbytes / 1024**3:.2f} GB")

        return X_4d, y

    def mostrar_estructura(self):
        """
        Muestra la estructura del tensor 4D de forma visual.
        """
        if self.X_4d is None:
            raise ValueError("Primero crea el tensor con crear_tensor_4d()")

        n, a, L, m = self.X_4d.shape

        print("="*70)
        print("TENSOR 4D MULTI-ACTIVO")
        print("="*70)
        print()
        print("DIMENSIONES:")
        print(f"  X ∈ ℝ^({n} × {a} × {L} × {m})")
        print(f"  n_samples = {n:,}")
        print(f"  n_activos = {a}")
        print(f"  lookback (L) = {L}")
        print(f"  n_transformaciones (m) = {m}")
        print()

        print("INTERPRETACIÓN:")
        print(f"  Cada sample contiene {a} activos simultáneamente")
        print(f"  Cada activo ve {L} barras de historia")
        print(f"  Cada barra tiene {m} transformaciones")
        print(f"  Total: {n:,} samples de contexto multi-activo")
        print()

        print("ACTIVOS INCLUIDOS:")
        for i, par in enumerate(self.lista_pares):
            print(f"  [{i}] {par}")
        print()

        print("ESTRUCTURA DEL TENSOR:")
        print()
        print("         ┌─────────────────────────────┐")
        print("        /│                             │")
        print(f"       / │      {self.lista_pares[0]:<20}│")
        print("      /  │    ┌─────────────┐          │")
        print("     /   │   /             /│          │")
        print(f"    /    │  /  L={L:<4}     / │          │")
        print(f"   /     │ /   m={m:<4}  /  │          │")
        print("  /      │/─────────────/   │          │")
        print(" /       ├─────────────────────────────┤")
        if a > 1:
            print(f"│        │      {self.lista_pares[1]:<20}│")
            print("│        │    ┌─────────────┐          │")
            print(f"│ a={a:<2}   │   /             /│          │")
            print(f"│        │  / Transforms  / │          │")
            print("│        │ /             /  │          │")
            print("│        │/─────────────/   │          │")
        if a > 2:
            print("│        ├─────────────────────────────┤")
            print(f"│        │      {self.lista_pares[2] if a > 2 else '...':<16}...  │")
        print("└────────┴─────────────────────────────┘")
        print()

        print("EJEMPLO DE UN SAMPLE (sample 0):")
        print(f"  X_4d[0] tiene shape: ({a}, {L}, {m})")
        print()

        # Mostrar estadísticas de cada activo en el sample 0
        sample_0 = self.X_4d[0]
        for i, par in enumerate(self.lista_pares[:3]):  # Primeros 3 activos
            datos_activo = sample_0[i]  # Shape: (L, m)
            print(f"  Activo {i} ({par}):")
            print(f"    Shape: ({L}, {m})")
            print(f"    Rango: [{datos_activo.min():.4f}, {datos_activo.max():.4f}]")
            print(f"    Media: {datos_activo.mean():.4f}")
            print()

        print("ESTADÍSTICAS GLOBALES:")
        print(f"  Tamaño en memoria: {self.X_4d.nbytes / 1024**3:.2f} GB")
        print(f"  Valores totales: {self.X_4d.size:,}")
        print(f"  Rango: [{self.X_4d.min():.6f}, {self.X_4d.max():.6f}]")
        print(f"  Media: {self.X_4d.mean():.6f}")
        print(f"  Std: {self.X_4d.std():.6f}")
        print()

        if self.y is not None:
            print("VECTOR TARGET:")
            print(f"  y ∈ ℝ^{len(self.y)}")
            print(f"  Rango: [{self.y.min():.6f}, {self.y.max():.6f}]")
            print(f"  Media: {self.y.mean():.6f}")
            print()

        print("="*70)

    def dividir_train_test(self, test_size: float = 0.2) -> Tuple:
        """
        Divide el tensor 4D en conjuntos de entrenamiento y prueba.

        Args:
            test_size: Proporción para test

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        if self.X_4d is None or self.y is None:
            raise ValueError("Primero crea X_4d e y")

        n = len(self.X_4d)
        split_idx = int(n * (1 - test_size))

        X_train = self.X_4d[:split_idx]
        X_test = self.X_4d[split_idx:]
        y_train = self.y[:split_idx]
        y_test = self.y[split_idx:]

        logger.info(f"División temporal del tensor 4D:")
        logger.info(f"  Train: {len(X_train):,} samples")
        logger.info(f"  Test:  {len(X_test):,} samples")

        return X_train, X_test, y_train, y_test

    def obtener_estadisticas(self) -> Dict:
        """
        Calcula estadísticas del tensor 4D.

        Returns:
            Diccionario con estadísticas
        """
        if self.X_4d is None:
            raise ValueError("Primero crea el tensor")

        stats = {
            'shape': self.X_4d.shape,
            'n_samples': self.X_4d.shape[0],
            'n_activos': self.X_4d.shape[1],
            'lookback': self.X_4d.shape[2],
            'n_features': self.X_4d.shape[3],
            'memoria_gb': self.X_4d.nbytes / 1024**3,
            'n_valores_totales': self.X_4d.size,
            'min': self.X_4d.min(),
            'max': self.X_4d.max(),
            'mean': self.X_4d.mean(),
            'std': self.X_4d.std(),
        }

        return stats


def ejemplo_uso():
    """
    Ejemplo completo de uso del tensor 4D multi-activo.
    """
    print("="*70)
    print("EJEMPLO: TENSOR 4D MULTI-ACTIVO")
    print("="*70)
    print()

    # Definir pares a usar
    pares = ['EUR_USD', 'GBP_USD', 'USD_JPY']
    features_dir = Path(__file__).parent.parent / 'datos' / 'features'

    if not features_dir.exists():
        print("⚠️  No se encontró directorio de features")
        print("   Ejecuta primero: aplicacion_sistematica.py para varios pares")
        return

    # Verificar que existan los archivos
    archivos_disponibles = []
    for par in pares:
        filepath = features_dir / f"{par}_H1_features.csv"
        if filepath.exists():
            archivos_disponibles.append(par)

    if len(archivos_disponibles) == 0:
        print("⚠️  No se encontraron archivos de features para los pares especificados")
        print("   Genera features para múltiples pares primero")
        return

    pares = archivos_disponibles[:3]  # Máximo 3 para demo
    print(f"📊 Usando {len(pares)} pares: {pares}")
    print()

    # 1. Inicializar tensor 4D
    print("PASO 1: Inicializar tensor 4D multi-activo")
    print("-" * 70)
    tensor = Tensor4DMultiActivo(pares, features_dir)
    print()

    # 2. Cargar matrices de todos los pares
    print("PASO 2: Cargar matrices de todos los pares")
    print("-" * 70)
    tensor.cargar_matrices(timeframe='H1')
    print()

    # 3. Alinear temporalmente
    print("PASO 3: Alinear temporalmente los pares")
    print("-" * 70)
    tensor.alinear_temporalmente()
    print()

    # 4. Crear tensor 4D
    print("PASO 4: Crear tensor 4D con ventanas deslizantes")
    print("-" * 70)
    X_4d, y = tensor.crear_tensor_4d(
        lookback=20,  # Lookback pequeño para demo
        horizonte=1,
        step=1
    )
    print()

    # 5. Mostrar estructura
    print("PASO 5: Visualizar estructura del tensor 4D")
    print("-" * 70)
    tensor.mostrar_estructura()

    print("\n" + "="*70)
    print("💡 USO CON MODELOS MULTI-INPUT")
    print("="*70)
    print()

    print("# EJEMPLO 1: CNN 3D (TensorFlow/Keras)")
    print("-" * 70)
    print("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense

# X_4d.shape = (n_samples, n_activos, lookback, n_features)
# Reshape para Conv3D: (n_samples, n_activos, lookback, n_features, 1)
X_train_5d = X_train[..., np.newaxis]

modelo = Sequential([
    Conv3D(32, (2, 3, 5), activation='relu',
           input_shape=(n_activos, lookback, n_features, 1)),
    MaxPooling3D((1, 2, 2)),
    Conv3D(64, (1, 3, 5), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])
    """)

    print("\n# EJEMPLO 2: Multi-Input LSTM (Keras)")
    print("-" * 70)
    print("""
from tensorflow.keras.layers import Input, LSTM, Concatenate

# Crear input por cada activo
inputs = []
lstms = []

for i in range(n_activos):
    inp = Input(shape=(lookback, n_features), name=f'activo_{i}')
    inputs.append(inp)

    lstm_out = LSTM(64)(inp)
    lstms.append(lstm_out)

# Concatenar todos los activos
merged = Concatenate()(lstms)
output = Dense(1)(merged)

modelo = Model(inputs=inputs, outputs=output)

# Preparar datos: lista de arrays por cada activo
X_train_list = [X_train[:, i, :, :] for i in range(n_activos)]
    """)

    print("\n# EJEMPLO 3: Attention entre Activos (PyTorch)")
    print("-" * 70)
    print("""
import torch
import torch.nn as nn

class MultiAssetAttention(nn.Module):
    def __init__(self, n_features, d_model=128):
        super().__init__()
        self.asset_encoder = nn.LSTM(n_features, d_model, batch_first=True)
        self.attention = nn.MultiheadAttention(d_model, num_heads=4)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, n_activos, lookback, n_features)
        batch, n_assets, L, m = x.shape

        # Codificar cada activo
        encoded = []
        for i in range(n_assets):
            _, (h, _) = self.asset_encoder(x[:, i, :, :])
            encoded.append(h[-1])

        # Stack: (batch, n_assets, d_model)
        encoded = torch.stack(encoded, dim=1)

        # Cross-attention entre activos
        attn_out, _ = self.attention(encoded, encoded, encoded)

        # Agregación y predicción
        pooled = attn_out.mean(dim=1)
        return self.fc(pooled)
    """)

    print("\n" + "="*70)
    print("VENTAJAS DEL TENSOR 4D:")
    print("-" * 70)
    print("✓ Captura correlaciones dinámicas entre activos")
    print("✓ El modelo ve divergencias y convergencias")
    print("✓ Permite estrategias de arbitraje estadístico")
    print("✓ Aprovecha información de mercados relacionados")
    print("✓ Más robusto ante ruido en un solo activo")
    print("="*70)


if __name__ == '__main__':
    ejemplo_uso()
