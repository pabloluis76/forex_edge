"""
MATRIZ 2D: TRANSFORMACIONES × TIEMPO
=====================================

Representación matricial de las transformaciones generadas para
machine learning en forex.

ESTRUCTURA BÁSICA:

                        Tiempo →
              t₁    t₂    t₃    t₄   ...   tₙ
            ┌─────┬─────┬─────┬─────┬─────┬─────┐
Trans_1     │ x₁₁ │ x₁₂ │ x₁₃ │ x₁₄ │ ... │ x₁ₙ │
            ├─────┼─────┼─────┼─────┼─────┼─────┤
Trans_2     │ x₂₁ │ x₂₂ │ x₂₃ │ x₂₄ │ ... │ x₂ₙ │
            ├─────┼─────┼─────┼─────┼─────┼─────┤
Trans_3     │ x₃₁ │ x₃₂ │ x₃₃ │ x₃₄ │ ... │ x₃ₙ │
            ├─────┼─────┼─────┼─────┼─────┼─────┤
    ⋮       │  ⋮  │  ⋮  │  ⋮  │  ⋮  │  ⋮  │  ⋮  │
            ├─────┼─────┼─────┼─────┼─────┼─────┤
Trans_m     │ xₘ₁ │ xₘ₂ │ xₘ₃ │ xₘ₄ │ ... │ xₘₙ │
            └─────┴─────┴─────┴─────┴─────┴─────┘

NOTACIÓN MATEMÁTICA:

X ∈ ℝᵐˣⁿ    Matriz de transformaciones (features)
y ∈ ℝⁿ      Vector de retornos futuros (target)

DIMENSIONES TÍPICAS:
- m ≈ 2,000-3,000 transformaciones
- n ≈ 100,000-500,000 observaciones temporales

EJEMPLO (5 años de datos H1):
- m = 762 transformaciones (EUR_USD)
- n = 31,094 velas
- Tamaño: 762 × 31,094 = 23,693,628 valores ≈ 180 MB (float64)

Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MatrizTransformacionesTiempo:
    """
    Clase para manejar la matriz 2D de transformaciones × tiempo.

    Esta es la estructura fundamental para machine learning:
    - Filas: Transformaciones (features)
    - Columnas: Tiempo (observaciones)
    """

    def __init__(self, df_features: pd.DataFrame):
        """
        Inicializa la matriz desde un DataFrame de features.

        Args:
            df_features: DataFrame con índice temporal y columnas = transformaciones
                        (output de aplicacion_sistematica.py)
        """
        self.df = df_features
        self.X = None  # Matriz de features
        self.y = None  # Vector target
        self.feature_names = None
        self.timestamps = None

        logger.info(f"Matriz inicializada: {df_features.shape}")
        logger.info(f"Período: {df_features.index[0]} a {df_features.index[-1]}")

    def construir_matriz(self, dropna: bool = True) -> np.ndarray:
        """
        Construye la matriz X ∈ ℝᵐˣⁿ.

        Args:
            dropna: Si True, elimina filas con NaN

        Returns:
            Matriz numpy de shape (n_observaciones, n_features)

        Note:
            En machine learning, la convención es X.shape = (n_samples, n_features)
            Aquí n_samples = n (tiempo) y n_features = m (transformaciones)
        """
        logger.info("Construyendo matriz X...")

        if dropna:
            df_clean = self.df.dropna()
            pct_removed = (1 - len(df_clean) / len(self.df)) * 100
            logger.info(f"Eliminadas {pct_removed:.1f}% filas con NaN")
            self.df = df_clean

        # Guardar metadatos
        self.feature_names = list(self.df.columns)
        self.timestamps = self.df.index

        # Convertir a numpy array
        # X.shape = (n_observaciones, n_transformaciones) = (n, m)
        self.X = self.df.values

        logger.info(f"✓ Matriz X construida: {self.X.shape}")
        logger.info(f"  n_observaciones (n) = {self.X.shape[0]:,}")
        logger.info(f"  n_transformaciones (m) = {self.X.shape[1]:,}")
        logger.info(f"  Tamaño en memoria: {self.X.nbytes / 1024**2:.1f} MB")

        return self.X

    def crear_target(self,
                     precio_col: str = 'close',
                     horizonte: int = 1,
                     tipo: str = 'retorno') -> np.ndarray:
        """
        Crea el vector target y ∈ ℝⁿ.

        Args:
            precio_col: Columna de precio a usar (debe existir en datos originales)
            horizonte: Número de períodos hacia adelante
            tipo: 'retorno' para retorno futuro, 'direccion' para signo

        Returns:
            Vector target de shape (n_observaciones,)
        """
        logger.info(f"Creando vector target (horizonte={horizonte}, tipo={tipo})...")

        # Nota: Necesitamos los precios originales
        # Por ahora, asumimos que están disponibles
        # En práctica, cargaríamos desde datos/ohlc/

        if precio_col in self.df.columns:
            precios = self.df[precio_col]
        else:
            logger.warning(f"Columna {precio_col} no encontrada. Usando primera columna.")
            # Buscar columna que contenga 'close'
            close_cols = [col for col in self.df.columns if 'close' in col.lower()]
            if close_cols:
                logger.info(f"Usando columna: {close_cols[0]}")
                precios = self.df[close_cols[0]]
            else:
                raise ValueError("No se encontró columna de precios")

        if tipo == 'retorno':
            # y_t = (P_{t+h} - P_t) / P_t
            self.y = precios.pct_change(horizonte).shift(-horizonte)
        elif tipo == 'log_retorno':
            # y_t = ln(P_{t+h} / P_t)
            self.y = np.log(precios / precios.shift(horizonte)).shift(-horizonte)
        elif tipo == 'direccion':
            # y_t = sign(P_{t+h} - P_t)
            retornos = precios.pct_change(horizonte).shift(-horizonte)
            self.y = np.sign(retornos)
        else:
            raise ValueError(f"Tipo desconocido: {tipo}")

        # Eliminar valores NaN al final (por el shift)
        valid_idx = ~self.y.isna()
        self.y = self.y[valid_idx].values
        self.X = self.X[valid_idx]
        self.timestamps = self.timestamps[valid_idx]

        logger.info(f"✓ Vector y creado: {self.y.shape}")
        logger.info(f"  min={self.y.min():.6f}, max={self.y.max():.6f}")
        logger.info(f"  mean={self.y.mean():.6f}, std={self.y.std():.6f}")

        return self.y

    def obtener_estadisticas(self) -> Dict:
        """
        Calcula estadísticas de la matriz X.

        Returns:
            Diccionario con estadísticas
        """
        if self.X is None:
            raise ValueError("Primero construye la matriz con construir_matriz()")

        stats = {
            'shape': self.X.shape,
            'n_observaciones': self.X.shape[0],
            'n_transformaciones': self.X.shape[1],
            'memoria_mb': self.X.nbytes / 1024**2,
            'n_valores_totales': self.X.size,
            'n_nan': np.isnan(self.X).sum(),
            'n_inf': np.isinf(self.X).sum(),
            'min': np.nanmin(self.X),
            'max': np.nanmax(self.X),
            'mean': np.nanmean(self.X),
            'std': np.nanstd(self.X),
        }

        return stats

    def mostrar_estructura(self):
        """
        Muestra la estructura de la matriz de forma visual.
        """
        if self.X is None:
            raise ValueError("Primero construye la matriz con construir_matriz()")

        n, m = self.X.shape

        print("="*70)
        print("ESTRUCTURA MATRICIAL 2D: TRANSFORMACIONES × TIEMPO")
        print("="*70)
        print()
        print("DIMENSIONES:")
        print(f"  X ∈ ℝ^({n} × {m})")
        print(f"  n_observaciones (tiempo) = {n:,}")
        print(f"  n_transformaciones (features) = {m:,}")
        print()

        print("REPRESENTACIÓN VISUAL:")
        print()
        print("                        Tiempo →")
        print("              t₁    t₂    t₃    t₄   ...   tₙ")
        print("            ┌─────┬─────┬─────┬─────┬─────┬─────┐")

        # Mostrar primeras 3 transformaciones
        for i in range(min(3, m)):
            feature_name = self.feature_names[i] if self.feature_names else f"Trans_{i+1}"
            feature_name = feature_name[:12].ljust(12)

            valores = [f"{self.X[j, i]:.2f}" if j < n else "..."
                      for j in range(min(4, n))]
            valores_str = " │ ".join(f"{v:>5}" for v in valores)

            print(f"{feature_name}│ {valores_str} │ ... │")
            if i < min(3, m) - 1:
                print("            ├─────┼─────┼─────┼─────┼─────┼─────┤")

        if m > 3:
            print("        ⋮   │  ⋮  │  ⋮  │  ⋮  │  ⋮  │  ⋮  │  ⋮  │")
            print("            ├─────┼─────┼─────┼─────┼─────┼─────┤")

            # Última transformación
            feature_name = self.feature_names[-1] if self.feature_names else f"Trans_{m}"
            feature_name = feature_name[:12].ljust(12)
            valores = [f"{self.X[j, -1]:.2f}" if j < n else "..."
                      for j in range(min(4, n))]
            valores_str = " │ ".join(f"{v:>5}" for v in valores)
            print(f"{feature_name}│ {valores_str} │ ... │")

        print("            └─────┴─────┴─────┴─────┴─────┴─────┘")
        print()

        print("ESTADÍSTICAS:")
        stats = self.obtener_estadisticas()
        print(f"  Tamaño en memoria: {stats['memoria_mb']:.1f} MB")
        print(f"  Valores totales: {stats['n_valores_totales']:,}")
        print(f"  Valores NaN: {stats['n_nan']:,}")
        print(f"  Valores Inf: {stats['n_inf']:,}")
        print(f"  Rango: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Media: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print()

        if self.y is not None:
            print("VECTOR TARGET:")
            print(f"  y ∈ ℝ^{len(self.y)}")
            print(f"  Rango: [{self.y.min():.6f}, {self.y.max():.6f}]")
            print(f"  Media: {self.y.mean():.6f}")
            print(f"  Std: {self.y.std():.6f}")
            print()

        print("PERÍODO TEMPORAL:")
        print(f"  Inicio: {self.timestamps[0]}")
        print(f"  Fin: {self.timestamps[-1]}")
        print(f"  Duración: {self.timestamps[-1] - self.timestamps[0]}")
        print()

        print("="*70)

    def dividir_train_test(self,
                          test_size: float = 0.2,
                          shuffle: bool = False) -> Tuple:
        """
        Divide la matriz en conjuntos de entrenamiento y prueba.

        Args:
            test_size: Proporción para test
            shuffle: Si True, mezcla aleatoriamente (NO recomendado para series temporales)

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        if self.X is None or self.y is None:
            raise ValueError("Primero construye X e y")

        n = len(self.X)
        split_idx = int(n * (1 - test_size))

        if shuffle:
            logger.warning("⚠️  Shuffle=True: Los datos temporales se mezclarán aleatoriamente")
            indices = np.random.permutation(n)
            X_train = self.X[indices[:split_idx]]
            X_test = self.X[indices[split_idx:]]
            y_train = self.y[indices[:split_idx]]
            y_test = self.y[indices[split_idx:]]
        else:
            # Split temporal (respeta el orden del tiempo)
            X_train = self.X[:split_idx]
            X_test = self.X[split_idx:]
            y_train = self.y[:split_idx]
            y_test = self.y[split_idx:]

        logger.info(f"División temporal:")
        logger.info(f"  Train: {len(X_train):,} observaciones ({self.timestamps[0]} a {self.timestamps[split_idx-1]})")
        logger.info(f"  Test:  {len(X_test):,} observaciones ({self.timestamps[split_idx]} a {self.timestamps[-1]})")

        return X_train, X_test, y_train, y_test


def cargar_desde_csv(filepath: Path) -> MatrizTransformacionesTiempo:
    """
    Carga features desde CSV generado por aplicacion_sistematica.py.

    Args:
        filepath: Path al archivo CSV

    Returns:
        Objeto MatrizTransformacionesTiempo
    """
    logger.info(f"Cargando features desde: {filepath}")

    df = pd.read_csv(filepath, index_col='time', parse_dates=True)

    logger.info(f"✓ Cargado: {df.shape[0]:,} observaciones × {df.shape[1]:,} features")
    logger.info(f"  Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    return MatrizTransformacionesTiempo(df)


def ejemplo_uso():
    """
    Ejemplo de uso de la matriz 2D.
    """
    print("="*70)
    print("EJEMPLO: MATRIZ 2D TRANSFORMACIONES × TIEMPO")
    print("="*70)
    print()

    # Buscar archivo de features
    features_dir = Path(__file__).parent.parent / 'datos' / 'features'

    if not features_dir.exists():
        print("⚠️  No se encontró directorio de features")
        print("   Ejecuta primero: aplicacion_sistematica.py")
        return

    # Buscar primer archivo CSV
    csv_files = list(features_dir.glob('*_features.csv'))

    if not csv_files:
        print("⚠️  No se encontraron archivos de features")
        print("   Ejecuta primero: aplicacion_sistematica.py")
        return

    filepath = csv_files[0]
    print(f"📂 Usando archivo: {filepath.name}")
    print()

    # Cargar matriz
    matriz = cargar_desde_csv(filepath)

    # Construir matriz X
    matriz.construir_matriz(dropna=True)

    # Mostrar estructura
    matriz.mostrar_estructura()

    print("\n💡 USO PARA MACHINE LEARNING:")
    print("-" * 70)
    print("# 1. Construir matriz y target")
    print("matriz = cargar_desde_csv('datos/features/EUR_USD_H1_features.csv')")
    print("X = matriz.construir_matriz(dropna=True)")
    print("y = matriz.crear_target(horizonte=1, tipo='retorno')")
    print()
    print("# 2. Dividir train/test (respetando tiempo)")
    print("X_train, X_test, y_train, y_test = matriz.dividir_train_test(test_size=0.2)")
    print()
    print("# 3. Entrenar modelo")
    print("from sklearn.linear_model import Ridge")
    print("modelo = Ridge().fit(X_train, y_train)")
    print()
    print("# 4. Evaluar")
    print("score = modelo.score(X_test, y_test)")
    print("="*70)


if __name__ == '__main__':
    ejemplo_uso()
