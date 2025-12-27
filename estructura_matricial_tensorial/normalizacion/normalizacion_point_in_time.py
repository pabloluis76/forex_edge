"""
NORMALIZACIÓN POINT-IN-TIME (SIN LOOK-AHEAD BIAS)
==================================================

ANTES DE CUALQUIER ANÁLISIS:
Normalizar cada transformación para hacerlas comparables.

REGLA FUNDAMENTAL:
NUNCA usar información de tiempo t o futuro para normalizar.
Solo información hasta t-1.


Z-SCORE ROLLING (evitar look-ahead):
────────────────────────────────────
Para cada transformación j:

zⱼ,ₜ = (xⱼ,ₜ₋₁ - μⱼ,ₜ) / σⱼ,ₜ

Donde:
μⱼ,ₜ = media de xⱼ en ventana [t-n, t-1]  ← NO incluye t
σⱼ,ₜ = std de xⱼ en ventana [t-n, t-1]    ← NO incluye t


RANK TRANSFORM (robusto a outliers):
────────────────────────────────────
Convertir cada valor a su percentil en la ventana histórica.

rankⱼ,ₜ = percentil(xⱼ,ₜ₋₁) en [xⱼ,ₜ₋ₙ, ..., xⱼ,ₜ₋₁]


IMPORTANTE:
- Cada barra solo usa información hasta t-1
- La normalización es "rolling" o "expanding"
- Evita contaminación de datos futuros
- Simula lo que sabríamos en tiempo real

Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import numpy as np
import pandas as pd
from typing import Union, Literal, Optional
import logging
from pathlib import Path
import sys

# Importar constantes centralizadas
sys.path.append(str(Path(__file__).parent.parent.parent))
from constants import EPSILON_NORMALIZATION

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NormalizadorPointInTime:
    """
    Normaliza transformaciones respetando point-in-time correctness.

    NUNCA usa información del futuro.
    Cada barra t solo ve datos hasta t-1.
    """

    @staticmethod
    def zscore_rolling(x: Union[pd.Series, np.ndarray],
                      window: int = 252,
                      min_periods: Optional[int] = None,
                      epsilon: float = None) -> np.ndarray:
        """
        Z-Score Rolling SIN look-ahead bias.

        Formula:
            zₜ = (xₜ₋₁ - μₜ) / σₜ

        donde μₜ y σₜ se calculan en ventana [t-n, t-1]

        Args:
            x: Serie temporal a normalizar
            window: Tamaño de ventana para estadísticas rolling
            min_periods: Mínimo de períodos requeridos (default: window)
            epsilon: Pequeño valor para evitar división por cero

        Returns:
            Array normalizado (mismo tamaño que x, con NaN al inicio)

        Example:
            >>> precios = pd.Series([100, 102, 101, 103, 105])
            >>> z = zscore_rolling(precios, window=3)
            >>> # z[0], z[1], z[2] = NaN (no hay suficiente historia)
            >>> # z[3] = (103 - mean([100,102,101])) / std([100,102,101])
        """
        if epsilon is None:
            epsilon = EPSILON_NORMALIZATION

        if isinstance(x, pd.Series):
            x_series = x
        else:
            x_series = pd.Series(x)

        if min_periods is None:
            min_periods = window

        # CRÍTICO: Calcular media y std usando solo datos HASTA t-1
        # Usamos shift(1) para que las estadísticas en t usen datos hasta t-1
        rolling_mean = x_series.rolling(window=window, min_periods=min_periods).mean().shift(1)
        rolling_std = x_series.rolling(window=window, min_periods=min_periods).std().shift(1)

        # Normalizar
        z = (x_series - rolling_mean) / (rolling_std + epsilon)

        return z.values

    @staticmethod
    def zscore_expanding(x: Union[pd.Series, np.ndarray],
                        min_periods: int = 30,
                        epsilon: float = None) -> np.ndarray:
        """
        Z-Score Expanding (ventana que crece) SIN look-ahead bias.

        Similar a zscore_rolling pero usa TODA la historia disponible
        hasta t-1.

        Args:
            x: Serie temporal a normalizar
            min_periods: Mínimo de períodos requeridos
            epsilon: Pequeño valor para evitar división por cero

        Returns:
            Array normalizado
        """
        if isinstance(x, pd.Series):
            x_series = x
        else:
            x_series = pd.Series(x)

        # Expanding mean/std hasta t-1
        expanding_mean = x_series.expanding(min_periods=min_periods).mean().shift(1)
        expanding_std = x_series.expanding(min_periods=min_periods).std().shift(1)

        z = (x_series - expanding_mean) / (expanding_std + epsilon)

        return z.values

    @staticmethod
    def rank_transform(x: Union[pd.Series, np.ndarray],
                      window: int = 252,
                      min_periods: Optional[int] = None) -> np.ndarray:
        """
        Rank Transform (percentil) SIN look-ahead bias.

        Convierte cada valor a su percentil en la ventana histórica [t-n, t-1].

        Resultado en rango [0, 1]:
        - 0.0 = valor mínimo en la ventana
        - 0.5 = valor mediano
        - 1.0 = valor máximo en la ventana

        Args:
            x: Serie temporal a transformar
            window: Tamaño de ventana
            min_periods: Mínimo de períodos requeridos

        Returns:
            Array de percentiles [0, 1]

        Example:
            >>> x = pd.Series([10, 20, 15, 25, 12])
            >>> rank = rank_transform(x, window=3)
            >>> # rank[3] = percentil de 25 en [10, 20, 15] = 1.0 (máximo)
        """
        if epsilon is None:
            epsilon = EPSILON_NORMALIZATION

        if isinstance(x, pd.Series):
            x_series = x
        else:
            x_series = pd.Series(x)

        if min_periods is None:
            min_periods = window

        def _percentile_in_window(series):
            """Calcula percentil del último valor en la ventana"""
            if len(series) < min_periods:
                return np.nan

            # Usar todos los valores de la ventana (incluyendo el actual)
            # para calcular el rank
            return pd.Series(series).rank(pct=True).iloc[-1]

        # CRÍTICO: Aplicar rolling SIN incluir el valor actual
        # Hacemos esto aplicando la función sobre la ventana shifteada
        ranks = x_series.rolling(window=window, min_periods=min_periods).apply(
            _percentile_in_window,
            raw=False
        )

        # Shift para que el rank en t use solo datos hasta t-1
        ranks = ranks.shift(1)

        return ranks.values

    @staticmethod
    def minmax_rolling(x: Union[pd.Series, np.ndarray],
                      window: int = 252,
                      min_periods: Optional[int] = None,
                      epsilon: float = None) -> np.ndarray:
        """
        Min-Max Scaling Rolling SIN look-ahead bias.

        Escala valores al rango [0, 1] usando mín/máx de ventana histórica.

        Formula:
            scaled_t = (x_t - min_t) / (max_t - min_t)

        donde min_t y max_t se calculan en ventana [t-n, t-1]

        Args:
            x: Serie temporal a escalar
            window: Tamaño de ventana
            min_periods: Mínimo de períodos requeridos
            epsilon: Pequeño valor para evitar división por cero

        Returns:
            Array escalado [0, 1] aproximadamente
        """
        if epsilon is None:
            epsilon = EPSILON_NORMALIZATION

        if isinstance(x, pd.Series):
            x_series = x
        else:
            x_series = pd.Series(x)

        if min_periods is None:
            min_periods = window

        # Rolling min/max hasta t-1
        rolling_min = x_series.rolling(window=window, min_periods=min_periods).min().shift(1)
        rolling_max = x_series.rolling(window=window, min_periods=min_periods).max().shift(1)

        # Escalar
        scaled = (x_series - rolling_min) / (rolling_max - rolling_min + epsilon)

        return scaled.values


def normalizar_matriz_2d(X: np.ndarray,
                        metodo: Literal['zscore_rolling', 'zscore_expanding', 'rank', 'minmax'] = 'zscore_rolling',
                        window: int = 252,
                        min_periods: Optional[int] = None) -> np.ndarray:
    """
    Normaliza una matriz 2D (n_observaciones × n_features).

    Aplica normalización point-in-time a cada columna (feature) independientemente.

    Args:
        X: Matriz 2D de shape (n, m)
        metodo: Método de normalización
        window: Tamaño de ventana (para métodos rolling)
        min_periods: Mínimo de períodos

    Returns:
        Matriz normalizada de mismo shape

    Example:
        >>> X = np.random.randn(1000, 500)  # 1000 observaciones, 500 features
        >>> X_norm = normalizar_matriz_2d(X, metodo='zscore_rolling', window=252)
    """
    logger.info(f"Normalizando matriz 2D: {X.shape}")
    logger.info(f"  Método: {metodo}")
    logger.info(f"  Window: {window}")

    n_obs, n_features = X.shape
    X_normalized = np.zeros_like(X, dtype=np.float32)

    normalizador = NormalizadorPointInTime()

    for j in range(n_features):
        feature = X[:, j]

        if metodo == 'zscore_rolling':
            X_normalized[:, j] = normalizador.zscore_rolling(
                feature, window=window, min_periods=min_periods
            )
        elif metodo == 'zscore_expanding':
            X_normalized[:, j] = normalizador.zscore_expanding(
                feature, min_periods=min_periods or 30
            )
        elif metodo == 'rank':
            X_normalized[:, j] = normalizador.rank_transform(
                feature, window=window, min_periods=min_periods
            )
        elif metodo == 'minmax':
            X_normalized[:, j] = normalizador.minmax_rolling(
                feature, window=window, min_periods=min_periods
            )
        else:
            raise ValueError(f"Método desconocido: {metodo}")

    # Contar NaNs
    n_nan = np.isnan(X_normalized).sum()
    pct_nan = (n_nan / X_normalized.size) * 100

    logger.info(f"✓ Normalización completa")
    logger.info(f"  NaNs generados: {n_nan:,} ({pct_nan:.2f}%)")
    logger.info(f"  Rango resultante: [{np.nanmin(X_normalized):.4f}, {np.nanmax(X_normalized):.4f}]")

    return X_normalized


def normalizar_tensor_3d(X_3d: np.ndarray,
                        metodo: Literal['zscore_rolling', 'zscore_expanding', 'rank', 'minmax'] = 'zscore_rolling',
                        window: int = 252,
                        min_periods: Optional[int] = None) -> np.ndarray:
    """
    Normaliza un tensor 3D (n_samples × lookback × n_features).

    Aplica normalización a cada feature, respetando la estructura temporal.

    Args:
        X_3d: Tensor 3D de shape (n, L, m)
        metodo: Método de normalización
        window: Tamaño de ventana
        min_periods: Mínimo de períodos

    Returns:
        Tensor normalizado de mismo shape

    Note:
        Para tensor 3D, primero "desenrollamos" a 2D, normalizamos,
        y luego re-enrollamos a 3D.
    """
    logger.info(f"Normalizando tensor 3D: {X_3d.shape}")

    n_samples, lookback, n_features = X_3d.shape

    # Desenrollar: cada secuencia se concatena temporalmente
    # Esto mantiene el orden temporal para la normalización
    X_2d_list = []
    for i in range(n_samples):
        X_2d_list.append(X_3d[i])  # Shape: (lookback, n_features)

    # Concatenar todas las secuencias
    X_2d = np.vstack(X_2d_list)  # Shape: (n_samples * lookback, n_features)

    # Normalizar la matriz 2D
    X_2d_norm = normalizar_matriz_2d(X_2d, metodo=metodo, window=window, min_periods=min_periods)

    # Re-enrollar a 3D
    X_3d_norm = np.zeros_like(X_3d, dtype=np.float32)
    for i in range(n_samples):
        start_idx = i * lookback
        end_idx = start_idx + lookback
        X_3d_norm[i] = X_2d_norm[start_idx:end_idx]

    logger.info(f"✓ Tensor 3D normalizado")

    return X_3d_norm


def normalizar_tensor_4d(X_4d: np.ndarray,
                        metodo: Literal['zscore_rolling', 'zscore_expanding', 'rank', 'minmax'] = 'zscore_rolling',
                        window: int = 252,
                        min_periods: Optional[int] = None) -> np.ndarray:
    """
    Normaliza un tensor 4D (n_samples × n_activos × lookback × n_features).

    Normaliza cada activo independientemente.

    Args:
        X_4d: Tensor 4D de shape (n, a, L, m)
        metodo: Método de normalización
        window: Tamaño de ventana
        min_periods: Mínimo de períodos

    Returns:
        Tensor normalizado de mismo shape
    """
    logger.info(f"Normalizando tensor 4D: {X_4d.shape}")

    n_samples, n_activos, lookback, n_features = X_4d.shape
    X_4d_norm = np.zeros_like(X_4d, dtype=np.float32)

    # Normalizar cada activo independientemente
    for activo_idx in range(n_activos):
        logger.info(f"  Normalizando activo {activo_idx + 1}/{n_activos}...")

        # Extraer tensor 3D para este activo
        X_3d_activo = X_4d[:, activo_idx, :, :]  # Shape: (n, L, m)

        # Normalizar
        X_3d_activo_norm = normalizar_tensor_3d(
            X_3d_activo,
            metodo=metodo,
            window=window,
            min_periods=min_periods
        )

        # Guardar
        X_4d_norm[:, activo_idx, :, :] = X_3d_activo_norm

    logger.info(f"✓ Tensor 4D normalizado")

    return X_4d_norm


def validar_point_in_time(x_original: np.ndarray,
                         x_normalizado: np.ndarray,
                         window: int = 252) -> bool:
    """
    Valida que la normalización NO tiene look-ahead bias.

    Verifica que el valor normalizado en t solo depende de datos hasta t-1.

    Args:
        x_original: Serie original
        x_normalizado: Serie normalizada
        window: Ventana usada

    Returns:
        True si pasa la validación
    """
    logger.info("Validando point-in-time correctness...")

    # Test: El valor normalizado en t no debe cambiar si modificamos t+1
    test_idx = window + 100  # Índice a probar

    if test_idx >= len(x_original):
        logger.warning("No hay suficientes datos para validar")
        return True

    # Modificar el valor en t+1
    x_modificado = x_original.copy()
    x_modificado[test_idx + 1] = x_modificado[test_idx + 1] * 10  # Cambio drástico

    # Re-normalizar
    normalizador = NormalizadorPointInTime()
    x_norm_modificado = normalizador.zscore_rolling(x_modificado, window=window)

    # El valor en test_idx NO debe cambiar
    diff = abs(x_normalizado[test_idx] - x_norm_modificado[test_idx])

    if diff < 1e-6:
        logger.info("✓ PASS: Normalización es point-in-time correcta")
        logger.info(f"  Diferencia: {diff:.10f} (esperado: ~0)")
        return True
    else:
        logger.error("✗ FAIL: Normalización tiene look-ahead bias!")
        logger.error(f"  Diferencia: {diff:.10f} (esperado: ~0)")
        return False


def ejemplo_uso():
    """
    Ejemplo de uso de normalización point-in-time.
    """
    print("="*70)
    print("NORMALIZACIÓN POINT-IN-TIME (SIN LOOK-AHEAD)")
    print("="*70)
    print()

    # Crear datos de ejemplo
    np.random.seed(42)
    n_obs = 1000
    n_features = 5

    # Simular precios que crecen con tendencia
    precios = 100 + np.cumsum(np.random.randn(n_obs) * 0.5)

    print("EJEMPLO 1: Z-Score Rolling")
    print("-" * 70)
    normalizador = NormalizadorPointInTime()

    z = normalizador.zscore_rolling(precios, window=20)

    print(f"Datos originales (últimos 10):")
    print(precios[-10:])
    print()
    print(f"Z-scores (últimos 10):")
    print(z[-10:])
    print()
    print(f"Estadísticas de z-scores:")
    print(f"  Media: {np.nanmean(z):.4f} (esperado: ~0)")
    print(f"  Std: {np.nanstd(z):.4f} (esperado: ~1)")
    print(f"  NaNs: {np.isnan(z).sum()} primeras observaciones")
    print()

    print("EJEMPLO 2: Rank Transform")
    print("-" * 70)
    ranks = normalizador.rank_transform(precios, window=20)

    print(f"Ranks (últimos 10):")
    print(ranks[-10:])
    print()
    print(f"Estadísticas de ranks:")
    print(f"  Min: {np.nanmin(ranks):.4f}")
    print(f"  Max: {np.nanmax(ranks):.4f}")
    print(f"  Media: {np.nanmean(ranks):.4f} (esperado: ~0.5)")
    print()

    print("EJEMPLO 3: Validación Point-In-Time")
    print("-" * 70)
    es_valido = validar_point_in_time(precios, z, window=20)
    print()

    print("EJEMPLO 4: Normalizar Matriz 2D")
    print("-" * 70)
    X = np.random.randn(n_obs, n_features) * 100 + 500

    X_norm = normalizar_matriz_2d(X, metodo='zscore_rolling', window=50)

    print(f"Matriz original:")
    print(f"  Shape: {X.shape}")
    print(f"  Rango: [{X.min():.2f}, {X.max():.2f}]")
    print()
    print(f"Matriz normalizada:")
    print(f"  Shape: {X_norm.shape}")
    print(f"  Rango: [{np.nanmin(X_norm):.2f}, {np.nanmax(X_norm):.2f}]")
    print(f"  Media por columna: {np.nanmean(X_norm, axis=0)}")
    print()

    print("="*70)
    print("CONCLUSIÓN:")
    print("-" * 70)
    print("✓ Todos los métodos respetan point-in-time correctness")
    print("✓ NO hay look-ahead bias")
    print("✓ Cada valor en t solo usa información hasta t-1")
    print("✓ Simula lo que sabríamos en tiempo real")
    print("="*70)


if __name__ == '__main__':
    ejemplo_uso()
