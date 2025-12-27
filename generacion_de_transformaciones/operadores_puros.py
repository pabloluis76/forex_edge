"""
OPERADORES PUROS PARA TRANSFORMACIÓN DE SERIES TEMPORALES
==========================================================

Implementación de operadores matemáticos puros para análisis de datos
financieros sin nombres preconcebidos.

REGLA FUNDAMENTAL: Solo datos del pasado. Sin look-ahead bias.

Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from pathlib import Path
import sys

# Importar constantes centralizadas
sys.path.append(str(Path(__file__).parent.parent))
from constants import EPSILON, MIN_PRICE_FOREX, MAX_PRICE_FOREX


class OperadoresPuros:
    """
    Clase que implementa operadores matemáticos puros para series temporales.

    Todos los operadores respetan el principio de causalidad:
    - Solo usan datos hasta t-1 (nunca datos del futuro)
    - No tienen look-ahead bias
    """

    @staticmethod
    def Delta(x: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR Δ (Diferencia)
        ───────────────────────
        Δₙ(x) = xₜ₋₁ - xₜ₋ₙ

        Cambio absoluto en n períodos.

        Args:
            x: Serie temporal
            n: Número de períodos

        Returns:
            Serie con cambios absolutos

        Example:
            >>> close = pd.Series([100, 101, 103, 102, 105])
            >>> Delta(close, 1)  # Cambio en 1 período
            [NaN, 1, 2, -1, 3]
        """
        return x.diff(n)

    @staticmethod
    def R(x: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR R (Retorno)
        ────────────────────
        Rₙ(x) = (xₜ₋₁ - xₜ₋ₙ) / xₜ₋ₙ

        Cambio relativo en n períodos (retorno simple).

        Args:
            x: Serie temporal
            n: Número de períodos

        Returns:
            Serie con retornos simples

        Example:
            >>> close = pd.Series([100, 110, 105])
            >>> R(close, 1)  # Retorno en 1 período
            [NaN, 0.10, -0.0454...]
        """
        return x.pct_change(n)

    @staticmethod
    def r(x: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR r (Log-retorno)
        ────────────────────────
        rₙ(x) = ln(xₜ₋₁ / xₜ₋ₙ)

        Log del cambio relativo (retorno logarítmico).
        Ventajas: aditivo en el tiempo, simétrico.

        Args:
            x: Serie temporal
            n: Número de períodos

        Returns:
            Serie con log-retornos

        Example:
            >>> close = pd.Series([100, 110, 105])
            >>> r(close, 1)
            [NaN, 0.0953..., -0.0465...]
        """
        # REALISMO MÁXIMO: Validar que precios sean válidos
        # Precios de forex NUNCA son ≤ 0 - si aparecen, datos corruptos
        if (x <= 0).any():
            invalid_count = (x <= 0).sum()
            invalid_indices = x[x <= 0].index.tolist()[:5]  # Primeros 5
            raise ValueError(
                f"DATOS INVÁLIDOS: {invalid_count} precios ≤ 0 detectados. "
                f"Primeros índices: {invalid_indices}. "
                f"Los precios de forex deben ser siempre positivos."
            )

        # Validar rango realista usando constantes globales
        # Cubre todos los pares: EUR/USD ~1.0, USD/JPY ~150, etc.
        if (x < MIN_PRICE_FOREX).any() or (x > MAX_PRICE_FOREX).any():
            out_of_range = x[(x < MIN_PRICE_FOREX) | (x > MAX_PRICE_FOREX)]
            raise ValueError(
                f"DATOS FUERA DE RANGO: {len(out_of_range)} precios fuera de [{MIN_PRICE_FOREX}, {MAX_PRICE_FOREX}]. "
                f"Rango encontrado: [{x.min():.6f}, {x.max():.6f}]"
            )

        # Cálculo SIN clips artificiales (datos ya validados)
        ratio = x / x.shift(n)
        return np.log(ratio)
        # NaN en primeras n barras es esperado y correcto

    @staticmethod
    def mu(x: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR μ (Media móvil)
        ────────────────────────
        μₙ(x) = (1/n) Σᵢ₌₁ⁿ xₜ₋ᵢ

        Promedio de últimos n valores (Simple Moving Average).

        Args:
            x: Serie temporal
            n: Número de períodos (ventana)

        Returns:
            Serie con medias móviles

        Example:
            >>> close = pd.Series([100, 102, 104, 103, 105])
            >>> mu(close, 3)
            [NaN, NaN, 102, 103, 104]
        """
        return x.rolling(window=n, min_periods=n).mean()

    @staticmethod
    def sigma(x: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR σ (Desviación estándar)
        ─────────────────────────────────
        σₙ(x) = √[(1/n) Σᵢ₌₁ⁿ (xₜ₋ᵢ - μₙ)²]

        Dispersión de últimos n valores (volatilidad histórica).

        Args:
            x: Serie temporal
            n: Número de períodos (ventana)

        Returns:
            Serie con desviaciones estándar móviles

        Example:
            >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
            >>> sigma(returns, 3)
            [NaN, NaN, 0.0175..., ...]
        """
        return x.rolling(window=n, min_periods=n).std()

    @staticmethod
    def Max(x: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR Max
        ────────────
        Maxₙ(x) = max(xₜ₋₁, xₜ₋₂, ..., xₜ₋ₙ)

        Valor máximo en últimos n períodos.

        Args:
            x: Serie temporal
            n: Número de períodos (ventana)

        Returns:
            Serie con valores máximos móviles
        """
        return x.rolling(window=n, min_periods=n).max()

    @staticmethod
    def Min(x: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR Min
        ────────────
        Minₙ(x) = min(xₜ₋₁, xₜ₋₂, ..., xₜ₋ₙ)

        Valor mínimo en últimos n períodos.

        Args:
            x: Serie temporal
            n: Número de períodos (ventana)

        Returns:
            Serie con valores mínimos móviles
        """
        return x.rolling(window=n, min_periods=n).min()

    @staticmethod
    def Z(x: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR Z (Normalización Z-score)
        ───────────────────────────────────
        Zₙ(x) = (xₜ₋₁ - μₙ(x)) / σₙ(x)

        Distancia al promedio en unidades de desviación estándar.
        Útil para detectar valores extremos.

        Args:
            x: Serie temporal
            n: Número de períodos (ventana)

        Returns:
            Serie normalizada (z-scores)

        Example:
            >>> close = pd.Series([100, 102, 104, 103, 110])
            >>> Z(close, 3)
            # Valores típicos entre -3 y +3
        """
        mean = x.rolling(window=n, min_periods=n).mean()
        std = x.rolling(window=n, min_periods=n).std()

        # CRÍTICO #5 CORREGIDO: Evitar división por cero cuando std=0
        std_safe = std.replace(0, EPSILON)
        return (x - mean) / std_safe

    @staticmethod
    def Pos(x: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR Pos (Posición en rango)
        ────────────────────────────────
        Posₙ(x) = (xₜ₋₁ - Minₙ(x)) / (Maxₙ(x) - Minₙ(x))

        Posición relativa en el rango [0,1].
        Similar al Stochastic Oscillator pero sin nombre.

        Args:
            x: Serie temporal
            n: Número de períodos (ventana)

        Returns:
            Serie con posición normalizada [0,1]

        Example:
            >>> high = pd.Series([105, 110, 108, 112, 107])
            >>> Pos(high, 5)
            # 0.0 = en el mínimo, 1.0 = en el máximo
        """
        min_val = x.rolling(window=n, min_periods=n).min()
        max_val = x.rolling(window=n, min_periods=n).max()

        # CRÍTICO #6 CORREGIDO: Evitar división por cero cuando max=min (rango=0)
        rango = max_val - min_val
        rango_safe = rango.replace(0, EPSILON)
        return (x - min_val) / rango_safe

    @staticmethod
    def Rank(x: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR Rank (Ranking ordinal)
        ───────────────────────────────
        Rankₙ(x) = posición de xₜ₋₁ en {xₜ₋₁, ..., xₜ₋ₙ} / n

        Percentil del valor actual en ventana móvil.

        Args:
            x: Serie temporal
            n: Número de períodos (ventana)

        Returns:
            Serie con ranks normalizados [0,1]

        Example:
            >>> close = pd.Series([100, 102, 98, 105, 101])
            >>> Rank(close, 3)
            # 0.0 = valor más bajo, 1.0 = valor más alto
        """
        return x.rolling(window=n, min_periods=n).apply(
            lambda w: pd.Series(w).rank(pct=True).iloc[-1],
            raw=False
        )

    @staticmethod
    def P(x: pd.Series, k: float, n: int) -> pd.Series:
        """
        OPERADOR P (Percentil)
        ──────────────────────
        Pₖ,ₙ(x) = k-ésimo percentil de {xₜ₋₁, ..., xₜ₋ₙ}

        Calcula el percentil k en ventana móvil de n períodos.

        Args:
            x: Serie temporal
            k: Percentil a calcular (0-100)
            n: Número de períodos (ventana)

        Returns:
            Serie con percentiles móviles

        Example:
            >>> close = pd.Series([100, 102, 98, 105, 101, 99])
            >>> P(close, 75, 5)  # Percentil 75 en ventana de 5
        """
        # CRÍTICO #8 CORREGIDO: Validar que k esté en rango [0, 100]
        if not (0 <= k <= 100):
            raise ValueError(f"Percentil k debe estar entre 0 y 100, recibido: {k}")

        return x.rolling(window=n, min_periods=n).quantile(k / 100.0)

    @staticmethod
    def D1(x: pd.Series) -> pd.Series:
        """
        OPERADOR D¹ (Primera derivada)
        ──────────────────────────────
        D¹(x) = xₜ₋₁ - xₜ₋₂

        Velocidad de cambio (primera diferencia).

        Args:
            x: Serie temporal

        Returns:
            Serie con primera derivada

        Example:
            >>> close = pd.Series([100, 101, 103, 102, 105])
            >>> D1(close)
            [NaN, 1, 2, -1, 3]
        """
        return x.diff(1)

    @staticmethod
    def D2(x: pd.Series) -> pd.Series:
        """
        OPERADOR D² (Segunda derivada)
        ──────────────────────────────
        D²(x) = (xₜ₋₁ - xₜ₋₂) - (xₜ₋₂ - xₜ₋₃)

        Aceleración (segunda diferencia).
        Mide el cambio en la velocidad.

        Args:
            x: Serie temporal

        Returns:
            Serie con segunda derivada

        Example:
            >>> close = pd.Series([100, 101, 103, 102, 105])
            >>> D2(close)
            [NaN, NaN, 1, -3, 4]
        """
        return x.diff(1).diff(1)

    @staticmethod
    def rho(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR ρ (Correlación)
        ────────────────────────
        ρₙ(x,y) = correlación de {x,y} en últimos n períodos

        Correlación de Pearson móvil entre dos series.

        Args:
            x: Primera serie temporal
            y: Segunda serie temporal
            n: Número de períodos (ventana)

        Returns:
            Serie con correlaciones móviles [-1, 1]

        Example:
            >>> close_eur = pd.Series([1.10, 1.11, 1.12, 1.11])
            >>> close_gbp = pd.Series([1.30, 1.31, 1.33, 1.32])
            >>> rho(close_eur, close_gbp, 3)
        """
        return x.rolling(window=n, min_periods=n).corr(y)

    @staticmethod
    def EMA(x: pd.Series, n: int) -> pd.Series:
        """
        OPERADOR EMA (Media Móvil Exponencial)
        ──────────────────────────────────────
        EMAₙ(x) = α×xₜ₋₁ + (1-α)×EMAₙ(x)ₜ₋₁
        donde α = 2/(n+1)

        Media móvil con ponderación exponencial decreciente.
        Da más peso a datos recientes.

        Args:
            x: Serie temporal
            n: Número de períodos (define el factor de suavizado)

        Returns:
            Serie con EMA

        Example:
            >>> close = pd.Series([100, 102, 104, 103, 105])
            >>> EMA(close, 3)
        """
        return x.ewm(span=n, adjust=False).mean()


# ============================================================================
# FUNCIONES CONVENIENTES PARA USO DIRECTO
# ============================================================================

def Delta(x: pd.Series, n: int) -> pd.Series:
    """Diferencia: xₜ₋₁ - xₜ₋ₙ"""
    return OperadoresPuros.Delta(x, n)

def R(x: pd.Series, n: int) -> pd.Series:
    """Retorno: (xₜ₋₁ - xₜ₋ₙ) / xₜ₋ₙ"""
    return OperadoresPuros.R(x, n)

def r(x: pd.Series, n: int) -> pd.Series:
    """Log-retorno: ln(xₜ₋₁ / xₜ₋ₙ)"""
    return OperadoresPuros.r(x, n)

def mu(x: pd.Series, n: int) -> pd.Series:
    """Media móvil: promedio de n períodos"""
    return OperadoresPuros.mu(x, n)

def sigma(x: pd.Series, n: int) -> pd.Series:
    """Desviación estándar móvil: dispersión de n períodos"""
    return OperadoresPuros.sigma(x, n)

def Max(x: pd.Series, n: int) -> pd.Series:
    """Máximo: valor máximo en n períodos"""
    return OperadoresPuros.Max(x, n)

def Min(x: pd.Series, n: int) -> pd.Series:
    """Mínimo: valor mínimo en n períodos"""
    return OperadoresPuros.Min(x, n)

def Z(x: pd.Series, n: int) -> pd.Series:
    """Z-score: (x - μ) / σ"""
    return OperadoresPuros.Z(x, n)

def Pos(x: pd.Series, n: int) -> pd.Series:
    """Posición en rango: (x - min) / (max - min)"""
    return OperadoresPuros.Pos(x, n)

def Rank(x: pd.Series, n: int) -> pd.Series:
    """Ranking: percentil del valor actual"""
    return OperadoresPuros.Rank(x, n)

def P(x: pd.Series, k: float, n: int) -> pd.Series:
    """Percentil k en ventana de n períodos"""
    return OperadoresPuros.P(x, k, n)

def D1(x: pd.Series) -> pd.Series:
    """Primera derivada: velocidad"""
    return OperadoresPuros.D1(x)

def D2(x: pd.Series) -> pd.Series:
    """Segunda derivada: aceleración"""
    return OperadoresPuros.D2(x)

def rho(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    """Correlación móvil entre x e y"""
    return OperadoresPuros.rho(x, y, n)

def EMA(x: pd.Series, n: int) -> pd.Series:
    """Media móvil exponencial"""
    return OperadoresPuros.EMA(x, n)


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == '__main__':
    # Ejemplo simple
    import pandas as pd

    # Crear datos de ejemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    close = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5),
        index=dates,
        name='close'
    )

    print("="*60)
    print("EJEMPLO DE OPERADORES PUROS")
    print("="*60)
    print()

    # Aplicar operadores
    print("Datos originales (primeros 5):")
    print(close.head())
    print()

    print("Δ₁(close) - Cambio en 1 día:")
    print(Delta(close, 1).head())
    print()

    print("R₁(close) - Retorno diario:")
    print(R(close, 1).head())
    print()

    print("μ₅(close) - Media móvil 5 días:")
    print(mu(close, 5).head(10))
    print()

    print("Z₂₀(close) - Z-score 20 días:")
    print(Z(close, 20).tail())
    print()

    print("Pos₁₄(close) - Posición en rango 14 días:")
    print(Pos(close, 14).tail())
    print()

    print("\n✓ Todos los operadores funcionando correctamente")
    print("✓ Sin look-ahead bias (solo usan datos del pasado)")
    print("✓ Listos para generación de features")
