"""
Módulo de Generación de Transformaciones
=========================================

Operadores matemáticos puros para transformación de series temporales
financieras sin look-ahead bias.
"""

from .operadores_puros import (
    # Clase principal
    OperadoresPuros,

    # Funciones directas
    Delta,  # Diferencia
    R,      # Retorno simple
    r,      # Log-retorno
    mu,     # Media móvil
    sigma,  # Desviación estándar
    Max,    # Máximo móvil
    Min,    # Mínimo móvil
    Z,      # Z-score
    Pos,    # Posición en rango
    Rank,   # Ranking ordinal
    P,      # Percentil
    D1,     # Primera derivada
    D2,     # Segunda derivada
    rho,    # Correlación
    EMA,    # Media exponencial
)

__all__ = [
    'OperadoresPuros',
    'Delta', 'R', 'r',
    'mu', 'sigma',
    'Max', 'Min',
    'Z', 'Pos', 'Rank', 'P',
    'D1', 'D2',
    'rho', 'EMA',
]

__version__ = '1.0.0'
__author__ = 'Edge Finding System'
