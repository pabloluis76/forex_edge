"""
APLICACIÓN SISTEMÁTICA DE OPERADORES
====================================

Genera TODAS las combinaciones posibles de transformaciones
sin sesgo humano ni conocimiento previo del dominio.

OBJETIVO: Crear un espacio masivo de features para que los datos
          hablen por sí mismos.

ADVERTENCIA: Este script genera MILES de features.
            Puede tomar bastante tiempo y memoria.

Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generacion_de_transformaciones import (
    Delta, R, r, mu, sigma, Max, Min, Z, Pos, Rank,
    D1, D2, EMA
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeneradorSistematicoFeatures:
    """
    Generador sistemático de features mediante aplicación
    exhaustiva de operadores matemáticos.
    """

    # Configuración de operadores y ventanas
    OPERADORES_SIMPLES = {
        'Delta': Delta,
        'R': R,
        'r': r,
        'mu': mu,
        'sigma': sigma,
        'Max': Max,
        'Min': Min,
        'Z': Z,
        'Pos': Pos,
        'Rank': Rank,
        'EMA': EMA,
    }

    OPERADORES_SIN_VENTANA = {
        'D1': D1,
        'D2': D2,
    }

    # Ventanas para operadores
    VENTANAS = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]

    # Variables base (OHLCV)
    VARIABLES_BASE = ['open', 'high', 'low', 'close', 'volume']

    def __init__(self, df: pd.DataFrame, nombre_par: str = 'UNKNOWN'):
        """
        Inicializa el generador.

        Args:
            df: DataFrame con columnas OHLCV
            nombre_par: Nombre del par (para logging)
        """
        self.df = df.copy()
        self.nombre_par = nombre_par
        self.features = {}
        self.features_generados = 0

        # Validar columnas requeridas
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas: {missing}")

        logger.info(f"Inicializado para {nombre_par}: {len(df)} velas")

    def _agregar_feature(self, nombre: str, serie: pd.Series):
        """Agrega un feature al diccionario si es válido"""
        # Verificar que no sea todo NaN
        if serie.isna().all():
            return False

        # Verificar que tenga varianza
        if serie.std() == 0:
            return False

        # Verificar valores infinitos
        if np.isinf(serie).any():
            return False

        self.features[nombre] = serie
        self.features_generados += 1
        return True

    def generar_combinaciones_basicas(self):
        """
        PASO 1: GENERAR TODAS LAS COMBINACIONES BÁSICAS
        ================================================
        Para cada variable base V ∈ {O, H, L, C, V}:
        Para cada operador Op ∈ {Δ, R, r, μ, σ, Max, Min, Z, Pos, Rank}:
        Para cada ventana n ∈ {1, 2, 3, 4, 5, 10, 20, 50, 100, 200}:
            Crear transformación: Op_n(V)
        """
        logger.info("\n" + "="*70)
        logger.info("PASO 1: Generando combinaciones básicas")
        logger.info("="*70)

        total = len(self.VARIABLES_BASE) * len(self.OPERADORES_SIMPLES) * len(self.VENTANAS)
        logger.info(f"Total de combinaciones: {total}")

        pbar = tqdm(total=total, desc="Combinaciones básicas")

        for variable in self.VARIABLES_BASE:
            serie = self.df[variable]

            # Operadores con ventana
            for op_name, op_func in self.OPERADORES_SIMPLES.items():
                for n in self.VENTANAS:
                    try:
                        resultado = op_func(serie, n)
                        nombre = f"{op_name}_{n}({variable})"
                        self._agregar_feature(nombre, resultado)
                    except Exception as e:
                        pass  # Silenciosamente ignorar errores
                    finally:
                        pbar.update(1)

        pbar.close()

        # Operadores sin ventana (D1, D2)
        logger.info("Generando derivadas...")
        for variable in self.VARIABLES_BASE:
            serie = self.df[variable]

            for op_name, op_func in self.OPERADORES_SIN_VENTANA.items():
                try:
                    resultado = op_func(serie)
                    nombre = f"{op_name}({variable})"
                    self._agregar_feature(nombre, resultado)
                except Exception:
                    pass

        logger.info(f"✓ Features básicos generados: {self.features_generados}")

    def generar_combinaciones_variables(self):
        """
        PASO 2: COMBINACIONES DE VARIABLES
        ===================================
        Ratios: C/O, H/L, C/H, C/L, (C-L)/(H-L), (H-C)/(H-L), (C-O)/(H-L)
        Diferencias: H-L, C-O, H-C, C-L
        Promedios típicos: HL2, HLC3, OHLC4
        """
        logger.info("\n" + "="*70)
        logger.info("PASO 2: Generando combinaciones de variables")
        logger.info("="*70)

        O = self.df['open']
        H = self.df['high']
        L = self.df['low']
        C = self.df['close']

        # Ratios
        # CRÍTICO #7 CORREGIDO: Usar np.where en lugar de offset después
        # Offset se sumaba DESPUÉS de (H-L), cuando debería validarse ANTES
        HL_diff = H - L

        ratios = {
            'C_O_ratio': C / O,
            'H_L_ratio': H / L,
            'C_H_ratio': C / H,
            'C_L_ratio': C / L,
            'CL_HL_ratio': np.where(HL_diff != 0, (C - L) / HL_diff, 0.0),
            'HC_HL_ratio': np.where(HL_diff != 0, (H - C) / HL_diff, 0.0),
            'CO_HL_ratio': np.where(HL_diff != 0, (C - O) / HL_diff, 0.0),
        }

        for nombre, serie in ratios.items():
            self._agregar_feature(nombre, serie)

        # Diferencias
        diferencias = {
            'H_L_diff': H - L,
            'C_O_diff': C - O,
            'H_C_diff': H - C,
            'C_L_diff': C - L,
        }

        for nombre, serie in diferencias.items():
            self._agregar_feature(nombre, serie)

        # Promedios típicos
        promedios = {
            'HL2': (H + L) / 2,
            'HLC3': (H + L + C) / 3,
            'OHLC4': (O + H + L + C) / 4,
        }

        for nombre, serie in promedios.items():
            self._agregar_feature(nombre, serie)

        logger.info(f"✓ Features combinados generados: {len(ratios) + len(diferencias) + len(promedios)}")

        # Ahora aplicar operadores sobre estas nuevas variables
        logger.info("Aplicando operadores sobre variables combinadas...")

        nuevas_variables = list(ratios.keys()) + list(diferencias.keys()) + list(promedios.keys())

        for var_name in tqdm(nuevas_variables, desc="Operadores sobre combinaciones"):
            if var_name not in self.features:
                continue

            serie = self.features[var_name]

            # Solo algunos operadores clave para no explotar la memoria
            operadores_clave = ['mu', 'sigma', 'Z', 'Pos', 'D1']
            ventanas_clave = [5, 20, 50]

            for op_name in operadores_clave:
                if op_name not in self.OPERADORES_SIMPLES:
                    continue

                op_func = self.OPERADORES_SIMPLES[op_name]

                for n in ventanas_clave:
                    try:
                        resultado = op_func(serie, n)
                        nombre = f"{op_name}_{n}({var_name})"
                        self._agregar_feature(nombre, resultado)
                    except Exception:
                        pass

        logger.info(f"✓ Total features hasta ahora: {self.features_generados}")

    def generar_composiciones(self):
        """
        PASO 3: COMPOSICIÓN DE OPERADORES
        ==================================
        Aplicar operadores sobre resultados de otros operadores:
        - Z_20(μ_10(C))
        - Pos_50(σ_20(C))
        - D¹(μ_20(C))
        - D²(Z_50(C))
        """
        logger.info("\n" + "="*70)
        logger.info("PASO 3: Generando composiciones de operadores")
        logger.info("="*70)

        # Definir composiciones importantes
        composiciones = [
            # Z-score de promedios
            ('Z', 20, 'mu', 10, 'close'),
            ('Z', 20, 'mu', 50, 'close'),
            ('Z', 50, 'EMA', 20, 'close'),

            # Posición de volatilidad
            ('Pos', 50, 'sigma', 20, 'close'),
            ('Pos', 100, 'sigma', 20, 'close'),

            # Velocidad de promedios
            ('D1', None, 'mu', 20, 'close'),
            ('D1', None, 'mu', 50, 'close'),
            ('D1', None, 'EMA', 20, 'close'),

            # Aceleración de z-scores
            ('D2', None, 'Z', 50, 'close'),
            ('D2', None, 'Z', 20, 'close'),

            # Z-score de momentum
            ('Z', 20, 'Delta', 1, 'close'),
            ('Z', 50, 'Delta', 5, 'close'),

            # Posición de retornos
            ('Pos', 20, 'R', 1, 'close'),
            ('Pos', 50, 'r', 1, 'close'),
        ]

        for comp in tqdm(composiciones, desc="Composiciones"):
            try:
                op1_name, n1, op2_name, n2, var = comp

                # Aplicar primer operador
                serie = self.df[var]
                op2_func = self.OPERADORES_SIMPLES[op2_name]
                intermedio = op2_func(serie, n2)

                # Aplicar segundo operador
                if op1_name in self.OPERADORES_SIN_VENTANA:
                    op1_func = self.OPERADORES_SIN_VENTANA[op1_name]
                    resultado = op1_func(intermedio)
                    nombre = f"{op1_name}({op2_name}_{n2}({var}))"
                else:
                    op1_func = self.OPERADORES_SIMPLES[op1_name]
                    resultado = op1_func(intermedio, n1)
                    nombre = f"{op1_name}_{n1}({op2_name}_{n2}({var}))"

                self._agregar_feature(nombre, resultado)

            except Exception as e:
                pass

        logger.info(f"✓ Composiciones generadas: {len(composiciones)}")

    def generar_comparaciones_temporales(self):
        """
        PASO 4: COMPARACIONES TEMPORALES
        =================================
        - μ_m(x) / μ_n(x)   → Ratio de promedios
        - σ_m(x) / σ_n(x)   → Ratio de volatilidades
        - μ_m(x) - μ_n(x)   → Diferencia de promedios
        """
        logger.info("\n" + "="*70)
        logger.info("PASO 4: Generando comparaciones temporales")
        logger.info("="*70)

        pares_ventanas = [
            (5, 20), (10, 50), (20, 100), (50, 200)
        ]

        serie = self.df['close']

        for m, n in tqdm(pares_ventanas, desc="Comparaciones temporales"):
            try:
                # Ratio de promedios (fast/slow)
                ma_m = mu(serie, m)
                ma_n = mu(serie, n)
                ratio = ma_m / ma_n
                self._agregar_feature(f"mu_{m}_mu_{n}_ratio", ratio)

                # Diferencia de promedios
                diff = ma_m - ma_n
                self._agregar_feature(f"mu_{m}_mu_{n}_diff", diff)

                # Ratio de volatilidades
                vol_m = sigma(serie, m)
                vol_n = sigma(serie, n)
                vol_ratio = vol_m / (vol_n + 1e-10)
                self._agregar_feature(f"sigma_{m}_sigma_{n}_ratio", vol_ratio)

                # EMA ratio
                ema_m = EMA(serie, m)
                ema_n = EMA(serie, n)
                ema_ratio = ema_m / ema_n
                self._agregar_feature(f"EMA_{m}_EMA_{n}_ratio", ema_ratio)

            except Exception:
                pass

        logger.info(f"✓ Comparaciones temporales generadas")

    def generar_informacion_temporal(self):
        """
        PASO 5: INFORMACIÓN TEMPORAL
        =============================
        De τ extraer:
        - hour_sin, hour_cos
        - day_sin, day_cos
        - es_lunes, es_viernes, es_fin_de_mes, etc.
        """
        logger.info("\n" + "="*70)
        logger.info("PASO 5: Generando información temporal")
        logger.info("="*70)

        # Verificar que el índice sea datetime
        if not isinstance(self.df.index, pd.DatetimeIndex):
            logger.warning("Índice no es DatetimeIndex, saltando features temporales")
            return

        # Hora del día (codificación cíclica)
        hora = self.df.index.hour
        self.features['hour_sin'] = np.sin(2 * np.pi * hora / 24)
        self.features['hour_cos'] = np.cos(2 * np.pi * hora / 24)

        # Día de la semana (codificación cíclica)
        dia_semana = self.df.index.dayofweek
        self.features['day_sin'] = np.sin(2 * np.pi * dia_semana / 5)
        self.features['day_cos'] = np.cos(2 * np.pi * dia_semana / 5)

        # Variables binarias
        self.features['es_lunes'] = (dia_semana == 0).astype(int)
        self.features['es_viernes'] = (dia_semana == 4).astype(int)

        # Día del mes
        dia_mes = self.df.index.day
        self.features['es_inicio_mes'] = (dia_mes <= 5).astype(int)
        self.features['es_fin_mes'] = (dia_mes >= 25).astype(int)

        # Mes del año (codificación cíclica)
        mes = self.df.index.month
        self.features['month_sin'] = np.sin(2 * np.pi * mes / 12)
        self.features['month_cos'] = np.cos(2 * np.pi * mes / 12)

        logger.info(f"✓ Features temporales generados: 10")

    def generar_todas_las_transformaciones(self):
        """
        Ejecuta todos los pasos de generación de features.
        """
        logger.info("\n" + "="*70)
        logger.info(f"GENERACIÓN SISTEMÁTICA DE FEATURES - {self.nombre_par}")
        logger.info("="*70)
        logger.info(f"Datos: {len(self.df)} velas")
        logger.info(f"Período: {self.df.index[0]} a {self.df.index[-1]}")

        # Ejecutar todos los pasos
        self.generar_combinaciones_basicas()
        self.generar_combinaciones_variables()
        self.generar_composiciones()
        self.generar_comparaciones_temporales()
        self.generar_informacion_temporal()

        logger.info("\n" + "="*70)
        logger.info("RESUMEN FINAL")
        logger.info("="*70)
        logger.info(f"Total de features generados: {self.features_generados}")
        logger.info(f"Features válidos: {len(self.features)}")

        # Crear DataFrame con todos los features
        df_transformaciones = pd.DataFrame(self.features, index=self.df.index)

        # Agregar columnas originales OHLCV al inicio
        columnas_originales = ['open', 'high', 'low', 'close', 'volume']
        df_features = pd.concat([
            self.df[columnas_originales],
            df_transformaciones
        ], axis=1)

        logger.info(f"\n  Columnas originales agregadas: {columnas_originales}")

        # Estadísticas
        logger.info(f"\nEstadísticas:")
        logger.info(f"  Tamaño en memoria: {df_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        logger.info(f"  % de NaN por feature: {df_features.isna().mean().mean()*100:.1f}%")

        return df_features


def procesar_par(par_dir: Path, nombre_par: str, timeframe: str = 'H1') -> pd.DataFrame:
    """
    Procesa un par específico y genera todos sus features.

    Args:
        par_dir: Directorio del par
        nombre_par: Nombre del par (ej: 'EUR_USD')
        timeframe: Timeframe a procesar (default: 'H1')

    Returns:
        DataFrame con todos los features generados
    """
    file_path = par_dir / f"{timeframe}.csv"

    if not file_path.exists():
        logger.error(f"Archivo no encontrado: {file_path}")
        return None

    # Cargar datos
    logger.info(f"\nProcesando {nombre_par} - {timeframe}")
    df = pd.read_csv(file_path, index_col='time', parse_dates=True)

    # Generar features
    generador = GeneradorSistematicoFeatures(df, nombre_par=f"{nombre_par}_{timeframe}")
    df_features = generador.generar_todas_las_transformaciones()

    return df_features


def main():
    """Función principal"""
    # Configuración
    DATA_DIR = Path(__file__).parent.parent / 'datos' / 'ohlc'
    OUTPUT_DIR = Path(__file__).parent.parent / 'datos' / 'features'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Procesar un par de ejemplo (EUR_USD H1)
    par = 'EUR_USD'
    timeframe = 'H1'

    par_dir = DATA_DIR / par

    if not par_dir.exists():
        logger.error(f"Directorio no encontrado: {par_dir}")
        return

    # Generar features
    df_features = procesar_par(par_dir, par, timeframe)

    if df_features is not None:
        # Guardar resultado
        output_file = OUTPUT_DIR / f"{par}_{timeframe}_features.csv"
        df_features.to_csv(output_file)
        logger.info(f"\n✓ Features guardados en: {output_file}")
        logger.info(f"  Tamaño del archivo: {output_file.stat().st_size / 1024**2:.1f} MB")


if __name__ == '__main__':
    main()
