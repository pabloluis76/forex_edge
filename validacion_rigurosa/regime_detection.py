"""
REGIME DETECTION - Identificación de Regímenes de Mercado
==========================================================

Detecta diferentes regímenes de mercado para validar que los features
funcionan consistentemente en todas las condiciones.

REGÍMENES DETECTADOS:
---------------------
1. TRENDING (direccional)
   - ADX > 25: Tendencia fuerte
   - Movimiento sostenido en una dirección

2. RANGING (lateral)
   - ADX < 20: Sin tendencia clara
   - Precio oscila en rango

3. HIGH VOLATILITY
   - ATR > percentil 75
   - Movimientos amplios

4. LOW VOLATILITY
   - ATR < percentil 25
   - Movimientos pequeños

VALIDACIÓN POR RÉGIMEN:
-----------------------
Un feature robusto debe mantener IC positivo en TODOS los regímenes,
no solo en uno específico.

Autor: Sistema de Edge Finding
Fecha: 2025-12-23
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import sys

# Importar constantes centralizadas
sys.path.append(str(Path(__file__).parent.parent))
from constants import EPSILON

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detector de regímenes de mercado.

    Identifica períodos de trending, ranging, alta/baja volatilidad.
    """

    def __init__(self, precios: pd.Series, high: pd.Series = None, low: pd.Series = None):
        """
        Inicializa el detector.

        Args:
            precios: Serie de precios (close)
            high: Serie de precios high (opcional, para ATR)
            low: Serie de precios low (opcional, para ATR)
        """
        self.precios = precios
        self.high = high if high is not None else precios
        self.low = low if low is not None else precios

        self.regimenes = None

        logger.info(f"RegimeDetector inicializado con {len(precios)} observaciones")

    def calcular_adx(self, ventana: int = 14) -> pd.Series:
        """
        Calcula Average Directional Index (ADX).

        ADX mide la fuerza de la tendencia (no la dirección).
        ADX > 25: Tendencia fuerte
        ADX < 20: Sin tendencia (ranging)

        Args:
            ventana: Período para cálculo (default: 14)

        Returns:
            Serie con ADX
        """
        high = self.high
        low = self.low
        close = self.precios

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        # ATR (Average True Range)
        atr = tr.rolling(ventana).mean()

        # Directional Movement
        plus_dm = high - high.shift(1)
        minus_dm = low.shift(1) - low

        # Solo movimientos positivos
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # Cuando ambos son positivos, solo cuenta el mayor
        plus_dm[(plus_dm > 0) & (minus_dm > 0) & (plus_dm < minus_dm)] = 0
        minus_dm[(plus_dm > 0) & (minus_dm > 0) & (minus_dm <= plus_dm)] = 0

        # Smoothed Directional Indicators
        plus_di = 100 * (plus_dm.rolling(ventana).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(ventana).mean() / atr)

        # Directional Index
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + EPSILON)

        # ADX es la media móvil de DX
        adx = dx.rolling(ventana).mean()

        return adx

    def calcular_atr(self, ventana: int = 14) -> pd.Series:
        """
        Calcula Average True Range (ATR).

        Mide volatilidad.

        Args:
            ventana: Período para cálculo (default: 14)

        Returns:
            Serie con ATR
        """
        high = self.high
        low = self.low
        close = self.precios

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        # ATR
        atr = tr.rolling(ventana).mean()

        return atr

    def detectar_regimenes(self,
                          ventana_adx: int = 14,
                          ventana_atr: int = 14,
                          umbral_trending: float = 25,
                          umbral_ranging: float = 20,
                          percentil_vol_alta: float = 75,
                          percentil_vol_baja: float = 25) -> pd.DataFrame:
        """
        Detecta todos los regímenes de mercado.

        Args:
            ventana_adx: Ventana para ADX
            ventana_atr: Ventana para ATR
            umbral_trending: ADX > umbral → trending
            umbral_ranging: ADX < umbral → ranging
            percentil_vol_alta: Percentil para alta volatilidad
            percentil_vol_baja: Percentil para baja volatilidad

        Returns:
            DataFrame con columnas:
            - 'trending': bool
            - 'ranging': bool
            - 'high_vol': bool
            - 'low_vol': bool
            - 'adx': float
            - 'atr': float
        """
        logger.info("="*70)
        logger.info("DETECTANDO REGÍMENES DE MERCADO")
        logger.info("="*70)

        # Calcular indicadores
        logger.info(f"Calculando ADX (ventana={ventana_adx})...")
        adx = self.calcular_adx(ventana_adx)

        logger.info(f"Calculando ATR (ventana={ventana_atr})...")
        atr = self.calcular_atr(ventana_atr)

        # Normalizar ATR (dividir por precio para comparar)
        atr_pct = atr / self.precios

        # Calcular percentiles de volatilidad
        atr_p75 = atr_pct.quantile(percentil_vol_alta / 100)
        atr_p25 = atr_pct.quantile(percentil_vol_baja / 100)

        # Crear DataFrame de regímenes
        regimenes = pd.DataFrame(index=self.precios.index)
        regimenes['adx'] = adx
        regimenes['atr'] = atr
        regimenes['atr_pct'] = atr_pct

        # Clasificar regímenes
        regimenes['trending'] = adx > umbral_trending
        regimenes['ranging'] = adx < umbral_ranging
        regimenes['high_vol'] = atr_pct > atr_p75
        regimenes['low_vol'] = atr_pct < atr_p25

        self.regimenes = regimenes

        # Estadísticas
        total_obs = len(regimenes.dropna())
        n_trending = regimenes['trending'].sum()
        n_ranging = regimenes['ranging'].sum()
        n_high_vol = regimenes['high_vol'].sum()
        n_low_vol = regimenes['low_vol'].sum()

        logger.info(f"\nResultados:")
        logger.info(f"  Observaciones válidas: {total_obs:,}")
        logger.info(f"  Trending: {n_trending:,} ({n_trending/total_obs*100:.1f}%)")
        logger.info(f"  Ranging: {n_ranging:,} ({n_ranging/total_obs*100:.1f}%)")
        logger.info(f"  High Volatility: {n_high_vol:,} ({n_high_vol/total_obs*100:.1f}%)")
        logger.info(f"  Low Volatility: {n_low_vol:,} ({n_low_vol/total_obs*100:.1f}%)")

        logger.info(f"\nUmbrales:")
        logger.info(f"  ADX trending: > {umbral_trending}")
        logger.info(f"  ADX ranging: < {umbral_ranging}")
        logger.info(f"  ATR high vol: > {atr_p75:.6f} ({percentil_vol_alta}th percentil)")
        logger.info(f"  ATR low vol: < {atr_p25:.6f} ({percentil_vol_baja}th percentil)")

        return regimenes

    def analizar_por_regimen(self,
                            feature: np.ndarray,
                            target: np.ndarray) -> Dict[str, Dict]:
        """
        Analiza performance de un feature en cada régimen.

        Args:
            feature: Array del feature
            target: Array del target (retornos)

        Returns:
            Dict con IC por régimen:
            {
                'trending': {'ic': float, 'p_value': float, 'n_obs': int},
                'ranging': {'ic': float, 'p_value': float, 'n_obs': int},
                ...
            }
        """
        if self.regimenes is None:
            raise ValueError("Primero ejecuta detectar_regimenes()")

        from scipy.stats import spearmanr

        resultados = {}

        for regimen_nombre in ['trending', 'ranging', 'high_vol', 'low_vol']:
            # Máscara del régimen
            mask_regimen = self.regimenes[regimen_nombre].fillna(False).values

            # Combinar con máscara de datos válidos
            mask_valid = ~(np.isnan(feature) | np.isnan(target))
            mask = mask_regimen & mask_valid

            if mask.sum() < 30:  # Mínimo de observaciones
                resultados[regimen_nombre] = {
                    'ic': np.nan,
                    'p_value': 1.0,
                    'n_obs': mask.sum(),
                    'significativo': False
                }
                continue

            # Calcular IC
            x = feature[mask]
            y = target[mask]

            ic, p_value = spearmanr(x, y)

            resultados[regimen_nombre] = {
                'ic': ic,
                'p_value': p_value,
                'n_obs': mask.sum(),
                'significativo': p_value < 0.01
            }

        return resultados

    def validar_feature_robusto(self,
                               feature: np.ndarray,
                               target: np.ndarray,
                               umbral_ic: float = 0.01) -> Tuple[bool, Dict]:
        """
        Valida que un feature sea robusto en todos los regímenes.

        Args:
            feature: Array del feature
            target: Array del target
            umbral_ic: IC mínimo aceptable

        Returns:
            (es_robusto, resultados_detallados)
        """
        resultados = self.analizar_por_regimen(feature, target)

        # Verificar que IC sea positivo y significativo en todos los regímenes
        es_robusto = True
        razones_rechazo = []

        for regimen, res in resultados.items():
            if np.isnan(res['ic']):
                razones_rechazo.append(f"{regimen}: Datos insuficientes")
                es_robusto = False
            elif res['ic'] < umbral_ic:
                razones_rechazo.append(f"{regimen}: IC={res['ic']:.4f} < {umbral_ic}")
                es_robusto = False
            elif not res['significativo']:
                razones_rechazo.append(f"{regimen}: No significativo (p={res['p_value']:.4f})")
                es_robusto = False

        resultado_final = {
            'es_robusto': es_robusto,
            'razones_rechazo': razones_rechazo,
            'resultados_por_regimen': resultados
        }

        return es_robusto, resultado_final


def ejemplo_uso():
    """
    Ejemplo de uso del detector de regímenes.
    """
    print("="*70)
    print("EJEMPLO: REGIME DETECTION")
    print("="*70)
    print()

    # Generar datos sintéticos
    np.random.seed(42)
    n = 5000

    # Simular precios con diferentes regímenes
    precios = [100.0]
    for i in range(1, n):
        if i < n/3:  # Trending up
            cambio = np.random.normal(0.001, 0.002)
        elif i < 2*n/3:  # Ranging
            cambio = np.random.normal(0, 0.001)
        else:  # High volatility
            cambio = np.random.normal(0, 0.005)

        precios.append(precios[-1] * (1 + cambio))

    precios = pd.Series(precios)

    # Crear detector
    detector = RegimeDetector(precios)

    # Detectar regímenes
    regimenes = detector.detectar_regimenes()

    print("\nRegímenes detectados exitosamente!")
    print(f"Shape: {regimenes.shape}")
    print(f"\nPrimeras filas:")
    print(regimenes.head(10))


if __name__ == '__main__':
    ejemplo_uso()
