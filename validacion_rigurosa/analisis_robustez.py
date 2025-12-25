"""
Análisis de Robustez

¿EL EDGE ES ROBUSTO O FRÁGIL?


A) SENSIBILIDAD A PARÁMETROS
────────────────────────────

Si la mejor transformación es μ₂₀(C) - μ₅₀(C):

Probar:
- μ₁₈(C) - μ₄₅(C)
- μ₁₉(C) - μ₄₈(C)
- μ₂₁(C) - μ₅₂(C)
- μ₂₂(C) - μ₅₅(C)

ROBUSTO: Todas dan IC similar
FRÁGIL: Solo funciona con exactamente 20 y 50


B) ESTABILIDAD TEMPORAL
───────────────────────

IC por año:
- Año 1: 0.025
- Año 2: 0.022
- Año 3: 0.028
- Año 4: 0.020
- Año 5: 0.024

ESTABLE: IC similar cada año
INESTABLE: IC varía mucho o es negativo en algunos años


C) CONSISTENCIA ENTRE ACTIVOS
─────────────────────────────

IC por par:
- EUR/USD: 0.022
- GBP/USD: 0.019
- USD/JPY: 0.025
- EUR/JPY: 0.021

CONSISTENTE: Funciona en múltiples pares
SOSPECHOSO: Solo funciona en 1 par

Author: Sistema de Edge-Finding Forex
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import sys
import warnings
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# Importar RegimeDetector
from .regime_detection import RegimeDetector


class AnalisisRobustez:
    """
    Análisis de Robustez para determinar si el edge es robusto o frágil.

    Prueba sensibilidad a:
    - Parámetros de transformaciones
    - Diferentes períodos temporales
    - Diferentes activos
    """

    def __init__(
        self,
        verbose: bool = True
    ):
        """
        Inicializa el Análisis de Robustez.

        Parameters:
        -----------
        verbose : bool
            Imprimir resultados detallados
        """
        self.verbose = verbose

        # Resultados
        self.resultados_parametros: Dict = {}
        self.resultados_temporales: Dict = {}
        self.resultados_activos: Dict = {}
        self.resultados_regimenes: Dict = {}

        if self.verbose:
            print("="*80)
            print("ANÁLISIS DE ROBUSTEZ - ¿EL EDGE ES ROBUSTO O FRÁGIL?")
            print("="*80)

    # ========================================================================
    # A) SENSIBILIDAD A PARÁMETROS
    # ========================================================================

    def analizar_sensibilidad_parametros(
        self,
        transformacion_base: str,
        features_df: pd.DataFrame,
        target: pd.Series,
        variacion_pct: float = 0.10,
        n_variaciones: int = 5
    ) -> pd.DataFrame:
        """
        Analiza sensibilidad a parámetros de transformación.

        Si la mejor transformación es μ₂₀(C) - μ₅₀(C):
        - Probar μ₁₈(C) - μ₄₅(C), μ₁₉(C) - μ₄₈(C), etc.
        - ROBUSTO: Todas dan IC similar
        - FRÁGIL: Solo funciona con parámetros exactos

        Parameters:
        -----------
        transformacion_base : str
            Nombre de la transformación base (ej: "mu_20_C_minus_mu_50_C")
        features_df : pd.DataFrame
            DataFrame con TODAS las transformaciones disponibles
        target : pd.Series
            Target (retornos futuros)
        variacion_pct : float
            Porcentaje de variación en parámetros (default: 10%)
        n_variaciones : int
            Número de variaciones a probar (default: 5)

        Returns:
        --------
        df_sensibilidad : pd.DataFrame
            Resultados de sensibilidad a parámetros
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"A) SENSIBILIDAD A PARÁMETROS")
            print(f"{'='*80}")
            print(f"Transformación base: {transformacion_base}")

        # Extraer parámetros de la transformación base
        parametros = self._extraer_parametros(transformacion_base)

        if not parametros:
            print(f"✗ No se pudieron extraer parámetros de '{transformacion_base}'")
            return pd.DataFrame()

        if self.verbose:
            print(f"Parámetros extraídos: {parametros}")

        # Generar variaciones de parámetros
        variaciones = self._generar_variaciones_parametros(
            parametros=parametros,
            variacion_pct=variacion_pct,
            n_variaciones=n_variaciones
        )

        if self.verbose:
            print(f"\nProbando {len(variaciones)} variaciones de parámetros...")

        # Calcular IC para cada variación
        resultados = []

        for i, params_var in enumerate(variaciones):
            # Construir nombre de transformación con parámetros variados
            nombre_var = self._construir_nombre_transformacion(
                transformacion_base,
                parametros,
                params_var
            )

            # Buscar transformación en features_df
            if nombre_var in features_df.columns:
                # Calcular IC
                ic = self._calcular_ic(features_df[nombre_var], target)

                resultados.append({
                    'Variacion': i + 1,
                    'Transformacion': nombre_var,
                    'Parametros': str(params_var),
                    'IC': ic
                })

                if self.verbose:
                    print(f"  {i+1}. {nombre_var}: IC = {ic:.4f}")
            else:
                if self.verbose:
                    print(f"  {i+1}. {nombre_var}: NO ENCONTRADA")

        # Crear DataFrame
        df_sensibilidad = pd.DataFrame(resultados)

        if len(df_sensibilidad) == 0:
            print(f"\n✗ No se encontraron transformaciones variadas en features_df")
            return df_sensibilidad

        # Calcular estadísticas
        ic_base = df_sensibilidad.iloc[0]['IC'] if len(df_sensibilidad) > 0 else np.nan
        ic_mean = df_sensibilidad['IC'].mean()
        ic_std = df_sensibilidad['IC'].std()
        ic_min = df_sensibilidad['IC'].min()
        ic_max = df_sensibilidad['IC'].max()

        # Evaluación de robustez
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RESULTADOS - SENSIBILIDAD A PARÁMETROS")
            print(f"{'='*80}")
            print(f"IC Base:        {ic_base:.4f}")
            print(f"IC Promedio:    {ic_mean:.4f}")
            print(f"IC Std:         {ic_std:.4f}")
            print(f"IC Rango:       [{ic_min:.4f}, {ic_max:.4f}]")

            print(f"\n{'='*80}")
            print(f"DIAGNÓSTICO:")
            print(f"{'='*80}")

            # Criterios de robustez:
            # 1. Std bajo (< 20% del IC promedio)
            # 2. Todos los ICs tienen el mismo signo
            # 3. Rango estrecho

            cv = abs(ic_std / ic_mean) if ic_mean != 0 else np.inf  # Coef. variación
            mismo_signo = all(df_sensibilidad['IC'] > 0) or all(df_sensibilidad['IC'] < 0)

            if cv < 0.20 and mismo_signo:
                print(f"✓ ROBUSTO A PARÁMETROS")
                print(f"  - Coeficiente de variación: {cv:.2%} (< 20%)")
                print(f"  - Todos los ICs tienen el mismo signo")
                print(f"  - El edge NO depende de parámetros exactos")
                robustez = "ROBUSTO"
            elif cv < 0.50 and mismo_signo:
                print(f"⚠ MODERADAMENTE ROBUSTO")
                print(f"  - Coeficiente de variación: {cv:.2%} (20-50%)")
                print(f"  - Hay cierta variabilidad pero mantiene el signo")
                print(f"  - Precaución: sensible a parámetros")
                robustez = "MODERADO"
            else:
                print(f"✗ FRÁGIL")
                print(f"  - Coeficiente de variación: {cv:.2%} (> 50%)")
                if not mismo_signo:
                    print(f"  - ICs cambian de signo con parámetros diferentes")
                print(f"  - El edge depende de parámetros MUY específicos")
                print(f"  - ALTO riesgo de overfitting")
                robustez = "FRÁGIL"

        # Guardar resultados
        self.resultados_parametros = {
            'df_sensibilidad': df_sensibilidad,
            'ic_base': ic_base,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'cv': cv,
            'robustez': robustez
        }

        return df_sensibilidad

    def _extraer_parametros(self, transformacion: str) -> List[int]:
        """
        Extrae parámetros numéricos de nombre de transformación.

        Ejemplo: "mu_20_C_minus_mu_50_C" → [20, 50]

        Parameters:
        -----------
        transformacion : str
            Nombre de transformación

        Returns:
        --------
        parametros : List[int]
            Lista de parámetros numéricos
        """
        # Buscar todos los números en el nombre
        numeros = re.findall(r'\d+', transformacion)
        parametros = [int(n) for n in numeros]
        return parametros

    def _generar_variaciones_parametros(
        self,
        parametros: List[int],
        variacion_pct: float,
        n_variaciones: int
    ) -> List[List[int]]:
        """
        Genera variaciones de parámetros.

        Parameters:
        -----------
        parametros : List[int]
            Parámetros base
        variacion_pct : float
            Porcentaje de variación
        n_variaciones : int
            Número de variaciones

        Returns:
        --------
        variaciones : List[List[int]]
            Lista de conjuntos de parámetros variados
        """
        variaciones = [parametros]  # Incluir parámetros base

        for i in range(1, n_variaciones):
            # Variar cada parámetro
            params_var = []
            for p in parametros:
                # Variación aleatoria dentro del rango
                delta = int(p * variacion_pct * (i / n_variaciones))
                if i % 2 == 0:
                    p_var = p + delta
                else:
                    p_var = p - delta

                p_var = max(2, p_var)  # Mínimo 2
                params_var.append(p_var)

            variaciones.append(params_var)

        return variaciones

    def _construir_nombre_transformacion(
        self,
        transformacion_base: str,
        parametros_base: List[int],
        parametros_nuevos: List[int]
    ) -> str:
        """
        Construye nombre de transformación con nuevos parámetros.

        Parameters:
        -----------
        transformacion_base : str
            Nombre base
        parametros_base : List[int]
            Parámetros originales
        parametros_nuevos : List[int]
            Parámetros nuevos

        Returns:
        --------
        nombre_nuevo : str
            Nombre con parámetros actualizados
        """
        nombre_nuevo = transformacion_base

        # Reemplazar cada parámetro
        for p_base, p_nuevo in zip(parametros_base, parametros_nuevos):
            # Reemplazar solo la primera ocurrencia
            nombre_nuevo = nombre_nuevo.replace(
                f'_{p_base}_',
                f'_{p_nuevo}_',
                1
            )

        return nombre_nuevo

    # ========================================================================
    # B) ESTABILIDAD TEMPORAL
    # ========================================================================

    def analizar_estabilidad_temporal(
        self,
        transformacion: pd.Series,
        target: pd.Series,
        periodos: Optional[List[Tuple[str, str]]] = None,
        n_periodos: int = 5
    ) -> pd.DataFrame:
        """
        Analiza estabilidad temporal del IC.

        IC por año:
        - Año 1: 0.025
        - Año 2: 0.022
        - Año 3: 0.028
        - ESTABLE: IC similar cada año
        - INESTABLE: IC varía mucho o negativo en algunos años

        Parameters:
        -----------
        transformacion : pd.Series
            Serie de transformación (con índice datetime)
        target : pd.Series
            Target (con índice datetime)
        periodos : List[Tuple[str, str]], optional
            Lista de (fecha_inicio, fecha_fin) para cada período
            Si None, divide automáticamente en n_periodos
        n_periodos : int
            Número de períodos si periodos=None (default: 5)

        Returns:
        --------
        df_temporal : pd.DataFrame
            IC por período temporal
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"B) ESTABILIDAD TEMPORAL")
            print(f"{'='*80}")

        # Alinear índices
        idx_comun = transformacion.index.intersection(target.index)
        transformacion = transformacion.loc[idx_comun]
        target = target.loc[idx_comun]

        # Si no se especifican períodos, dividir automáticamente
        if periodos is None:
            periodos = self._dividir_periodos(transformacion.index, n_periodos)

        if self.verbose:
            print(f"Analizando {len(periodos)} períodos temporales...")

        # Calcular IC por período
        resultados = []

        for i, (fecha_inicio, fecha_fin) in enumerate(periodos):
            # Filtrar datos del período
            mask = (transformacion.index >= fecha_inicio) & (transformacion.index <= fecha_fin)
            trans_periodo = transformacion.loc[mask]
            target_periodo = target.loc[mask]

            if len(trans_periodo) < 30:  # Mínimo 30 observaciones
                continue

            # Calcular IC
            ic = self._calcular_ic(trans_periodo, target_periodo)

            resultados.append({
                'Periodo': i + 1,
                'Fecha_Inicio': fecha_inicio,
                'Fecha_Fin': fecha_fin,
                'N_Obs': len(trans_periodo),
                'IC': ic
            })

            if self.verbose:
                print(f"  Período {i+1}: {fecha_inicio.date()} → {fecha_fin.date()}")
                print(f"    N obs: {len(trans_periodo)}, IC: {ic:.4f}")

        # Crear DataFrame
        df_temporal = pd.DataFrame(resultados)

        if len(df_temporal) == 0:
            print(f"\n✗ No se pudieron calcular ICs temporales")
            return df_temporal

        # Calcular estadísticas
        ic_mean = df_temporal['IC'].mean()
        ic_std = df_temporal['IC'].std()
        ic_min = df_temporal['IC'].min()
        ic_max = df_temporal['IC'].max()
        n_positivos = (df_temporal['IC'] > 0).sum()
        n_total = len(df_temporal)

        # Evaluación de estabilidad
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RESULTADOS - ESTABILIDAD TEMPORAL")
            print(f"{'='*80}")
            print(f"IC Promedio:    {ic_mean:.4f}")
            print(f"IC Std:         {ic_std:.4f}")
            print(f"IC Rango:       [{ic_min:.4f}, {ic_max:.4f}]")
            print(f"Períodos IC>0:  {n_positivos}/{n_total} ({n_positivos/n_total*100:.1f}%)")

            print(f"\n{'='*80}")
            print(f"DIAGNÓSTICO:")
            print(f"{'='*80}")

            # Criterios de estabilidad:
            # 1. Mayoría de períodos con IC > 0 (>= 80%)
            # 2. Std bajo (< 50% del IC promedio)
            # 3. No hay períodos con IC muy negativo (< -0.02)

            pct_positivos = n_positivos / n_total
            cv = abs(ic_std / ic_mean) if ic_mean != 0 else np.inf
            tiene_muy_negativos = ic_min < -0.02

            if pct_positivos >= 0.80 and cv < 0.50 and not tiene_muy_negativos:
                print(f"✓ ESTABLE TEMPORALMENTE")
                print(f"  - {n_positivos}/{n_total} períodos con IC > 0")
                print(f"  - Coeficiente de variación: {cv:.2%}")
                print(f"  - Funciona consistentemente en el tiempo")
                print(f"  - Edge probablemente estructural")
                estabilidad = "ESTABLE"
            elif pct_positivos >= 0.60:
                print(f"⚠ MODERADAMENTE ESTABLE")
                print(f"  - {n_positivos}/{n_total} períodos con IC > 0")
                print(f"  - Cierta variabilidad temporal")
                print(f"  - Posible dependencia de régimen de mercado")
                estabilidad = "MODERADO"
            else:
                print(f"✗ INESTABLE")
                print(f"  - Solo {n_positivos}/{n_total} períodos con IC > 0")
                if tiene_muy_negativos:
                    print(f"  - Algunos períodos con IC muy negativo")
                print(f"  - Alta variabilidad temporal")
                print(f"  - Edge probablemente espurio o régimen-específico")
                estabilidad = "INESTABLE"

        # Guardar resultados
        self.resultados_temporales = {
            'df_temporal': df_temporal,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'pct_positivos': pct_positivos,
            'estabilidad': estabilidad
        }

        return df_temporal

    def _dividir_periodos(
        self,
        index: pd.DatetimeIndex,
        n_periodos: int
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Divide índice temporal en n períodos iguales.

        Parameters:
        -----------
        index : pd.DatetimeIndex
            Índice temporal
        n_periodos : int
            Número de períodos

        Returns:
        --------
        periodos : List[Tuple[pd.Timestamp, pd.Timestamp]]
            Lista de (fecha_inicio, fecha_fin)
        """
        fecha_inicio = index[0]
        fecha_fin = index[-1]
        delta_total = (fecha_fin - fecha_inicio) / n_periodos

        periodos = []
        for i in range(n_periodos):
            inicio = fecha_inicio + delta_total * i
            fin = fecha_inicio + delta_total * (i + 1)
            periodos.append((inicio, fin))

        return periodos

    # ========================================================================
    # C) CONSISTENCIA ENTRE ACTIVOS
    # ========================================================================

    def analizar_consistencia_activos(
        self,
        transformaciones_por_activo: Dict[str, pd.Series],
        targets_por_activo: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Analiza consistencia entre diferentes activos.

        IC por par:
        - EUR/USD: 0.022
        - GBP/USD: 0.019
        - USD/JPY: 0.025
        - CONSISTENTE: Funciona en múltiples pares
        - SOSPECHOSO: Solo funciona en 1 par

        Parameters:
        -----------
        transformaciones_por_activo : Dict[str, pd.Series]
            {activo: transformacion}
        targets_por_activo : Dict[str, pd.Series]
            {activo: target}

        Returns:
        --------
        df_activos : pd.DataFrame
            IC por activo
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"C) CONSISTENCIA ENTRE ACTIVOS")
            print(f"{'='*80}")
            print(f"Analizando {len(transformaciones_por_activo)} activos...")

        # Calcular IC por activo
        resultados = []

        for activo in transformaciones_por_activo.keys():
            if activo not in targets_por_activo:
                continue

            transformacion = transformaciones_por_activo[activo]
            target = targets_por_activo[activo]

            # Alinear índices
            idx_comun = transformacion.index.intersection(target.index)
            if len(idx_comun) < 30:
                continue

            transformacion = transformacion.loc[idx_comun]
            target = target.loc[idx_comun]

            # Calcular IC
            ic = self._calcular_ic(transformacion, target)

            resultados.append({
                'Activo': activo,
                'N_Obs': len(transformacion),
                'IC': ic
            })

            if self.verbose:
                print(f"  {activo}: IC = {ic:.4f} (N = {len(transformacion)})")

        # Crear DataFrame
        df_activos = pd.DataFrame(resultados)

        if len(df_activos) == 0:
            print(f"\n✗ No se pudieron calcular ICs por activo")
            return df_activos

        # Calcular estadísticas
        ic_mean = df_activos['IC'].mean()
        ic_std = df_activos['IC'].std()
        n_positivos = (df_activos['IC'] > 0).sum()
        n_total = len(df_activos)

        # Evaluación de consistencia
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RESULTADOS - CONSISTENCIA ENTRE ACTIVOS")
            print(f"{'='*80}")
            print(f"IC Promedio:    {ic_mean:.4f}")
            print(f"IC Std:         {ic_std:.4f}")
            print(f"Activos IC>0:   {n_positivos}/{n_total} ({n_positivos/n_total*100:.1f}%)")

            print(f"\n{'='*80}")
            print(f"DIAGNÓSTICO:")
            print(f"{'='*80}")

            # Criterios de consistencia:
            # 1. Mayoría de activos con IC > 0 (>= 75%)
            # 2. Funciona en al menos 3 activos diferentes

            pct_positivos = n_positivos / n_total

            if n_positivos >= 3 and pct_positivos >= 0.75:
                print(f"✓ CONSISTENTE ENTRE ACTIVOS")
                print(f"  - Funciona en {n_positivos}/{n_total} activos")
                print(f"  - Edge generalizable, no específico de un par")
                print(f"  - Mayor confianza en edge estructural")
                consistencia = "CONSISTENTE"
            elif n_positivos >= 2 and pct_positivos >= 0.50:
                print(f"⚠ PARCIALMENTE CONSISTENTE")
                print(f"  - Funciona en {n_positivos}/{n_total} activos")
                print(f"  - Cierta generalización pero no universal")
                consistencia = "MODERADO"
            else:
                print(f"✗ SOSPECHOSO")
                print(f"  - Solo funciona en {n_positivos}/{n_total} activos")
                print(f"  - Edge probablemente específico de activo")
                print(f"  - Alto riesgo de overfitting a ruido particular")
                consistencia = "SOSPECHOSO"

        # Guardar resultados
        self.resultados_activos = {
            'df_activos': df_activos,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'pct_positivos': pct_positivos,
            'consistencia': consistencia
        }

        return df_activos

    # ========================================================================
    # D) ROBUSTEZ POR RÉGIMEN DE MERCADO
    # ========================================================================

    def analizar_robustez_regimenes(
        self,
        transformacion: pd.Series,
        target: pd.Series,
        precios: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        umbral_ic: float = 0.01
    ) -> pd.DataFrame:
        """
        Analiza robustez del feature en diferentes regímenes de mercado.

        Un feature ROBUSTO debe funcionar en TODOS los regímenes:
        - Trending (ADX > 25)
        - Ranging (ADX < 20)
        - High Volatility (ATR > percentil 75)
        - Low Volatility (ATR < percentil 25)

        Si solo funciona en un régimen específico, es FRÁGIL.

        Parameters:
        -----------
        transformacion : pd.Series
            Serie de transformación (feature)
        target : pd.Series
            Target (retornos futuros)
        precios : pd.Series
            Serie de precios (close)
        high : pd.Series, optional
            Precios high (para ATR más preciso)
        low : pd.Series, optional
            Precios low (para ATR más preciso)
        umbral_ic : float
            IC mínimo requerido por régimen (default: 0.01)

        Returns:
        --------
        df_regimenes : pd.DataFrame
            IC y significancia por régimen de mercado
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"D) ROBUSTEZ POR RÉGIMEN DE MERCADO")
            print(f"{'='*80}")

        # Alinear índices
        idx_comun = transformacion.index.intersection(target.index).intersection(precios.index)
        transformacion_aligned = transformacion.loc[idx_comun]
        target_aligned = target.loc[idx_comun]
        precios_aligned = precios.loc[idx_comun]

        if high is not None and low is not None:
            high_aligned = high.loc[idx_comun]
            low_aligned = low.loc[idx_comun]
        else:
            high_aligned = None
            low_aligned = None

        # Crear detector de regímenes
        detector = RegimeDetector(
            precios=precios_aligned,
            high=high_aligned,
            low=low_aligned
        )

        # Detectar regímenes
        if self.verbose:
            print(f"Detectando regímenes de mercado...")

        regimenes = detector.detectar_regimenes()

        # Analizar por régimen
        if self.verbose:
            print(f"\nCalculando IC por régimen...")

        resultados_regimen = detector.analizar_por_regimen(
            feature=transformacion_aligned.values,
            target=target_aligned.values
        )

        # Crear DataFrame de resultados
        resultados = []
        for regimen_nombre, res in resultados_regimen.items():
            resultados.append({
                'Regimen': regimen_nombre,
                'IC': res['ic'],
                'P_Value': res['p_value'],
                'N_Obs': res['n_obs'],
                'Significativo': res['significativo']
            })

        df_regimenes = pd.DataFrame(resultados)

        # Evaluación de robustez por régimen
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RESULTADOS - ROBUSTEZ POR RÉGIMEN")
            print(f"{'='*80}")

            for _, row in df_regimenes.iterrows():
                regimen = row['Regimen']
                ic = row['IC']
                p_val = row['P_Value']
                n_obs = row['N_Obs']
                sig = row['Significativo']

                status = "✓" if (ic > umbral_ic and sig) else "✗"
                print(f"{status} {regimen:15s}: IC={ic:7.4f}, p={p_val:.4f}, N={n_obs:5d}")

            print(f"\n{'='*80}")
            print(f"DIAGNÓSTICO:")
            print(f"{'='*80}")

            # Criterios de robustez por régimen:
            # 1. IC > umbral en TODOS los regímenes
            # 2. Significativo (p < 0.01) en TODOS los regímenes
            # 3. Mismo signo en todos los regímenes

            ics_positivos = (df_regimenes['IC'] > umbral_ic).sum()
            ics_significativos = df_regimenes['Significativo'].sum()
            n_regimenes = len(df_regimenes)
            mismo_signo = all(df_regimenes['IC'] > 0) or all(df_regimenes['IC'] < 0)

            if ics_positivos == n_regimenes and ics_significativos == n_regimenes and mismo_signo:
                print(f"✓ ROBUSTO EN TODOS LOS REGÍMENES")
                print(f"  - Funciona en {ics_positivos}/{n_regimenes} regímenes (IC > {umbral_ic})")
                print(f"  - Significativo en {ics_significativos}/{n_regimenes} regímenes")
                print(f"  - Edge genuino, no depende de condiciones específicas")
                print(f"  - Alta confianza para trading real")
                robustez_regimen = "ROBUSTO"
            elif ics_positivos >= n_regimenes * 0.75 and ics_significativos >= n_regimenes * 0.75:
                print(f"⚠ MODERADAMENTE ROBUSTO")
                print(f"  - Funciona en {ics_positivos}/{n_regimenes} regímenes")
                print(f"  - Significativo en {ics_significativos}/{n_regimenes} regímenes")
                print(f"  - Cierta dependencia de régimen de mercado")
                print(f"  - Requiere monitoreo de condiciones")
                robustez_regimen = "MODERADO"
            else:
                print(f"✗ RÉGIMEN-ESPECÍFICO (FRÁGIL)")
                print(f"  - Solo funciona en {ics_positivos}/{n_regimenes} regímenes")
                print(f"  - Significativo solo en {ics_significativos}/{n_regimenes} regímenes")
                print(f"  - Edge depende de condiciones específicas")
                print(f"  - ALTO riesgo: puede fallar en cambio de régimen")
                print(f"  - NO RECOMENDADO sin gestión de régimen")
                robustez_regimen = "FRÁGIL"

        # Guardar resultados
        self.resultados_regimenes = {
            'df_regimenes': df_regimenes,
            'ics_positivos': ics_positivos,
            'ics_significativos': ics_significativos,
            'robustez_regimen': robustez_regimen
        }

        return df_regimenes

    # ========================================================================
    # UTILIDADES
    # ========================================================================

    def _calcular_ic(
        self,
        transformacion: pd.Series,
        target: pd.Series
    ) -> float:
        """
        Calcula Information Coefficient (correlación).

        Parameters:
        -----------
        transformacion : pd.Series
            Serie de transformación
        target : pd.Series
            Target

        Returns:
        --------
        ic : float
            Information Coefficient
        """
        # Eliminar NaN
        mask = ~(transformacion.isna() | target.isna())
        transformacion = transformacion[mask]
        target = target[mask]

        if len(transformacion) < 2:
            return np.nan

        ic = np.corrcoef(transformacion, target)[0, 1]
        return ic

    def generar_informe_completo(self) -> pd.DataFrame:
        """
        Genera informe completo de robustez.

        Returns:
        --------
        df_informe : pd.DataFrame
            Resumen de todos los análisis de robustez
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"INFORME COMPLETO DE ROBUSTEZ")
            print(f"{'='*80}")

        informe = []

        # A) Sensibilidad a parámetros
        if self.resultados_parametros:
            informe.append({
                'Dimensión': 'A) Parámetros',
                'Evaluación': self.resultados_parametros['robustez'],
                'IC Mean': self.resultados_parametros['ic_mean'],
                'IC Std': self.resultados_parametros['ic_std'],
                'CV': self.resultados_parametros['cv']
            })

        # B) Estabilidad temporal
        if self.resultados_temporales:
            informe.append({
                'Dimensión': 'B) Temporal',
                'Evaluación': self.resultados_temporales['estabilidad'],
                'IC Mean': self.resultados_temporales['ic_mean'],
                'IC Std': self.resultados_temporales['ic_std'],
                'Pct Positivos': self.resultados_temporales['pct_positivos']
            })

        # C) Consistencia activos
        if self.resultados_activos:
            informe.append({
                'Dimensión': 'C) Activos',
                'Evaluación': self.resultados_activos['consistencia'],
                'IC Mean': self.resultados_activos['ic_mean'],
                'IC Std': self.resultados_activos['ic_std'],
                'Pct Positivos': self.resultados_activos['pct_positivos']
            })

        # D) Robustez por régimen
        if hasattr(self, 'resultados_regimenes') and self.resultados_regimenes:
            df_reg = self.resultados_regimenes['df_regimenes']
            informe.append({
                'Dimensión': 'D) Regímenes',
                'Evaluación': self.resultados_regimenes['robustez_regimen'],
                'IC Mean': df_reg['IC'].mean(),
                'IC Std': df_reg['IC'].std(),
                'Pct Significativos': self.resultados_regimenes['ics_significativos'] / len(df_reg)
            })

        df_informe = pd.DataFrame(informe)

        if self.verbose and len(df_informe) > 0:
            print(df_informe.to_string(index=False))

            print(f"\n{'='*80}")
            print(f"EVALUACIÓN FINAL")
            print(f"{'='*80}")

            # Contar evaluaciones
            evaluaciones = df_informe['Evaluación'].tolist()
            n_robusto = sum([1 for e in evaluaciones if e in ['ROBUSTO', 'ESTABLE', 'CONSISTENTE']])
            n_total = len(evaluaciones)

            if n_robusto == n_total:
                print(f"✓ EDGE ALTAMENTE ROBUSTO")
                print(f"  - Robusto en TODAS las dimensiones ({n_robusto}/{n_total})")
                print(f"  - Alta confianza en edge genuino")
                print(f"  - Bajo riesgo de overfitting")
            elif n_robusto >= n_total * 0.6:
                print(f"⚠ EDGE MODERADAMENTE ROBUSTO")
                print(f"  - Robusto en {n_robusto}/{n_total} dimensiones")
                print(f"  - Requiere monitoreo adicional")
            else:
                print(f"✗ EDGE FRÁGIL")
                print(f"  - Solo robusto en {n_robusto}/{n_total} dimensiones")
                print(f"  - Alto riesgo de overfitting")
                print(f"  - NO RECOMENDADO para trading real")

        return df_informe


def ejemplo_uso():
    """
    Ejemplo de uso del Análisis de Robustez.
    """
    print("="*80)
    print("EJEMPLO: Análisis de Robustez")
    print("="*80)

    # Simular datos
    np.random.seed(42)
    n_obs = 1000

    # Crear índice temporal (5 años)
    dates = pd.date_range(start='2019-01-01', periods=n_obs, freq='D')

    # Simular transformación y target
    transformacion = pd.Series(
        np.random.normal(0, 1, n_obs),
        index=dates,
        name='mu_20_C_minus_mu_50_C'
    )

    target = pd.Series(
        0.05 * transformacion + np.random.normal(0, 0.01, n_obs),
        index=dates,
        name='retorno_futuro'
    )

    # Inicializar análisis
    analisis = AnalisisRobustez(verbose=True)

    # B) Estabilidad temporal
    print("\n" + "="*80)
    print("Ejecutando análisis de estabilidad temporal...")
    print("="*80)

    df_temporal = analisis.analizar_estabilidad_temporal(
        transformacion=transformacion,
        target=target,
        n_periodos=5
    )

    # C) Consistencia entre activos (simulado)
    print("\n" + "="*80)
    print("Ejecutando análisis de consistencia entre activos...")
    print("="*80)

    # Simular múltiples activos
    activos = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY']
    transformaciones_por_activo = {}
    targets_por_activo = {}

    for activo in activos:
        trans = pd.Series(
            np.random.normal(0, 1, n_obs),
            index=dates,
            name=f'{activo}_transform'
        )
        tgt = pd.Series(
            0.04 * trans + np.random.normal(0, 0.01, n_obs),
            index=dates,
            name=f'{activo}_target'
        )
        transformaciones_por_activo[activo] = trans
        targets_por_activo[activo] = tgt

    df_activos = analisis.analizar_consistencia_activos(
        transformaciones_por_activo=transformaciones_por_activo,
        targets_por_activo=targets_por_activo
    )

    # Generar informe completo
    df_informe = analisis.generar_informe_completo()

    print("\n✓ Análisis de Robustez completado")


if __name__ == "__main__":
    """
    Ejecutar Análisis de Robustez.

    OBJETIVO:
    Determinar si el edge descubierto es robusto o frágil.

    DIMENSIONES:
    A) Sensibilidad a parámetros (¿solo funciona con valores exactos?)
    B) Estabilidad temporal (¿funciona en diferentes años?)
    C) Consistencia entre activos (¿generalizable a múltiples pares?)
    """
    ejemplo_uso()
