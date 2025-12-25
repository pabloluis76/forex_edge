"""
Walk-Forward Validation

SIMULAR TRADING REAL:
No uses información futura para entrenar.

PROCESO:
Año 1  Año 2  Año 3  Año 4  Año 5

Ventana 1: [TRAIN][TEST]
Ventana 2:   [TRAIN][TEST]
Ventana 3:     [TRAIN][TEST]

RESULTADO FINAL = Concatenar todos los TEST

EN CADA VENTANA:
1. Entrenar todos los modelos en TRAIN
2. Seleccionar transformaciones por consenso
3. Predecir en TEST (datos no vistos)
4. Registrar métricas

Si funciona en TODOS los TEST → Probablemente real
Si funciona en algunos TEST → Inestable, cuidado
Si falla en TEST → Overfitting

Author: Sistema de Edge-Finding Forex
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import warnings
warnings.filterwarnings('ignore')

# Importar módulos del sistema
sys.path.append(str(Path(__file__).parent.parent))

from analisis_multi_metodo.analisis_estadistico import AnalizadorEstadistico
from analisis_multi_metodo.machine_learning import AnalizadorML
from analisis_multi_metodo.deep_learning import ModelosDeepLearning
from analisis_multi_metodo.metodos_fisica import MetodosFisica
from consenso_metodos.proceso_consenso import ProcesoConsenso


class WalkForwardValidation:
    """
    Walk-Forward Validation para simular trading real.

    Nunca usa información futura para entrenar.
    Divide datos en ventanas deslizantes: TRAIN → TEST
    Evalúa estabilidad temporal de transformaciones.
    """

    def __init__(
        self,
        ruta_features: str,
        ruta_target: str,
        train_years: int = 2,
        test_months: int = 6,
        step_months: int = 3,
        umbral_consenso: int = 3,
        modo: str = 'rolling',
        retrain_threshold_ic: float = 0.01,
        verbose: bool = True
    ):
        """
        Inicializa el sistema de Walk-Forward Validation.

        Parameters:
        -----------
        ruta_features : str
            Ruta al CSV con las transformaciones (matriz 2D)
        ruta_target : str
            Ruta al CSV con los retornos futuros (target)
        train_years : int
            Años de datos para TRAIN (default: 2)
        test_months : int
            Meses de datos para TEST (default: 6)
        step_months : int
            Meses para avanzar la ventana (default: 3)
        umbral_consenso : int
            Número mínimo de métodos que deben concordar (default: 3)
        modo : str
            'rolling': ventana deslizante de tamaño fijo
            'expanding': ventana que crece (anchored walk-forward)
        retrain_threshold_ic : float
            IC mínimo para considerar degradación (default: 0.01)
        verbose : bool
            Imprimir progreso detallado
        """
        self.ruta_features = ruta_features
        self.ruta_target = ruta_target
        self.train_years = train_years
        self.test_months = test_months
        self.step_months = step_months
        self.umbral_consenso = umbral_consenso
        self.modo = modo
        self.retrain_threshold_ic = retrain_threshold_ic
        self.verbose = verbose

        # Resultados
        self.ventanas: List[Dict] = []
        self.resultados_test: List[pd.DataFrame] = []
        self.metricas_por_ventana: List[Dict] = []

        if self.verbose:
            print("="*80)
            print("WALK-FORWARD VALIDATION - SIMULACIÓN TRADING REAL")
            print("="*80)
            print(f"Configuración:")
            print(f"  - Train: {train_years} años")
            print(f"  - Test: {test_months} meses")
            print(f"  - Step: {step_months} meses")
            print(f"  - Modo: {modo}")
            print(f"  - Umbral consenso: {umbral_consenso} métodos")
            print(f"  - Re-train threshold IC: {retrain_threshold_ic}")
            print("="*80)

    def cargar_datos(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carga features y target desde CSV.

        Returns:
        --------
        features : pd.DataFrame
            Matriz de transformaciones (index = timestamp)
        target : pd.Series
            Retornos futuros (index = timestamp)
        """
        if self.verbose:
            print("\n[1/5] Cargando datos...")

        # Cargar features
        features = pd.read_csv(self.ruta_features, index_col=0, parse_dates=True)

        # Cargar target
        target = pd.read_csv(self.ruta_target, index_col=0, parse_dates=True)
        if isinstance(target, pd.DataFrame):
            target = target.iloc[:, 0]  # Primera columna si es DataFrame

        # Alinear índices
        idx_comun = features.index.intersection(target.index)
        features = features.loc[idx_comun]
        target = target.loc[idx_comun]

        if self.verbose:
            print(f"  ✓ Features: {features.shape}")
            print(f"  ✓ Target: {target.shape}")
            print(f"  ✓ Período: {features.index[0]} → {features.index[-1]}")

        return features, target

    def generar_ventanas(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> List[Dict]:
        """
        Genera ventanas de Train/Test deslizantes.

        PROCESO:
        Año 1  Año 2  Año 3  Año 4  Año 5
        [TRAIN][TEST]
           [TRAIN][TEST]
              [TRAIN][TEST]

        Parameters:
        -----------
        features : pd.DataFrame
            Matriz de transformaciones
        target : pd.Series
            Target

        Returns:
        --------
        ventanas : List[Dict]
            Lista de ventanas con índices train/test
        """
        if self.verbose:
            print("\n[2/5] Generando ventanas Train/Test...")

        ventanas = []
        fecha_inicio = features.index[0]
        fecha_fin = features.index[-1]

        # Calcular deltas de tiempo
        train_delta = timedelta(days=365 * self.train_years)
        test_delta = timedelta(days=30 * self.test_months)
        step_delta = timedelta(days=30 * self.step_months)

        # Generar ventanas
        ventana_num = 1
        fecha_train_inicio_inicial = fecha_inicio  # Para modo expanding
        fecha_train_inicio = fecha_inicio

        while True:
            # Modo EXPANDING: inicio de train es siempre el comienzo de los datos
            # Modo ROLLING: inicio de train avanza con cada ventana
            if self.modo == 'expanding' and ventana_num > 1:
                fecha_train_inicio = fecha_train_inicio_inicial
                fecha_train_fin = fecha_train_inicio_inicial + train_delta + (step_delta * (ventana_num - 1))
            else:
                fecha_train_fin = fecha_train_inicio + train_delta

            fecha_test_fin = fecha_train_fin + test_delta

            # Verificar que hay datos suficientes
            if fecha_test_fin > fecha_fin:
                break

            # Obtener índices
            idx_train = (features.index >= fecha_train_inicio) & (features.index < fecha_train_fin)
            idx_test = (features.index >= fecha_train_fin) & (features.index < fecha_test_fin)

            # Verificar que hay observaciones
            if idx_train.sum() < 100 or idx_test.sum() < 20:
                break

            ventana = {
                'num': ventana_num,
                'fecha_train_inicio': fecha_train_inicio,
                'fecha_train_fin': fecha_train_fin,
                'fecha_test_fin': fecha_test_fin,
                'idx_train': idx_train,
                'idx_test': idx_test,
                'n_train': idx_train.sum(),
                'n_test': idx_test.sum(),
                'modo': self.modo
            }

            ventanas.append(ventana)

            if self.verbose:
                modo_str = "EXPANDING" if self.modo == 'expanding' else "ROLLING"
                print(f"\n  Ventana {ventana_num} ({modo_str}):")
                print(f"    TRAIN: {fecha_train_inicio.date()} → {fecha_train_fin.date()} ({idx_train.sum()} obs)")
                print(f"    TEST:  {fecha_train_fin.date()} → {fecha_test_fin.date()} ({idx_test.sum()} obs)")

            # Avanzar ventana (solo para modo rolling)
            if self.modo == 'rolling':
                fecha_train_inicio += step_delta
            ventana_num += 1

        if self.verbose:
            print(f"\n  ✓ Total ventanas generadas: {len(ventanas)}")

        self.ventanas = ventanas
        return ventanas

    def procesar_ventana(
        self,
        ventana: Dict,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Dict:
        """
        Procesa una ventana: TRAIN → CONSENSO → TEST

        EN CADA VENTANA:
        1. Entrenar todos los modelos en TRAIN
        2. Seleccionar transformaciones por consenso
        3. Predecir en TEST (datos no vistos)
        4. Registrar métricas

        Parameters:
        -----------
        ventana : Dict
            Información de la ventana
        features : pd.DataFrame
            Matriz de transformaciones
        target : pd.Series
            Target

        Returns:
        --------
        resultado : Dict
            Predicciones, métricas y transformaciones seleccionadas
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"PROCESANDO VENTANA {ventana['num']}")
            print(f"{'='*80}")

        # Extraer datos de TRAIN y TEST
        X_train = features.loc[ventana['idx_train']].copy()
        y_train = target.loc[ventana['idx_train']].copy()
        X_test = features.loc[ventana['idx_test']].copy()
        y_test = target.loc[ventana['idx_test']].copy()

        # Eliminar NaN
        idx_train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
        idx_test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())

        X_train = X_train.loc[idx_train_valid]
        y_train = y_train.loc[idx_train_valid]
        X_test = X_test.loc[idx_test_valid]
        y_test = y_test.loc[idx_test_valid]

        if self.verbose:
            print(f"\n[Paso 1] Entrenar modelos en TRAIN...")
            print(f"  X_train: {X_train.shape}")
            print(f"  y_train: {y_train.shape}")

        # ============================================================
        # PASO 1: ENTRENAR TODOS LOS MODELOS EN TRAIN
        # ============================================================

        rankings = {}

        # 1.1 Análisis Estadístico
        try:
            if self.verbose:
                print(f"\n  [1.1] Análisis Estadístico...")
            analizador_est = AnalizadorEstadistico(X_train, y_train)

            # IC
            ic = analizador_est.calcular_ic()
            ic_abs = ic.abs().sort_values(ascending=False)
            rankings['IC'] = ic_abs.head(100).index.tolist()

            # MI
            mi = analizador_est.calcular_mutual_information()
            mi_sorted = mi.sort_values(ascending=False)
            rankings['MI'] = mi_sorted.head(100).index.tolist()

            # Lasso
            lasso_coef = analizador_est.entrenar_lasso(alpha=0.01)
            lasso_abs = lasso_coef.abs().sort_values(ascending=False)
            lasso_nonzero = lasso_abs[lasso_abs > 0]
            rankings['Lasso'] = lasso_nonzero.head(100).index.tolist()

            if self.verbose:
                print(f"    ✓ IC: {len(rankings['IC'])} features")
                print(f"    ✓ MI: {len(rankings['MI'])} features")
                print(f"    ✓ Lasso: {len(rankings['Lasso'])} features")
        except Exception as e:
            if self.verbose:
                print(f"    ✗ Error: {e}")
            rankings['IC'] = []
            rankings['MI'] = []
            rankings['Lasso'] = []

        # 1.2 Machine Learning
        try:
            if self.verbose:
                print(f"\n  [1.2] Machine Learning...")
            analizador_ml = AnalizadorML(X_train, y_train)

            # Random Forest
            rf_result = analizador_ml.entrenar_random_forest(
                tarea='regresion',
                n_estimators=50
            )
            rf_imp = rf_result['feature_importance_mdi'].sort_values(ascending=False)
            rankings['RF'] = rf_imp.head(100).index.tolist()

            # XGBoost
            xgb_result = analizador_ml.entrenar_xgboost(
                tarea='regresion',
                n_estimators=50
            )
            xgb_imp = xgb_result['feature_importance'].sort_values(ascending=False)
            rankings['XGB'] = xgb_imp.head(100).index.tolist()

            if self.verbose:
                print(f"    ✓ RF: {len(rankings['RF'])} features")
                print(f"    ✓ XGB: {len(rankings['XGB'])} features")
        except Exception as e:
            if self.verbose:
                print(f"    ✗ Error: {e}")
            rankings['RF'] = []
            rankings['XGB'] = []

        # 1.3 Deep Learning (opcional - puede ser lento)
        # Se puede activar si hay suficiente tiempo y recursos
        # Por ahora se omite para rapidez

        # 1.4 Métodos de Física (opcional)
        # Similar a DL, se puede activar si se desea

        # ============================================================
        # PASO 2: SELECCIONAR TRANSFORMACIONES POR CONSENSO
        # ============================================================

        if self.verbose:
            print(f"\n[Paso 2] Selección por consenso (umbral={self.umbral_consenso})...")

        # Contar votos
        todas_features = features.columns.tolist()
        votos = {f: 0 for f in todas_features}

        for metodo, top_features in rankings.items():
            for f in top_features:
                if f in votos:
                    votos[f] += 1

        # Filtrar por umbral
        features_consenso = [f for f, v in votos.items() if v >= self.umbral_consenso]
        features_consenso_sorted = sorted(
            features_consenso,
            key=lambda f: votos[f],
            reverse=True
        )

        if self.verbose:
            print(f"  ✓ Features con consenso: {len(features_consenso)}")
            if len(features_consenso) > 0:
                print(f"  ✓ Top features:")
                for f in features_consenso_sorted[:10]:
                    print(f"      {f}: {votos[f]} votos")

        # Si no hay consenso, usar top 50 por IC
        if len(features_consenso) == 0:
            if self.verbose:
                print(f"  ⚠ No hay consenso, usando top-50 por IC...")
            features_consenso = rankings.get('IC', [])[:50]

        # ============================================================
        # PASO 3: PREDECIR EN TEST (DATOS NO VISTOS)
        # ============================================================

        if self.verbose:
            print(f"\n[Paso 3] Predecir en TEST (datos no vistos)...")
            print(f"  X_test: {X_test.shape}")
            print(f"  y_test: {y_test.shape}")

        # Entrenar modelo simple con features de consenso
        X_train_consenso = X_train[features_consenso]
        X_test_consenso = X_test[features_consenso]

        try:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler

            # Normalizar
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_consenso)
            X_test_scaled = scaler.transform(X_test_consenso)

            # Entrenar Ridge
            modelo = Ridge(alpha=1.0)
            modelo.fit(X_train_scaled, y_train)

            # Predecir en TEST
            y_pred_test = modelo.predict(X_test_scaled)

            # Crear DataFrame de resultados
            df_test = pd.DataFrame({
                'timestamp': X_test.index,
                'y_true': y_test.values,
                'y_pred': y_pred_test
            })
            df_test.set_index('timestamp', inplace=True)

            if self.verbose:
                print(f"  ✓ Predicciones generadas: {len(y_pred_test)}")

        except Exception as e:
            if self.verbose:
                print(f"  ✗ Error en predicción: {e}")
            df_test = pd.DataFrame({
                'timestamp': X_test.index,
                'y_true': y_test.values,
                'y_pred': np.nan
            })
            df_test.set_index('timestamp', inplace=True)

        # ============================================================
        # PASO 4: REGISTRAR MÉTRICAS
        # ============================================================

        if self.verbose:
            print(f"\n[Paso 4] Calcular métricas...")

        metricas = self._calcular_metricas(df_test['y_true'], df_test['y_pred'])

        if self.verbose:
            print(f"\n  MÉTRICAS TEST:")
            print(f"    IC: {metricas['IC']:.4f}")
            print(f"    Sharpe: {metricas['Sharpe']:.4f}")
            print(f"    Return Anualizado: {metricas['Return_Anual']:.2f}%")
            print(f"    Volatilidad: {metricas['Volatilidad']:.2f}%")

        # Resultado de la ventana
        resultado = {
            'ventana': ventana['num'],
            'df_test': df_test,
            'metricas': metricas,
            'features_consenso': features_consenso_sorted,
            'n_features_consenso': len(features_consenso),
            'votos': votos
        }

        return resultado

    def _calcular_metricas(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict:
        """
        Calcula métricas de rendimiento.

        Parameters:
        -----------
        y_true : pd.Series
            Retornos reales
        y_pred : pd.Series
            Retornos predichos

        Returns:
        --------
        metricas : Dict
            IC, Sharpe, Return Anualizado, Volatilidad, etc.
        """
        # Eliminar NaN
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 2:
            return {
                'IC': np.nan,
                'Sharpe': np.nan,
                'Return_Anual': np.nan,
                'Volatilidad': np.nan,
                'Max_Drawdown': np.nan
            }

        # IC (Information Coefficient)
        ic = np.corrcoef(y_true, y_pred)[0, 1]

        # Retornos basados en señales (signo de predicción)
        señales = np.sign(y_pred)
        retornos_estrategia = señales * y_true

        # Return Anualizado (asumiendo datos diarios)
        # Si es H1, ajustar por 24*252 en lugar de 252
        # Para simplificar, asumimos datos diarios
        ret_medio = retornos_estrategia.mean()
        ret_anual = ret_medio * 252 * 100  # En porcentaje

        # Volatilidad
        vol = retornos_estrategia.std() * np.sqrt(252) * 100  # En porcentaje

        # Sharpe Ratio
        sharpe = (ret_anual / vol) if vol > 0 else 0

        # Max Drawdown
        cumret = (1 + retornos_estrategia).cumprod()
        running_max = cumret.cummax()
        drawdown = (cumret - running_max) / running_max
        max_dd = drawdown.min() * 100  # En porcentaje

        return {
            'IC': ic,
            'Sharpe': sharpe,
            'Return_Anual': ret_anual,
            'Volatilidad': vol,
            'Max_Drawdown': max_dd
        }

    def _detectar_degradacion(self, df_metricas: pd.DataFrame):
        """
        Detecta degradación de performance a lo largo del tiempo.

        SEÑALES DE DEGRADACIÓN:
        - IC cayendo consistentemente
        - Ventanas recientes con IC < umbral
        - Sharpe decreciente

        Parameters:
        -----------
        df_metricas : pd.DataFrame
            Métricas por ventana
        """
        if len(df_metricas) < 3:
            print(f"  ⚠ Pocas ventanas para detectar tendencias")
            return

        ics = df_metricas['IC'].values
        ventanas = df_metricas['ventana'].values

        # Detectar tendencia en IC usando regresión simple
        mask_valid = ~np.isnan(ics)
        if mask_valid.sum() < 3:
            print(f"  ⚠ Insuficientes datos válidos")
            return

        x = ventanas[mask_valid]
        y = ics[mask_valid]

        # Regresión lineal simple
        pendiente = np.polyfit(x, y, 1)[0]

        # Últimas 3 ventanas
        ultimas_3_ics = ics[-3:]
        promedio_reciente = np.nanmean(ultimas_3_ics)

        print(f"  Tendencia IC: {'decreciente' if pendiente < 0 else 'creciente'} (pendiente={pendiente:.6f})")
        print(f"  IC promedio últimas 3 ventanas: {promedio_reciente:.4f}")

        # Diagnóstico
        degradacion_detectada = False
        razones = []

        if pendiente < -0.001:  # Tendencia decreciente significativa
            degradacion_detectada = True
            razones.append("IC decreciente a lo largo del tiempo")

        if promedio_reciente < self.retrain_threshold_ic:
            degradacion_detectada = True
            razones.append(f"IC reciente ({promedio_reciente:.4f}) < umbral ({self.retrain_threshold_ic})")

        # Verificar últimas 2 ventanas negativas
        if len(ultimas_3_ics) >= 2:
            if ultimas_3_ics[-1] < 0 and ultimas_3_ics[-2] < 0:
                degradacion_detectada = True
                razones.append("Últimas 2 ventanas con IC negativo")

        if degradacion_detectada:
            print(f"\n  ⚠ DEGRADACIÓN DETECTADA:")
            for razon in razones:
                print(f"    - {razon}")
            print(f"  → RECOMENDACIÓN: Re-entrenar modelos o revisar features")
        else:
            print(f"\n  ✓ No se detectó degradación significativa")
            print(f"  → Performance estable")

    def ejecutar_walk_forward(self) -> pd.DataFrame:
        """
        Ejecuta el proceso completo de Walk-Forward Validation.

        PROCESO COMPLETO:
        1. Cargar datos
        2. Generar ventanas Train/Test
        3. Para cada ventana:
            - Entrenar modelos en TRAIN
            - Seleccionar por consenso
            - Predecir en TEST
            - Registrar métricas
        4. Concatenar todos los TEST
        5. Evaluación final

        Returns:
        --------
        df_resultados : pd.DataFrame
            Concatenación de todos los TEST con predicciones
        """
        # [1/5] Cargar datos
        features, target = self.cargar_datos()

        # [2/5] Generar ventanas
        ventanas = self.generar_ventanas(features, target)

        if len(ventanas) == 0:
            print("\n✗ ERROR: No se pudieron generar ventanas.")
            print("  Verifica que hay suficientes datos.")
            return pd.DataFrame()

        # [3/5] Procesar cada ventana
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[3/5] PROCESANDO {len(ventanas)} VENTANAS")
            print(f"{'='*80}")

        self.resultados_test = []
        self.metricas_por_ventana = []

        for ventana in ventanas:
            resultado = self.procesar_ventana(ventana, features, target)
            self.resultados_test.append(resultado['df_test'])
            self.metricas_por_ventana.append({
                'ventana': resultado['ventana'],
                **resultado['metricas'],
                'n_features': resultado['n_features_consenso']
            })

        # [4/5] Concatenar todos los TEST
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[4/5] CONCATENANDO RESULTADOS TEST")
            print(f"{'='*80}")

        df_resultados = pd.concat(self.resultados_test, axis=0)
        df_resultados.sort_index(inplace=True)

        if self.verbose:
            print(f"  ✓ Total observaciones TEST: {len(df_resultados)}")
            print(f"  ✓ Período: {df_resultados.index[0]} → {df_resultados.index[-1]}")

        # [5/5] Evaluación final
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[5/5] EVALUACIÓN FINAL")
            print(f"{'='*80}")

        self._evaluacion_final(df_resultados)

        return df_resultados

    def _evaluacion_final(self, df_resultados: pd.DataFrame):
        """
        Evaluación final de walk-forward validation.

        Si funciona en TODOS los TEST → Probablemente real
        Si funciona en algunos TEST → Inestable, cuidado
        Si falla en TEST → Overfitting

        Parameters:
        -----------
        df_resultados : pd.DataFrame
            Resultados concatenados de todos los TEST
        """
        # Métricas globales
        metricas_global = self._calcular_metricas(
            df_resultados['y_true'],
            df_resultados['y_pred']
        )

        print(f"\nMÉTRICAS GLOBALES (TODOS LOS TEST):")
        print(f"  IC:                {metricas_global['IC']:.4f}")
        print(f"  Sharpe Ratio:      {metricas_global['Sharpe']:.4f}")
        print(f"  Return Anualizado: {metricas_global['Return_Anual']:.2f}%")
        print(f"  Volatilidad:       {metricas_global['Volatilidad']:.2f}%")
        print(f"  Max Drawdown:      {metricas_global['Max_Drawdown']:.2f}%")

        # Métricas por ventana
        df_metricas = pd.DataFrame(self.metricas_por_ventana)

        print(f"\nMÉTRICAS POR VENTANA:")
        print(df_metricas.to_string(index=False))

        # Análisis de estabilidad
        print(f"\nANÁLISIS DE ESTABILIDAD:")

        # Contar ventanas con IC positivo
        ics = df_metricas['IC'].dropna()
        n_positivo = (ics > 0).sum()
        n_total = len(ics)
        pct_positivo = (n_positivo / n_total * 100) if n_total > 0 else 0

        print(f"  IC > 0: {n_positivo}/{n_total} ventanas ({pct_positivo:.1f}%)")

        # Contar ventanas con Sharpe > 0
        sharpes = df_metricas['Sharpe'].dropna()
        n_sharpe_pos = (sharpes > 0).sum()
        pct_sharpe = (n_sharpe_pos / len(sharpes) * 100) if len(sharpes) > 0 else 0

        print(f"  Sharpe > 0: {n_sharpe_pos}/{len(sharpes)} ventanas ({pct_sharpe:.1f}%)")

        # Detección de degradación de performance
        print(f"\nDETECCIÓN DE DEGRADACIÓN:")
        self._detectar_degradacion(df_metricas)

        # Diagnóstico final
        print(f"\n{'='*80}")
        print(f"DIAGNÓSTICO FINAL:")
        print(f"{'='*80}")

        if pct_positivo >= 80 and metricas_global['IC'] > 0.02:
            print("✓ PROBABLEMENTE REAL")
            print("  - Funciona consistentemente en TODOS los períodos TEST")
            print("  - IC global positivo y estable")
            print("  - Evidencia de edge genuino")
        elif pct_positivo >= 50:
            print("⚠ INESTABLE - PRECAUCIÓN")
            print("  - Funciona en ALGUNOS períodos TEST, falla en otros")
            print("  - Posible sensibilidad a régimen de mercado")
            print("  - Requiere análisis adicional de condiciones de mercado")
        else:
            print("✗ PROBABLE OVERFITTING")
            print("  - Falla en la mayoría de períodos TEST")
            print("  - IC inconsistente o negativo")
            print("  - No hay evidencia de edge real")

        print(f"{'='*80}")

    def guardar_resultados(
        self,
        df_resultados: pd.DataFrame,
        ruta_salida: Optional[str] = None
    ):
        """
        Guarda resultados de walk-forward validation.

        Parameters:
        -----------
        df_resultados : pd.DataFrame
            Resultados concatenados
        ruta_salida : str, optional
            Ruta para guardar (default: auto-genera nombre)
        """
        if ruta_salida is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_salida = f"walk_forward_resultados_{timestamp}.csv"

        df_resultados.to_csv(ruta_salida)

        # Guardar métricas por ventana
        df_metricas = pd.DataFrame(self.metricas_por_ventana)
        ruta_metricas = ruta_salida.replace('.csv', '_metricas.csv')
        df_metricas.to_csv(ruta_metricas, index=False)

        if self.verbose:
            print(f"\n✓ Resultados guardados:")
            print(f"  - Predicciones: {ruta_salida}")
            print(f"  - Métricas: {ruta_metricas}")


def ejemplo_uso():
    """
    Ejemplo de uso del sistema Walk-Forward Validation.
    """
    print("="*80)
    print("EJEMPLO: Walk-Forward Validation")
    print("="*80)

    # Rutas de ejemplo
    ruta_features = "../datos/features/transformaciones_EUR_USD_H1.csv"
    ruta_target = "../datos/features/target_retornos_futuros.csv"

    # Inicializar
    wfv = WalkForwardValidation(
        ruta_features=ruta_features,
        ruta_target=ruta_target,
        train_years=2,          # 2 años de TRAIN
        test_months=6,          # 6 meses de TEST
        step_months=3,          # Avanzar 3 meses cada vez
        umbral_consenso=3,      # Mínimo 3 métodos deben concordar
        verbose=True
    )

    # Ejecutar walk-forward validation
    df_resultados = wfv.ejecutar_walk_forward()

    # Guardar resultados
    wfv.guardar_resultados(df_resultados)

    print("\n✓ Walk-Forward Validation completado")


if __name__ == "__main__":
    """
    Ejecutar Walk-Forward Validation.

    IMPORTANTE:
    - Asegúrate de que las rutas de features y target sean correctas
    - Este proceso puede tardar varios minutos dependiendo del tamaño de datos
    - Ajusta train_years, test_months, step_months según tus necesidades
    """
    ejemplo_uso()
