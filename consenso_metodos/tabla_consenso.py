"""
TABLA DE CONSENSO DE MÉTODOS
=============================

Evalúa CADA transformación con TODOS los métodos de análisis
y genera una tabla de consenso para identificar transformaciones
con evidencia convergente de múltiples métodos.


CADA TRANSFORMACIÓN ES EVALUADA POR TODOS LOS MÉTODOS:

Trans_ID │ IC    │ MI   │ Lasso │ RF   │ XGB  │ MLP  │ LSTM │ Hurst │ CONSENSO
─────────┼───────┼──────┼───────┼──────┼──────┼──────┼──────┼───────┼──────────
T_0247   │ ✓0.028│ ✓    │ ✓     │ ✓top10│ ✓top10│ ✓   │ ✓    │ n/a  │ 7/7 ✓✓✓
T_0891   │ ✓0.024│ ✓    │ ✓     │ ✓top20│ ✓top15│ ✓   │ ✓    │ n/a  │ 7/7 ✓✓✓
T_0156   │ ✓0.021│ ✓    │ ✓     │ ✓top25│ ✓top20│ ✗   │ ✓    │ n/a  │ 6/7 ✓✓
T_0403   │ ✓0.019│ ✗    │ ✓     │ ✗     │ ✓top50│ ✓   │ ✗    │ n/a  │ 4/7 ✓
T_1234   │ ✗0.008│ ✗    │ ✗     │ ✓top30│ ✗     │ ✗   │ ✗    │ n/a  │ 1/7 ✗
...      │       │      │       │       │       │      │      │       │


CRITERIO DE SELECCIÓN:
──────────────────────
- 6-7 métodos coinciden: Muy probablemente edge real (✓✓✓)
- 4-5 métodos coinciden: Posiblemente edge, verificar más (✓✓)
- 1-3 métodos coinciden: Probablemente ruido o overfitting (✓ o ✗)


MÉTODOS EVALUADOS:
──────────────────
1. IC (Information Coefficient): ✓ si |IC| > 0.01
2. MI (Mutual Information): ✓ si MI > 0.01
3. Lasso: ✓ si β ≠ 0 (feature seleccionado)
4. RF (Random Forest): ✓ si está en top-50 por importancia
5. XGB (XGBoost): ✓ si está en top-50 por importancia
6. MLP: ✓ si peso absoluto promedio > umbral
7. LSTM: ✓ si peso absoluto promedio > umbral
8. Hurst: n/a (se aplica a la serie, no a features individuales)


Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

# Importar módulos de análisis
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TablaConsenso:
    """
    Genera tabla de consenso evaluando cada transformación
    con múltiples métodos de análisis.
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 nombres_features: List[str]):
        """
        Inicializa tabla de consenso.

        Args:
            X: Matriz de features (n_observaciones, n_features)
            y: Vector de retornos futuros
            nombres_features: Nombres de las transformaciones
        """
        self.X = X
        self.y = y
        self.nombres_features = nombres_features
        self.n_features = len(nombres_features)

        # Resultados por método
        self.resultados_metodos = {}

        # Tabla de consenso
        self.tabla_consenso = None

        logger.info(f"Tabla de Consenso inicializada:")
        logger.info(f"  Features: {self.n_features:,}")
        logger.info(f"  Observaciones: {len(X):,}")

    def evaluar_ic(self, umbral: float = 0.01) -> pd.DataFrame:
        """
        1. Information Coefficient

        Args:
            umbral: Umbral mínimo de |IC| para considerar ✓

        Returns:
            DataFrame con IC por feature
        """
        logger.info("Evaluando IC (Information Coefficient)...")

        from analisis_multi_metodo.analisis_estadistico import AnalizadorEstadistico

        analizador = AnalizadorEstadistico(self.X, self.y, self.nombres_features)
        df_ic = analizador.calcular_information_coefficient(
            metodo='pearson',
            correccion_multipletests=True
        )

        # Marcar ✓ o ✗
        df_ic['IC_pass'] = df_ic['abs_IC'] > umbral

        self.resultados_metodos['IC'] = df_ic

        n_pass = df_ic['IC_pass'].sum()
        logger.info(f"  Features con |IC| > {umbral}: {n_pass}/{self.n_features}")

        return df_ic

    def evaluar_mi(self, umbral: float = 0.01) -> pd.DataFrame:
        """
        2. Mutual Information

        Args:
            umbral: Umbral mínimo de MI para considerar ✓

        Returns:
            DataFrame con MI por feature
        """
        logger.info("Evaluando MI (Mutual Information)...")

        from analisis_multi_metodo.analisis_estadistico import AnalizadorEstadistico

        analizador = AnalizadorEstadistico(self.X, self.y, self.nombres_features)
        df_mi = analizador.calcular_informacion_mutua()

        # Marcar ✓ o ✗
        df_mi['MI_pass'] = df_mi['MI'] > umbral

        self.resultados_metodos['MI'] = df_mi

        n_pass = df_mi['MI_pass'].sum()
        logger.info(f"  Features con MI > {umbral}: {n_pass}/{self.n_features}")

        return df_mi

    def evaluar_lasso(self) -> pd.DataFrame:
        """
        3. Lasso Regression

        Returns:
            DataFrame con coeficientes Lasso
        """
        logger.info("Evaluando Lasso...")

        from analisis_multi_metodo.analisis_estadistico import AnalizadorEstadistico

        analizador = AnalizadorEstadistico(self.X, self.y, self.nombres_features)
        resultados = analizador.regresion_lasso()

        df_lasso = resultados['coeficientes']

        # Marcar ✓ si β ≠ 0
        df_lasso['Lasso_pass'] = df_lasso['non_zero']

        self.resultados_metodos['Lasso'] = df_lasso

        n_pass = df_lasso['Lasso_pass'].sum()
        logger.info(f"  Features seleccionados por Lasso: {n_pass}/{self.n_features}")

        return df_lasso

    def evaluar_random_forest(self, top_n: int = 50) -> pd.DataFrame:
        """
        4. Random Forest

        Args:
            top_n: Número de top features a marcar como ✓

        Returns:
            DataFrame con feature importance
        """
        logger.info("Evaluando Random Forest...")

        from analisis_multi_metodo.machine_learning import AnalizadorML

        analizador = AnalizadorML(self.X, self.y, self.nombres_features)
        resultados = analizador.entrenar_random_forest(
            tarea='regresion',
            n_estimators=100,
            calcular_permutation=False
        )

        df_rf = resultados['feature_importance']

        # Marcar top-N como ✓
        df_rf['RF_rank'] = df_rf['Importance_MDI'].rank(ascending=False)
        df_rf['RF_pass'] = df_rf['RF_rank'] <= top_n

        self.resultados_metodos['RF'] = df_rf

        n_pass = df_rf['RF_pass'].sum()
        logger.info(f"  Features en top-{top_n}: {n_pass}/{self.n_features}")

        return df_rf

    def evaluar_xgboost(self, top_n: int = 50) -> Optional[pd.DataFrame]:
        """
        5. XGBoost (si disponible)

        Args:
            top_n: Número de top features a marcar como ✓

        Returns:
            DataFrame con feature importance o None si no disponible
        """
        logger.info("Evaluando XGBoost...")

        from analisis_multi_metodo.machine_learning import AnalizadorML

        analizador = AnalizadorML(self.X, self.y, self.nombres_features)

        try:
            resultados = analizador.entrenar_gradient_boosting(
                metodo='xgboost',
                tarea='regresion',
                n_estimators=100
            )

            if resultados is None:
                logger.warning("  XGBoost no disponible")
                return None

            df_xgb = resultados['feature_importance']

            # Marcar top-N como ✓
            df_xgb['XGB_rank'] = df_xgb['Importance'].rank(ascending=False)
            df_xgb['XGB_pass'] = df_xgb['XGB_rank'] <= top_n

            self.resultados_metodos['XGB'] = df_xgb

            n_pass = df_xgb['XGB_pass'].sum()
            logger.info(f"  Features en top-{top_n}: {n_pass}/{self.n_features}")

            return df_xgb

        except Exception as e:
            logger.warning(f"  Error en XGBoost: {e}")
            return None

    def construir_tabla_consenso(self) -> pd.DataFrame:
        """
        Construye la tabla de consenso final.

        Returns:
            DataFrame con consenso de todos los métodos
        """
        logger.info("="*70)
        logger.info("CONSTRUYENDO TABLA DE CONSENSO")
        logger.info("="*70)

        # Inicializar tabla
        df_consenso = pd.DataFrame({
            'Trans_ID': self.nombres_features
        })

        # Agregar resultados de cada método
        metodos_disponibles = []

        # IC
        if 'IC' in self.resultados_metodos:
            df_ic = self.resultados_metodos['IC']
            df_consenso = df_consenso.merge(
                df_ic[['Feature', 'IC', 'IC_pass']],
                left_on='Trans_ID',
                right_on='Feature',
                how='left'
            ).drop('Feature', axis=1)
            metodos_disponibles.append('IC_pass')

        # MI
        if 'MI' in self.resultados_metodos:
            df_mi = self.resultados_metodos['MI']
            df_consenso = df_consenso.merge(
                df_mi[['Feature', 'MI', 'MI_pass']],
                left_on='Trans_ID',
                right_on='Feature',
                how='left'
            ).drop('Feature', axis=1)
            metodos_disponibles.append('MI_pass')

        # Lasso
        if 'Lasso' in self.resultados_metodos:
            df_lasso = self.resultados_metodos['Lasso']
            df_consenso = df_consenso.merge(
                df_lasso[['Feature', 'Lasso_pass']],
                left_on='Trans_ID',
                right_on='Feature',
                how='left'
            ).drop('Feature', axis=1)
            metodos_disponibles.append('Lasso_pass')

        # RF
        if 'RF' in self.resultados_metodos:
            df_rf = self.resultados_metodos['RF']
            df_consenso = df_consenso.merge(
                df_rf[['Feature', 'Importance_MDI', 'RF_pass', 'RF_rank']],
                left_on='Trans_ID',
                right_on='Feature',
                how='left'
            ).drop('Feature', axis=1)
            metodos_disponibles.append('RF_pass')

        # XGB
        if 'XGB' in self.resultados_metodos and self.resultados_metodos['XGB'] is not None:
            df_xgb = self.resultados_metodos['XGB']
            df_consenso = df_consenso.merge(
                df_xgb[['Feature', 'Importance', 'XGB_pass', 'XGB_rank']],
                left_on='Trans_ID',
                right_on='Feature',
                how='left'
            ).drop('Feature', axis=1)
            metodos_disponibles.append('XGB_pass')

        # Calcular consenso
        df_consenso['N_metodos_total'] = len(metodos_disponibles)
        df_consenso['N_metodos_pass'] = df_consenso[metodos_disponibles].sum(axis=1)
        df_consenso['Consenso_pct'] = df_consenso['N_metodos_pass'] / df_consenso['N_metodos_total']

        # Categorizar consenso
        def categorizar_consenso(pct, n_pass, n_total):
            if n_pass >= 6 or (n_pass >= n_total * 0.85):
                return "✓✓✓ MUY PROBABLE"
            elif n_pass >= 4 or (n_pass >= n_total * 0.6):
                return "✓✓ POSIBLE"
            elif n_pass >= 2:
                return "✓ DÉBIL"
            else:
                return "✗ RUIDO"

        df_consenso['Categoria'] = df_consenso.apply(
            lambda row: categorizar_consenso(
                row['Consenso_pct'],
                row['N_metodos_pass'],
                row['N_metodos_total']
            ),
            axis=1
        )

        # Ordenar por consenso descendente
        df_consenso = df_consenso.sort_values('N_metodos_pass', ascending=False)

        self.tabla_consenso = df_consenso

        # Estadísticas
        logger.info(f"\nEstadísticas de consenso:")
        logger.info(f"  Métodos evaluados: {len(metodos_disponibles)}")
        for cat in ["✓✓✓ MUY PROBABLE", "✓✓ POSIBLE", "✓ DÉBIL", "✗ RUIDO"]:
            n = (df_consenso['Categoria'] == cat).sum()
            pct = n / len(df_consenso) * 100
            logger.info(f"  {cat}: {n} ({pct:.1f}%)")

        return df_consenso

    def mostrar_tabla_consenso(self, top_n: int = 20):
        """
        Muestra tabla de consenso en formato bonito.

        Args:
            top_n: Número de top features a mostrar
        """
        if self.tabla_consenso is None:
            logger.error("Primero construye la tabla con construir_tabla_consenso()")
            return

        print("\n" + "="*100)
        print("TABLA DE CONSENSO - TOP TRANSFORMACIONES")
        print("="*100)
        print()

        df_top = self.tabla_consenso.head(top_n)

        # Crear tabla formateada
        for _, row in df_top.iterrows():
            trans_id = row['Trans_ID'][:15].ljust(15)

            # IC
            ic_val = row.get('IC', np.nan)
            ic_pass = row.get('IC_pass', False)
            ic_str = f"{'✓' if ic_pass else '✗'}{ic_val:.3f}" if not np.isnan(ic_val) else "n/a"
            ic_str = ic_str.ljust(8)

            # MI
            mi_val = row.get('MI', np.nan)
            mi_pass = row.get('MI_pass', False)
            mi_str = f"{'✓' if mi_pass else '✗'}" if not np.isnan(mi_val) else "n/a"
            mi_str = mi_str.ljust(5)

            # Lasso
            lasso_pass = row.get('Lasso_pass', False)
            lasso_str = f"{'✓' if lasso_pass else '✗'}"
            lasso_str = lasso_str.ljust(7)

            # RF
            rf_pass = row.get('RF_pass', False)
            rf_rank = row.get('RF_rank', np.nan)
            rf_str = f"{'✓' if rf_pass else '✗'}top{int(rf_rank)}" if not np.isnan(rf_rank) else "n/a"
            rf_str = rf_str.ljust(10)

            # XGB
            xgb_pass = row.get('XGB_pass', False)
            xgb_rank = row.get('XGB_rank', np.nan)
            xgb_str = f"{'✓' if xgb_pass else '✗'}top{int(xgb_rank)}" if not np.isnan(xgb_rank) else "n/a"
            xgb_str = xgb_str.ljust(10)

            # Consenso
            consenso_str = f"{int(row['N_metodos_pass'])}/{int(row['N_metodos_total'])}"
            categoria = row['Categoria']

            print(f"{trans_id} │ {ic_str} │ {mi_str} │ {lasso_str} │ {rf_str} │ {xgb_str} │ {consenso_str} {categoria}")

        print()
        print("="*100)

    def exportar_consenso(self, filepath: Path):
        """
        Exporta tabla de consenso a CSV.

        Args:
            filepath: Ruta del archivo CSV
        """
        if self.tabla_consenso is None:
            logger.error("Primero construye la tabla con construir_tabla_consenso()")
            return

        self.tabla_consenso.to_csv(filepath, index=False)
        logger.info(f"Tabla de consenso exportada a: {filepath}")

    def filtrar_por_consenso(self,
                           categoria_minima: str = "✓✓ POSIBLE") -> Tuple[np.ndarray, List[str]]:
        """
        Filtra features por nivel de consenso.

        Args:
            categoria_minima: Categoría mínima a incluir

        Returns:
            (X_filtrada, nombres_filtrados)
        """
        if self.tabla_consenso is None:
            logger.error("Primero construye la tabla con construir_tabla_consenso()")
            return self.X, self.nombres_features

        categorias_orden = [
            "✓✓✓ MUY PROBABLE",
            "✓✓ POSIBLE",
            "✓ DÉBIL",
            "✗ RUIDO"
        ]

        idx_minimo = categorias_orden.index(categoria_minima)
        categorias_incluir = categorias_orden[:idx_minimo+1]

        # Filtrar
        mask = self.tabla_consenso['Categoria'].isin(categorias_incluir)
        features_seleccionados = self.tabla_consenso[mask]['Trans_ID'].tolist()

        # Encontrar índices
        indices = [i for i, name in enumerate(self.nombres_features) if name in features_seleccionados]

        X_filtrada = self.X[:, indices]
        nombres_filtrados = [self.nombres_features[i] for i in indices]

        logger.info(f"Filtrado por consenso >= '{categoria_minima}':")
        logger.info(f"  Features originales: {self.n_features}")
        logger.info(f"  Features filtrados: {len(nombres_filtrados)}")
        logger.info(f"  Reducción: {(1 - len(nombres_filtrados)/self.n_features)*100:.1f}%")

        return X_filtrada, nombres_filtrados


def ejemplo_uso():
    """
    Ejemplo de uso de la tabla de consenso.
    """
    print("="*70)
    print("EJEMPLO: TABLA DE CONSENSO DE MÉTODOS")
    print("="*70)
    print()

    # Generar datos sintéticos
    np.random.seed(42)
    n_obs = 2000
    n_features = 100

    # Features con diferentes niveles de señal
    X = np.random.randn(n_obs, n_features)

    # Target con señal en primeros 10 features
    y = np.zeros(n_obs)
    for i in range(10):
        y += (0.05 - 0.01 * i) * X[:, i]

    y += np.random.randn(n_obs) * 0.01  # Ruido

    nombres = [f"Feature_{i:04d}" for i in range(n_features)]

    # Crear tabla de consenso
    tabla = TablaConsenso(X, y, nombres)

    # Evaluar todos los métodos
    print("\nEvaluando métodos...")
    print("-" * 70)

    tabla.evaluar_ic()
    tabla.evaluar_mi()
    tabla.evaluar_lasso()
    tabla.evaluar_random_forest(top_n=20)
    # tabla.evaluar_xgboost(top_n=20)  # Si está disponible

    # Construir tabla de consenso
    df_consenso = tabla.construir_tabla_consenso()

    # Mostrar tabla
    tabla.mostrar_tabla_consenso(top_n=15)

    # Filtrar por consenso
    X_filtrada, nombres_filtrados = tabla.filtrar_por_consenso(
        categoria_minima="✓✓ POSIBLE"
    )

    print(f"\nFeatures seleccionados por consenso:")
    for i, nombre in enumerate(nombres_filtrados[:10]):
        print(f"  {i+1}. {nombre}")

    # Exportar
    # tabla.exportar_consenso(Path("tabla_consenso.csv"))


if __name__ == '__main__':
    ejemplo_uso()
