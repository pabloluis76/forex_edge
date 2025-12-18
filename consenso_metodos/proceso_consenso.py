"""
PROCESO DE CONSENSO DE MÉTODOS
===============================

Pipeline completo de 3 pasos para identificar transformaciones
con evidencia convergente de múltiples métodos independientes.


PASO 1: CADA MÉTODO GENERA SU RANKING
──────────────────────────────────────

- IC: Top 100 transformaciones por |IC|
- MI: Top 100 por información mutua
- Lasso: Transformaciones con β ≠ 0
- RF: Top 100 por feature importance
- XGBoost: Top 100 por feature importance
- MLP: Top 100 por gradient-based importance
- LSTM: Top 100 por attention weights


PASO 2: INTERSECCIÓN
────────────────────

Transformaciones que aparecen en top de MÚLTIPLES métodos.

Consenso_fuerte = transformaciones en top de ≥5 métodos
Consenso_medio = transformaciones en top de 3-4 métodos
Sin_consenso = transformaciones en top de ≤2 métodos


PASO 3: VERIFICACIÓN CRUZADA
────────────────────────────

Para cada transformación en consenso_fuerte:

- ¿IC es estable en diferentes años?
- ¿RF y XGBoost coinciden en importancia?
- ¿MLP y LSTM capturan la misma señal?
- ¿Tiene sentido económico? (opcional pero útil)


RESULTADO FINAL:
────────────────

Lista de transformaciones APROBADAS para usar en producción.


Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcesoConsenso:
    """
    Pipeline completo de consenso de métodos en 3 pasos.

    1. Generar rankings por método
    2. Calcular intersecciones
    3. Verificación cruzada
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 nombres_features: List[str],
                 top_n: int = 100):
        """
        Inicializa proceso de consenso.

        Args:
            X: Matriz de features (n_observaciones, n_features)
            y: Vector de retornos futuros
            nombres_features: Nombres de transformaciones
            top_n: Número de top features por método
        """
        self.X = X
        self.y = y
        self.nombres_features = nombres_features
        self.n_features = len(nombres_features)
        self.top_n = top_n

        # Rankings por método
        self.rankings = {}

        # Conjuntos de intersección
        self.consenso_fuerte = set()
        self.consenso_medio = set()
        self.sin_consenso = set()

        # Verificación cruzada
        self.features_aprobados = []

        logger.info(f"Proceso de Consenso inicializado:")
        logger.info(f"  Features totales: {self.n_features:,}")
        logger.info(f"  Top-N por método: {top_n}")

    # ==================== PASO 1: RANKINGS ====================

    def paso1_generar_rankings(self):
        """
        PASO 1: Generar ranking de top features para cada método.
        """
        logger.info("="*70)
        logger.info("PASO 1: GENERANDO RANKINGS POR MÉTODO")
        logger.info("="*70)

        self._ranking_ic()
        self._ranking_mi()
        self._ranking_lasso()
        self._ranking_rf()
        self._ranking_xgboost()
        # self._ranking_mlp()  # Opcional si se implementa
        # self._ranking_lstm()  # Opcional si se implementa

        # Resumen
        logger.info("\nRankings generados:")
        for metodo, features in self.rankings.items():
            logger.info(f"  {metodo}: {len(features)} features")

    def _ranking_ic(self):
        """Ranking por Information Coefficient."""
        logger.info("\n1. Ranking por IC...")

        from analisis_multi_metodo.analisis_estadistico import AnalizadorEstadistico

        analizador = AnalizadorEstadistico(self.X, self.y, self.nombres_features)
        df_ic = analizador.calcular_information_coefficient()

        # Top-N por |IC|
        top_features = df_ic.nlargest(self.top_n, 'abs_IC')['Feature'].tolist()
        self.rankings['IC'] = set(top_features)

        logger.info(f"   Top IC: {df_ic.iloc[0]['Feature']} (|IC|={df_ic.iloc[0]['abs_IC']:.4f})")

    def _ranking_mi(self):
        """Ranking por Mutual Information."""
        logger.info("\n2. Ranking por MI...")

        from analisis_multi_metodo.analisis_estadistico import AnalizadorEstadistico

        analizador = AnalizadorEstadistico(self.X, self.y, self.nombres_features)
        df_mi = analizador.calcular_informacion_mutua()

        # Top-N por MI
        top_features = df_mi.nlargest(self.top_n, 'MI')['Feature'].tolist()
        self.rankings['MI'] = set(top_features)

        logger.info(f"   Top MI: {df_mi.iloc[0]['Feature']} (MI={df_mi.iloc[0]['MI']:.4f})")

    def _ranking_lasso(self):
        """Ranking por Lasso (features seleccionados)."""
        logger.info("\n3. Ranking por Lasso...")

        from analisis_multi_metodo.analisis_estadistico import AnalizadorEstadistico

        analizador = AnalizadorEstadistico(self.X, self.y, self.nombres_features)
        resultados = analizador.regresion_lasso()

        # Features con β ≠ 0
        df_lasso = resultados['coeficientes']
        features_seleccionados = df_lasso[df_lasso['non_zero']]['Feature'].tolist()

        # Limitar a top-N si hay más
        if len(features_seleccionados) > self.top_n:
            features_seleccionados = df_lasso[df_lasso['non_zero']].nlargest(
                self.top_n, 'abs_Coef'
            )['Feature'].tolist()

        self.rankings['Lasso'] = set(features_seleccionados)

        logger.info(f"   Lasso seleccionó: {len(features_seleccionados)} features")

    def _ranking_rf(self):
        """Ranking por Random Forest importance."""
        logger.info("\n4. Ranking por Random Forest...")

        from analisis_multi_metodo.machine_learning import AnalizadorML

        analizador = AnalizadorML(self.X, self.y, self.nombres_features)
        resultados = analizador.entrenar_random_forest(
            tarea='regresion',
            n_estimators=100,
            calcular_permutation=False
        )

        # Top-N por importance
        df_rf = resultados['feature_importance']
        top_features = df_rf.nlargest(self.top_n, 'Importance_MDI')['Feature'].tolist()
        self.rankings['RF'] = set(top_features)

        logger.info(f"   Top RF: {df_rf.iloc[0]['Feature']} (Imp={df_rf.iloc[0]['Importance_MDI']:.4f})")

    def _ranking_xgboost(self):
        """Ranking por XGBoost importance."""
        logger.info("\n5. Ranking por XGBoost...")

        from analisis_multi_metodo.machine_learning import AnalizadorML

        analizador = AnalizadorML(self.X, self.y, self.nombres_features)

        try:
            resultados = analizador.entrenar_gradient_boosting(
                metodo='xgboost',
                tarea='regresion',
                n_estimators=100
            )

            if resultados is not None:
                df_xgb = resultados['feature_importance']
                top_features = df_xgb.nlargest(self.top_n, 'Importance')['Feature'].tolist()
                self.rankings['XGBoost'] = set(top_features)

                logger.info(f"   Top XGBoost: {df_xgb.iloc[0]['Feature']} (Imp={df_xgb.iloc[0]['Importance']:.4f})")
            else:
                logger.warning("   XGBoost no disponible")

        except Exception as e:
            logger.warning(f"   Error en XGBoost: {e}")

    # ==================== PASO 2: INTERSECCIÓN ====================

    def paso2_calcular_intersecciones(self):
        """
        PASO 2: Calcular intersecciones entre rankings.
        """
        logger.info("="*70)
        logger.info("PASO 2: CALCULANDO INTERSECCIONES")
        logger.info("="*70)

        # Contar en cuántos métodos aparece cada feature
        feature_counts = defaultdict(int)
        feature_metodos = defaultdict(list)

        for metodo, features in self.rankings.items():
            for feature in features:
                feature_counts[feature] += 1
                feature_metodos[feature].append(metodo)

        n_metodos_total = len(self.rankings)

        # Clasificar por consenso
        for feature, count in feature_counts.items():
            if count >= 5 or count >= n_metodos_total * 0.8:
                self.consenso_fuerte.add(feature)
            elif count >= 3:
                self.consenso_medio.add(feature)
            else:
                self.sin_consenso.add(feature)

        # Estadísticas
        logger.info(f"\nResultados de intersección:")
        logger.info(f"  Métodos evaluados: {n_metodos_total}")
        logger.info(f"  Consenso FUERTE (≥5 métodos): {len(self.consenso_fuerte)}")
        logger.info(f"  Consenso MEDIO (3-4 métodos): {len(self.consenso_medio)}")
        logger.info(f"  Sin consenso (≤2 métodos): {len(self.sin_consenso)}")

        # Mostrar top features de consenso fuerte
        if len(self.consenso_fuerte) > 0:
            logger.info(f"\nTop 10 features con consenso fuerte:")
            # Ordenar por número de métodos
            features_ordenados = sorted(
                [(f, feature_counts[f], feature_metodos[f]) for f in self.consenso_fuerte],
                key=lambda x: x[1],
                reverse=True
            )

            for i, (feature, count, metodos) in enumerate(features_ordenados[:10]):
                metodos_str = ', '.join(metodos)
                logger.info(f"   {i+1}. {feature}: {count}/{n_metodos_total} métodos ({metodos_str})")

        return feature_counts, feature_metodos

    # ==================== PASO 3: VERIFICACIÓN CRUZADA ====================

    def paso3_verificacion_cruzada(self,
                                   feature_counts: Dict,
                                   feature_metodos: Dict):
        """
        PASO 3: Verificación cruzada de features en consenso fuerte.
        """
        logger.info("="*70)
        logger.info("PASO 3: VERIFICACIÓN CRUZADA")
        logger.info("="*70)

        features_aprobados = []
        features_rechazados = []

        for feature in self.consenso_fuerte:
            # Verificaciones
            checks_passed = 0
            checks_total = 0

            # 1. ¿Está en múltiples métodos?
            n_metodos = feature_counts[feature]
            if n_metodos >= 5:
                checks_passed += 1
            checks_total += 1

            # 2. ¿RF y XGBoost coinciden?
            if 'RF' in self.rankings and 'XGBoost' in self.rankings:
                if feature in self.rankings['RF'] and feature in self.rankings['XGBoost']:
                    checks_passed += 1
                checks_total += 1

            # 3. ¿IC y MI coinciden?
            if 'IC' in self.rankings and 'MI' in self.rankings:
                if feature in self.rankings['IC'] and feature in self.rankings['MI']:
                    checks_passed += 1
                checks_total += 1

            # 4. ¿Lasso lo seleccionó?
            if 'Lasso' in self.rankings:
                if feature in self.rankings['Lasso']:
                    checks_passed += 1
                checks_total += 1

            # Criterio de aprobación: ≥75% de checks
            if checks_passed / checks_total >= 0.75:
                features_aprobados.append({
                    'Feature': feature,
                    'N_metodos': n_metodos,
                    'Metodos': feature_metodos[feature],
                    'Checks_passed': checks_passed,
                    'Checks_total': checks_total,
                    'Aprobado': True
                })
            else:
                features_rechazados.append({
                    'Feature': feature,
                    'N_metodos': n_metodos,
                    'Checks_passed': checks_passed,
                    'Checks_total': checks_total,
                    'Aprobado': False
                })

        self.features_aprobados = features_aprobados

        # Resultados
        logger.info(f"\nResultados de verificación cruzada:")
        logger.info(f"  Features en consenso fuerte: {len(self.consenso_fuerte)}")
        logger.info(f"  Features APROBADOS: {len(features_aprobados)}")
        logger.info(f"  Features rechazados: {len(features_rechazados)}")

        if len(features_aprobados) > 0:
            logger.info(f"\nTop 10 features APROBADOS:")
            for i, f in enumerate(features_aprobados[:10]):
                metodos_str = ', '.join(f['Metodos'])
                logger.info(
                    f"   {i+1}. {f['Feature']}: "
                    f"{f['N_metodos']} métodos, "
                    f"{f['Checks_passed']}/{f['Checks_total']} checks "
                    f"({metodos_str})"
                )

        return features_aprobados, features_rechazados

    def verificar_estabilidad_temporal(self,
                                      feature: str,
                                      n_splits: int = 3) -> bool:
        """
        Verifica estabilidad del IC en diferentes períodos temporales.

        Args:
            feature: Nombre del feature
            n_splits: Número de splits temporales

        Returns:
            True si IC es estable
        """
        # Encontrar índice del feature
        try:
            idx = self.nombres_features.index(feature)
        except ValueError:
            return False

        feature_data = self.X[:, idx]

        # Dividir en períodos
        n = len(feature_data)
        split_size = n // n_splits

        ics = []

        for i in range(n_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < n_splits - 1 else n

            x_split = feature_data[start:end]
            y_split = self.y[start:end]

            # Eliminar NaN
            mask = ~(np.isnan(x_split) | np.isnan(y_split))
            if mask.sum() < 10:
                continue

            # Calcular IC
            from scipy.stats import pearsonr
            ic, _ = pearsonr(x_split[mask], y_split[mask])
            ics.append(ic)

        if len(ics) < 2:
            return False

        # IC es estable si:
        # 1. Todos tienen el mismo signo
        # 2. Desviación estándar baja
        mismo_signo = all(ic * ics[0] > 0 for ic in ics)
        std_bajo = np.std(ics) < 0.01

        return mismo_signo and std_bajo

    # ==================== EJECUCIÓN COMPLETA ====================

    def ejecutar_proceso_completo(self) -> pd.DataFrame:
        """
        Ejecuta el proceso completo de consenso en 3 pasos.

        Returns:
            DataFrame con features aprobados
        """
        logger.info("\n" + "="*70)
        logger.info("EJECUTANDO PROCESO COMPLETO DE CONSENSO")
        logger.info("="*70 + "\n")

        # Paso 1
        self.paso1_generar_rankings()

        # Paso 2
        feature_counts, feature_metodos = self.paso2_calcular_intersecciones()

        # Paso 3
        features_aprobados, features_rechazados = self.paso3_verificacion_cruzada(
            feature_counts, feature_metodos
        )

        # Crear DataFrame final
        if len(features_aprobados) > 0:
            df_aprobados = pd.DataFrame(features_aprobados)
            df_aprobados = df_aprobados.sort_values('N_metodos', ascending=False)

            logger.info("\n" + "="*70)
            logger.info("PROCESO COMPLETO FINALIZADO")
            logger.info("="*70)
            logger.info(f"\nFeatures APROBADOS para producción: {len(df_aprobados)}")
            logger.info(f"Reducción: {(1 - len(df_aprobados)/self.n_features)*100:.1f}%")

            return df_aprobados
        else:
            logger.warning("No hay features aprobados")
            return pd.DataFrame()

    def exportar_resultados(self, output_dir: Path):
        """
        Exporta resultados del proceso de consenso.

        Args:
            output_dir: Directorio de salida
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Features aprobados
        if len(self.features_aprobados) > 0:
            df_aprobados = pd.DataFrame(self.features_aprobados)
            filepath = output_dir / 'features_aprobados.csv'
            df_aprobados.to_csv(filepath, index=False)
            logger.info(f"Exportado: {filepath}")

        # 2. Rankings por método
        for metodo, features in self.rankings.items():
            df = pd.DataFrame({
                'Feature': list(features),
                'Metodo': metodo
            })
            filepath = output_dir / f'ranking_{metodo.lower()}.csv'
            df.to_csv(filepath, index=False)

        logger.info(f"Resultados exportados a: {output_dir}")

    def filtrar_matriz_aprobados(self) -> Tuple[np.ndarray, List[str]]:
        """
        Filtra matriz X dejando solo features aprobados.

        Returns:
            (X_filtrada, nombres_filtrados)
        """
        if len(self.features_aprobados) == 0:
            logger.warning("No hay features aprobados para filtrar")
            return self.X, self.nombres_features

        nombres_aprobados = [f['Feature'] for f in self.features_aprobados]

        # Encontrar índices
        indices = [i for i, name in enumerate(self.nombres_features) if name in nombres_aprobados]

        X_filtrada = self.X[:, indices]
        nombres_filtrados = [self.nombres_features[i] for i in indices]

        logger.info(f"Matriz filtrada:")
        logger.info(f"  Original: ({self.X.shape[0]}, {self.X.shape[1]})")
        logger.info(f"  Filtrada: ({X_filtrada.shape[0]}, {X_filtrada.shape[1]})")
        logger.info(f"  Reducción: {(1 - X_filtrada.shape[1]/self.X.shape[1])*100:.1f}%")

        return X_filtrada, nombres_filtrados


def ejemplo_uso():
    """
    Ejemplo de uso del proceso de consenso.
    """
    print("="*70)
    print("EJEMPLO: PROCESO COMPLETO DE CONSENSO")
    print("="*70)
    print()

    # Generar datos sintéticos
    np.random.seed(42)
    n_obs = 2000
    n_features = 200

    # Features con diferentes niveles de señal
    X = np.random.randn(n_obs, n_features)

    # Target con señal en primeros 15 features
    y = np.zeros(n_obs)
    for i in range(15):
        y += (0.05 - 0.003 * i) * X[:, i]

    y += np.random.randn(n_obs) * 0.01

    nombres = [f"Feature_{i:04d}" for i in range(n_features)]

    # Ejecutar proceso de consenso
    proceso = ProcesoConsenso(X, y, nombres, top_n=50)

    # Ejecutar proceso completo
    df_aprobados = proceso.ejecutar_proceso_completo()

    # Mostrar resultados
    if len(df_aprobados) > 0:
        print("\n" + "="*70)
        print("FEATURES APROBADOS (Top 10):")
        print("="*70)
        print(df_aprobados.head(10).to_string(index=False))

    # Filtrar matriz
    X_filtrada, nombres_filtrados = proceso.filtrar_matriz_aprobados()

    print(f"\nMatriz filtrada lista para producción:")
    print(f"  Shape: {X_filtrada.shape}")
    print(f"  Features: {len(nombres_filtrados)}")

    # Exportar resultados
    # proceso.exportar_resultados(Path("resultados_consenso"))


if __name__ == '__main__':
    ejemplo_uso()
