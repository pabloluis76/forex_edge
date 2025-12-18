"""
MACHINE LEARNING PARA EVALUACIÓN DE TRANSFORMACIONES
=====================================================

Métodos de ML para identificar transformaciones predictivas
y eliminar redundancias.


A) RANDOM FOREST
────────────────
Ensemble de árboles de decisión.

Entrenar para predecir:
- Retorno continuo (regresión)
- Signo del retorno (clasificación)

OUTPUT CLAVE: Feature Importance

Ranking de transformaciones por:
- Mean Decrease Impurity (MDI)
- Permutation Importance

Las transformaciones con alta importancia son predictivas.


B) GRADIENT BOOSTING (XGBoost, LightGBM)
────────────────────────────────────────
Árboles secuenciales que corrigen errores.

Generalmente más preciso que Random Forest.
También proporciona feature importance.


C) SVM
──────
Clasificador de margen máximo.

Con kernel RBF para capturar no linealidades.

Útil para clasificar: ¿Retorno positivo o negativo?


D) CLUSTERING
─────────────
Agrupar transformaciones similares.

K-means o Hierarchical clustering sobre la matriz de correlaciones.

Si muchas transformaciones están correlacionadas:
→ Son redundantes
→ Elegir 1 representante por cluster


Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Literal
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import logging
import warnings

warnings.filterwarnings('ignore')

# Intentar importar XGBoost y LightGBM (opcionales)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost no disponible")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM no disponible")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalizadorML:
    """
    Análisis de transformaciones mediante Machine Learning.

    Identifica features predictivos y elimina redundancias.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, nombres_features: Optional[List[str]] = None):
        """
        Inicializa el analizador ML.

        Args:
            X: Matriz de features (n_observaciones, n_features)
            y: Vector de retornos futuros (n_observaciones,)
            nombres_features: Nombres de los features
        """
        self.X = X
        self.y = y
        self.n_obs, self.n_features = X.shape

        if nombres_features is None:
            self.nombres_features = [f"Feature_{i}" for i in range(self.n_features)]
        else:
            self.nombres_features = nombres_features

        # Crear versión clasificación (signo del retorno)
        self.y_clase = np.sign(y).astype(int)

        # Resultados
        self.rf_results = None
        self.gb_results = None
        self.svm_results = None
        self.clustering_results = None

        logger.info(f"Analizador ML inicializado:")
        logger.info(f"  Observaciones: {self.n_obs:,}")
        logger.info(f"  Features: {self.n_features:,}")
        logger.info(f"  Target range: [{y.min():.6f}, {y.max():.6f}]")
        logger.info(f"  Clases: {np.unique(self.y_clase, return_counts=True)}")

    def entrenar_random_forest(self,
                              tarea: Literal['regresion', 'clasificacion'] = 'regresion',
                              n_estimators: int = 100,
                              max_depth: int = 10,
                              test_size: float = 0.2,
                              calcular_permutation: bool = True) -> Dict:
        """
        A) RANDOM FOREST

        Entrena Random Forest para regresión o clasificación.

        Args:
            tarea: 'regresion' o 'clasificacion'
            n_estimators: Número de árboles
            max_depth: Profundidad máxima de árboles
            test_size: Proporción para test
            calcular_permutation: Calcular permutation importance (lento)

        Returns:
            Diccionario con resultados y feature importance
        """
        logger.info("="*70)
        logger.info(f"A) RANDOM FOREST ({tarea.upper()})")
        logger.info("="*70)

        # Eliminar NaNs
        mask = ~(np.isnan(self.X).any(axis=1) | np.isnan(self.y))
        X_clean = self.X[mask]

        if tarea == 'regresion':
            y_clean = self.y[mask]
        else:
            y_clean = self.y_clase[mask]

        # Split temporal (NO shuffle para series temporales)
        split_idx = int(len(X_clean) * (1 - test_size))
        X_train = X_clean[:split_idx]
        X_test = X_clean[split_idx:]
        y_train = y_clean[:split_idx]
        y_test = y_clean[split_idx:]

        logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entrenar modelo
        if tarea == 'regresion':
            modelo = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:
            modelo = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )

        logger.info("Entrenando modelo...")
        modelo.fit(X_train_scaled, y_train)

        # Predicciones
        y_pred_train = modelo.predict(X_train_scaled)
        y_pred_test = modelo.predict(X_test_scaled)

        # Métricas
        if tarea == 'regresion':
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mse_test = mean_squared_error(y_test, y_pred_test)

            logger.info(f"R² Train: {r2_train:.6f}")
            logger.info(f"R² Test:  {r2_test:.6f}")
            logger.info(f"MSE Test: {mse_test:.6f}")

            metricas = {
                'r2_train': r2_train,
                'r2_test': r2_test,
                'mse_test': mse_test
            }
        else:
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

            logger.info(f"Accuracy Train: {acc_train:.4f}")
            logger.info(f"Accuracy Test:  {acc_test:.4f}")
            logger.info(f"Precision:      {precision:.4f}")
            logger.info(f"Recall:         {recall:.4f}")
            logger.info(f"F1-Score:       {f1:.4f}")

            metricas = {
                'accuracy_train': acc_train,
                'accuracy_test': acc_test,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # Feature Importance - Mean Decrease Impurity (MDI)
        importance_mdi = modelo.feature_importances_

        # Permutation Importance (más robusto pero lento)
        if calcular_permutation:
            logger.info("Calculando Permutation Importance...")
            perm_importance = permutation_importance(
                modelo, X_test_scaled, y_test,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            importance_perm = perm_importance.importances_mean
        else:
            importance_perm = None

        # DataFrame de importancias
        df_importance = pd.DataFrame({
            'Feature': self.nombres_features,
            'Importance_MDI': importance_mdi,
        })

        if importance_perm is not None:
            df_importance['Importance_Perm'] = importance_perm

        df_importance = df_importance.sort_values('Importance_MDI', ascending=False)

        # Resultados
        resultados = {
            'tarea': tarea,
            'modelo': modelo,
            'scaler': scaler,
            'metricas': metricas,
            'feature_importance': df_importance,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        }

        self.rf_results = resultados

        logger.info(f"\nTop 10 Features (MDI):")
        logger.info(f"\n{df_importance.head(10).to_string(index=False)}")

        return resultados

    def entrenar_gradient_boosting(self,
                                   metodo: Literal['sklearn', 'xgboost', 'lightgbm'] = 'sklearn',
                                   tarea: Literal['regresion', 'clasificacion'] = 'regresion',
                                   n_estimators: int = 100,
                                   max_depth: int = 5,
                                   learning_rate: float = 0.1,
                                   test_size: float = 0.2) -> Dict:
        """
        B) GRADIENT BOOSTING

        Entrena Gradient Boosting (sklearn, XGBoost o LightGBM).

        Args:
            metodo: 'sklearn', 'xgboost', 'lightgbm'
            tarea: 'regresion' o 'clasificacion'
            n_estimators: Número de árboles
            max_depth: Profundidad máxima
            learning_rate: Tasa de aprendizaje
            test_size: Proporción para test

        Returns:
            Diccionario con resultados
        """
        logger.info("="*70)
        logger.info(f"B) GRADIENT BOOSTING - {metodo.upper()} ({tarea.upper()})")
        logger.info("="*70)

        # Verificar disponibilidad
        if metodo == 'xgboost' and not XGBOOST_AVAILABLE:
            logger.error("XGBoost no está instalado. Usa: pip install xgboost")
            return None
        if metodo == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM no está instalado. Usa: pip install lightgbm")
            return None

        # Eliminar NaNs
        mask = ~(np.isnan(self.X).any(axis=1) | np.isnan(self.y))
        X_clean = self.X[mask]

        if tarea == 'regresion':
            y_clean = self.y[mask]
        else:
            y_clean = self.y_clase[mask]

        # Split temporal
        split_idx = int(len(X_clean) * (1 - test_size))
        X_train = X_clean[:split_idx]
        X_test = X_clean[split_idx:]
        y_train = y_clean[:split_idx]
        y_test = y_clean[split_idx:]

        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Seleccionar modelo
        if metodo == 'sklearn':
            if tarea == 'regresion':
                modelo = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42
                )
            else:
                modelo = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42
                )

        elif metodo == 'xgboost':
            if tarea == 'regresion':
                modelo = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42
                )
            else:
                modelo = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42
                )

        elif metodo == 'lightgbm':
            if tarea == 'regresion':
                modelo = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42,
                    verbose=-1
                )
            else:
                modelo = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42,
                    verbose=-1
                )

        # Entrenar
        logger.info("Entrenando modelo...")
        modelo.fit(X_train_scaled, y_train)

        # Predicciones
        y_pred_test = modelo.predict(X_test_scaled)

        # Métricas
        if tarea == 'regresion':
            r2 = r2_score(y_test, y_pred_test)
            mse = mean_squared_error(y_test, y_pred_test)
            logger.info(f"R² Test: {r2:.6f}")
            logger.info(f"MSE Test: {mse:.6f}")
            metricas = {'r2_test': r2, 'mse_test': mse}
        else:
            acc = accuracy_score(y_test, y_pred_test)
            logger.info(f"Accuracy Test: {acc:.4f}")
            metricas = {'accuracy_test': acc}

        # Feature Importance
        importance = modelo.feature_importances_

        df_importance = pd.DataFrame({
            'Feature': self.nombres_features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        resultados = {
            'metodo': metodo,
            'tarea': tarea,
            'modelo': modelo,
            'scaler': scaler,
            'metricas': metricas,
            'feature_importance': df_importance
        }

        self.gb_results = resultados

        logger.info(f"\nTop 10 Features:")
        logger.info(f"\n{df_importance.head(10).to_string(index=False)}")

        return resultados

    def entrenar_svm(self,
                    kernel: str = 'rbf',
                    C: float = 1.0,
                    test_size: float = 0.2) -> Dict:
        """
        C) SVM (Support Vector Machine)

        Clasificador de margen máximo para predecir signo del retorno.

        Args:
            kernel: 'rbf', 'linear', 'poly'
            C: Parámetro de regularización
            test_size: Proporción para test

        Returns:
            Diccionario con resultados
        """
        logger.info("="*70)
        logger.info(f"C) SVM (kernel={kernel.upper()}, C={C})")
        logger.info("="*70)

        # Eliminar NaNs
        mask = ~(np.isnan(self.X).any(axis=1) | np.isnan(self.y_clase))
        X_clean = self.X[mask]
        y_clean = self.y_clase[mask]

        # Split temporal
        split_idx = int(len(X_clean) * (1 - test_size))
        X_train = X_clean[:split_idx]
        X_test = X_clean[split_idx:]
        y_train = y_clean[:split_idx]
        y_test = y_clean[split_idx:]

        # Normalizar (crucial para SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entrenar SVM
        logger.info("Entrenando SVM...")
        modelo = SVC(kernel=kernel, C=C, random_state=42)
        modelo.fit(X_train_scaled, y_train)

        # Predicciones
        y_pred_train = modelo.predict(X_train_scaled)
        y_pred_test = modelo.predict(X_test_scaled)

        # Métricas
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

        logger.info(f"Accuracy Train: {acc_train:.4f}")
        logger.info(f"Accuracy Test:  {acc_test:.4f}")
        logger.info(f"Precision:      {precision:.4f}")
        logger.info(f"Recall:         {recall:.4f}")
        logger.info(f"F1-Score:       {f1:.4f}")

        resultados = {
            'modelo': modelo,
            'scaler': scaler,
            'kernel': kernel,
            'C': C,
            'metricas': {
                'accuracy_train': acc_train,
                'accuracy_test': acc_test,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }

        self.svm_results = resultados

        return resultados

    def clustering_features(self,
                           metodo: Literal['kmeans', 'hierarchical'] = 'hierarchical',
                           n_clusters: int = 20,
                           umbral_corr: float = 0.7) -> Dict:
        """
        D) CLUSTERING DE FEATURES

        Agrupa transformaciones similares para identificar redundancia.

        Args:
            metodo: 'kmeans' o 'hierarchical'
            n_clusters: Número de clusters
            umbral_corr: Umbral de correlación para considerar similares

        Returns:
            Diccionario con clusters y representantes
        """
        logger.info("="*70)
        logger.info(f"D) CLUSTERING DE FEATURES ({metodo.upper()})")
        logger.info("="*70)

        # Eliminar NaNs
        mask = ~np.isnan(self.X).any(axis=1)
        X_clean = self.X[mask]

        # Calcular matriz de correlación
        logger.info("Calculando matriz de correlación...")
        corr_matrix = np.corrcoef(X_clean.T)

        # Convertir correlación a distancia
        # distancia = 1 - |correlación|
        dist_matrix = 1 - np.abs(corr_matrix)

        # Clustering
        if metodo == 'kmeans':
            logger.info(f"Ejecutando K-Means con {n_clusters} clusters...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X_clean.T)

        elif metodo == 'hierarchical':
            logger.info(f"Ejecutando Hierarchical Clustering...")
            # Convertir a forma condensada para linkage
            dist_condensed = squareform(dist_matrix, checks=False)
            linkage_matrix = linkage(dist_condensed, method='average')

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(dist_matrix)

        # Crear DataFrame con clusters
        df_clusters = pd.DataFrame({
            'Feature': self.nombres_features,
            'Cluster': labels
        })

        # Seleccionar un representante por cluster
        # Criterio: el feature con mayor varianza en su cluster
        representantes = []

        for cluster_id in range(n_clusters):
            features_en_cluster = df_clusters[df_clusters['Cluster'] == cluster_id]['Feature'].tolist()
            indices_en_cluster = [i for i, name in enumerate(self.nombres_features) if name in features_en_cluster]

            if len(indices_en_cluster) == 0:
                continue

            # Calcular varianza de cada feature en el cluster
            varianzas = np.var(X_clean[:, indices_en_cluster], axis=0)
            idx_max_var = indices_en_cluster[np.argmax(varianzas)]

            representantes.append({
                'Cluster': cluster_id,
                'Representante': self.nombres_features[idx_max_var],
                'N_Features': len(features_en_cluster),
                'Varianza': np.max(varianzas)
            })

        df_representantes = pd.DataFrame(representantes)

        # Estadísticas
        features_por_cluster = df_clusters.groupby('Cluster').size()

        logger.info(f"Clustering completado:")
        logger.info(f"  Clusters: {n_clusters}")
        logger.info(f"  Features por cluster (media): {features_por_cluster.mean():.1f}")
        logger.info(f"  Features por cluster (min): {features_por_cluster.min()}")
        logger.info(f"  Features por cluster (max): {features_por_cluster.max()}")
        logger.info(f"  Representantes seleccionados: {len(representantes)}")

        resultados = {
            'metodo': metodo,
            'n_clusters': n_clusters,
            'labels': labels,
            'corr_matrix': corr_matrix,
            'dist_matrix': dist_matrix,
            'df_clusters': df_clusters,
            'df_representantes': df_representantes,
            'features_por_cluster': features_por_cluster
        }

        if metodo == 'hierarchical':
            resultados['linkage_matrix'] = linkage_matrix

        self.clustering_results = resultados

        logger.info(f"\nRepresentantes por cluster:")
        logger.info(f"\n{df_representantes.to_string(index=False)}")

        return resultados

    def generar_reporte_completo(self) -> None:
        """
        Genera reporte completo de todos los análisis ML.
        """
        print("\n" + "="*70)
        print("REPORTE COMPLETO - MACHINE LEARNING")
        print("="*70)
        print()

        # Random Forest
        if self.rf_results is not None:
            print("A) RANDOM FOREST")
            print("-"*70)
            print(f"Tarea: {self.rf_results['tarea']}")
            print(f"Métricas: {self.rf_results['metricas']}")
            print()
            print("Top 10 Features:")
            print(self.rf_results['feature_importance'].head(10).to_string(index=False))
            print()

        # Gradient Boosting
        if self.gb_results is not None:
            print("B) GRADIENT BOOSTING")
            print("-"*70)
            print(f"Método: {self.gb_results['metodo']}")
            print(f"Métricas: {self.gb_results['metricas']}")
            print()

        # SVM
        if self.svm_results is not None:
            print("C) SVM")
            print("-"*70)
            print(f"Kernel: {self.svm_results['kernel']}")
            print(f"Métricas: {self.svm_results['metricas']}")
            print()

        # Clustering
        if self.clustering_results is not None:
            print("D) CLUSTERING")
            print("-"*70)
            print(f"Clusters: {self.clustering_results['n_clusters']}")
            print(f"Representantes seleccionados: {len(self.clustering_results['df_representantes'])}")
            print()

        print("="*70)


def ejemplo_uso():
    """
    Ejemplo de uso de análisis ML.
    """
    print("="*70)
    print("EJEMPLO: MACHINE LEARNING PARA TRANSFORMACIONES")
    print("="*70)
    print()

    # Generar datos sintéticos
    np.random.seed(42)
    n_obs = 5000
    n_features = 100

    # Features (algunos con señal, otros ruido)
    X = np.random.randn(n_obs, n_features)

    # Target con señal en primeros 5 features
    y = (
        0.05 * X[:, 0] +
        -0.03 * X[:, 1] +
        0.02 * X[:, 2] +
        np.random.randn(n_obs) * 0.01
    )

    nombres = [f"Feature_{i}" for i in range(n_features)]

    # Crear analizador
    analizador = AnalizadorML(X, y, nombres)

    # A) Random Forest
    print("\n")
    analizador.entrenar_random_forest(
        tarea='regresion',
        n_estimators=50,
        calcular_permutation=False  # Rápido para demo
    )

    # B) Gradient Boosting (sklearn)
    print("\n")
    analizador.entrenar_gradient_boosting(
        metodo='sklearn',
        tarea='regresion',
        n_estimators=50
    )

    # C) SVM
    print("\n")
    analizador.entrenar_svm(kernel='rbf')

    # D) Clustering
    print("\n")
    analizador.clustering_features(
        metodo='hierarchical',
        n_clusters=10
    )

    # Reporte completo
    analizador.generar_reporte_completo()


if __name__ == '__main__':
    ejemplo_uso()
