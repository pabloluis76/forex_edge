"""
Análisis de Correlación y Causalidad
=====================================

CORRELACIÓN:

ρ(X,Y) = Cov(X,Y) / (σₓ σᵧ)

CUIDADO: Correlación ≠ Causalidad

Dos variables pueden estar correlacionadas porque:
1. X causa Y
2. Y causa X
3. Z causa ambas (confounding)
4. Coincidencia (spurious correlation)


MATRIZ DE CORRELACIÓN:

          │ Ret_1h │ Ret_4h │ Vol   │ RSI  │
──────────┼────────┼────────┼───────┼──────┤
Ret_1h    │  1.00  │  0.75  │ -0.20 │ 0.30 │
Ret_4h    │  0.75  │  1.00  │ -0.15 │ 0.35 │
Vol       │ -0.20  │ -0.15  │  1.00 │-0.10 │
RSI       │  0.30  │  0.35  │ -0.10 │ 1.00 │

Features con |ρ| > 0.7 son muy similares → Redundantes


CAUSALIDAD DE GRANGER:

"X Granger-causa Y si los valores pasados de X
 ayudan a predecir Y más allá de lo que predicen
 los valores pasados de Y solos."

Prueba estadística:
1. Modelo 1: Yₜ = f(Yₜ₋₁, Yₜ₋₂, ...)
2. Modelo 2: Yₜ = f(Yₜ₋₁, Yₜ₋₂, ..., Xₜ₋₁, Xₜ₋₂, ...)
3. Si Modelo 2 es significativamente mejor → X Granger-causa Y

NOTA: Granger-causalidad es sobre PREDICCIÓN, no causalidad verdadera.


INFORMACIÓN MUTUA (No lineal):

I(X;Y) = Σ p(x,y) log[p(x,y) / (p(x)p(y))]

Captura dependencias NO LINEALES que la correlación no ve.

Correlación = 0 pero Información Mutua > 0:
→ Hay relación no lineal entre X e Y
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import digamma
from pathlib import Path
import warnings
from sklearn.neighbors import NearestNeighbors


class AnalisisCorrelacion:
    """
    Análisis de correlación entre features

    Calcula y visualiza correlaciones de Pearson, Spearman y Kendall.
    Identifica features redundantes (altamente correlacionadas).
    """

    def __init__(self):
        """Inicializa el análisis de correlación"""
        self.corr_matrix_ = None
        self.feature_names_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None,
            method: str = 'pearson') -> 'AnalisisCorrelacion':
        """
        Calcula matriz de correlación

        Args:
            X: Datos (n_samples, n_features)
            feature_names: Nombres de features
            method: Tipo de correlación ('pearson', 'spearman', 'kendall')

        Returns:
            self
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D")

        self.n_features_ = X.shape[1]

        # Nombres de features
        if feature_names is None:
            self.feature_names_ = [f'Feature_{i+1}' for i in range(self.n_features_)]
        else:
            if len(feature_names) != self.n_features_:
                raise ValueError(f"feature_names debe tener {self.n_features_} elementos")
            self.feature_names_ = feature_names

        # Calcular correlación
        if method == 'pearson':
            # ρ(X,Y) = Cov(X,Y) / (σₓ σᵧ)
            self.corr_matrix_ = np.corrcoef(X.T)

        elif method == 'spearman':
            # Correlación de rangos
            from scipy.stats import spearmanr
            self.corr_matrix_, _ = spearmanr(X, axis=0)

        elif method == 'kendall':
            # Correlación de Kendall
            self.corr_matrix_ = np.ones((self.n_features_, self.n_features_))
            for i in range(self.n_features_):
                for j in range(i+1, self.n_features_):
                    tau, _ = stats.kendalltau(X[:, i], X[:, j])
                    self.corr_matrix_[i, j] = tau
                    self.corr_matrix_[j, i] = tau

        else:
            raise ValueError("method debe ser 'pearson', 'spearman' o 'kendall'")

        return self

    def get_redundant_features(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Identifica pares de features altamente correlacionadas (redundantes)

        Args:
            threshold: Umbral de correlación (default: 0.7)

        Returns:
            Lista de tuplas (feature1, feature2, correlación)
        """
        if self.corr_matrix_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        redundant = []

        for i in range(self.n_features_):
            for j in range(i+1, self.n_features_):
                corr = abs(self.corr_matrix_[i, j])

                if corr >= threshold:
                    redundant.append((
                        self.feature_names_[i],
                        self.feature_names_[j],
                        self.corr_matrix_[i, j]
                    ))

        # Ordenar por correlación absoluta descendente
        redundant.sort(key=lambda x: abs(x[2]), reverse=True)

        return redundant

    def get_correlation_with_target(self, y: np.ndarray) -> pd.DataFrame:
        """
        Calcula correlación de cada feature con variable objetivo

        Args:
            y: Variable objetivo (n_samples,)

        Returns:
            DataFrame con correlaciones ordenadas
        """
        if self.corr_matrix_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        y = np.asarray(y)

        correlations = []
        for i, name in enumerate(self.feature_names_):
            corr = np.corrcoef(self.corr_matrix_[i], y)[0, 1]
            correlations.append({
                'Feature': name,
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })

        df = pd.DataFrame(correlations)
        df = df.sort_values('Abs_Correlation', ascending=False)

        return df

    def summary(self, threshold: float = 0.7) -> str:
        """
        Genera resumen del análisis de correlación

        Args:
            threshold: Umbral para identificar features redundantes

        Returns:
            String con resumen
        """
        if self.corr_matrix_ is None:
            return "Análisis no realizado. Llame a fit() primero."

        lines = []
        lines.append("=" * 78)
        lines.append("ANÁLISIS DE CORRELACIÓN")
        lines.append("=" * 78)
        lines.append("")

        # Información general
        lines.append(f"Número de features:  {self.n_features_}")
        lines.append("")

        # Estadísticas de correlación
        # Extraer triángulo superior (sin diagonal)
        triu_indices = np.triu_indices_from(self.corr_matrix_, k=1)
        correlations = self.corr_matrix_[triu_indices]

        lines.append("Estadísticas de Correlación:")
        lines.append("-" * 78)
        lines.append(f"Media:             {np.mean(correlations):>10.4f}")
        lines.append(f"Mediana:           {np.median(correlations):>10.4f}")
        lines.append(f"Máxima:            {np.max(correlations):>10.4f}")
        lines.append(f"Mínima:            {np.min(correlations):>10.4f}")
        lines.append(f"Desv. Estándar:    {np.std(correlations):>10.4f}")
        lines.append("")

        # Features redundantes
        redundant = self.get_redundant_features(threshold)

        lines.append(f"Features Redundantes (|ρ| > {threshold}):")
        lines.append("-" * 78)

        if redundant:
            lines.append(f"Se encontraron {len(redundant)} pares de features redundantes:")
            lines.append("")

            for feat1, feat2, corr in redundant:
                if corr > 0:
                    relationship = "↑ Positivamente correlacionadas"
                else:
                    relationship = "↓ Negativamente correlacionadas"

                lines.append(f"  • {feat1:<20} ↔ {feat2:<20} : ρ = {corr:>7.4f}")
                lines.append(f"    └─ {relationship}")

            lines.append("")
            lines.append("ADVERTENCIA:")
            lines.append("  Features muy correlacionadas son redundantes.")
            lines.append("  Considere:")
            lines.append("    - Eliminar una de cada par")
            lines.append("    - Usar PCA para combinarlas")
            lines.append("    - Usar regularización (Lasso) para selección")

        else:
            lines.append(f"✓ No se encontraron features redundantes (|ρ| > {threshold})")

        lines.append("")
        lines.append("=" * 78)

        return "\n".join(lines)

    def plot_heatmap(self, save_path: Optional[Path] = None, annotate: bool = True):
        """
        Visualiza matriz de correlación como heatmap

        Args:
            save_path: Ruta para guardar gráfico
            annotate: Si True, muestra valores en el heatmap
        """
        if self.corr_matrix_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        fig, ax = plt.subplots(figsize=(max(10, self.n_features_ * 0.6),
                                        max(8, self.n_features_ * 0.5)))

        # Crear máscara para triángulo superior
        mask = np.triu(np.ones_like(self.corr_matrix_, dtype=bool), k=1)

        sns.heatmap(
            self.corr_matrix_,
            mask=mask,
            annot=annotate,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlación de Pearson'},
            xticklabels=self.feature_names_,
            yticklabels=self.feature_names_,
            ax=ax
        )

        ax.set_title('Matriz de Correlación\n(|ρ| > 0.7 indica redundancia)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()

    def plot_correlation_network(self, threshold: float = 0.5, save_path: Optional[Path] = None):
        """
        Visualiza correlaciones como grafo/red

        Args:
            threshold: Mostrar solo correlaciones con |ρ| > threshold
            save_path: Ruta para guardar gráfico
        """
        if self.corr_matrix_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        try:
            import networkx as nx
        except ImportError:
            print("networkx no está instalado. Instale con: pip install networkx")
            return

        # Crear grafo
        G = nx.Graph()

        # Agregar nodos
        for name in self.feature_names_:
            G.add_node(name)

        # Agregar edges (correlaciones significativas)
        for i in range(self.n_features_):
            for j in range(i+1, self.n_features_):
                corr = self.corr_matrix_[i, j]

                if abs(corr) >= threshold:
                    G.add_edge(
                        self.feature_names_[i],
                        self.feature_names_[j],
                        weight=abs(corr),
                        correlation=corr
                    )

        # Visualizar
        fig, ax = plt.subplots(figsize=(14, 10))

        pos = nx.spring_layout(G, k=1.5, iterations=50)

        # Dibujar nodos
        nx.draw_networkx_nodes(
            G, pos,
            node_size=3000,
            node_color='lightblue',
            edgecolors='black',
            linewidths=2,
            ax=ax
        )

        # Dibujar edges con colores según correlación
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        correlations = [G[u][v]['correlation'] for u, v in edges]

        # Color: rojo si negativo, azul si positivo
        edge_colors = ['red' if c < 0 else 'blue' for c in correlations]

        nx.draw_networkx_edges(
            G, pos,
            width=[w * 3 for w in weights],
            edge_color=edge_colors,
            alpha=0.6,
            ax=ax
        )

        # Etiquetas
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )

        # Etiquetas de correlación en edges
        edge_labels = {(u, v): f"{G[u][v]['correlation']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels,
            font_size=8,
            ax=ax
        )

        ax.set_title(f'Red de Correlaciones (|ρ| > {threshold})\n'
                    f'Azul: Positiva, Rojo: Negativa',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()


class GrangerCausalidad:
    """
    Prueba de Causalidad de Granger

    "X Granger-causa Y si los valores pasados de X
     ayudan a predecir Y más allá de lo que predicen
     los valores pasados de Y solos."

    NOTA: Granger-causalidad es sobre PREDICCIÓN, no causalidad verdadera.
    """

    def __init__(self, max_lag: int = 5):
        """
        Inicializa prueba de Granger

        Args:
            max_lag: Número máximo de lags a probar
        """
        self.max_lag = max_lag

    def test(self, y: np.ndarray, x: np.ndarray, lag: Optional[int] = None) -> Dict:
        """
        Prueba si X Granger-causa Y

        Args:
            y: Serie temporal objetivo
            x: Serie temporal predictora
            lag: Número de lags (si None, usa max_lag)

        Returns:
            Diccionario con resultados:
            - f_statistic: Estadístico F
            - p_value: p-valor
            - granger_causes: True si X Granger-causa Y (p < 0.05)
            - lag: Número de lags usado
        """
        y = np.asarray(y)
        x = np.asarray(x)

        if len(y) != len(x):
            raise ValueError("y y x deben tener la misma longitud")

        if lag is None:
            lag = self.max_lag

        # Modelo 1: Yₜ = β₀ + β₁Yₜ₋₁ + ... + βₚYₜ₋ₚ + εₜ
        # (Solo valores pasados de Y)

        # Crear matriz de lags de Y
        n = len(y) - lag
        Y_matrix = np.zeros((n, lag))
        y_target = y[lag:]

        for i in range(lag):
            Y_matrix[:, i] = y[lag-i-1:-i-1] if i > 0 else y[lag-1:-1]

        # Agregar intercepto
        Y_matrix_with_intercept = np.column_stack([np.ones(n), Y_matrix])

        # Ajustar Modelo 1
        try:
            beta1 = np.linalg.lstsq(Y_matrix_with_intercept, y_target, rcond=None)[0]
            y_pred1 = Y_matrix_with_intercept @ beta1
            rss1 = np.sum((y_target - y_pred1) ** 2)  # Residual sum of squares
        except np.linalg.LinAlgError:
            return {
                'f_statistic': np.nan,
                'p_value': 1.0,
                'granger_causes': False,
                'lag': lag
            }

        # Modelo 2: Yₜ = β₀ + β₁Yₜ₋₁ + ... + βₚYₜ₋ₚ + γ₁Xₜ₋₁ + ... + γₚXₜ₋ₚ + εₜ
        # (Valores pasados de Y + valores pasados de X)

        # Crear matriz de lags de X
        X_matrix = np.zeros((n, lag))

        for i in range(lag):
            X_matrix[:, i] = x[lag-i-1:-i-1] if i > 0 else x[lag-1:-1]

        # Combinar Y_matrix y X_matrix
        YX_matrix = np.column_stack([Y_matrix, X_matrix])
        YX_matrix_with_intercept = np.column_stack([np.ones(n), YX_matrix])

        # Ajustar Modelo 2
        try:
            beta2 = np.linalg.lstsq(YX_matrix_with_intercept, y_target, rcond=None)[0]
            y_pred2 = YX_matrix_with_intercept @ beta2
            rss2 = np.sum((y_target - y_pred2) ** 2)
        except np.linalg.LinAlgError:
            return {
                'f_statistic': np.nan,
                'p_value': 1.0,
                'granger_causes': False,
                'lag': lag
            }

        # Prueba F
        # F = [(RSS₁ - RSS₂) / q] / [RSS₂ / (n - k)]
        # q = número de restricciones (lag)
        # k = número de parámetros en Modelo 2

        q = lag
        k = YX_matrix_with_intercept.shape[1]
        df1 = q
        df2 = n - k

        if df2 <= 0 or rss2 <= 0:
            return {
                'f_statistic': np.nan,
                'p_value': 1.0,
                'granger_causes': False,
                'lag': lag
            }

        f_stat = ((rss1 - rss2) / q) / (rss2 / df2)

        # p-valor
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'granger_causes': p_value < 0.05,
            'lag': lag
        }

    def test_all_pairs(self, data: np.ndarray, feature_names: List[str],
                      lag: Optional[int] = None) -> pd.DataFrame:
        """
        Prueba Granger-causalidad para todos los pares de features

        Args:
            data: Datos (n_samples, n_features)
            feature_names: Nombres de features
            lag: Número de lags

        Returns:
            DataFrame con resultados
        """
        n_features = data.shape[1]
        results = []

        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    continue

                # Probar si feature i Granger-causa feature j
                result = self.test(data[:, j], data[:, i], lag=lag)

                results.append({
                    'Causa': feature_names[i],
                    'Efecto': feature_names[j],
                    'F-statistic': result['f_statistic'],
                    'p-value': result['p_value'],
                    'Granger-causa': result['granger_causes'],
                    'Lag': result['lag']
                })

        df = pd.DataFrame(results)
        df = df.sort_values('p-value')

        return df

    def plot_causality_graph(self, results_df: pd.DataFrame,
                            threshold: float = 0.05,
                            save_path: Optional[Path] = None):
        """
        Visualiza relaciones de Granger-causalidad como grafo dirigido

        Args:
            results_df: DataFrame de resultados de test_all_pairs
            threshold: Umbral de p-valor para considerar causalidad
            save_path: Ruta para guardar gráfico
        """
        try:
            import networkx as nx
        except ImportError:
            print("networkx no está instalado. Instale con: pip install networkx")
            return

        # Crear grafo dirigido
        G = nx.DiGraph()

        # Filtrar solo relaciones significativas
        significant = results_df[results_df['p-value'] < threshold]

        # Agregar edges
        for _, row in significant.iterrows():
            G.add_edge(
                row['Causa'],
                row['Efecto'],
                weight=1 - row['p-value'],  # Menor p-valor = mayor peso
                f_stat=row['F-statistic'],
                p_value=row['p-value']
            )

        if len(G.edges()) == 0:
            print(f"No se encontraron relaciones de Granger-causalidad con p < {threshold}")
            return

        # Visualizar
        fig, ax = plt.subplots(figsize=(14, 10))

        pos = nx.spring_layout(G, k=2, iterations=50)

        # Nodos
        nx.draw_networkx_nodes(
            G, pos,
            node_size=4000,
            node_color='lightgreen',
            edgecolors='black',
            linewidths=2,
            ax=ax
        )

        # Edges (flechas)
        nx.draw_networkx_edges(
            G, pos,
            width=2,
            alpha=0.6,
            edge_color='black',
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )

        # Etiquetas de nodos
        nx.draw_networkx_labels(
            G, pos,
            font_size=11,
            font_weight='bold',
            ax=ax
        )

        # Etiquetas de edges (p-valores)
        edge_labels = {(u, v): f"p={G[u][v]['p_value']:.3f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels,
            font_size=8,
            ax=ax
        )

        ax.set_title(f'Granger-Causalidad (p < {threshold})\n'
                    f'A → B significa "A Granger-causa B"',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()


class InformacionMutua:
    """
    Información Mutua: Mide dependencia NO LINEAL entre variables

    I(X;Y) = Σ p(x,y) log[p(x,y) / (p(x)p(y))]

    Captura relaciones no lineales que la correlación de Pearson no detecta.

    Correlación = 0 pero I(X;Y) > 0 → Relación no lineal
    """

    def __init__(self, n_neighbors: int = 3):
        """
        Inicializa cálculo de información mutua

        Args:
            n_neighbors: Número de vecinos para estimación KNN
        """
        self.n_neighbors = n_neighbors

    def estimate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Estima información mutua entre X e Y usando método KNN

        Args:
            x: Variable 1 (n_samples,)
            y: Variable 2 (n_samples,)

        Returns:
            Información mutua estimada (en nats)
        """
        x = np.asarray(x).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        n_samples = len(x)

        # Combinar x e y
        xy = np.hstack([x, y])

        # KNN en espacio conjunto (x,y)
        nn_xy = NearestNeighbors(n_neighbors=self.n_neighbors, metric='chebyshev')
        nn_xy.fit(xy)
        distances_xy, _ = nn_xy.kneighbors(xy)
        epsilon = distances_xy[:, -1]  # Distancia al k-ésimo vecino

        # KNN en espacio marginal x
        nn_x = NearestNeighbors(metric='chebyshev')
        nn_x.fit(x)

        # KNN en espacio marginal y
        nn_y = NearestNeighbors(metric='chebyshev')
        nn_y.fit(y)

        # Contar vecinos dentro de epsilon
        n_x = np.array([len(nn_x.radius_neighbors([xi], radius=eps, return_distance=False)[0]) - 1
                        for xi, eps in zip(x, epsilon)])
        n_y = np.array([len(nn_y.radius_neighbors([yi], radius=eps, return_distance=False)[0]) - 1
                        for yi, eps in zip(y, epsilon)])

        # Estimador de información mutua
        mi = (digamma(n_samples) + digamma(self.n_neighbors) -
              np.mean(digamma(n_x + 1) + digamma(n_y + 1)))

        return max(0, mi)  # No puede ser negativa

    def compute_matrix(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calcula matriz de información mutua entre todas las features

        Args:
            X: Datos (n_samples, n_features)
            feature_names: Nombres de features

        Returns:
            DataFrame con matriz de información mutua
        """
        n_features = X.shape[1]

        if feature_names is None:
            feature_names = [f'Feature_{i+1}' for i in range(n_features)]

        # Matriz de información mutua
        mi_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:
                    # Información mutua consigo misma = entropía
                    mi_matrix[i, j] = np.nan
                else:
                    mi = self.estimate(X[:, i], X[:, j])
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi

        df = pd.DataFrame(mi_matrix, columns=feature_names, index=feature_names)

        return df

    def compare_with_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Compara correlación lineal vs información mutua

        Útil para detectar relaciones no lineales

        Args:
            x: Variable 1
            y: Variable 2

        Returns:
            Diccionario con correlación, información mutua y tipo de relación
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # Correlación de Pearson
        corr = np.corrcoef(x, y)[0, 1]

        # Información mutua
        mi = self.estimate(x, y)

        # Determinar tipo de relación
        if abs(corr) > 0.7:
            if mi > 0.5:
                tipo = "Fuerte relación lineal"
            else:
                tipo = "Relación lineal moderada"
        elif abs(corr) < 0.3 and mi > 0.5:
            tipo = "Relación NO LINEAL (correlación baja pero MI alta)"
        elif abs(corr) < 0.3 and mi < 0.3:
            tipo = "Sin relación significativa"
        else:
            tipo = "Relación mixta (lineal + no lineal)"

        return {
            'correlation': corr,
            'mutual_information': mi,
            'relationship_type': tipo
        }


def ejemplo_correlacion_redundancia():
    """
    Ejemplo de análisis de correlación y detección de features redundantes
    """
    print("\n" + "=" * 78)
    print("EJEMPLO: Análisis de Correlación y Redundancia")
    print("=" * 78 + "\n")

    # Generar datos sintéticos
    np.random.seed(42)
    n_samples = 500

    # Features correlacionadas (redundantes)
    ret_1h = np.random.randn(n_samples) * 0.01
    ret_2h = ret_1h * 1.8 + np.random.randn(n_samples) * 0.002  # Muy correlacionado con ret_1h
    ret_4h = ret_1h * 3.2 + np.random.randn(n_samples) * 0.003

    # Features independientes
    volatilidad = np.abs(np.random.randn(n_samples) * 0.005)
    rsi = np.random.uniform(30, 70, n_samples)

    # Feature con correlación negativa
    spread = -ret_1h * 0.5 + np.random.randn(n_samples) * 0.001

    X = np.column_stack([ret_1h, ret_2h, ret_4h, volatilidad, rsi, spread])
    feature_names = ['Ret_1h', 'Ret_2h', 'Ret_4h', 'Volatilidad', 'RSI', 'Spread']

    print(f"Datos: {n_samples} muestras, {X.shape[1]} features")
    print()

    # Análisis de correlación
    analisis = AnalisisCorrelacion()
    analisis.fit(X, feature_names=feature_names, method='pearson')

    print(analisis.summary(threshold=0.7))

    # Visualizaciones
    try:
        analisis.plot_heatmap()
        analisis.plot_correlation_network(threshold=0.5)
    except Exception as e:
        print(f"No se pudieron generar gráficos: {e}")


def ejemplo_granger_causalidad():
    """
    Ejemplo de prueba de Granger-causalidad
    """
    print("\n" + "=" * 78)
    print("EJEMPLO: Granger-Causalidad")
    print("=" * 78 + "\n")

    # Generar series temporales con causalidad
    np.random.seed(42)
    n_samples = 200

    # X causa Y (con lag)
    x = np.random.randn(n_samples)
    y = np.zeros(n_samples)

    for t in range(5, n_samples):
        # Y depende de sus propios lags + lags de X
        y[t] = 0.3 * y[t-1] + 0.5 * x[t-2] + np.random.randn() * 0.1

    # Z es independiente
    z = np.random.randn(n_samples)

    # Combinar datos
    data = np.column_stack([x, y, z])
    feature_names = ['X', 'Y', 'Z']

    print("Series temporales generadas:")
    print("  • X → Y (X Granger-causa Y con lag 2)")
    print("  • Z es independiente")
    print()

    # Prueba de Granger
    granger = GrangerCausalidad(max_lag=5)

    # Probar todos los pares
    results = granger.test_all_pairs(data, feature_names, lag=3)

    print("Resultados de Granger-Causalidad:")
    print("=" * 78)
    print(results.to_string(index=False))
    print()

    # Filtrar relaciones significativas
    significant = results[results['Granger-causa'] == True]

    if len(significant) > 0:
        print("\nRelaciones de Granger-causalidad detectadas:")
        print("-" * 78)
        for _, row in significant.iterrows():
            print(f"  • {row['Causa']} → {row['Efecto']} "
                  f"(F={row['F-statistic']:.2f}, p={row['p-value']:.4f})")
    else:
        print("\nNo se detectaron relaciones de Granger-causalidad significativas")

    print()

    # Visualizar
    try:
        granger.plot_causality_graph(results, threshold=0.05)
    except Exception as e:
        print(f"No se pudo generar gráfico: {e}")


def ejemplo_informacion_mutua():
    """
    Ejemplo de información mutua para detectar relaciones no lineales
    """
    print("\n" + "=" * 78)
    print("EJEMPLO: Información Mutua (Relaciones No Lineales)")
    print("=" * 78 + "\n")

    np.random.seed(42)
    n_samples = 500

    # Caso 1: Relación LINEAL
    x1 = np.random.randn(n_samples)
    y1 = 0.8 * x1 + np.random.randn(n_samples) * 0.3

    # Caso 2: Relación NO LINEAL (cuadrática)
    x2 = np.random.randn(n_samples)
    y2 = x2**2 + np.random.randn(n_samples) * 0.5

    # Caso 3: Sin relación
    x3 = np.random.randn(n_samples)
    y3 = np.random.randn(n_samples)

    # Análisis
    mi = InformacionMutua(n_neighbors=5)

    print("Comparación: Correlación vs Información Mutua")
    print("=" * 78)
    print()

    casos = [
        ("Relación LINEAL", x1, y1),
        ("Relación NO LINEAL (cuadrática)", x2, y2),
        ("Sin relación", x3, y3)
    ]

    for nombre, x, y in casos:
        result = mi.compare_with_correlation(x, y)

        print(f"{nombre}:")
        print("-" * 78)
        print(f"  Correlación de Pearson: {result['correlation']:>10.4f}")
        print(f"  Información Mutua:      {result['mutual_information']:>10.4f}")
        print(f"  Tipo: {result['relationship_type']}")
        print()

    print("INTERPRETACIÓN:")
    print("-" * 78)
    print("• Relación lineal:")
    print("    Correlación ALTA + MI ALTA")
    print()
    print("• Relación NO lineal:")
    print("    Correlación BAJA + MI ALTA")
    print("    → La correlación de Pearson NO detecta esta relación")
    print("    → La información mutua SÍ la detecta")
    print()
    print("• Sin relación:")
    print("    Correlación BAJA + MI BAJA")
    print()


if __name__ == '__main__':
    # Ejecutar ejemplos
    ejemplo_correlacion_redundancia()
    ejemplo_granger_causalidad()
    ejemplo_informacion_mutua()
