"""
Principal Component Analysis (PCA)
===================================

PROBLEMA: Muchas features están correlacionadas (redundantes).

SOLUCIÓN: PCA reduce dimensionalidad encontrando direcciones de máxima varianza.


CONCEPTO:

100 features originales → 10-20 componentes principales

Cada componente es una COMBINACIÓN LINEAL de features originales.


MATEMÁTICAMENTE:

1. Calcular matriz de covarianza: Σ = X'X / n

2. Descomposición en eigenvalores/eigenvectores:
   Σ = VΛV'

   V = matriz de eigenvectores (direcciones principales)
   Λ = matriz diagonal de eigenvalues (varianza explicada)

3. Proyectar datos a nuevas dimensiones:
   Z = XV


INTERPRETACIÓN:

PC1 = Dirección de MÁXIMA varianza
PC2 = Dirección de segunda máxima varianza (ortogonal a PC1)
...
PCk = k-ésima dirección

Los primeros k componentes capturan la mayor parte de la información.


EJEMPLO EN TRADING:

Features originales:
- Retorno 1h, 2h, 4h, 8h, 12h, 24h (6 features)
- Todas muy correlacionadas entre sí

Después de PCA:
- PC1 ≈ "Momentum general" (combinación de todos)
- PC2 ≈ "Diferencia corto vs largo plazo"

6 features → 2 componentes que capturan 95% de la información


VARIANZA EXPLICADA:

Σᵢ λᵢ = varianza total

% varianza de PCk = λₖ / Σᵢ λᵢ

Típicamente:
- PC1-PC5 capturan 70-80% de varianza
- PC1-PC10 capturan 90%+ de varianza


USO EN MODELOS:

En vez de usar 100 features:
1. Aplicar PCA
2. Quedarse con k componentes (ej: 20)
3. Entrenar modelo con los componentes

VENTAJAS:
- Reduce overfitting
- Elimina multicolinealidad
- Reduce ruido
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings


class PCA:
    """
    Implementación de Principal Component Analysis (PCA)

    Reduce dimensionalidad encontrando las direcciones de máxima varianza
    en los datos y proyectando a un espacio de menor dimensión.

    Attributes:
        n_components (int): Número de componentes principales a extraer
        components_ (np.ndarray): Eigenvectores (componentes principales)
        explained_variance_ (np.ndarray): Varianza explicada por cada componente
        explained_variance_ratio_ (np.ndarray): Proporción de varianza explicada
        singular_values_ (np.ndarray): Valores singulares
        mean_ (np.ndarray): Media de cada feature (para centrar datos)
        std_ (np.ndarray): Desviación estándar de cada feature
        feature_names_ (List[str]): Nombres de las features originales
        n_features_ (int): Número de features originales
        n_samples_ (int): Número de muestras
    """

    def __init__(self, n_components: Optional[int] = None, whiten: bool = False):
        """
        Inicializa PCA

        Args:
            n_components: Número de componentes a extraer (si None, usa todas)
            whiten: Si True, los componentes tienen varianza unitaria
        """
        self.n_components = n_components
        self.whiten = whiten

        # Atributos a llenar después de fit
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.std_ = None
        self.feature_names_ = None
        self.n_features_ = None
        self.n_samples_ = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'PCA':
        """
        Ajusta PCA a los datos

        Pasos:
        1. Centrar y estandarizar datos
        2. Calcular matriz de covarianza: Σ = X'X / n
        3. Calcular eigenvalores y eigenvectores de Σ
        4. Ordenar por eigenvalores decrecientes
        5. Seleccionar top k componentes

        Args:
            X: Datos (n_samples, n_features)
            feature_names: Nombres opcionales de las features

        Returns:
            self: Modelo ajustado
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D")

        self.n_samples_, self.n_features_ = X.shape

        # Nombres de features
        if feature_names is None:
            self.feature_names_ = [f'Feature_{i+1}' for i in range(self.n_features_)]
        else:
            if len(feature_names) != self.n_features_:
                raise ValueError(f"feature_names debe tener {self.n_features_} elementos")
            self.feature_names_ = feature_names

        # Número de componentes
        if self.n_components is None:
            self.n_components = min(self.n_samples_, self.n_features_)
        else:
            if self.n_components > min(self.n_samples_, self.n_features_):
                warnings.warn(
                    f"n_components={self.n_components} es mayor que "
                    f"min(n_samples, n_features)={min(self.n_samples_, self.n_features_)}. "
                    f"Reduciendo a {min(self.n_samples_, self.n_features_)}"
                )
                self.n_components = min(self.n_samples_, self.n_features_)

        # 1. Centrar y estandarizar datos
        # ─────────────────────────────────
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Evitar división por cero
        self.std_[self.std_ == 0] = 1.0

        X_centered = (X - self.mean_) / self.std_

        # 2. Calcular matriz de covarianza
        # ─────────────────────────────────
        # Σ = X'X / (n-1)
        cov_matrix = (X_centered.T @ X_centered) / (self.n_samples_ - 1)

        # 3. Eigendecomposición
        # ─────────────────────────────────
        # Σ = VΛV'
        # V = eigenvectores (componentes principales)
        # Λ = eigenvalores (varianza explicada)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Ordenar por eigenvalores decrecientes
        # ─────────────────────────────────────────
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. Seleccionar top k componentes
        # ─────────────────────────────────
        self.components_ = eigenvectors[:, :self.n_components].T  # (n_components, n_features)
        self.explained_variance_ = eigenvalues[:self.n_components]

        # Calcular proporción de varianza explicada
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        # Valores singulares
        self.singular_values_ = np.sqrt(self.explained_variance_ * (self.n_samples_ - 1))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Proyecta datos a espacio de componentes principales

        Z = XV

        Args:
            X: Datos originales (n_samples, n_features)

        Returns:
            Datos transformados (n_samples, n_components)
        """
        if self.components_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        X = np.asarray(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(f"X debe tener {self.n_features_} features")

        # Centrar y estandarizar
        X_centered = (X - self.mean_) / self.std_

        # Proyectar: Z = XV
        X_transformed = X_centered @ self.components_.T

        # Whitening (opcional)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)

        return X_transformed

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Ajusta PCA y transforma los datos

        Args:
            X: Datos originales
            feature_names: Nombres de features

        Returns:
            Datos transformados
        """
        return self.fit(X, feature_names).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Proyecta datos de vuelta al espacio original

        X ≈ ZV'

        Args:
            X_transformed: Datos en espacio de componentes principales

        Returns:
            Datos reconstruidos en espacio original
        """
        if self.components_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        X_transformed = np.asarray(X_transformed)

        # Whitening inverso
        if self.whiten:
            X_transformed = X_transformed * np.sqrt(self.explained_variance_)

        # Proyección inversa: X = ZV'
        X_reconstructed = X_transformed @ self.components_

        # Des-estandarizar
        X_reconstructed = X_reconstructed * self.std_ + self.mean_

        return X_reconstructed

    def get_component_loadings(self, component_idx: int = 0) -> pd.DataFrame:
        """
        Obtiene los loadings (pesos) de un componente principal

        Los loadings indican qué features originales contribuyen más al componente.

        Args:
            component_idx: Índice del componente (0 = PC1, 1 = PC2, ...)

        Returns:
            DataFrame con features y sus loadings
        """
        if self.components_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        if component_idx >= self.n_components:
            raise ValueError(f"component_idx debe ser < {self.n_components}")

        loadings = self.components_[component_idx]

        df = pd.DataFrame({
            'Feature': self.feature_names_,
            'Loading': loadings,
            'Abs_Loading': np.abs(loadings)
        })

        df = df.sort_values('Abs_Loading', ascending=False)

        return df

    def get_cumulative_variance(self) -> np.ndarray:
        """
        Calcula varianza explicada acumulada

        Returns:
            Array con varianza acumulada para cada componente
        """
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        return np.cumsum(self.explained_variance_ratio_)

    def n_components_for_variance(self, threshold: float = 0.95) -> int:
        """
        Determina cuántos componentes se necesitan para explicar
        cierto porcentaje de varianza

        Args:
            threshold: Proporción de varianza deseada (ej: 0.95 = 95%)

        Returns:
            Número de componentes necesarios
        """
        cumsum = self.get_cumulative_variance()
        return int(np.argmax(cumsum >= threshold) + 1)

    def summary(self) -> str:
        """
        Genera resumen de PCA

        Returns:
            String con resumen
        """
        if self.components_ is None:
            return "PCA no ajustado. Llame a fit() primero."

        lines = []
        lines.append("=" * 78)
        lines.append("PRINCIPAL COMPONENT ANALYSIS (PCA)")
        lines.append("=" * 78)
        lines.append("")

        # Información general
        lines.append(f"Número de muestras:           {self.n_samples_:>10}")
        lines.append(f"Features originales:          {self.n_features_:>10}")
        lines.append(f"Componentes principales:      {self.n_components:>10}")
        lines.append("")

        # Reducción de dimensionalidad
        reduction_pct = (1 - self.n_components / self.n_features_) * 100
        lines.append(f"Reducción de dimensionalidad: {reduction_pct:>9.1f}%")
        lines.append(f"  ({self.n_features_} features → {self.n_components} componentes)")
        lines.append("")

        # Varianza explicada
        cumsum = self.get_cumulative_variance()

        lines.append("Varianza Explicada por Componente:")
        lines.append("-" * 78)
        lines.append(f"{'PC':<6} {'Varianza':>15} {'% Varianza':>15} {'% Acumulado':>15}")
        lines.append("-" * 78)

        # Mostrar primeros 10 componentes (o todos si son menos)
        n_to_show = min(10, self.n_components)

        for i in range(n_to_show):
            variance = self.explained_variance_[i]
            ratio = self.explained_variance_ratio_[i]
            cum = cumsum[i]

            lines.append(
                f"PC{i+1:<3} {variance:>15.6f} {ratio*100:>14.2f}% {cum*100:>14.2f}%"
            )

        if self.n_components > n_to_show:
            lines.append(f"... ({self.n_components - n_to_show} componentes más)")

        lines.append("-" * 78)
        lines.append("")

        # Interpretación
        lines.append("INTERPRETACIÓN:")
        lines.append("-" * 78)

        # ¿Cuántos componentes para 80%, 90%, 95%?
        for threshold in [0.80, 0.90, 0.95]:
            n_needed = self.n_components_for_variance(threshold)
            if n_needed <= self.n_components:
                lines.append(
                    f"  • PC1-PC{n_needed} capturan {threshold*100:.0f}% de la varianza"
                )

        lines.append("")

        # Componentes principales más importantes
        lines.append("Componentes Principales Más Importantes:")
        lines.append("-" * 78)

        for i in range(min(3, self.n_components)):
            lines.append(f"\nPC{i+1} ({self.explained_variance_ratio_[i]*100:.1f}% varianza):")

            loadings_df = self.get_component_loadings(i)
            top_features = loadings_df.head(5)

            for _, row in top_features.iterrows():
                loading = row['Loading']
                if loading > 0:
                    sign = "+"
                else:
                    sign = "-"

                lines.append(f"  {sign} {abs(loading):.4f} × {row['Feature']}")

        lines.append("")
        lines.append("=" * 78)

        return "\n".join(lines)

    def plot_scree(self, save_path: Optional[Path] = None):
        """
        Scree Plot: Varianza explicada por componente

        Ayuda a determinar cuántos componentes mantener.

        Args:
            save_path: Ruta opcional para guardar gráfico
        """
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Varianza individual
        components = np.arange(1, self.n_components + 1)

        ax1.bar(components, self.explained_variance_ratio_ * 100, alpha=0.7, color='steelblue')
        ax1.plot(components, self.explained_variance_ratio_ * 100, 'ro-', linewidth=2, markersize=6)
        ax1.set_xlabel('Componente Principal', fontsize=12)
        ax1.set_ylabel('% Varianza Explicada', fontsize=12)
        ax1.set_title('Scree Plot: Varianza por Componente', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(components[::max(1, self.n_components // 10)])

        # Plot 2: Varianza acumulada
        cumsum = self.get_cumulative_variance()

        ax2.plot(components, cumsum * 100, 'o-', linewidth=2, markersize=6, color='darkgreen')
        ax2.axhline(y=80, color='r', linestyle='--', label='80%', alpha=0.7)
        ax2.axhline(y=90, color='orange', linestyle='--', label='90%', alpha=0.7)
        ax2.axhline(y=95, color='purple', linestyle='--', label='95%', alpha=0.7)
        ax2.set_xlabel('Número de Componentes', fontsize=12)
        ax2.set_ylabel('% Varianza Acumulada', fontsize=12)
        ax2.set_title('Varianza Explicada Acumulada', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_ylim([0, 105])
        ax2.set_xticks(components[::max(1, self.n_components // 10)])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()

    def plot_biplot(self, X_transformed: np.ndarray, pc1: int = 0, pc2: int = 1,
                    labels: Optional[np.ndarray] = None, save_path: Optional[Path] = None):
        """
        Biplot: Visualiza datos en espacio de componentes principales

        Muestra:
        - Puntos: muestras proyectadas en PC1 vs PC2
        - Flechas: dirección de features originales

        Args:
            X_transformed: Datos transformados por PCA
            pc1: Índice del primer componente a plotear
            pc2: Índice del segundo componente a plotear
            labels: Etiquetas opcionales para colorear puntos
            save_path: Ruta para guardar gráfico
        """
        if self.components_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Scatter de puntos
        if labels is None:
            ax.scatter(X_transformed[:, pc1], X_transformed[:, pc2],
                      alpha=0.6, s=50, c='steelblue')
        else:
            scatter = ax.scatter(X_transformed[:, pc1], X_transformed[:, pc2],
                               alpha=0.6, s=50, c=labels, cmap='viridis')
            plt.colorbar(scatter, ax=ax)

        # Flechas de loadings
        scale_factor = 3  # Factor de escala para visualización

        for i in range(self.n_features_):
            loading_pc1 = self.components_[pc1, i] * scale_factor
            loading_pc2 = self.components_[pc2, i] * scale_factor

            ax.arrow(0, 0, loading_pc1, loading_pc2,
                    head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)

            # Etiquetas de features
            ax.text(loading_pc1 * 1.1, loading_pc2 * 1.1,
                   self.feature_names_[i],
                   fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

        ax.set_xlabel(f'PC{pc1+1} ({self.explained_variance_ratio_[pc1]*100:.1f}% varianza)',
                     fontsize=12)
        ax.set_ylabel(f'PC{pc2+1} ({self.explained_variance_ratio_[pc2]*100:.1f}% varianza)',
                     fontsize=12)
        ax.set_title('Biplot: Datos y Features en Espacio de Componentes Principales',
                    fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()

    def plot_loadings_heatmap(self, n_components_to_show: int = 10, save_path: Optional[Path] = None):
        """
        Heatmap de loadings: qué features contribuyen a cada componente

        Args:
            n_components_to_show: Número de componentes a mostrar
            save_path: Ruta para guardar gráfico
        """
        if self.components_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        n_to_show = min(n_components_to_show, self.n_components)

        # Crear matriz de loadings
        loadings_matrix = self.components_[:n_to_show, :].T  # (n_features, n_components)

        fig, ax = plt.subplots(figsize=(max(10, n_to_show), max(8, len(self.feature_names_) * 0.3)))

        sns.heatmap(
            loadings_matrix,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Loading'},
            xticklabels=[f'PC{i+1}' for i in range(n_to_show)],
            yticklabels=self.feature_names_,
            ax=ax
        )

        ax.set_title('Loadings de Componentes Principales\n(Qué features contribuyen a cada PC)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Componente Principal', fontsize=12)
        ax.set_ylabel('Feature Original', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()


def ejemplo_forex_retornos_correlacionados():
    """
    Ejemplo con datos de forex: retornos en múltiples timeframes

    Las features están correlacionadas (redundantes).
    PCA las reduce a pocos componentes.
    """
    print("\n" + "=" * 78)
    print("EJEMPLO: PCA con Retornos Correlacionados en Forex")
    print("=" * 78 + "\n")

    # Generar datos sintéticos
    np.random.seed(42)
    n_samples = 1000

    # Simular retornos en diferentes timeframes
    # Todos correlacionados porque reflejan el mismo momentum subyacente

    # Momentum subyacente (factor latente)
    momentum = np.random.randn(n_samples) * 0.01

    # Ruido específico de cada timeframe
    noise_level = 0.005

    # Retornos en diferentes timeframes (altamente correlacionados)
    ret_1h = momentum + np.random.randn(n_samples) * noise_level
    ret_2h = momentum * 1.8 + np.random.randn(n_samples) * noise_level
    ret_4h = momentum * 3.2 + np.random.randn(n_samples) * noise_level
    ret_8h = momentum * 5.5 + np.random.randn(n_samples) * noise_level
    ret_12h = momentum * 7.8 + np.random.randn(n_samples) * noise_level
    ret_24h = momentum * 12.0 + np.random.randn(n_samples) * noise_level

    # Construir matriz de features
    X = np.column_stack([ret_1h, ret_2h, ret_4h, ret_8h, ret_12h, ret_24h])

    feature_names = ['Ret_1h', 'Ret_2h', 'Ret_4h', 'Ret_8h', 'Ret_12h', 'Ret_24h']

    print(f"Datos: {n_samples} muestras, {X.shape[1]} features (retornos)")
    print(f"Features: {', '.join(feature_names)}")
    print()

    # Calcular matriz de correlación
    print("Matriz de Correlación:")
    print("-" * 78)
    corr_matrix = np.corrcoef(X.T)
    corr_df = pd.DataFrame(corr_matrix, columns=feature_names, index=feature_names)
    print(corr_df.round(3))
    print()
    print("→ Las features están ALTAMENTE correlacionadas (redundantes)")
    print()

    # Aplicar PCA
    print("\n" + "─" * 78)
    print("Aplicando PCA...")
    print("─" * 78 + "\n")

    pca = PCA(n_components=6)  # Extraer todos los componentes
    X_transformed = pca.fit_transform(X, feature_names=feature_names)

    # Mostrar resumen
    print(pca.summary())

    # Determinar cuántos componentes mantener
    n_for_90 = pca.n_components_for_variance(0.90)
    n_for_95 = pca.n_components_for_variance(0.95)

    print("\n" + "=" * 78)
    print("REDUCCIÓN DE DIMENSIONALIDAD")
    print("=" * 78 + "\n")

    print(f"6 features originales → {n_for_90} componentes (90% varianza)")
    print(f"6 features originales → {n_for_95} componentes (95% varianza)")
    print()

    # Interpretar componentes principales
    print("\n" + "=" * 78)
    print("INTERPRETACIÓN DE COMPONENTES")
    print("=" * 78 + "\n")

    print("PC1 (Componente Principal 1):")
    print("-" * 78)
    pc1_loadings = pca.get_component_loadings(0)
    print(pc1_loadings.to_string(index=False))
    print()
    print("→ PC1 ≈ 'Momentum General' (combinación de todos los retornos)")
    print()

    if pca.n_components >= 2:
        print("PC2 (Componente Principal 2):")
        print("-" * 78)
        pc2_loadings = pca.get_component_loadings(1)
        print(pc2_loadings.to_string(index=False))
        print()
        print("→ PC2 ≈ 'Diferencia Corto vs Largo Plazo'")
        print()

    # Reconstrucción con componentes reducidos
    print("\n" + "=" * 78)
    print("RECONSTRUCCIÓN CON COMPONENTES REDUCIDOS")
    print("=" * 78 + "\n")

    # Usar solo los primeros k componentes
    k = n_for_95
    pca_reduced = PCA(n_components=k)
    X_reduced = pca_reduced.fit_transform(X, feature_names=feature_names)

    # Reconstruir
    X_reconstructed = pca_reduced.inverse_transform(X_reduced)

    # Calcular error de reconstrucción
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)

    print(f"Usando {k} componentes (de 6 originales):")
    print(f"  Varianza capturada:      {pca_reduced.get_cumulative_variance()[-1]*100:.2f}%")
    print(f"  Error de reconstrucción: {reconstruction_error:.8f}")
    print()

    # Visualizaciones
    try:
        print("\nGenerando visualizaciones...")

        # Scree plot
        pca.plot_scree()

        # Biplot (solo si hay suficientes componentes)
        if pca.n_components >= 2:
            pca.plot_biplot(X_transformed)

        # Heatmap de loadings
        pca.plot_loadings_heatmap()

    except Exception as e:
        print(f"No se pudieron generar gráficos: {e}")


def ejemplo_uso_en_modelo():
    """
    Ejemplo de usar PCA como preprocesamiento para un modelo predictivo
    """
    print("\n" + "=" * 78)
    print("EJEMPLO: PCA como Preprocesamiento para Modelo Predictivo")
    print("=" * 78 + "\n")

    # Generar datos con MUCHAS features
    np.random.seed(42)
    n_samples = 500
    n_features = 50  # MUCHAS features

    print(f"Generando datos con {n_features} features...")
    print()

    # Crear features correlacionadas y con ruido
    # Solo las primeras 10 son relevantes
    X = np.random.randn(n_samples, n_features)

    # Agregar correlaciones entre features
    for i in range(1, 20):
        X[:, i] = X[:, 0] * (0.9 - i*0.03) + np.random.randn(n_samples) * 0.3

    # Variable objetivo (depende solo de las primeras features)
    y = (
        0.5 * X[:, 0] +
        0.3 * X[:, 1] +
        -0.2 * X[:, 2] +
        np.random.randn(n_samples) * 0.1
    )

    feature_names = [f'Feature_{i+1}' for i in range(n_features)]

    # Dividir datos
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ==========================================
    # CASO 1: Sin PCA (todas las features)
    # ==========================================
    print("CASO 1: Sin PCA (usar todas las 50 features)")
    print("-" * 78)

    from regresion_lineal import RegresionLineal

    modelo_sin_pca = RegresionLineal()
    modelo_sin_pca.fit(X_train, y_train, feature_names=feature_names)

    r2_train_sin_pca = modelo_sin_pca.score(X_train, y_train)
    r2_test_sin_pca = modelo_sin_pca.score(X_test, y_test)

    print(f"R² train: {r2_train_sin_pca:.6f}")
    print(f"R² test:  {r2_test_sin_pca:.6f}")
    print()

    # ==========================================
    # CASO 2: Con PCA (reducir a k componentes)
    # ==========================================
    print("CASO 2: Con PCA (reducir a componentes que explican 95% varianza)")
    print("-" * 78)

    # Aplicar PCA
    pca = PCA()
    pca.fit(X_train, feature_names=feature_names)

    # Determinar k componentes para 95% varianza
    k = pca.n_components_for_variance(0.95)
    print(f"→ {n_features} features reducidas a {k} componentes")
    print()

    # Re-aplicar con k componentes
    pca_reduced = PCA(n_components=k)
    X_train_pca = pca_reduced.fit_transform(X_train, feature_names=feature_names)
    X_test_pca = pca_reduced.transform(X_test)

    # Entrenar modelo con componentes
    pc_names = [f'PC{i+1}' for i in range(k)]
    modelo_con_pca = RegresionLineal()
    modelo_con_pca.fit(X_train_pca, y_train, feature_names=pc_names)

    r2_train_con_pca = modelo_con_pca.score(X_train_pca, y_train)
    r2_test_con_pca = modelo_con_pca.score(X_test_pca, y_test)

    print(f"R² train: {r2_train_con_pca:.6f}")
    print(f"R² test:  {r2_test_con_pca:.6f}")
    print()

    # Comparación
    print("=" * 78)
    print("COMPARACIÓN")
    print("=" * 78 + "\n")

    comparison = pd.DataFrame({
        'Método': ['Sin PCA', 'Con PCA'],
        'N Features': [n_features, k],
        'R² Train': [r2_train_sin_pca, r2_train_con_pca],
        'R² Test': [r2_test_sin_pca, r2_test_con_pca]
    })

    print(comparison.to_string(index=False))
    print()

    print("VENTAJAS DE PCA:")
    print("-" * 78)
    print(f"• Reducción de {n_features} → {k} features ({(1-k/n_features)*100:.1f}% reducción)")
    print("• Elimina multicolinealidad")
    print("• Reduce overfitting")
    print("• Modelo más simple e interpretable")
    print()


if __name__ == '__main__':
    # Ejecutar ejemplos
    ejemplo_forex_retornos_correlacionados()
    ejemplo_uso_en_modelo()
