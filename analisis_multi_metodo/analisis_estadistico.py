"""
ANÁLISIS ESTADÍSTICO DE TRANSFORMACIONES
=========================================

Evaluación multi-método para identificar transformaciones con poder predictivo.

PARA CADA TRANSFORMACIÓN j:

A) INFORMATION COEFFICIENT (IC)
───────────────────────────────
ICⱼ = correlación(Xⱼ[t-1], retorno[t])

Interpretación:
- IC > 0: Correlación positiva con retornos futuros
- IC < 0: Correlación negativa con retornos futuros
- IC ≈ 0: Sin relación

Expectativas realistas:
- IC > 0.05: Sospechoso, verificar errores
- IC ∈ [0.01, 0.03]: Típico para edge real
- IC < 0.01: Probablemente ruido


B) SIGNIFICANCIA ESTADÍSTICA
────────────────────────────
t-statistic = IC × √n

p-value = probabilidad de ver este IC por azar

Criterio: p < 0.001 (con corrección por múltiples tests)


C) INFORMACIÓN MUTUA
────────────────────
Iⱼ = I(Xⱼ; retorno)

Captura dependencias NO LINEALES que IC no ve.

Si Iⱼ > 0 pero ICⱼ ≈ 0:
→ Hay relación no lineal


D) REGRESIÓN LINEAL
───────────────────
retorno = β₀ + β₁X₁ + β₂X₂ + ... + βₘXₘ + ε

Coeficientes βⱼ indican importancia de cada transformación.
R² indica varianza explicada (en finanzas, R² > 0.01 es significativo).


E) REGRESIÓN RIDGE/LASSO
────────────────────────
Con regularización para evitar overfitting:

Ridge: Minimizar ||y - Xβ||² + λ||β||²
Lasso: Minimizar ||y - Xβ||² + λ|β|₁

Lasso automáticamente selecciona transformaciones importantes
(las que quedan con β ≠ 0).


F) PCA
──────
Reducir dimensionalidad encontrando direcciones principales.

Si las primeras 20 componentes capturan 80% de varianza:
→ Hay estructura en los datos, no es todo ruido


Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.multitest import multipletests
import logging
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalizadorEstadistico:
    """
    Análisis estadístico completo de transformaciones.

    Evalúa poder predictivo mediante múltiples métodos.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, nombres_features: Optional[List[str]] = None):
        """
        Inicializa el analizador.

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

        # Resultados
        self.ic_results = None
        self.mi_results = None
        self.regression_results = None
        self.ridge_results = None
        self.lasso_results = None
        self.pca_results = None

        logger.info(f"Analizador inicializado:")
        logger.info(f"  Observaciones: {self.n_obs:,}")
        logger.info(f"  Features: {self.n_features:,}")
        logger.info(f"  Target range: [{y.min():.6f}, {y.max():.6f}]")

    def calcular_information_coefficient(self,
                                        metodo: str = 'pearson',
                                        correccion_multipletests: bool = True) -> pd.DataFrame:
        """
        A) INFORMATION COEFFICIENT

        Calcula correlación entre cada feature y retornos futuros.

        Args:
            metodo: 'pearson' o 'spearman'
            correccion_multipletests: Aplicar corrección Bonferroni

        Returns:
            DataFrame con IC, t-stat, p-value para cada feature
        """
        logger.info("="*70)
        logger.info("A) CALCULANDO INFORMATION COEFFICIENT (IC)")
        logger.info("="*70)

        ic_values = []
        t_stats = []
        p_values = []

        for j in range(self.n_features):
            # Eliminar NaNs
            mask = ~(np.isnan(self.X[:, j]) | np.isnan(self.y))
            x_clean = self.X[mask, j]
            y_clean = self.y[mask]

            if len(x_clean) < 10:
                ic_values.append(np.nan)
                t_stats.append(np.nan)
                p_values.append(1.0)
                continue

            # Calcular correlación
            if metodo == 'pearson':
                ic, p_val = stats.pearsonr(x_clean, y_clean)
            elif metodo == 'spearman':
                ic, p_val = spearmanr(x_clean, y_clean)
            else:
                raise ValueError(f"Método desconocido: {metodo}")

            # t-statistic
            n = len(x_clean)
            t_stat = ic * np.sqrt(n)

            ic_values.append(ic)
            t_stats.append(t_stat)
            p_values.append(p_val)

        # Aplicar corrección por múltiples tests (Bonferroni)
        if correccion_multipletests:
            p_values_array = np.array(p_values)
            valid_mask = ~np.isnan(p_values_array)

            if valid_mask.sum() > 0:
                reject, p_corrected, _, _ = multipletests(
                    p_values_array[valid_mask],
                    alpha=0.001,
                    method='bonferroni'
                )

                p_values_corregidos = np.full(len(p_values), np.nan)
                p_values_corregidos[valid_mask] = p_corrected
            else:
                p_values_corregidos = p_values_array
        else:
            p_values_corregidos = np.array(p_values)

        # Crear DataFrame
        df = pd.DataFrame({
            'Feature': self.nombres_features,
            'IC': ic_values,
            't_stat': t_stats,
            'p_value': p_values,
            'p_value_corrected': p_values_corregidos,
            'abs_IC': np.abs(ic_values)
        })

        # Ordenar por IC absoluto descendente
        df = df.sort_values('abs_IC', ascending=False)

        self.ic_results = df

        # Estadísticas
        logger.info(f"IC Calculados: {self.n_features}")
        logger.info(f"  IC medio: {df['IC'].mean():.6f}")
        logger.info(f"  |IC| medio: {df['abs_IC'].mean():.6f}")
        logger.info(f"  IC max: {df['IC'].max():.6f}")
        logger.info(f"  IC min: {df['IC'].min():.6f}")
        logger.info(f"  Features con |IC| > 0.01: {(df['abs_IC'] > 0.01).sum()}")
        logger.info(f"  Features con |IC| > 0.03: {(df['abs_IC'] > 0.03).sum()}")
        logger.info(f"  Features con p < 0.001: {(df['p_value_corrected'] < 0.001).sum()}")

        return df

    def calcular_informacion_mutua(self, n_neighbors: int = 3) -> pd.DataFrame:
        """
        C) INFORMACIÓN MUTUA

        Calcula información mutua entre cada feature y retornos.
        Captura dependencias NO LINEALES.

        Args:
            n_neighbors: Número de vecinos para estimación

        Returns:
            DataFrame con MI para cada feature
        """
        logger.info("="*70)
        logger.info("C) CALCULANDO INFORMACIÓN MUTUA (MI)")
        logger.info("="*70)

        # Eliminar columnas con todos NaN
        valid_features = []
        valid_indices = []

        for j in range(self.n_features):
            if not np.all(np.isnan(self.X[:, j])):
                valid_features.append(j)
                valid_indices.append(j)

        if len(valid_features) == 0:
            logger.warning("No hay features válidos para MI")
            return pd.DataFrame()

        X_valid = self.X[:, valid_features]

        # Eliminar filas con NaN
        mask = ~(np.isnan(X_valid).any(axis=1) | np.isnan(self.y))
        X_clean = X_valid[mask]
        y_clean = self.y[mask]

        logger.info(f"Calculando MI para {len(valid_features)} features...")

        # Calcular MI
        mi_scores = mutual_info_regression(
            X_clean, y_clean,
            n_neighbors=n_neighbors,
            random_state=42
        )

        # Crear DataFrame completo (con NaN para features inválidos)
        mi_values = np.full(self.n_features, np.nan)
        for i, idx in enumerate(valid_features):
            mi_values[idx] = mi_scores[i]

        df = pd.DataFrame({
            'Feature': self.nombres_features,
            'MI': mi_values
        })

        # Merge con IC results si existen
        if self.ic_results is not None:
            df = df.merge(
                self.ic_results[['Feature', 'IC', 'abs_IC']],
                on='Feature',
                how='left'
            )
            df['MI/abs_IC_ratio'] = df['MI'] / (df['abs_IC'] + 1e-10)

        df = df.sort_values('MI', ascending=False)

        self.mi_results = df

        logger.info(f"MI Calculados: {len(valid_features)}")
        logger.info(f"  MI medio: {df['MI'].mean():.6f}")
        logger.info(f"  MI max: {df['MI'].max():.6f}")
        logger.info(f"  Features con MI > 0.01: {(df['MI'] > 0.01).sum()}")

        return df

    def regresion_lineal(self) -> Dict:
        """
        D) REGRESIÓN LINEAL

        retorno = β₀ + β₁X₁ + ... + βₘXₘ + ε

        Returns:
            Diccionario con R², coeficientes, etc.
        """
        logger.info("="*70)
        logger.info("D) REGRESIÓN LINEAL MULTIVARIADA")
        logger.info("="*70)

        # Eliminar NaNs
        mask = ~(np.isnan(self.X).any(axis=1) | np.isnan(self.y))
        X_clean = self.X[mask]
        y_clean = self.y[mask]

        logger.info(f"Observaciones válidas: {len(X_clean):,}")

        # Normalizar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Regresión
        modelo = LinearRegression()
        modelo.fit(X_scaled, y_clean)

        # Métricas
        r2 = modelo.score(X_scaled, y_clean)
        y_pred = modelo.predict(X_scaled)
        mse = np.mean((y_clean - y_pred)**2)
        mae = np.mean(np.abs(y_clean - y_pred))

        # Coeficientes
        coefs = pd.DataFrame({
            'Feature': self.nombres_features,
            'Coef': modelo.coef_,
            'abs_Coef': np.abs(modelo.coef_)
        }).sort_values('abs_Coef', ascending=False)

        resultados = {
            'R2': r2,
            'MSE': mse,
            'MAE': mae,
            'intercept': modelo.intercept_,
            'coeficientes': coefs,
            'modelo': modelo,
            'scaler': scaler
        }

        self.regression_results = resultados

        logger.info(f"R² = {r2:.6f}")
        logger.info(f"MSE = {mse:.6f}")
        logger.info(f"MAE = {mae:.6f}")
        logger.info(f"Top 5 coeficientes:")
        logger.info(f"\n{coefs.head().to_string(index=False)}")

        return resultados

    def regresion_ridge(self, alpha: float = 1.0) -> Dict:
        """
        E) REGRESIÓN RIDGE

        Ridge: Minimizar ||y - Xβ||² + λ||β||²

        Args:
            alpha: Parámetro de regularización λ

        Returns:
            Diccionario con resultados
        """
        logger.info("="*70)
        logger.info("E.1) REGRESIÓN RIDGE (L2 Regularization)")
        logger.info("="*70)

        # Eliminar NaNs
        mask = ~(np.isnan(self.X).any(axis=1) | np.isnan(self.y))
        X_clean = self.X[mask]
        y_clean = self.y[mask]

        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Ridge
        modelo = Ridge(alpha=alpha)
        modelo.fit(X_scaled, y_clean)

        # Métricas
        r2 = modelo.score(X_scaled, y_clean)
        y_pred = modelo.predict(X_scaled)
        mse = np.mean((y_clean - y_pred)**2)

        # Coeficientes
        coefs = pd.DataFrame({
            'Feature': self.nombres_features,
            'Coef': modelo.coef_,
            'abs_Coef': np.abs(modelo.coef_)
        }).sort_values('abs_Coef', ascending=False)

        resultados = {
            'R2': r2,
            'MSE': mse,
            'alpha': alpha,
            'coeficientes': coefs,
            'modelo': modelo,
            'scaler': scaler
        }

        self.ridge_results = resultados

        logger.info(f"Alpha (λ) = {alpha}")
        logger.info(f"R² = {r2:.6f}")
        logger.info(f"MSE = {mse:.6f}")

        return resultados

    def regresion_lasso(self, alpha: Optional[float] = None, cv: int = 5) -> Dict:
        """
        E) REGRESIÓN LASSO

        Lasso: Minimizar ||y - Xβ||² + λ|β|₁

        Lasso automáticamente selecciona features importantes (β ≠ 0).

        Args:
            alpha: Parámetro λ (si None, usa CV para encontrar óptimo)
            cv: Número de folds para cross-validation

        Returns:
            Diccionario con resultados
        """
        logger.info("="*70)
        logger.info("E.2) REGRESIÓN LASSO (L1 Regularization)")
        logger.info("="*70)

        # Eliminar NaNs
        mask = ~(np.isnan(self.X).any(axis=1) | np.isnan(self.y))
        X_clean = self.X[mask]
        y_clean = self.y[mask]

        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Lasso con CV si alpha no especificado
        if alpha is None:
            logger.info(f"Buscando alpha óptimo con {cv}-fold CV...")
            modelo = LassoCV(cv=cv, random_state=42, max_iter=10000)
            modelo.fit(X_scaled, y_clean)
            alpha_optimo = modelo.alpha_
            logger.info(f"Alpha óptimo encontrado: {alpha_optimo:.6f}")
        else:
            modelo = Lasso(alpha=alpha, max_iter=10000)
            modelo.fit(X_scaled, y_clean)
            alpha_optimo = alpha

        # Métricas
        r2 = modelo.score(X_scaled, y_clean)
        y_pred = modelo.predict(X_scaled)
        mse = np.mean((y_clean - y_pred)**2)

        # Coeficientes
        coefs = pd.DataFrame({
            'Feature': self.nombres_features,
            'Coef': modelo.coef_,
            'abs_Coef': np.abs(modelo.coef_),
            'non_zero': modelo.coef_ != 0
        }).sort_values('abs_Coef', ascending=False)

        n_seleccionados = (modelo.coef_ != 0).sum()

        resultados = {
            'R2': r2,
            'MSE': mse,
            'alpha': alpha_optimo,
            'n_features_seleccionados': n_seleccionados,
            'coeficientes': coefs,
            'features_seleccionados': coefs[coefs['non_zero']]['Feature'].tolist(),
            'modelo': modelo,
            'scaler': scaler
        }

        self.lasso_results = resultados

        logger.info(f"Alpha (λ) = {alpha_optimo:.6f}")
        logger.info(f"R² = {r2:.6f}")
        logger.info(f"MSE = {mse:.6f}")
        logger.info(f"Features seleccionados: {n_seleccionados}/{self.n_features}")
        logger.info(f"Tasa de selección: {n_seleccionados/self.n_features*100:.1f}%")

        return resultados

    def analisis_pca(self, n_components: int = 50) -> Dict:
        """
        F) PCA (Principal Component Analysis)

        Reducción de dimensionalidad.

        Args:
            n_components: Número de componentes principales

        Returns:
            Diccionario con resultados de PCA
        """
        logger.info("="*70)
        logger.info("F) ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)")
        logger.info("="*70)

        # Eliminar NaNs
        mask = ~np.isnan(self.X).any(axis=1)
        X_clean = self.X[mask]

        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # PCA
        n_components = min(n_components, X_clean.shape[1], X_clean.shape[0])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Varianza explicada
        varianza_explicada = pca.explained_variance_ratio_
        varianza_acumulada = np.cumsum(varianza_explicada)

        # Encontrar cuántos componentes para 80%, 90%, 95%
        n_80 = np.argmax(varianza_acumulada >= 0.80) + 1
        n_90 = np.argmax(varianza_acumulada >= 0.90) + 1
        n_95 = np.argmax(varianza_acumulada >= 0.95) + 1

        resultados = {
            'n_components': n_components,
            'varianza_explicada': varianza_explicada,
            'varianza_acumulada': varianza_acumulada,
            'n_para_80pct': n_80,
            'n_para_90pct': n_90,
            'n_para_95pct': n_95,
            'componentes': X_pca,
            'pca': pca,
            'scaler': scaler
        }

        self.pca_results = resultados

        logger.info(f"Componentes calculados: {n_components}")
        logger.info(f"Varianza explicada por primeros 10 componentes:")
        for i in range(min(10, n_components)):
            logger.info(f"  PC{i+1}: {varianza_explicada[i]*100:.2f}% (acum: {varianza_acumulada[i]*100:.2f}%)")
        logger.info(f"")
        logger.info(f"Componentes para 80% varianza: {n_80}")
        logger.info(f"Componentes para 90% varianza: {n_90}")
        logger.info(f"Componentes para 95% varianza: {n_95}")

        return resultados

    def generar_reporte_completo(self) -> None:
        """
        Genera reporte completo de todos los análisis.
        """
        print("\n" + "="*70)
        print("REPORTE COMPLETO DE ANÁLISIS ESTADÍSTICO")
        print("="*70)
        print()

        # IC
        if self.ic_results is not None:
            print("A) INFORMATION COEFFICIENT (IC)")
            print("-"*70)
            print("Top 10 features por |IC|:")
            print(self.ic_results[['Feature', 'IC', 't_stat', 'p_value_corrected']].head(10).to_string(index=False))
            print()

        # MI
        if self.mi_results is not None:
            print("C) INFORMACIÓN MUTUA (MI)")
            print("-"*70)
            print("Top 10 features por MI:")
            print(self.mi_results[['Feature', 'MI']].head(10).to_string(index=False))
            print()

        # Regresión lineal
        if self.regression_results is not None:
            print("D) REGRESIÓN LINEAL")
            print("-"*70)
            print(f"R² = {self.regression_results['R2']:.6f}")
            print()

        # Lasso
        if self.lasso_results is not None:
            print("E) REGRESIÓN LASSO")
            print("-"*70)
            print(f"Features seleccionados: {self.lasso_results['n_features_seleccionados']}/{self.n_features}")
            print()
            print("Top 10 features seleccionados:")
            selected = self.lasso_results['coeficientes'][self.lasso_results['coeficientes']['non_zero']]
            print(selected[['Feature', 'Coef']].head(10).to_string(index=False))
            print()

        # PCA
        if self.pca_results is not None:
            print("F) PCA")
            print("-"*70)
            print(f"Componentes para 80% varianza: {self.pca_results['n_para_80pct']}")
            print(f"Componentes para 90% varianza: {self.pca_results['n_para_90pct']}")
            print()

        print("="*70)


def ejemplo_uso():
    """
    Ejemplo de uso del análisis estadístico.
    """
    print("="*70)
    print("EJEMPLO: ANÁLISIS ESTADÍSTICO DE TRANSFORMACIONES")
    print("="*70)
    print()

    # Generar datos sintéticos
    np.random.seed(42)
    n_obs = 5000
    n_features = 100

    # Features aleatorios
    X = np.random.randn(n_obs, n_features)

    # Target con algo de señal en los primeros 5 features
    y = (
        0.05 * X[:, 0] +
        -0.03 * X[:, 1] +
        0.02 * X[:, 2]**2 +  # Relación no lineal
        np.random.randn(n_obs) * 0.01  # Ruido
    )

    nombres = [f"Feature_{i}" for i in range(n_features)]

    # Crear analizador
    analizador = AnalizadorEstadistico(X, y, nombres)

    # Ejecutar análisis
    print("\n")
    analizador.calcular_information_coefficient()

    print("\n")
    analizador.calcular_informacion_mutua()

    print("\n")
    analizador.regresion_lineal()

    print("\n")
    analizador.regresion_ridge(alpha=1.0)

    print("\n")
    analizador.regresion_lasso()

    print("\n")
    analizador.analisis_pca(n_components=20)

    # Reporte completo
    analizador.generar_reporte_completo()


if __name__ == '__main__':
    ejemplo_uso()
