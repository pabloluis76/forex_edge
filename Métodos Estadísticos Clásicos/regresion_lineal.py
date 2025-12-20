"""
Regresión Lineal - Modelo Estadístico Clásico
==============================================

MODELO MÁS SIMPLE:

y = β₀ + β₁x₁ + β₂x₂ + ... + βₘxₘ + ε

Donde:
- y = retorno futuro
- xᵢ = feature i
- βᵢ = coeficiente (peso) de feature i
- ε = error (ruido)


INTERPRETACIÓN:

βᵢ > 0: Cuando xᵢ aumenta, el retorno tiende a aumentar
βᵢ < 0: Cuando xᵢ aumenta, el retorno tiende a disminuir
βᵢ ≈ 0: xᵢ no tiene relación con el retorno


ESTIMACIÓN (Ordinary Least Squares):

β = (X'X)⁻¹X'y

Minimiza la suma de errores al cuadrado.


MÉTRICAS DE EVALUACIÓN:

1. R² (Coeficiente de determinación)
   ─────────────────────────────────
   R² = 1 - SS_res / SS_tot

   R² = 0.01: El modelo explica 1% de la varianza
   R² = 0.05: El modelo explica 5% de la varianza

   En finanzas, R² > 0.02 ya es significativo.

2. t-statistic de cada coeficiente
   ─────────────────────────────────
   t = β / SE(β)

   |t| > 2: Coeficiente estadísticamente significativo

3. F-statistic del modelo completo
   ─────────────────────────────────
   ¿El modelo en conjunto es significativo?


LIMITACIONES:

- Asume relación LINEAL
- Sensible a outliers
- No captura interacciones no lineales
- Puede sufrir multicolinealidad
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional, List
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class RegresionLineal:
    """
    Implementación de Regresión Lineal Ordinaria (OLS)

    Este modelo estima la relación lineal entre variables predictoras (features)
    y una variable objetivo (retornos futuros en trading).

    Attributes:
        coef_ (np.ndarray): Coeficientes β del modelo (sin intercepto)
        intercept_ (float): Intercepto β₀ del modelo
        r_squared_ (float): R² - proporción de varianza explicada
        adjusted_r_squared_ (float): R² ajustado por número de features
        f_statistic_ (float): F-statistic del modelo completo
        f_pvalue_ (float): p-valor del F-statistic
        t_statistics_ (np.ndarray): t-statistics de cada coeficiente
        p_values_ (np.ndarray): p-valores de cada coeficiente
        std_errors_ (np.ndarray): Errores estándar de cada coeficiente
        residuals_ (np.ndarray): Residuales (errores) del modelo
        feature_names_ (List[str]): Nombres de las features
        n_samples_ (int): Número de observaciones
        n_features_ (int): Número de features
    """

    def __init__(self):
        """Inicializa el modelo de regresión lineal"""
        self.coef_ = None
        self.intercept_ = None
        self.r_squared_ = None
        self.adjusted_r_squared_ = None
        self.f_statistic_ = None
        self.f_pvalue_ = None
        self.t_statistics_ = None
        self.p_values_ = None
        self.std_errors_ = None
        self.residuals_ = None
        self.feature_names_ = None
        self.n_samples_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'RegresionLineal':
        """
        Ajusta el modelo de regresión lineal usando Ordinary Least Squares

        Implementa: β = (X'X)⁻¹X'y

        Args:
            X: Array de features (n_samples, n_features)
            y: Array de variable objetivo (n_samples,)
            feature_names: Nombres opcionales de las features

        Returns:
            self: Modelo ajustado

        Raises:
            ValueError: Si X o y tienen dimensiones incorrectas
        """
        # Validación de entrada
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y.ndim != 1:
            raise ValueError("y debe ser un array 1D")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X y y deben tener el mismo número de muestras. X: {X.shape[0]}, y: {y.shape[0]}")

        self.n_samples_, self.n_features_ = X.shape

        # Guardar nombres de features
        if feature_names is None:
            self.feature_names_ = [f'X{i+1}' for i in range(self.n_features_)]
        else:
            if len(feature_names) != self.n_features_:
                raise ValueError(f"feature_names debe tener {self.n_features_} elementos")
            self.feature_names_ = feature_names

        # Agregar columna de 1s para el intercepto
        X_with_intercept = np.column_stack([np.ones(self.n_samples_), X])

        # Calcular coeficientes: β = (X'X)⁻¹X'y
        try:
            XtX = X_with_intercept.T @ X_with_intercept
            Xty = X_with_intercept.T @ y

            # Resolver el sistema de ecuaciones
            beta = np.linalg.solve(XtX, Xty)

        except np.linalg.LinAlgError:
            # Si la matriz es singular, usar pseudo-inversa
            warnings.warn("Matriz singular detectada. Usando pseudo-inversa.", RuntimeWarning)
            beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

        # Separar intercepto y coeficientes
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

        # Calcular predicciones y residuales
        y_pred = X_with_intercept @ beta
        self.residuals_ = y - y_pred

        # Calcular métricas de evaluación
        self._calculate_statistics(X, y, y_pred)

        return self

    def _calculate_statistics(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        """
        Calcula todas las estadísticas del modelo

        Args:
            X: Features
            y: Variable objetivo
            y_pred: Predicciones del modelo
        """
        # 1. R² (Coeficiente de determinación)
        # ────────────────────────────────────
        # R² = 1 - SS_res / SS_tot

        ss_res = np.sum(self.residuals_ ** 2)  # Suma de cuadrados de residuales
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Suma de cuadrados total

        self.r_squared_ = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # R² ajustado (penaliza por número de features)
        n = self.n_samples_
        p = self.n_features_

        if n > p + 1:
            self.adjusted_r_squared_ = 1 - (1 - self.r_squared_) * (n - 1) / (n - p - 1)
        else:
            self.adjusted_r_squared_ = self.r_squared_

        # 2. Estimación de varianza del error
        # ────────────────────────────────────
        if n > p + 1:
            mse = ss_res / (n - p - 1)  # Mean Squared Error
        else:
            mse = ss_res / n

        # 3. Errores estándar de los coeficientes
        # ────────────────────────────────────────
        # SE(β) = sqrt(MSE * diag((X'X)⁻¹))

        X_with_intercept = np.column_stack([np.ones(n), X])

        try:
            XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            var_beta = mse * np.diag(XtX_inv)
            std_errors_all = np.sqrt(var_beta)

            # Separar error estándar del intercepto y coeficientes
            self.std_errors_ = std_errors_all[1:]  # Sin intercepto
            intercept_se = std_errors_all[0]

        except np.linalg.LinAlgError:
            warnings.warn("No se pudieron calcular errores estándar", RuntimeWarning)
            self.std_errors_ = np.full(p, np.nan)
            intercept_se = np.nan

        # 4. t-statistics de cada coeficiente
        # ────────────────────────────────────
        # t = β / SE(β)
        # |t| > 2: Coeficiente estadísticamente significativo

        if not np.isnan(self.std_errors_).any():
            self.t_statistics_ = self.coef_ / self.std_errors_

            # p-valores (bilateral)
            df = n - p - 1  # grados de libertad
            self.p_values_ = 2 * (1 - stats.t.cdf(np.abs(self.t_statistics_), df))
        else:
            self.t_statistics_ = np.full(p, np.nan)
            self.p_values_ = np.full(p, np.nan)

        # 5. F-statistic del modelo completo
        # ──────────────────────────────────
        # ¿El modelo en conjunto es significativo?

        if p > 0 and n > p + 1:
            ss_reg = ss_tot - ss_res  # Suma de cuadrados de regresión

            ms_reg = ss_reg / p  # Mean square regression
            ms_res = ss_res / (n - p - 1)  # Mean square residual

            if ms_res > 0:
                self.f_statistic_ = ms_reg / ms_res
                self.f_pvalue_ = 1 - stats.f.cdf(self.f_statistic_, p, n - p - 1)
            else:
                self.f_statistic_ = np.nan
                self.f_pvalue_ = np.nan
        else:
            self.f_statistic_ = np.nan
            self.f_pvalue_ = np.nan

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice valores usando el modelo ajustado

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicciones (n_samples,)

        Raises:
            RuntimeError: Si el modelo no ha sido ajustado
        """
        if self.coef_ is None:
            raise RuntimeError("El modelo debe ser ajustado primero con fit()")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_:
            raise ValueError(f"X debe tener {self.n_features_} features")

        return self.intercept_ + X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula R² en datos de prueba

        Args:
            X: Features
            y: Variable objetivo

        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def summary(self) -> str:
        """
        Genera un resumen estadístico del modelo al estilo de statsmodels/R

        Returns:
            String con el resumen del modelo
        """
        if self.coef_ is None:
            return "Modelo no ajustado. Llame a fit() primero."

        lines = []
        lines.append("=" * 78)
        lines.append("REGRESIÓN LINEAL - RESUMEN DEL MODELO")
        lines.append("=" * 78)
        lines.append("")

        # Información general
        lines.append(f"Número de observaciones:  {self.n_samples_:>10}")
        lines.append(f"Número de features:       {self.n_features_:>10}")
        lines.append("")

        # Métricas del modelo
        lines.append("Métricas del modelo:")
        lines.append("-" * 78)
        lines.append(f"R²:                       {self.r_squared_:>10.6f}")
        lines.append(f"R² ajustado:              {self.adjusted_r_squared_:>10.6f}")
        lines.append(f"F-statistic:              {self.f_statistic_:>10.4f}")
        lines.append(f"Prob (F-statistic):       {self.f_pvalue_:>10.6f}")
        lines.append("")

        # Interpretación de R²
        if self.r_squared_ >= 0.02:
            lines.append(f"✓ R² = {self.r_squared_:.4f} → El modelo explica {self.r_squared_*100:.2f}% de la varianza")
            lines.append("  En finanzas, R² > 0.02 ya es significativo.")
        else:
            lines.append(f"✗ R² = {self.r_squared_:.4f} → El modelo explica {self.r_squared_*100:.2f}% de la varianza")
            lines.append("  R² muy bajo. Considere agregar más features o usar modelos no lineales.")
        lines.append("")

        # Tabla de coeficientes
        lines.append("Coeficientes:")
        lines.append("-" * 78)
        lines.append(f"{'Feature':<20} {'β':>12} {'SE(β)':>12} {'t-stat':>12} {'p-value':>12} {'Sig':>6}")
        lines.append("-" * 78)

        for i, name in enumerate(self.feature_names_):
            coef = self.coef_[i]
            se = self.std_errors_[i]
            t = self.t_statistics_[i]
            p = self.p_values_[i]

            # Determinar significancia
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            elif p < 0.10:
                sig = "."
            else:
                sig = ""

            lines.append(f"{name:<20} {coef:>12.6f} {se:>12.6f} {t:>12.4f} {p:>12.6f} {sig:>6}")

            # Interpretación del coeficiente
            if abs(t) > 2:  # Estadísticamente significativo
                if coef > 0:
                    interpretation = "↑ aumenta retorno"
                else:
                    interpretation = "↓ disminuye retorno"
            else:
                interpretation = "≈ no significativo"

            lines.append(f"  └─ {interpretation}")

        lines.append("-" * 78)
        lines.append("")
        lines.append(f"Intercepto (β₀): {self.intercept_:.6f}")
        lines.append("")

        # Leyenda de significancia
        lines.append("Códigos de significancia: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        lines.append("")

        # Interpretación general
        lines.append("Interpretación:")
        lines.append("-" * 78)
        significant_features = [name for i, name in enumerate(self.feature_names_)
                              if abs(self.t_statistics_[i]) > 2]

        if significant_features:
            lines.append(f"✓ Features significativas: {', '.join(significant_features)}")
        else:
            lines.append("✗ Ninguna feature es estadísticamente significativa (|t| > 2)")

        lines.append("")

        # Advertencias
        lines.append("LIMITACIONES:")
        lines.append("-" * 78)
        lines.append("• Asume relación LINEAL entre features y retornos")
        lines.append("• Sensible a outliers")
        lines.append("• No captura interacciones no lineales")
        lines.append("• Puede sufrir multicolinealidad")
        lines.append("")
        lines.append("=" * 78)

        return "\n".join(lines)

    def plot_diagnostics(self, save_path: Optional[Path] = None):
        """
        Genera gráficos de diagnóstico del modelo

        Args:
            save_path: Ruta opcional para guardar los gráficos
        """
        if self.coef_ is None:
            raise RuntimeError("El modelo debe ser ajustado primero con fit()")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Diagnósticos de Regresión Lineal', fontsize=16, fontweight='bold')

        # 1. Residuales vs Valores Ajustados
        # ────────────────────────────────────
        # Debe mostrar patrón aleatorio. Si hay patrón, la relación no es lineal.
        ax = axes[0, 0]
        fitted_values = np.arange(len(self.residuals_))  # Placeholder
        ax.scatter(fitted_values, self.residuals_, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Valores Ajustados')
        ax.set_ylabel('Residuales')
        ax.set_title('Residuales vs Valores Ajustados\n(Debe verse aleatorio, sin patrones)')
        ax.grid(True, alpha=0.3)

        # 2. Q-Q Plot (Normalidad de Residuales)
        # ────────────────────────────────────────
        # Si los residuales son normales, deben seguir la línea diagonal
        ax = axes[0, 1]
        stats.probplot(self.residuals_, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot\n(Debe seguir línea diagonal si residuales son normales)')
        ax.grid(True, alpha=0.3)

        # 3. Histograma de Residuales
        # ────────────────────────────
        ax = axes[1, 0]
        ax.hist(self.residuals_, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Residuales')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Residuales\n(Debe ser aproximadamente normal)')
        ax.grid(True, alpha=0.3)

        # 4. Coeficientes con Intervalos de Confianza
        # ─────────────────────────────────────────────
        ax = axes[1, 1]

        # Calcular intervalos de confianza (95%)
        ci = 1.96 * self.std_errors_  # 95% CI

        y_pos = np.arange(len(self.feature_names_))

        # Plotear coeficientes
        ax.barh(y_pos, self.coef_, xerr=ci, capsize=5, alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.feature_names_)
        ax.set_xlabel('Coeficiente β')
        ax.set_title('Coeficientes con Intervalos de Confianza 95%\n(Significativo si no cruza 0)')
        ax.grid(True, alpha=0.3, axis='x')

        # Destacar coeficientes significativos
        for i, (coef, t_stat) in enumerate(zip(self.coef_, self.t_statistics_)):
            if abs(t_stat) > 2:
                ax.get_children()[i].set_color('green')
                ax.get_children()[i].set_alpha(0.8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráficos guardados en: {save_path}")
        else:
            plt.show()


def ejemplo_uso_forex():
    """
    Ejemplo de uso con datos de forex

    Predice el retorno futuro (1 hora) basado en:
    - Retorno actual
    - RSI
    - MACD
    - Volatilidad
    """
    print("\n" + "=" * 78)
    print("EJEMPLO: Predicción de Retornos en Forex")
    print("=" * 78 + "\n")

    # Generar datos sintéticos de ejemplo
    np.random.seed(42)
    n_samples = 1000

    # Features simuladas
    retorno_actual = np.random.randn(n_samples) * 0.001  # Retornos pequeños
    rsi = np.random.uniform(30, 70, n_samples)  # RSI entre 30-70
    macd = np.random.randn(n_samples) * 0.0001
    volatilidad = np.abs(np.random.randn(n_samples) * 0.002)

    # Variable objetivo: retorno futuro
    # Modelo verdadero (desconocido en la práctica):
    # retorno_futuro = 0.0001 - 0.5 * retorno_actual + 0.00001 * (rsi - 50) + 2.0 * macd - 0.1 * volatilidad + ruido

    retorno_futuro = (
        0.0001 +  # Intercepto (drift)
        -0.5 * retorno_actual +  # Mean reversion
        0.00001 * (rsi - 50) +  # Efecto RSI
        2.0 * macd +  # Efecto MACD
        -0.1 * volatilidad +  # Penalización por volatilidad
        np.random.randn(n_samples) * 0.003  # Ruido (la mayor parte)
    )

    # Preparar datos
    X = np.column_stack([retorno_actual, rsi, macd, volatilidad])
    y = retorno_futuro

    feature_names = ['Retorno_Actual', 'RSI', 'MACD', 'Volatilidad']

    # Dividir en train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Datos de prueba:        {X_test.shape[0]} muestras")
    print(f"Features:               {', '.join(feature_names)}")
    print("")

    # Ajustar modelo
    print("Ajustando modelo de regresión lineal...")
    modelo = RegresionLineal()
    modelo.fit(X_train, y_train, feature_names=feature_names)

    # Evaluar en test
    r2_train = modelo.score(X_train, y_train)
    r2_test = modelo.score(X_test, y_test)

    print(f"\nR² en training:  {r2_train:.6f}")
    print(f"R² en test:      {r2_test:.6f}")

    # Mostrar resumen
    print("\n")
    print(modelo.summary())

    # Hacer predicciones de ejemplo
    print("\n" + "=" * 78)
    print("PREDICCIONES DE EJEMPLO")
    print("=" * 78 + "\n")

    for i in range(min(5, len(X_test))):
        y_pred = modelo.predict(X_test[i:i+1])[0]
        y_true = y_test[i]
        error = y_pred - y_true

        print(f"Muestra {i+1}:")
        print(f"  Features:  Retorno={X_test[i, 0]:.6f}, RSI={X_test[i, 1]:.1f}, "
              f"MACD={X_test[i, 2]:.6f}, Vol={X_test[i, 3]:.6f}")
        print(f"  Predicción: {y_pred:>12.6f}")
        print(f"  Real:       {y_true:>12.6f}")
        print(f"  Error:      {error:>12.6f}")
        print()

    # Generar gráficos de diagnóstico (opcional)
    try:
        modelo.plot_diagnostics()
    except Exception as e:
        print(f"No se pudieron generar gráficos: {e}")


if __name__ == '__main__':
    # Ejecutar ejemplo
    ejemplo_uso_forex()
