"""
Regresión Ridge y Lasso (Regularización)
=========================================

PROBLEMA: Con muchas features, la regresión lineal overfittea.

SOLUCIÓN: Agregar penalización a los coeficientes.


RIDGE (L2 Regularization):
──────────────────────────

Minimizar: ||y - Xβ||² + λ||β||²

El término λ||β||² penaliza coeficientes grandes.

Efecto:
- Coeficientes se "encogen" hacia cero
- Reduce varianza, aumenta sesgo
- Nunca elimina features completamente


LASSO (L1 Regularization):
──────────────────────────

Minimizar: ||y - Xβ||² + λ|β|₁

El término λ|β|₁ penaliza la suma de valores absolutos.

Efecto:
- Algunos coeficientes se vuelven EXACTAMENTE cero
- Automáticamente selecciona features
- Produce modelos más interpretables


ELASTIC NET (Combinación):
──────────────────────────

Minimizar: ||y - Xβ||² + λ₁|β|₁ + λ₂||β||²

Combina beneficios de Ridge y Lasso.


SELECCIÓN DE λ (Hiperparámetro):
───────────────────────────────

Usar cross-validation:
1. Probar varios valores de λ
2. Para cada λ, evaluar error en datos de validación
3. Elegir λ que minimiza error de validación


INTERPRETACIÓN EN TRADING:

Las features con β ≠ 0 después de Lasso son las "importantes".
Si Lasso elimina una feature → Probablemente no tiene edge.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Literal
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import sys

# Importar constantes centralizadas
sys.path.append(str(Path(__file__).parent.parent))
from constants import EPSILON, COEF_ZERO_THRESHOLD


class RegresionRegularizada:
    """
    Implementación de regresiones regularizadas: Ridge, Lasso y Elastic Net

    Estos modelos agregan penalización a los coeficientes para prevenir overfitting
    y seleccionar features importantes.

    Attributes:
        tipo (str): Tipo de regularización: 'ridge', 'lasso', 'elastic_net'
        alpha (float): Parámetro de regularización λ
        l1_ratio (float): Ratio de L1 en Elastic Net (0=Ridge, 1=Lasso)
        coef_ (np.ndarray): Coeficientes β del modelo
        intercept_ (float): Intercepto β₀ del modelo
        feature_names_ (List[str]): Nombres de las features
        n_features_selected_ (int): Número de features con β ≠ 0
        feature_importance_ (np.ndarray): Importancia de cada feature (|β|)
    """

    def __init__(
        self,
        tipo: Literal['ridge', 'lasso', 'elastic_net'] = 'ridge',
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 10000,
        tol: float = 1e-4
    ):
        """
        Inicializa el modelo de regresión regularizada

        Args:
            tipo: Tipo de regularización ('ridge', 'lasso', 'elastic_net')
            alpha: Parámetro λ de regularización (mayor = más penalización)
            l1_ratio: Ratio L1/(L1+L2) para Elastic Net (0=Ridge, 1=Lasso)
            max_iter: Número máximo de iteraciones para optimización
            tol: Tolerancia para convergencia
        """
        if tipo not in ['ridge', 'lasso', 'elastic_net']:
            raise ValueError("tipo debe ser 'ridge', 'lasso' o 'elastic_net'")

        if alpha < 0:
            raise ValueError("alpha debe ser >= 0")

        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio debe estar entre 0 y 1")

        self.tipo = tipo
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

        # Atributos a llenar después de fit
        self.coef_ = None
        self.intercept_ = None
        self.feature_names_ = None
        self.n_features_selected_ = None
        self.feature_importance_ = None
        self.n_samples_ = None
        self.n_features_ = None

    def _ridge_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solución analítica para Ridge Regression

        β = (X'X + λI)⁻¹X'y

        Args:
            X: Features con intercepto (n_samples, n_features+1)
            y: Variable objetivo

        Returns:
            Coeficientes β (incluyendo intercepto)
        """
        n_features = X.shape[1]

        # Matriz de penalización (no penalizar intercepto)
        lambda_matrix = self.alpha * np.eye(n_features)
        lambda_matrix[0, 0] = 0  # No penalizar intercepto

        # Solución: β = (X'X + λI)⁻¹X'y
        XtX = X.T @ X
        Xty = X.T @ y

        try:
            beta = np.linalg.solve(XtX + lambda_matrix, Xty)
        except np.linalg.LinAlgError:
            # Si es singular, usar pseudo-inversa
            beta = np.linalg.pinv(XtX + lambda_matrix) @ Xty

        return beta

    def _lasso_objective(self, beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Función objetivo de Lasso

        L(β) = ||y - Xβ||² + λ|β|₁

        Args:
            beta: Coeficientes (incluyendo intercepto)
            X: Features con intercepto
            y: Variable objetivo

        Returns:
            Valor de la función objetivo
        """
        residuals = y - X @ beta
        mse = np.sum(residuals ** 2)

        # Penalización L1 (no penalizar intercepto)
        l1_penalty = self.alpha * np.sum(np.abs(beta[1:]))

        return mse + l1_penalty

    def _elastic_net_objective(self, beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Función objetivo de Elastic Net

        L(β) = ||y - Xβ||² + λ₁|β|₁ + λ₂||β||²

        Args:
            beta: Coeficientes (incluyendo intercepto)
            X: Features con intercepto
            y: Variable objetivo

        Returns:
            Valor de la función objetivo
        """
        residuals = y - X @ beta
        mse = np.sum(residuals ** 2)

        # Penalización L1 (no penalizar intercepto)
        l1_penalty = self.alpha * self.l1_ratio * np.sum(np.abs(beta[1:]))

        # Penalización L2 (no penalizar intercepto)
        l2_penalty = self.alpha * (1 - self.l1_ratio) * np.sum(beta[1:] ** 2)

        return mse + l1_penalty + l2_penalty

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'RegresionRegularizada':
        """
        Ajusta el modelo de regresión regularizada

        Args:
            X: Array de features (n_samples, n_features)
            y: Array de variable objetivo (n_samples,)
            feature_names: Nombres opcionales de las features

        Returns:
            self: Modelo ajustado
        """
        # Validación
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y.ndim != 1:
            raise ValueError("y debe ser un array 1D")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X y y deben tener el mismo número de muestras")

        self.n_samples_, self.n_features_ = X.shape

        # Guardar nombres de features
        if feature_names is None:
            self.feature_names_ = [f'X{i+1}' for i in range(self.n_features_)]
        else:
            if len(feature_names) != self.n_features_:
                raise ValueError(f"feature_names debe tener {self.n_features_} elementos")
            self.feature_names_ = feature_names

        # Agregar columna de 1s para intercepto
        X_with_intercept = np.column_stack([np.ones(self.n_samples_), X])

        # Ajustar modelo según el tipo
        if self.tipo == 'ridge':
            # Ridge tiene solución analítica
            beta = self._ridge_solution(X_with_intercept, y)

        elif self.tipo == 'lasso':
            # Lasso requiere optimización numérica
            beta_init = np.zeros(X_with_intercept.shape[1])

            result = minimize(
                fun=self._lasso_objective,
                x0=beta_init,
                args=(X_with_intercept, y),
                method='L-BFGS-B',
                options={'maxiter': self.max_iter, 'ftol': self.tol}
            )

            if not result.success:
                warnings.warn(f"Lasso no convergió: {result.message}", RuntimeWarning)

            beta = result.x

        else:  # elastic_net
            # Elastic Net requiere optimización numérica
            beta_init = np.zeros(X_with_intercept.shape[1])

            result = minimize(
                fun=self._elastic_net_objective,
                x0=beta_init,
                args=(X_with_intercept, y),
                method='L-BFGS-B',
                options={'maxiter': self.max_iter, 'ftol': self.tol}
            )

            if not result.success:
                warnings.warn(f"Elastic Net no convergió: {result.message}", RuntimeWarning)

            beta = result.x

        # Separar intercepto y coeficientes
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

        # Calcular features seleccionadas (coeficientes no cero)
        threshold = COEF_ZERO_THRESHOLD  # Umbral para considerar un coeficiente como cero
        self.n_features_selected_ = np.sum(np.abs(self.coef_) > threshold)

        # Calcular importancia de features
        self.feature_importance_ = np.abs(self.coef_)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice valores usando el modelo ajustado

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicciones (n_samples,)
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

    def get_selected_features(self, threshold: float = None) -> List[str]:
        """
        Obtiene las features seleccionadas (β ≠ 0)

        Args:
            threshold: Umbral para considerar un coeficiente como no cero

        Returns:
            Lista de nombres de features seleccionadas
        """
        if self.coef_ is None:
            raise RuntimeError("El modelo debe ser ajustado primero con fit()")

        if threshold is None:
            threshold = COEF_ZERO_THRESHOLD

        selected_idx = np.where(np.abs(self.coef_) > threshold)[0]
        return [self.feature_names_[i] for i in selected_idx]

    def summary(self) -> str:
        """
        Genera un resumen del modelo regularizado

        Returns:
            String con el resumen
        """
        if self.coef_ is None:
            return "Modelo no ajustado. Llame a fit() primero."

        lines = []
        lines.append("=" * 78)

        if self.tipo == 'ridge':
            lines.append("RIDGE REGRESSION (L2 Regularization)")
            lines.append(f"Minimiza: ||y - Xβ||² + {self.alpha}||β||²")
        elif self.tipo == 'lasso':
            lines.append("LASSO REGRESSION (L1 Regularization)")
            lines.append(f"Minimiza: ||y - Xβ||² + {self.alpha}|β|₁")
        else:
            lines.append("ELASTIC NET REGRESSION (L1 + L2)")
            l1_weight = self.alpha * self.l1_ratio
            l2_weight = self.alpha * (1 - self.l1_ratio)
            lines.append(f"Minimiza: ||y - Xβ||² + {l1_weight:.4f}|β|₁ + {l2_weight:.4f}||β||²")

        lines.append("=" * 78)
        lines.append("")

        # Información general
        lines.append(f"Número de observaciones:      {self.n_samples_:>10}")
        lines.append(f"Número total de features:     {self.n_features_:>10}")
        lines.append(f"Features seleccionadas (β≠0): {self.n_features_selected_:>10}")
        lines.append(f"Parámetro α (lambda):         {self.alpha:>10.6f}")

        if self.tipo == 'elastic_net':
            lines.append(f"Ratio L1:                     {self.l1_ratio:>10.2f}")

        lines.append("")

        # Tabla de coeficientes
        lines.append("Coeficientes:")
        lines.append("-" * 78)
        lines.append(f"{'Feature':<25} {'β':>15} {'|β|':>15} {'Status':>15}")
        lines.append("-" * 78)

        # Ordenar features por importancia
        sorted_idx = np.argsort(self.feature_importance_)[::-1]

        for idx in sorted_idx:
            name = self.feature_names_[idx]
            coef = self.coef_[idx]
            importance = self.feature_importance_[idx]

            if abs(coef) > COEF_ZERO_THRESHOLD:
                status = "✓ Seleccionada"
                if coef > 0:
                    direction = "↑"
                else:
                    direction = "↓"
            else:
                status = "✗ Eliminada"
                direction = " "

            lines.append(f"{name:<25} {coef:>15.6f} {importance:>15.6f} {status:>15}")

            if abs(coef) > COEF_ZERO_THRESHOLD:
                if coef > 0:
                    interpretation = "aumenta retorno"
                else:
                    interpretation = "disminuye retorno"
                lines.append(f"  └─ {direction} {interpretation}")

        lines.append("-" * 78)
        lines.append("")
        lines.append(f"Intercepto (β₀): {self.intercept_:.6f}")
        lines.append("")

        # Features seleccionadas
        selected = self.get_selected_features()
        if selected:
            lines.append("Features Importantes:")
            lines.append("-" * 78)
            for name in selected:
                idx = self.feature_names_.index(name)
                lines.append(f"  • {name:<20} (β = {self.coef_[idx]:>10.6f})")
        else:
            lines.append("✗ Ninguna feature fue seleccionada (todas fueron eliminadas)")

        lines.append("")

        # Interpretación
        lines.append("INTERPRETACIÓN EN TRADING:")
        lines.append("-" * 78)

        if self.tipo == 'lasso' or self.tipo == 'elastic_net':
            eliminated = [name for name in self.feature_names_ if name not in selected]
            if eliminated:
                lines.append("Features eliminadas (probablemente sin edge):")
                for name in eliminated:
                    lines.append(f"  ✗ {name}")
                lines.append("")

        if selected:
            lines.append("Las features seleccionadas son las que tienen edge predictivo.")
        else:
            lines.append("Ninguna feature tiene edge predictivo significativo.")
            lines.append("Considere:")
            lines.append("  - Reducir α (menos penalización)")
            lines.append("  - Crear mejores features")
            lines.append("  - Usar modelos no lineales")

        lines.append("")
        lines.append("=" * 78)

        return "\n".join(lines)

    def plot_coefficients(self, save_path: Optional[Path] = None):
        """
        Visualiza los coeficientes del modelo

        Args:
            save_path: Ruta opcional para guardar el gráfico
        """
        if self.coef_ is None:
            raise RuntimeError("El modelo debe ser ajustado primero con fit()")

        fig, ax = plt.subplots(figsize=(12, max(6, len(self.feature_names_) * 0.3)))

        # Ordenar por importancia
        sorted_idx = np.argsort(self.feature_importance_)
        sorted_names = [self.feature_names_[i] for i in sorted_idx]
        sorted_coefs = self.coef_[sorted_idx]

        # Colores: verde si seleccionado, rojo si eliminado
        colors = ['green' if abs(c) > COEF_ZERO_THRESHOLD else 'red' for c in sorted_coefs]
        alphas = [0.8 if abs(c) > COEF_ZERO_THRESHOLD else 0.3 for c in sorted_coefs]

        # Plotear
        y_pos = np.arange(len(sorted_names))
        bars = ax.barh(y_pos, sorted_coefs, color=colors, alpha=alphas)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Coeficiente β')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')

        # Título según tipo
        if self.tipo == 'ridge':
            title = f'Ridge Regression (α={self.alpha:.4f})'
        elif self.tipo == 'lasso':
            title = f'Lasso Regression (α={self.alpha:.4f})\n{self.n_features_selected_}/{self.n_features_} features seleccionadas'
        else:
            title = f'Elastic Net (α={self.alpha:.4f}, L1_ratio={self.l1_ratio:.2f})\n{self.n_features_selected_}/{self.n_features_} features seleccionadas'

        ax.set_title(title, fontweight='bold', fontsize=12)

        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.8, label='Seleccionada (β ≠ 0)'),
            Patch(facecolor='red', alpha=0.3, label='Eliminada (β ≈ 0)')
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()


class RegularizacionCV:
    """
    Selección automática del hiperparámetro α usando Cross-Validation

    SELECCIÓN DE λ (Hiperparámetro):
    ───────────────────────────────

    Usar cross-validation:
    1. Probar varios valores de λ
    2. Para cada λ, evaluar error en datos de validación
    3. Elegir λ que minimiza error de validación
    """

    def __init__(
        self,
        tipo: Literal['ridge', 'lasso', 'elastic_net'] = 'lasso',
        alphas: Optional[np.ndarray] = None,
        l1_ratio: float = 0.5,
        cv: int = 5,
        scoring: str = 'r2'
    ):
        """
        Inicializa CV para selección de α

        Args:
            tipo: Tipo de regularización
            alphas: Valores de α a probar (si None, usa grid automático)
            l1_ratio: Ratio L1 para Elastic Net
            cv: Número de folds para cross-validation
            scoring: Métrica ('r2' o 'mse')
        """
        self.tipo = tipo
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.scoring = scoring

        # Grid de alphas
        if alphas is None:
            self.alphas = np.logspace(-4, 2, 50)  # De 0.0001 a 100
        else:
            self.alphas = alphas

        # Resultados
        self.best_alpha_ = None
        self.best_score_ = None
        self.cv_scores_ = None
        self.best_model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'RegularizacionCV':
        """
        Encuentra el mejor α usando cross-validation

        Args:
            X: Features
            y: Variable objetivo
            feature_names: Nombres de features

        Returns:
            self con best_alpha_ y best_model_
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Estandarizar features (importante para regularización)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validation
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        cv_scores = []

        print(f"\n{'='*60}")
        print(f"Cross-Validation para {self.tipo.upper()}")
        print(f"{'='*60}")
        print(f"Probando {len(self.alphas)} valores de α con {self.cv}-fold CV...")
        print()

        for alpha in self.alphas:
            fold_scores = []

            for train_idx, val_idx in kf.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Entrenar modelo
                model = RegresionRegularizada(
                    tipo=self.tipo,
                    alpha=alpha,
                    l1_ratio=self.l1_ratio
                )
                model.fit(X_train, y_train)

                # Evaluar
                if self.scoring == 'r2':
                    score = model.score(X_val, y_val)
                else:  # mse
                    y_pred = model.predict(X_val)
                    score = -np.mean((y_val - y_pred) ** 2)  # Negativo para maximizar

                fold_scores.append(score)

            mean_score = np.mean(fold_scores)
            cv_scores.append(mean_score)

        self.cv_scores_ = np.array(cv_scores)

        # Mejor α
        best_idx = np.argmax(self.cv_scores_)
        self.best_alpha_ = self.alphas[best_idx]
        self.best_score_ = self.cv_scores_[best_idx]

        print(f"✓ Mejor α encontrado: {self.best_alpha_:.6f}")
        print(f"  Score CV ({self.scoring}): {self.best_score_:.6f}")
        print()

        # Entrenar modelo final con mejor α
        self.best_model_ = RegresionRegularizada(
            tipo=self.tipo,
            alpha=self.best_alpha_,
            l1_ratio=self.l1_ratio
        )
        self.best_model_.fit(X_scaled, y, feature_names=feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice usando el mejor modelo"""
        if self.best_model_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Estandarizar
        scaler = StandardScaler()
        scaler.fit(X)  # En práctica, usar el scaler del entrenamiento
        X_scaled = scaler.transform(X)

        return self.best_model_.predict(X_scaled)

    def plot_cv_scores(self, save_path: Optional[Path] = None):
        """
        Visualiza scores de CV vs α

        Args:
            save_path: Ruta para guardar gráfico
        """
        if self.cv_scores_ is None:
            raise RuntimeError("Debe llamar a fit() primero")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.alphas, self.cv_scores_, 'b-', linewidth=2, label='CV Score')
        ax.axvline(x=self.best_alpha_, color='r', linestyle='--', linewidth=2,
                   label=f'Mejor α = {self.best_alpha_:.6f}')

        ax.set_xscale('log')
        ax.set_xlabel('α (Parámetro de Regularización)', fontsize=12)
        ax.set_ylabel(f'Score ({self.scoring.upper()})', fontsize=12)
        ax.set_title(f'Cross-Validation: {self.tipo.upper()}\nSelección de Hiperparámetro α',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()


def ejemplo_comparacion_ridge_lasso():
    """
    Ejemplo comparando Ridge, Lasso y Elastic Net con datos de forex
    """
    print("\n" + "=" * 78)
    print("EJEMPLO: Comparación Ridge vs Lasso vs Elastic Net")
    print("=" * 78 + "\n")

    # Generar datos sintéticos con MUCHAS features (algunas irrelevantes)
    np.random.seed(42)
    n_samples = 500

    # Features RELEVANTES (tienen edge real)
    retorno_actual = np.random.randn(n_samples) * 0.001
    rsi = np.random.uniform(30, 70, n_samples)
    macd = np.random.randn(n_samples) * 0.0001
    volatilidad = np.abs(np.random.randn(n_samples) * 0.002)

    # Features IRRELEVANTES (ruido)
    ruido1 = np.random.randn(n_samples)
    ruido2 = np.random.randn(n_samples)
    ruido3 = np.random.randn(n_samples)
    ruido4 = np.random.randn(n_samples)
    ruido5 = np.random.randn(n_samples)

    # Variable objetivo
    retorno_futuro = (
        0.0001 +
        -0.5 * retorno_actual +  # Relevante
        0.00001 * (rsi - 50) +    # Relevante
        2.0 * macd +               # Relevante
        -0.1 * volatilidad +       # Relevante
        # Features de ruido NO tienen efecto
        np.random.randn(n_samples) * 0.003  # Ruido
    )

    # Preparar datos
    X = np.column_stack([
        retorno_actual, rsi, macd, volatilidad,  # Relevantes
        ruido1, ruido2, ruido3, ruido4, ruido5   # Irrelevantes
    ])
    y = retorno_futuro

    feature_names = [
        'Retorno_Actual', 'RSI', 'MACD', 'Volatilidad',  # Relevantes
        'Ruido_1', 'Ruido_2', 'Ruido_3', 'Ruido_4', 'Ruido_5'  # Irrelevantes
    ]

    # Dividir datos
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Estandarizar (importante para regularización)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Datos: {n_samples} muestras, {X.shape[1]} features")
    print(f"  - 4 features RELEVANTES")
    print(f"  - 5 features de RUIDO (sin edge)")
    print()

    # Valor fijo de alpha para comparación
    alpha = 0.01

    # ==========================================
    # 1. RIDGE REGRESSION
    # ==========================================
    print("\n" + "─" * 78)
    print("1. RIDGE REGRESSION (L2)")
    print("─" * 78)

    ridge = RegresionRegularizada(tipo='ridge', alpha=alpha)
    ridge.fit(X_train_scaled, y_train, feature_names=feature_names)

    r2_train_ridge = ridge.score(X_train_scaled, y_train)
    r2_test_ridge = ridge.score(X_test_scaled, y_test)

    print(f"R² train: {r2_train_ridge:.6f}")
    print(f"R² test:  {r2_test_ridge:.6f}")
    print(ridge.summary())

    # ==========================================
    # 2. LASSO REGRESSION
    # ==========================================
    print("\n" + "─" * 78)
    print("2. LASSO REGRESSION (L1)")
    print("─" * 78)

    lasso = RegresionRegularizada(tipo='lasso', alpha=alpha)
    lasso.fit(X_train_scaled, y_train, feature_names=feature_names)

    r2_train_lasso = lasso.score(X_train_scaled, y_train)
    r2_test_lasso = lasso.score(X_test_scaled, y_test)

    print(f"R² train: {r2_train_lasso:.6f}")
    print(f"R² test:  {r2_test_lasso:.6f}")
    print(lasso.summary())

    # ==========================================
    # 3. ELASTIC NET
    # ==========================================
    print("\n" + "─" * 78)
    print("3. ELASTIC NET (L1 + L2)")
    print("─" * 78)

    elastic = RegresionRegularizada(tipo='elastic_net', alpha=alpha, l1_ratio=0.5)
    elastic.fit(X_train_scaled, y_train, feature_names=feature_names)

    r2_train_elastic = elastic.score(X_train_scaled, y_train)
    r2_test_elastic = elastic.score(X_test_scaled, y_test)

    print(f"R² train: {r2_train_elastic:.6f}")
    print(f"R² test:  {r2_test_elastic:.6f}")
    print(elastic.summary())

    # ==========================================
    # COMPARACIÓN
    # ==========================================
    print("\n" + "=" * 78)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 78 + "\n")

    comparison = pd.DataFrame({
        'Modelo': ['Ridge', 'Lasso', 'Elastic Net'],
        'R² Train': [r2_train_ridge, r2_train_lasso, r2_train_elastic],
        'R² Test': [r2_test_ridge, r2_test_lasso, r2_test_elastic],
        'Features Seleccionadas': [
            ridge.n_features_selected_,
            lasso.n_features_selected_,
            elastic.n_features_selected_
        ]
    })

    print(comparison.to_string(index=False))
    print()

    print("CONCLUSIONES:")
    print("─" * 78)
    print("• Ridge: Mantiene todas las features, pero reduce coeficientes")
    print("• Lasso: Elimina features irrelevantes (selección automática)")
    print("• Elastic Net: Balance entre Ridge y Lasso")
    print()
    print("EN TRADING:")
    print("  → Usar Lasso para identificar features con edge real")
    print("  → Features eliminadas por Lasso probablemente no tienen edge")
    print()

    # Visualización
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, (modelo, nombre) in enumerate([(ridge, 'Ridge'), (lasso, 'Lasso'), (elastic, 'Elastic Net')]):
            ax = axes[idx]

            sorted_idx = np.argsort(modelo.feature_importance_)
            sorted_names = [modelo.feature_names_[i] for i in sorted_idx]
            sorted_coefs = modelo.coef_[sorted_idx]

            colors = ['green' if abs(c) > COEF_ZERO_THRESHOLD else 'red' for c in sorted_coefs]
            alphas_plot = [0.8 if abs(c) > COEF_ZERO_THRESHOLD else 0.3 for c in sorted_coefs]

            y_pos = np.arange(len(sorted_names))
            ax.barh(y_pos, sorted_coefs, color=colors, alpha=alphas_plot)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Coeficiente β')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_title(f'{nombre}\n{modelo.n_features_selected_}/{modelo.n_features_} features',
                        fontweight='bold')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"No se pudieron generar gráficos: {e}")


def ejemplo_cv_lasso():
    """
    Ejemplo de selección automática de α usando Cross-Validation
    """
    print("\n" + "=" * 78)
    print("EJEMPLO: Selección de α con Cross-Validation")
    print("=" * 78 + "\n")

    # Generar datos
    np.random.seed(42)
    n_samples = 500

    retorno_actual = np.random.randn(n_samples) * 0.001
    rsi = np.random.uniform(30, 70, n_samples)
    macd = np.random.randn(n_samples) * 0.0001
    volatilidad = np.abs(np.random.randn(n_samples) * 0.002)
    ruido1 = np.random.randn(n_samples)
    ruido2 = np.random.randn(n_samples)

    retorno_futuro = (
        0.0001 - 0.5 * retorno_actual + 0.00001 * (rsi - 50) +
        2.0 * macd - 0.1 * volatilidad +
        np.random.randn(n_samples) * 0.003
    )

    X = np.column_stack([retorno_actual, rsi, macd, volatilidad, ruido1, ruido2])
    y = retorno_futuro

    feature_names = ['Retorno_Actual', 'RSI', 'MACD', 'Volatilidad', 'Ruido_1', 'Ruido_2']

    # Cross-Validation
    cv_lasso = RegularizacionCV(
        tipo='lasso',
        alphas=np.logspace(-5, -1, 30),
        cv=5,
        scoring='r2'
    )

    cv_lasso.fit(X, y, feature_names=feature_names)

    print(cv_lasso.best_model_.summary())

    # Visualizar
    try:
        cv_lasso.plot_cv_scores()
    except Exception as e:
        print(f"No se pudo generar gráfico: {e}")


if __name__ == '__main__':
    # Ejecutar ejemplos
    ejemplo_comparacion_ridge_lasso()
    ejemplo_cv_lasso()
