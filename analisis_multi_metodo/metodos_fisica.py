"""
MÉTODOS DE FÍSICA Y TEORÍA DE SISTEMAS
=======================================

Análisis avanzado inspirado en física, teoría del caos,
procesamiento de señales y teoría de redes.


MODELAR EL PRECIO COMO PROCESO ESTOCÁSTICO:
───────────────────────────────────────────


RANDOM WALK (Modelo más simple):

dP = σ dW

P sigue un paseo aleatorio.
No hay predicción posible.


GEOMETRIC BROWNIAN MOTION:

dP/P = μdt + σdW

El precio tiene:
- Drift μ (tendencia)
- Volatilidad σ
- Componente aleatorio dW

Usado en Black-Scholes para opciones.


ORNSTEIN-UHLENBECK (Mean Reversion):

dX = θ(μ - X)dt + σdW

El proceso tiende a revertir hacia μ.
θ = velocidad de reversión
Half-life = ln(2)/θ

SI el mercado sigue OU:
- Cuando X está lejos de μ → Esperar reversión
- θ alto → Reversión rápida
- θ bajo → Reversión lenta


JUMP DIFFUSION:

dP/P = μdt + σdW + J dN

Agrega "saltos" (J) que ocurren con probabilidad λ.
Modela eventos extremos (crashes, rallies).


ESTIMACIÓN DE PARÁMETROS:

Para OU:

θ = -ln(ρ) / Δt
donde ρ = autocorrelación lag-1

σ² = Var(X) × 2θ

μ = Media de largo plazo


APLICACIÓN:

1. Estimar parámetros del proceso
2. Si θ > 0 significativo → Hay mean reversion explotable
3. Si θ ≈ 0 → Random walk, no hay edge
4. Calcular half-life para sizing de trades


A) PROCESO ORNSTEIN-UHLENBECK
─────────────────────────────
Modelar la serie como proceso de mean reversion:

dX = θ(μ - X)dt + σdW

Estimar parámetros:
- θ = velocidad de reversión
- μ = nivel de equilibrio
- σ = volatilidad

Si θ > 0 significativamente:
→ Hay mean reversion explotable

Half-life = ln(2)/θ
→ Tiempo esperado para revertir a la mitad


B) HURST EXPONENT
─────────────────
H = log(R/S) / log(n)

H > 0.5: Serie PERSISTENTE (momentum funciona)
H = 0.5: Random walk (no hay edge)
H < 0.5: Serie ANTI-PERSISTENTE (mean reversion funciona)

Calcular H para cada par:
→ Indica qué tipo de estrategia aplicar


C) LYAPUNOV EXPONENT
────────────────────
λ = lim (1/n) Σ ln|f'(xᵢ)|

λ > 0: Sistema caótico, predictibilidad limitada
λ < 0: Sistema estable

Si λ > 0:
→ Solo predicción de muy corto plazo es posible


D) WAVELETS
───────────
Descomponer la serie en múltiples escalas temporales.

Niveles:
- Nivel 1-2: Ruido de alta frecuencia
- Nivel 3-4: Movimientos intradía
- Nivel 5-6: Tendencias de días
- Nivel 7+: Tendencias de semanas

Identificar en qué escala hay estructura predecible.


E) TEORÍA DE INFORMACIÓN
────────────────────────
Transfer Entropy: TE(X→Y)

Mide flujo de información de X a Y.

Si TE(Transform_j → retorno) > 0:
→ La transformación j contiene información sobre retornos futuros


F) ANÁLISIS DE REDES
────────────────────
Construir grafo de correlaciones entre activos.

Métricas:
- Centralidad: ¿Qué activo "lidera"?
- Clustering: ¿Hay grupos de activos?
- Cambios en estructura: ¿Cambio de régimen?


Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import logging
import warnings

warnings.filterwarnings('ignore')

# Librerías opcionales
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logging.warning("PyWavelets no disponible. Instala con: pip install PyWavelets")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX no disponible. Instala con: pip install networkx")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetodosFisica:
    """
    Métodos de física y teoría de sistemas para análisis financiero.
    """

    @staticmethod
    def ornstein_uhlenbeck(serie: np.ndarray,
                          dt: float = 1.0) -> Dict:
        """
        A) PROCESO ORNSTEIN-UHLENBECK

        Estima parámetros de proceso de mean reversion.

        Args:
            serie: Serie temporal
            dt: Incremento de tiempo (default: 1 período)

        Returns:
            Diccionario con parámetros θ, μ, σ y half-life
        """
        logger.info("="*70)
        logger.info("A) PROCESO ORNSTEIN-UHLENBECK")
        logger.info("="*70)

        # Eliminar NaNs
        serie_clean = serie[~np.isnan(serie)]

        if len(serie_clean) < 10:
            logger.error("Serie muy corta para estimar OU")
            return None

        # Método de regresión lineal para estimar parámetros
        # dX ≈ θ(μ - X)dt
        # Regresión: ΔX = a + b*X + ε
        # donde: b = -θ*dt, a = θ*μ*dt

        X = serie_clean[:-1]
        dX = np.diff(serie_clean)

        # CRÍTICO #4 CORREGIDO: Split temporal para evitar data leakage
        # Estimar parámetros en primera mitad, validar en segunda mitad
        split_idx = len(X) // 2
        X_train = X[:split_idx]
        dX_train = dX[:split_idx]
        X_test = X[split_idx:]
        dX_test = dX[split_idx:]

        # Regresión lineal en train set
        from sklearn.linear_model import LinearRegression

        modelo = LinearRegression()
        modelo.fit(X_train.reshape(-1, 1), dX_train)

        a = modelo.intercept_
        b = modelo.coef_[0]

        # Estimar parámetros OU
        theta = -b / dt
        mu = -a / b if b != 0 else np.mean(serie_clean)

        # Estimar σ (volatilidad de residuos en train)
        residuos_train = dX_train - (a + b * X_train)
        sigma = np.std(residuos_train) / np.sqrt(dt)

        # Half-life (tiempo para revertir a la mitad)
        if theta > 0:
            half_life = np.log(2) / theta
        else:
            half_life = np.inf

        # Test estadístico: ¿es θ significativamente > 0?
        # t-statistic para b (que es -θ*dt)
        n_train = len(X_train)

        # ALTO #1 CORREGIDO: Validar std antes de división por cero
        std_X_train = np.std(X_train)
        std_residuos = np.std(residuos_train)

        if std_X_train > 1e-10 and std_residuos > 0:
            se_b = std_residuos / (std_X_train * np.sqrt(n_train))

            # ALTO #2 CORREGIDO: Validar se_b antes de división
            if se_b > 1e-10:
                t_stat = b / se_b
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_train-2))
            else:
                t_stat = 0.0
                p_value = 1.0
        else:
            # Serie constante o sin variación - no hay mean reversion
            t_stat = 0.0
            p_value = 1.0

        # R² en train y test para validar estabilidad de parámetros
        r2_train = modelo.score(X_train.reshape(-1, 1), dX_train)
        r2_test = modelo.score(X_test.reshape(-1, 1), dX_test)

        resultados = {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': half_life,
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_reversion': theta > 0 and p_value < 0.05,
            'r_squared': r2_test,  # Reportar R² de test
            'r_squared_train': r2_train,
            'r_squared_test': r2_test
        }

        logger.info(f"Parámetros estimados:")
        logger.info(f"  θ (velocidad reversión): {theta:.6f}")
        logger.info(f"  μ (nivel equilibrio):    {mu:.6f}")
        logger.info(f"  σ (volatilidad):         {sigma:.6f}")
        logger.info(f"  Half-life:               {half_life:.2f} períodos")
        logger.info(f"  t-statistic:             {t_stat:.4f}")
        logger.info(f"  p-value:                 {p_value:.6f}")
        logger.info(f"  R² Train:                {r2_train:.6f}")
        logger.info(f"  R² Test:                 {r2_test:.6f}")

        if resultados['mean_reversion']:
            logger.info(f"✓ Mean reversion detectado (θ > 0, p < 0.05)")
        else:
            logger.info(f"✗ No hay evidencia significativa de mean reversion")

        return resultados

    @staticmethod
    def random_walk_test(serie: np.ndarray) -> Dict:
        """
        Test de Random Walk

        Determina si la serie sigue un paseo aleatorio (no hay predicción posible).

        Args:
            serie: Serie temporal (precios o log-precios)

        Returns:
            Diccionario con resultados del test
        """
        logger.info("="*70)
        logger.info("RANDOM WALK TEST")
        logger.info("="*70)

        # Eliminar NaNs
        serie_clean = serie[~np.isnan(serie)]

        # Calcular retornos
        retornos = np.diff(serie_clean)

        # Test de autocorrelación
        from statsmodels.stats.diagnostic import acorr_ljungbox

        # Ljung-Box test para autocorrelación
        lb_result = acorr_ljungbox(retornos, lags=[1, 5, 10], return_df=True)

        # Si p-value > 0.05 → No hay autocorrelación significativa → Random Walk
        is_random_walk = (lb_result['lb_pvalue'] > 0.05).all()

        # Varianza ratio test (Lo-MacKinlay)
        # VR = Var(retorno k-período) / (k × Var(retorno 1-período))
        # Si VR ≈ 1 → Random Walk

        k = 5
        var_1 = np.var(retornos)
        retornos_k = np.diff(serie_clean[::k])
        var_k = np.var(retornos_k)
        variance_ratio = var_k / (k * var_1) if var_1 > 0 else 0

        resultados = {
            'is_random_walk': is_random_walk,
            'ljungbox_pvalue': lb_result['lb_pvalue'].iloc[0],
            'variance_ratio': variance_ratio,
            'autocorr_lag1': np.corrcoef(retornos[:-1], retornos[1:])[0, 1]
        }

        logger.info(f"Ljung-Box p-value (lag 1): {resultados['ljungbox_pvalue']:.6f}")
        logger.info(f"Variance Ratio (k={k}):    {variance_ratio:.4f}")
        logger.info(f"Autocorrelación lag-1:     {resultados['autocorr_lag1']:.6f}")

        if is_random_walk:
            logger.info("✓ Serie sigue RANDOM WALK → No hay predicción posible")
        else:
            logger.info("✗ Serie NO es Random Walk → Posible edge predictivo")

        return resultados

    @staticmethod
    def geometric_brownian_motion(serie: np.ndarray, dt: float = 1.0) -> Dict:
        """
        Estima parámetros de Geometric Brownian Motion

        dP/P = μdt + σdW

        Args:
            serie: Serie de precios
            dt: Incremento de tiempo

        Returns:
            Diccionario con μ (drift) y σ (volatilidad)
        """
        logger.info("="*70)
        logger.info("GEOMETRIC BROWNIAN MOTION")
        logger.info("="*70)

        # Eliminar NaNs
        serie_clean = serie[~np.isnan(serie)]

        # Calcular log-retornos
        log_retornos = np.diff(np.log(serie_clean))

        # Estimación de parámetros
        # μ = E[log(P_t+1 / P_t)] / dt + σ²/2
        # σ² = Var[log(P_t+1 / P_t)] / dt

        media_log_ret = np.mean(log_retornos)
        var_log_ret = np.var(log_retornos)

        sigma_squared = var_log_ret / dt
        sigma = np.sqrt(sigma_squared)

        # Ajuste de drift
        mu = media_log_ret / dt + sigma_squared / 2

        # Intervalo de confianza para μ
        n = len(log_retornos)
        se_mu = sigma / np.sqrt(n * dt)
        ci_lower = mu - 1.96 * se_mu
        ci_upper = mu + 1.96 * se_mu

        resultados = {
            'mu': mu,
            'sigma': sigma,
            'mu_anualizado': mu * 252,  # Para datos diarios
            'sigma_anualizada': sigma * np.sqrt(252),
            'mu_ci_lower': ci_lower,
            'mu_ci_upper': ci_upper,
            'sharpe_ratio': mu / sigma if sigma > 0 else 0
        }

        logger.info(f"Drift (μ):              {mu:.6f} por período")
        logger.info(f"Volatilidad (σ):        {sigma:.6f} por período")
        logger.info(f"μ anualizado (252 días): {resultados['mu_anualizado']:.4f}")
        logger.info(f"σ anualizada (252 días): {resultados['sigma_anualizada']:.4f}")
        logger.info(f"Sharpe ratio:            {resultados['sharpe_ratio']:.4f}")
        logger.info(f"95% CI para μ:           [{ci_lower:.6f}, {ci_upper:.6f}]")

        if ci_lower > 0:
            logger.info("✓ Drift positivo significativo (tendencia alcista)")
        elif ci_upper < 0:
            logger.info("✓ Drift negativo significativo (tendencia bajista)")
        else:
            logger.info("○ Drift no significativamente diferente de cero")

        return resultados

    @staticmethod
    def jump_diffusion_detect(serie: np.ndarray,
                              threshold_std: float = 3.0) -> Dict:
        """
        Detecta "saltos" en la serie (Jump Diffusion model)

        dP/P = μdt + σdW + J dN

        Args:
            serie: Serie de precios
            threshold_std: Umbral en desviaciones estándar para detectar salto

        Returns:
            Diccionario con saltos detectados y parámetros
        """
        logger.info("="*70)
        logger.info("JUMP DIFFUSION - DETECCIÓN DE SALTOS")
        logger.info("="*70)

        # Eliminar NaNs
        serie_clean = serie[~np.isnan(serie)]

        # Calcular retornos
        retornos = np.diff(serie_clean) / serie_clean[:-1]

        # Identificar outliers (saltos)
        media = np.mean(retornos)
        std = np.std(retornos)

        # Saltos = retornos > threshold × std
        umbral_superior = media + threshold_std * std
        umbral_inferior = media - threshold_std * std

        saltos_indices = np.where((retornos > umbral_superior) | (retornos < umbral_inferior))[0]
        saltos_valores = retornos[saltos_indices]

        # Frecuencia de saltos (λ)
        n_periodos = len(retornos)
        lambda_saltos = len(saltos_indices) / n_periodos

        # Magnitud promedio de saltos
        if len(saltos_valores) > 0:
            magnitud_salto_promedio = np.mean(np.abs(saltos_valores))
            magnitud_salto_std = np.std(saltos_valores)
        else:
            magnitud_salto_promedio = 0
            magnitud_salto_std = 0

        # Retornos sin saltos (componente de difusión)
        retornos_sin_saltos = retornos[(retornos <= umbral_superior) & (retornos >= umbral_inferior)]
        sigma_difusion = np.std(retornos_sin_saltos)

        resultados = {
            'n_saltos': len(saltos_indices),
            'lambda_saltos': lambda_saltos,
            'saltos_indices': saltos_indices,
            'saltos_valores': saltos_valores,
            'magnitud_promedio': magnitud_salto_promedio,
            'magnitud_std': magnitud_salto_std,
            'sigma_difusion': sigma_difusion,
            'sigma_total': std,
            'threshold_std': threshold_std
        }

        logger.info(f"Saltos detectados:           {len(saltos_indices)}")
        logger.info(f"Frecuencia de saltos (λ):    {lambda_saltos:.6f} (prob. por período)")
        logger.info(f"Magnitud promedio de salto:  {magnitud_salto_promedio:.6f}")
        logger.info(f"Std de magnitud de saltos:   {magnitud_salto_std:.6f}")
        logger.info(f"σ (difusión sin saltos):     {sigma_difusion:.6f}")
        logger.info(f"σ (total con saltos):        {std:.6f}")

        if len(saltos_indices) > 0:
            logger.info(f"\nPrimeros saltos detectados:")
            for i in range(min(5, len(saltos_indices))):
                idx = saltos_indices[i]
                valor = saltos_valores[i]
                logger.info(f"  Período {idx}: {valor:>10.6f} ({valor*100:.2f}%)")

        if lambda_saltos > 0.01:
            logger.info("\n⚠ Saltos frecuentes detectados → Modelo Jump Diffusion apropiado")
        else:
            logger.info("\n○ Pocos saltos → GBM estándar puede ser suficiente")

        return resultados

    @staticmethod
    def hurst_exponent(serie: np.ndarray,
                      min_lag: int = 2,
                      max_lag: int = 100) -> Dict:
        """
        B) HURST EXPONENT

        Calcula el exponente de Hurst usando R/S analysis.

        Args:
            serie: Serie temporal
            min_lag: Lag mínimo
            max_lag: Lag máximo

        Returns:
            Diccionario con H y interpretación
        """
        logger.info("="*70)
        logger.info("B) HURST EXPONENT")
        logger.info("="*70)

        # Eliminar NaNs
        serie_clean = serie[~np.isnan(serie)]

        if len(serie_clean) < max_lag:
            max_lag = len(serie_clean) // 2

        lags = range(min_lag, max_lag)
        rs_values = []

        for lag in lags:
            # Dividir serie en chunks de tamaño lag
            n_chunks = len(serie_clean) // lag

            rs_chunk = []
            for i in range(n_chunks):
                chunk = serie_clean[i*lag:(i+1)*lag]

                # Mean
                mean = np.mean(chunk)

                # Cumulative deviation
                Y = np.cumsum(chunk - mean)

                # Range
                R = np.max(Y) - np.min(Y)

                # Standard deviation
                S = np.std(chunk)

                if S > 0:
                    rs_chunk.append(R / S)

            if len(rs_chunk) > 0:
                rs_values.append(np.mean(rs_chunk))

        # Regresión log-log
        log_lags = np.log(list(lags))
        log_rs = np.log(rs_values)

        # Eliminar infinitos o NaN
        valid = np.isfinite(log_lags) & np.isfinite(log_rs)
        log_lags = log_lags[valid]
        log_rs = log_rs[valid]

        if len(log_lags) < 2:
            logger.error("No hay suficientes datos para calcular Hurst")
            return None

        # Regresión lineal
        slope, intercept = np.polyfit(log_lags, log_rs, 1)
        H = slope

        # Interpretación
        if H > 0.55:
            interpretacion = "PERSISTENTE (momentum funciona)"
            estrategia = "Trend-following"
        elif H < 0.45:
            interpretacion = "ANTI-PERSISTENTE (mean reversion funciona)"
            estrategia = "Mean reversion"
        else:
            interpretacion = "RANDOM WALK (no hay edge claro)"
            estrategia = "Ninguna (o muy difícil)"

        resultados = {
            'H': H,
            'interpretacion': interpretacion,
            'estrategia_sugerida': estrategia,
            'persistente': H > 0.55,
            'anti_persistente': H < 0.45,
            'random_walk': 0.45 <= H <= 0.55
        }

        logger.info(f"Hurst Exponent: H = {H:.4f}")
        logger.info(f"Interpretación: {interpretacion}")
        logger.info(f"Estrategia sugerida: {estrategia}")

        return resultados

    @staticmethod
    def lyapunov_exponent(serie: np.ndarray,
                         embedding_dim: int = 3,
                         tau: int = 1,
                         n_iterations: int = 50) -> Dict:
        """
        C) LYAPUNOV EXPONENT

        Estima el máximo exponente de Lyapunov (aproximación simplificada).

        Args:
            serie: Serie temporal
            embedding_dim: Dimensión de embedding
            tau: Delay para embedding
            n_iterations: Número de iteraciones

        Returns:
            Diccionario con λ y predictibilidad
        """
        logger.info("="*70)
        logger.info("C) LYAPUNOV EXPONENT")
        logger.info("="*70)

        # Eliminar NaNs
        serie_clean = serie[~np.isnan(serie)]

        # Time-delay embedding
        n = len(serie_clean)
        m = embedding_dim

        # Simplified Lyapunov calculation
        # Usamos aproximación de divergencia de trayectorias cercanas

        divergences = []

        for _ in range(n_iterations):
            # Seleccionar punto inicial aleatorio
            idx = np.random.randint(m * tau, n - 10)

            # Encontrar punto cercano
            point = serie_clean[idx]
            distances = np.abs(serie_clean[m*tau:n-10] - point)
            distances[idx - m*tau] = np.inf  # Excluir el mismo punto

            nearest_idx = np.argmin(distances) + m * tau

            # Seguir evolución de ambos puntos
            steps = min(10, n - max(idx, nearest_idx) - 1)

            if steps < 2:
                continue

            initial_dist = abs(serie_clean[idx] - serie_clean[nearest_idx])
            final_dist = abs(serie_clean[idx + steps] - serie_clean[nearest_idx + steps])

            if initial_dist > 0 and final_dist > 0:
                divergence = np.log(final_dist / initial_dist) / steps
                divergences.append(divergence)

        if len(divergences) == 0:
            logger.error("No se pudo calcular Lyapunov exponent")
            return None

        lambda_max = np.mean(divergences)

        # Interpretación
        if lambda_max > 0.01:
            interpretacion = "CAÓTICO - Predictibilidad limitada"
            horizonte = "Muy corto plazo"
        elif lambda_max < -0.01:
            interpretacion = "ESTABLE - Mayor predictibilidad"
            horizonte = "Corto a medio plazo"
        else:
            interpretacion = "NEUTRAL"
            horizonte = "Incierto"

        resultados = {
            'lambda': lambda_max,
            'interpretacion': interpretacion,
            'horizonte_prediccion': horizonte,
            'caotico': lambda_max > 0.01,
            'estable': lambda_max < -0.01
        }

        logger.info(f"Lyapunov Exponent: λ = {lambda_max:.6f}")
        logger.info(f"Interpretación: {interpretacion}")
        logger.info(f"Horizonte de predicción: {horizonte}")

        return resultados

    @staticmethod
    def analisis_wavelets(serie: np.ndarray,
                         wavelet: str = 'db4',
                         nivel: int = 6) -> Optional[Dict]:
        """
        D) WAVELETS

        Descomposición multi-escala usando wavelets.

        Args:
            serie: Serie temporal
            wavelet: Tipo de wavelet ('db4', 'sym5', etc.)
            nivel: Número de niveles de descomposición

        Returns:
            Diccionario con coeficientes y energía por nivel
        """
        if not PYWT_AVAILABLE:
            logger.error("PyWavelets no disponible. Instala con: pip install PyWavelets")
            return None

        logger.info("="*70)
        logger.info("D) ANÁLISIS WAVELETS")
        logger.info("="*70)

        # Eliminar NaNs
        serie_clean = serie[~np.isnan(serie)]

        # Descomposición wavelet
        coeffs = pywt.wavedec(serie_clean, wavelet, level=nivel)

        # Calcular energía por nivel
        energias = []
        for i, c in enumerate(coeffs):
            energia = np.sum(c**2)
            energias.append(energia)

        energia_total = sum(energias)
        energia_pct = [e / energia_total * 100 for e in energias]

        # Interpretación de niveles
        interpretaciones = [
            "Aproximación (tendencia de largo plazo)",
            "Detalle nivel 1 (ruido alta frecuencia)",
            "Detalle nivel 2 (ruido alta frecuencia)",
            "Detalle nivel 3 (movimientos intradía)",
            "Detalle nivel 4 (movimientos intradía)",
            "Detalle nivel 5 (tendencias de días)",
            "Detalle nivel 6 (tendencias de días)",
            "Detalle nivel 7+ (tendencias de semanas)"
        ]

        resultados = {
            'wavelet': wavelet,
            'nivel': nivel,
            'coeffs': coeffs,
            'energias': energias,
            'energia_pct': energia_pct
        }

        logger.info(f"Wavelet: {wavelet}, Niveles: {nivel}")
        logger.info(f"Energía por nivel:")
        for i, (e, pct) in enumerate(zip(energias, energia_pct)):
            nivel_name = f"Nivel {i}" if i > 0 else "Aproximación"
            interp = interpretaciones[i] if i < len(interpretaciones) else ""
            logger.info(f"  {nivel_name}: {pct:.2f}% - {interp}")

        # Identificar nivel dominante
        nivel_dominante = np.argmax(energia_pct[1:]) + 1  # Excluir aproximación
        logger.info(f"Nivel dominante: {nivel_dominante} ({energia_pct[nivel_dominante]:.2f}% energía)")

        return resultados

    @staticmethod
    def transfer_entropy(X: np.ndarray,
                        Y: np.ndarray,
                        k: int = 1,
                        bins: int = 10) -> float:
        """
        E) TRANSFER ENTROPY

        Calcula TE(X→Y): flujo de información de X a Y.

        Args:
            X: Serie fuente
            Y: Serie destino
            k: Número de lags
            bins: Número de bins para discretización

        Returns:
            Transfer entropy
        """
        logger.info("="*70)
        logger.info("E) TRANSFER ENTROPY")
        logger.info("="*70)

        # Eliminar NaNs y alinear
        mask = ~(np.isnan(X) | np.isnan(Y))
        X_clean = X[mask]
        Y_clean = Y[mask]

        n = len(X_clean)

        if n < k + 1:
            logger.error("Serie muy corta para TE")
            return 0.0

        # Discretizar
        X_disc = np.digitize(X_clean, np.linspace(X_clean.min(), X_clean.max(), bins))
        Y_disc = np.digitize(Y_clean, np.linspace(Y_clean.min(), Y_clean.max(), bins))

        # Calcular probabilidades (implementación simplificada)
        # TE(X→Y) ≈ I(Y_future; X_past | Y_past)

        # Para simplicidad, usamos mutual information como proxy
        # (implementación completa de TE es más compleja)

        from sklearn.metrics import mutual_info_score

        # Y_future
        Y_future = Y_disc[k:]
        # X_past
        X_past = X_disc[:-k]
        # Y_past
        Y_past = Y_disc[:-k]

        # MI(Y_future, X_past)
        mi_xy = mutual_info_score(Y_future[:len(X_past)], X_past)

        # Esta es una aproximación simplificada
        te = mi_xy

        logger.info(f"Transfer Entropy (aproximado): TE(X→Y) = {te:.6f}")

        if te > 0.01:
            logger.info("✓ Hay flujo de información de X a Y")
        else:
            logger.info("✗ Flujo de información bajo o nulo")

        return te

    @staticmethod
    def analisis_redes(matriz_correlacion: np.ndarray,
                      nombres_activos: List[str],
                      umbral: float = 0.5) -> Optional[Dict]:
        """
        F) ANÁLISIS DE REDES

        Construye y analiza red de correlaciones entre activos.

        Args:
            matriz_correlacion: Matriz de correlación (n × n)
            nombres_activos: Nombres de los activos
            umbral: Umbral de correlación para crear edge

        Returns:
            Diccionario con métricas de red
        """
        if not NETWORKX_AVAILABLE:
            logger.error("NetworkX no disponible. Instala con: pip install networkx")
            return None

        logger.info("="*70)
        logger.info("F) ANÁLISIS DE REDES")
        logger.info("="*70)

        n = len(nombres_activos)

        # Crear grafo
        G = nx.Graph()

        # Añadir nodos
        for nombre in nombres_activos:
            G.add_node(nombre)

        # Añadir edges (correlaciones > umbral)
        for i in range(n):
            for j in range(i+1, n):
                corr = abs(matriz_correlacion[i, j])
                if corr > umbral:
                    G.add_edge(nombres_activos[i], nombres_activos[j], weight=corr)

        # Métricas de red
        # 1. Centralidad de grado
        degree_centrality = nx.degree_centrality(G)

        # 2. Centralidad de betweenness
        betweenness_centrality = nx.betweenness_centrality(G)

        # 3. Clustering coefficient
        clustering = nx.clustering(G)

        # 4. Comunidades (si hay suficientes nodos)
        if len(G.nodes()) > 2:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(G))
        else:
            communities = []

        # Activo más central (líder)
        lider = max(degree_centrality, key=degree_centrality.get)

        resultados = {
            'grafo': G,
            'n_nodos': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'densidad': nx.density(G),
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'clustering': clustering,
            'n_comunidades': len(communities),
            'lider': lider
        }

        logger.info(f"Red construida:")
        logger.info(f"  Nodos: {resultados['n_nodos']}")
        logger.info(f"  Edges: {resultados['n_edges']}")
        logger.info(f"  Densidad: {resultados['densidad']:.4f}")
        logger.info(f"  Comunidades: {resultados['n_comunidades']}")
        logger.info(f"  Activo líder (mayor centralidad): {lider}")
        logger.info(f"  Top 3 centralidad:")
        top_3 = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        for nombre, cent in top_3:
            logger.info(f"    {nombre}: {cent:.4f}")

        return resultados


def ejemplo_uso():
    """
    Ejemplo de uso de métodos de física.
    """
    print("="*70)
    print("EJEMPLO: MÉTODOS DE FÍSICA Y TEORÍA DE SISTEMAS")
    print("="*70)
    print()

    # Generar series sintéticas
    np.random.seed(42)
    n = 1000

    metodos = MetodosFisica()

    # ========================================
    # 1. RANDOM WALK
    # ========================================
    print("\n" + "="*70)
    print("1. RANDOM WALK")
    print("="*70)

    # Serie que sigue random walk
    serie_rw = np.cumsum(np.random.randn(n) * 0.01)

    rw_results = metodos.random_walk_test(serie_rw)

    # ========================================
    # 2. GEOMETRIC BROWNIAN MOTION
    # ========================================
    print("\n" + "="*70)
    print("2. GEOMETRIC BROWNIAN MOTION")
    print("="*70)

    # Serie GBM con drift positivo
    mu_gbm = 0.0005  # 0.05% por período
    sigma_gbm = 0.02  # 2% volatilidad
    precios_gbm = [100]

    for _ in range(n-1):
        dP = mu_gbm * precios_gbm[-1] + sigma_gbm * precios_gbm[-1] * np.random.randn()
        precios_gbm.append(precios_gbm[-1] + dP)

    precios_gbm = np.array(precios_gbm)

    gbm_results = metodos.geometric_brownian_motion(precios_gbm)

    # ========================================
    # 3. ORNSTEIN-UHLENBECK (MEAN REVERSION)
    # ========================================
    print("\n" + "="*70)
    print("3. ORNSTEIN-UHLENBECK (MEAN REVERSION)")
    print("="*70)

    # Serie con mean reversion (proceso OU)
    theta_true = 0.1
    mu_true = 100
    sigma_true = 1.0

    serie_ou = [mu_true]
    for _ in range(n-1):
        dX = theta_true * (mu_true - serie_ou[-1]) + sigma_true * np.random.randn()
        serie_ou.append(serie_ou[-1] + dX)

    serie_ou = np.array(serie_ou)

    ou_results = metodos.ornstein_uhlenbeck(serie_ou)

    # ========================================
    # 4. JUMP DIFFUSION
    # ========================================
    print("\n" + "="*70)
    print("4. JUMP DIFFUSION")
    print("="*70)

    # Serie con saltos ocasionales
    precios_jump = [100]
    lambda_jump = 0.02  # 2% probabilidad de salto por período

    for _ in range(n-1):
        # Componente de difusión
        dP = 0.0001 * precios_jump[-1] + 0.01 * precios_jump[-1] * np.random.randn()

        # Componente de salto
        if np.random.rand() < lambda_jump:
            jump = np.random.choice([-1, 1]) * 0.05 * precios_jump[-1]  # Salto de ±5%
            dP += jump

        precios_jump.append(precios_jump[-1] + dP)

    precios_jump = np.array(precios_jump)

    jump_results = metodos.jump_diffusion_detect(precios_jump, threshold_std=2.5)

    # B) Hurst Exponent
    print("\n")
    hurst_results = metodos.hurst_exponent(serie_ou)

    # C) Lyapunov
    print("\n")
    lyapunov_results = metodos.lyapunov_exponent(serie_ou)

    # D) Wavelets (si disponible)
    if PYWT_AVAILABLE:
        print("\n")
        wavelet_results = metodos.analisis_wavelets(serie_ou)

    # E) Transfer Entropy
    print("\n")
    # Serie Y que depende de X con lag
    serie_x = np.random.randn(n)
    serie_y = 0.5 * np.roll(serie_x, 1) + 0.5 * np.random.randn(n)
    te = metodos.transfer_entropy(serie_x, serie_y)

    # F) Análisis de Redes (si disponible)
    if NETWORKX_AVAILABLE:
        print("\n")
        # Matriz de correlación sintética
        n_activos = 5
        nombres = [f"Asset_{i}" for i in range(n_activos)]
        corr_matrix = np.random.rand(n_activos, n_activos)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)

        red_results = metodos.analisis_redes(corr_matrix, nombres, umbral=0.6)

    print("\n" + "="*70)


if __name__ == '__main__':
    ejemplo_uso()
