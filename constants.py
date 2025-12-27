"""
Constantes globales del sistema Forex Edge.

CONFIGURACIÓN: MÁXIMO REALISMO
===============================
Sistema configurado para producir resultados 100% realistas:
- NO rellena NaN con valores artificiales
- VALIDA rangos de precios estrictamente
- FALLA rápido si encuentra datos inválidos
"""

# ==============================================================================
# CONSTANTES NUMÉRICAS
# ==============================================================================

# Epsilon para operaciones matemáticas
EPSILON = 1e-10  # Valor epsilon para evitar división por cero
EPSILON_NORMALIZATION = 1e-8  # Valor epsilon específico para normalización (más conservador)

# Tolerancias para comparaciones de floats
FLOAT_TOLERANCE = 1e-10

# ==============================================================================
# VALIDACIÓN DE DATOS (MÁXIMO REALISMO)
# ==============================================================================

# Rangos válidos para precios de forex
MIN_PRICE_FOREX = 0.0001   # 1 pip mínimo absoluto (0.01 pips)
MAX_PRICE_FOREX = 10000    # Máximo razonable (cubre USD/JPY, etc.)

# Rangos válidos para ATR
MIN_ATR_VALUE = 0.00001    # ATR mínimo válido
MAX_ATR_VALUE = 1.0        # ATR máximo razonable

# Control de validación
ENABLE_STRICT_VALIDATION = True   # FAIL FAST si datos inválidos
ALLOW_FILLNA_ATR = False          # NO rellenar ATR con valores artificiales

# Mínimos de historia
MIN_ATR_BARS = 15          # Mínimo de barras para ATR válido (rolling window 14 + 1)
MIN_HISTORY_BARS = 50      # Mínimo de historia antes de operar

# ==============================================================================
# BACKTEST Y COSTOS
# ==============================================================================

# Constantes de backtest
BARS_PER_YEAR_M15 = 35_040  # 96 bars/día * 365 días

# Slippage realista
BASE_SLIPPAGE_PIPS = 0.5           # Base realista pero alcanzable
MIN_SLIPPAGE_PIPS = 0.2            # Mínimo en condiciones ideales
MAX_SLIPPAGE_PIPS = 3.0            # Máximo en alta volatilidad
SLIPPAGE_LOW_LIQUIDITY_MULT = 1.5  # Multiplicador en horas de baja liquidez

# Horas de baja liquidez (UTC)
LOW_LIQUIDITY_HOURS = [0, 1, 2, 22, 23]

# ==============================================================================
# FEATURE SELECTION
# ==============================================================================

# Umbrales para validación de features
FEATURE_VARIANCE_THRESHOLD = 1e-10  # Umbral mínimo de varianza para considerar un feature válido
COEF_ZERO_THRESHOLD = 1e-10         # Umbral para considerar un coeficiente como cero en regresión

# ==============================================================================
# LOGGING Y DEBUGGING
# ==============================================================================

# Control de logging
LOG_ARTIFICIAL_VALUES = True   # Loguear uso de valores artificiales
LOG_NAN_HANDLING = True        # Loguear manejo de NaN
LOG_TRADES_REJECTED = True     # Loguear trades rechazados
