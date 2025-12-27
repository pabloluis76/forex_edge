"""
Script para corregir el sistema hacia MÁXIMO REALISMO

Aplica las recomendaciones de auditoria_realismo.md
"""

print("=" * 80)
print("CORRECCIONES PARA MÁXIMO REALISMO")
print("=" * 80)

print("""
Este script implementa las siguientes correcciones:

🔴 ALTA PRIORIDAD:
   1. ATR: FAIL FAST en lugar de fillna con valores artificiales
   2. Log-retorno: VALIDAR datos de entrada (precios > 0)
   3. Validación estricta de rangos de precios

🟡 MEDIA PRIORIDAD:
   4. Logging de todos los valores artificiales usados
   5. Estadísticas de NaN encontrados y manejados

🟢 BAJA PRIORIDAD:
   6. Slippage ajustado por hora del día

IMPORTANTE:
-----------
Estas correcciones harán el sistema MÁS ESTRICTO:
- FALLARÁ si encuentra datos inválidos
- NO rellenará NaN con valores inventados
- RECHAZARÁ precios fuera de rango realista

¿Quieres aplicar estas correcciones? Esto podría causar que el sistema
falle en lugares donde antes funcionaba con valores artificiales.

Opciones:
---------
A) Aplicar TODO (máximo realismo - puede fallar más)
B) Solo ATR (no rellenar NaN)
C) Solo validación de precios
D) Solo logging (no cambiar comportamiento)
E) Cancelar

""")

respuesta = input("Selecciona opción (A/B/C/D/E): ").strip().upper()

if respuesta == "E":
    print("\n❌ Cancelado. No se aplicaron cambios.")
    exit(0)

print(f"\n✅ Opción seleccionada: {respuesta}")
print("\nGenerando archivos de corrección...\n")

# ============================================================================
# CORRECCIÓN 1: ATR con validación estricta
# ============================================================================

if respuesta in ["A", "B"]:
    print("📝 Generando: correcciones_atr_estricto.py")

    with open("correcciones_atr_estricto.py", "w") as f:
        f.write("""
# CORRECCIÓN: ATR con validación estricta (NO fillna artificial)
# Aplicar en: backtest/motor_backtest_completo.py

# ========== CÓDIGO ACTUAL (líneas 300-316) ==========
# nan_count = atr.isna().sum()
# if nan_count > 0:
#     if self.verbose:
#         print(f"  ⚠️  {par}: {nan_count} valores NaN en ATR")
#     atr_min_valid = atr.dropna().min() if atr.notna().any() else 0.0001
#     atr = atr.fillna(atr_min_valid)  # ← VALOR ARTIFICIAL

# ========== CÓDIGO NUEVO (REALISMO MÁXIMO) ==========

nan_count = atr.isna().sum()
if nan_count > 0:
    print(f"⚠️  {par}: ATR con {nan_count} NaN ({nan_count/len(atr)*100:.1f}%)")

    # OPCIÓN 1: FAIL FAST (más seguro)
    # raise ValueError(f"{par}: ATR inválido - verificar datos de entrada")

    # OPCIÓN 2: No operar en barras con NaN (más permisivo)
    # Dejar NaN como está - se validará en abrir_posicion()
    print(f"   → No se rellenará - trades rechazados en barras con ATR=NaN")

# NO hacer fillna - mantener NaN como indicador de dato inválido

# AGREGAR en abrir_posicion() (línea 486):
atr = row.get('ATR', None)

# VALIDACIÓN ESTRICTA
if atr is None or pd.isna(atr) or atr <= 0:
    if self.verbose:
        print(f"⚠️  ATR inválido ({atr}) en {timestamp} - Trade RECHAZADO")
    return  # No abrir posición

# Continuar solo si ATR es válido...
""")

    print("   ✅ Creado: correcciones_atr_estricto.py")

# ============================================================================
# CORRECCIÓN 2: Validación de precios en log-retorno
# ============================================================================

if respuesta in ["A", "C"]:
    print("📝 Generando: correcciones_precios_validos.py")

    with open("correcciones_precios_validos.py", "w") as f:
        f.write("""
# CORRECCIÓN: Validación estricta de precios
# Aplicar en: generacion_de_transformaciones/operadores_puros.py

# ========== CÓDIGO ACTUAL (líneas 102-105) ==========
# ratio = x / x.shift(n).clip(lower=EPSILON)  # ← Enmascara datos inválidos
# return np.log(ratio.clip(lower=EPSILON))

# ========== CÓDIGO NUEVO (REALISMO MÁXIMO) ==========

@staticmethod
def r(x: pd.Series, n: int) -> pd.Series:
    \"\"\"
    OPERADOR r (Log-retorno) - CON VALIDACIÓN ESTRICTA
    \"\"\"
    # VALIDACIÓN 1: Precios positivos
    if (x <= 0).any():
        invalid_count = (x <= 0).sum()
        raise ValueError(
            f"Precios ≤ 0 detectados ({invalid_count} valores) - "
            f"datos de entrada INVÁLIDOS. Rango forex debe ser > 0."
        )

    # VALIDACIÓN 2: Rango realista para forex
    # EUR/USD típicamente en [0.80, 1.60]
    # GBP/USD típicamente en [1.00, 2.00]
    # USD/JPY típicamente en [80, 160]
    MIN_PRICE = 0.0001  # 0.01 pips (mínimo absoluto)
    MAX_PRICE = 10000   # 10,000 (máximo razonable para cualquier par)

    if (x < MIN_PRICE).any() or (x > MAX_PRICE).any():
        raise ValueError(
            f"Precios fuera de rango realista [{MIN_PRICE}, {MAX_PRICE}] - "
            f"Rango encontrado: [{x.min():.6f}, {x.max():.6f}]"
        )

    # CÁLCULO sin clips artificiales
    ratio = x / x.shift(n)

    # VALIDACIÓN 3: Detectar infinitos o NaN resultantes
    if np.isinf(ratio).any():
        raise ValueError("División generó infinitos - verificar datos")

    return np.log(ratio)
    # NaN en primeras n barras es esperado y correcto
""")

    print("   ✅ Creado: correcciones_precios_validos.py")

# ============================================================================
# CORRECCIÓN 3: Logging extensivo
# ============================================================================

if respuesta in ["A", "D"]:
    print("📝 Generando: logging_valores_artificiales.py")

    with open("logging_valores_artificiales.py", "w") as f:
        f.write("""
# CORRECCIÓN: Logging de valores artificiales
# Aplicar en: múltiples archivos

import logging
from collections import defaultdict

class ValorArtificialLogger:
    \"\"\"Logger para rastrear uso de valores artificiales/protecciones\"\"\"

    def __init__(self):
        self.contadores = defaultdict(int)
        self.logger = logging.getLogger(__name__)

    def log_epsilon_usado(self, ubicacion: str, valor_original: float):
        \"\"\"Registra uso de epsilon en división\"\"\"
        self.contadores['epsilon_divisiones'] += 1
        if valor_original == 0:
            self.logger.warning(
                f"{ubicacion}: División por cero exacto - usando epsilon"
            )

    def log_clip_usado(self, ubicacion: str, valores_clipped: int):
        \"\"\"Registra uso de clip\"\"\"
        self.contadores['clips'] += 1
        self.logger.warning(
            f"{ubicacion}: {valores_clipped} valores clipped"
        )

    def log_nan_rellenado(self, ubicacion: str, cantidad: int):
        \"\"\"Registra NaN rellenados\"\"\"
        self.contadores['nan_fills'] += 1
        self.logger.warning(
            f"{ubicacion}: {cantidad} NaN rellenados con valor artificial"
        )

    def reporte_final(self):
        \"\"\"Genera reporte de valores artificiales usados\"\"\"
        print("\\n" + "=" * 80)
        print("REPORTE: Valores Artificiales Usados")
        print("=" * 80)

        if not self.contadores:
            print("✅ No se usaron valores artificiales")
        else:
            for tipo, cantidad in self.contadores.items():
                print(f"⚠️  {tipo}: {cantidad} veces")

        print("=" * 80)

# USAR en backtest:
logger_artificial = ValorArtificialLogger()

# Antes de:
# std_safe = std.replace(0, EPSILON)
# HACER:
if (std == 0).any():
    logger_artificial.log_epsilon_usado("Z-score", std[std==0].iloc[0])
std_safe = std.replace(0, EPSILON)

# Al final:
logger_artificial.reporte_final()
""")

    print("   ✅ Creado: logging_valores_artificiales.py")

# ============================================================================
# CORRECCIÓN 4: Actualizar constants.py
# ============================================================================

print("📝 Generando: constants_realismo_maximo.py")

with open("constants_realismo_maximo.py", "w") as f:
    f.write("""
# constants.py - CONFIGURACIÓN PARA REALISMO MÁXIMO

# Constantes numéricas
EPSILON = 1e-10                    # Solo para comparaciones de floats
EPSILON_NORMALIZATION = 1e-8       # Solo para normalización

# Validación estricta
ENABLE_STRICT_VALIDATION = True    # Fail fast en datos inválidos
ALLOW_FILLNA_ATR = False           # NO rellenar ATR con valores artificiales
MIN_ATR_BARS = 20                  # Mínimo de barras para ATR válido
MIN_HISTORY_BARS = 50              # Mínimo de historia antes de operar

# Rangos realistas para forex
MIN_PRICE_FOREX = 0.0001           # 0.01 pips (mínimo absoluto)
MAX_PRICE_FOREX = 10000            # Máximo razonable
MIN_ATR_VALUE = 0.00001            # ATR mínimo válido
MAX_ATR_VALUE = 1.0                # ATR máximo razonable

# Slippage realista
BASE_SLIPPAGE_PIPS = 0.5           # Base realista
MIN_SLIPPAGE_PIPS = 0.2            # Mínimo en condiciones ideales
MAX_SLIPPAGE_PIPS = 3.0            # Máximo en alta volatilidad
SLIPPAGE_LOW_LIQUIDITY_MULT = 1.5  # Multiplicador en horas de baja liquidez

# Horas de baja liquidez (UTC)
LOW_LIQUIDITY_HOURS = [0, 1, 2, 22, 23]

# Tolerancias
FLOAT_TOLERANCE = 1e-10
FEATURE_VARIANCE_THRESHOLD = 1e-10
COEF_ZERO_THRESHOLD = 1e-10

# Logging
LOG_ARTIFICIAL_VALUES = True       # Loguear uso de valores artificiales
LOG_NAN_HANDLING = True            # Loguear manejo de NaN
""")

print("   ✅ Creado: constants_realismo_maximo.py")

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "=" * 80)
print("RESUMEN DE ARCHIVOS GENERADOS")
print("=" * 80)

archivos_generados = []

if respuesta in ["A", "B"]:
    archivos_generados.append("correcciones_atr_estricto.py")

if respuesta in ["A", "C"]:
    archivos_generados.append("correcciones_precios_validos.py")

if respuesta in ["A", "D"]:
    archivos_generados.append("logging_valores_artificiales.py")

archivos_generados.append("constants_realismo_maximo.py")

for i, archivo in enumerate(archivos_generados, 1):
    print(f"{i}. {archivo}")

print("\n" + "=" * 80)
print("PRÓXIMOS PASOS")
print("=" * 80)

print("""
1. Revisar los archivos generados
2. Aplicar las correcciones manualmente a los archivos correspondientes
3. Ejecutar tests para verificar que el sistema falla correctamente
4. Ajustar configuración según necesidades

IMPORTANTE:
-----------
Con estas correcciones, el sistema:
✅ Será MÁS REALISTA
✅ FALLARÁ si encuentra datos inválidos (en lugar de ocultarlos)
✅ Producirá resultados más confiables
⚠️  Requerirá datos de entrada de ALTA CALIDAD
⚠️  Puede fallar en lugares donde antes funcionaba

¿Es esto lo que quieres? (Sí/No): """)

confirmacion = input().strip().lower()

if confirmacion in ['si', 'sí', 's', 'yes', 'y']:
    print("\n✅ Correcciones listas para aplicar")
    print("   Revisa los archivos generados e implementa los cambios manualmente")
else:
    print("\n⚠️  Revisa los archivos antes de aplicar")

print("\n" + "=" * 80)
