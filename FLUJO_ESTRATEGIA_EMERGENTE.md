# 🔄 FLUJO DE DATOS HACIA ESTRATEGIA EMERGENTE

## 📊 INPUT REQUERIDO

`estrategia_emergente` necesita un **DataFrame** con transformaciones validadas:

```python
transformaciones_validadas = pd.DataFrame({
    'Transformacion': ['R_1_C', 'Pos_20_C', 'R_24_C_minus_R_96_C', ...],
    'IC': [0.016, -0.028, 0.024, ...],
    'Robusto': ['Sí', 'Sí', 'Sí', ...],
    'Estable': ['Sí', 'Sí', 'Sí', ...],
    'P_Value': [0.0001, 0.0005, 0.0010, ...]
})
```

---

## 🗺️ MÓDULOS QUE ALIMENTAN LA ESTRATEGIA

### **PIPELINE COMPLETO:**

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. GENERACIÓN DE TRANSFORMACIONES                               │
│    Módulo: generacion_de_transformaciones/                      │
│    Script: ejecutar_generacion_transformaciones.py              │
│                                                                  │
│    Input:  datos/ohlc/*.parquet (OHLCV raw)                    │
│    Output: datos/features/*.parquet (~1,700 transformaciones)   │
│                                                                  │
│    ┌───────────────────────────────────────────────────┐        │
│    │ • Delta, R, r, mu, sigma, Max, Min, Z, Pos, Rank │        │
│    │ • Ventanas: [1,2,3,4,5,10,20,50,100,200]        │        │
│    │ • Combinaciones de variables (C/O, H-L, etc.)    │        │
│    │ • Composiciones (Z(mu(C)), D²(σ(C)), etc.)       │        │
│    └───────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. ANÁLISIS MULTI-MÉTODO                                        │
│    Módulo: analisis_multi_metodo/                               │
│    Script: ejecutar_analisis_multimetodo.py                     │
│                                                                  │
│    Input:  datos/features/*.parquet                             │
│    Output: datos/analisis_multimetodo/                          │
│            ├── EUR_USD_M15_analisis_IC.csv                      │
│            ├── EUR_USD_M15_analisis_MI.csv                      │
│            ├── EUR_USD_M15_analisis_RF_importance.csv           │
│            └── EUR_USD_M15_analisis_completo.json               │
│                                                                  │
│    ┌───────────────────────────────────────────────────┐        │
│    │ AnalizadorEstadistico:                            │        │
│    │ • Information Coefficient (IC)                    │        │
│    │ • Información Mutua (MI)                          │        │
│    │ • Regresión Lasso (selección features)            │        │
│    │ • PCA (componentes principales)                   │        │
│    │                                                    │        │
│    │ AnalizadorML:                                     │        │
│    │ • Random Forest (feature importance)              │        │
│    │ • Gradient Boosting                               │        │
│    │ • XGBoost/LightGBM (opcional)                     │        │
│    └───────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. MÉTODOS ESTADÍSTICOS CLÁSICOS (Opcional)                     │
│    Módulo: Métodos Estadísticos Clásicos/                       │
│    Script: ejecutar_metodos_estadisticos_clasicos.py            │
│                                                                  │
│    Input:  datos/features/*.parquet                             │
│    Output: datos/metodos_estadisticos_clasicos/                 │
│            ├── regresion_lineal/*.csv (coeficientes)            │
│            ├── regresion_regularizada/*.csv (lasso features)    │
│            ├── pca/*.csv                                        │
│            └── correlacion/*.csv                                │
│                                                                  │
│    ┌───────────────────────────────────────────────────┐        │
│    │ • Regresión Lineal OLS (coeficientes β)          │        │
│    │ • Ridge/Lasso (regularización)                    │        │
│    │ • PCA (reducción dimensionalidad)                 │        │
│    │ • Correlación (features redundantes)              │        │
│    └───────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. CONSENSO DE MÉTODOS ⭐ CLAVE                                 │
│    Módulo: consenso_metodos/                                    │
│    Script: ejecutar_consenso_metodos.py                         │
│                                                                  │
│    Input:  datos/features/*.parquet                             │
│            datos/analisis_multimetodo/*.csv                     │
│    Output: datos/consenso_metodos/                              │
│            ├── rankings/*.csv (por método)                      │
│            ├── consenso/*.csv (tabla de consenso)               │
│            └── features_aprobados/*.csv ← ARCHIVO CLAVE         │
│                                                                  │
│    ┌───────────────────────────────────────────────────┐        │
│    │ TablaConsenso:                                    │        │
│    │ • Evalúa CADA transformación con TODOS métodos    │        │
│    │ • Cuenta "votos" (≥5 métodos = consenso fuerte)   │        │
│    │                                                    │        │
│    │ ProcesoConsenso:                                  │        │
│    │ PASO 1: Generar rankings por método               │        │
│    │ PASO 2: Calcular intersecciones                   │        │
│    │ PASO 3: Verificación cruzada                      │        │
│    │                                                    │        │
│    │ → Features que aparecen en ≥5 métodos             │        │
│    └───────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. VALIDACIÓN RIGUROSA ⭐⭐⭐ CRÍTICO                           │
│    Módulo: validacion_rigurosa/                                 │
│    Script: ejecutar_validacion_rigurosa.py                      │
│                                                                  │
│    Input:  datos/features/*.parquet                             │
│            datos/consenso_metodos/features_aprobados/*.csv      │
│    Output: datos/validacion_rigurosa/                           │
│            ├── walk_forward/*.json                              │
│            ├── bootstrap/*.json                                 │
│            ├── permutation/*.json                               │
│            ├── robustez/*.json                                  │
│            └── features_validados/*.csv ← USAR ESTOS            │
│                                                                  │
│    ┌───────────────────────────────────────────────────┐        │
│    │ WalkForwardValidation:                            │        │
│    │ • Ventanas deslizantes [TRAIN][TEST]              │        │
│    │ • Sin información futura                          │        │
│    │ • Evalúa estabilidad temporal                     │        │
│    │                                                    │        │
│    │ BootstrapIntervalosConfianza:                     │        │
│    │ • Resampling 10,000 iteraciones                   │        │
│    │ • IC 95% para métricas                            │        │
│    │ • Si IC incluye 0 → No significativo              │        │
│    │                                                    │        │
│    │ PermutationTest:                                  │        │
│    │ • Destruye relación temporal                      │        │
│    │ • p-value < 0.001 → Edge real                     │        │
│    │                                                    │        │
│    │ AnalisisRobustez:                                 │        │
│    │ • IC estable por año                              │        │
│    │ • Sensibilidad a parámetros                       │        │
│    │ • Consistencia entre activos                      │        │
│    │                                                    │        │
│    │ → Features que pasan ≥3/4 validaciones            │        │
│    └───────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. ESTRATEGIA EMERGENTE 🎯 DESTINO FINAL                        │
│    Módulo: estrategia_emergente/                                │
│                                                                  │
│    Input:  datos/validacion_rigurosa/features_validados/*.csv  │
│                                                                  │
│    DataFrame esperado:                                          │
│    ┌────────────────┬────────┬──────────┬─────────┬──────────┐ │
│    │ Transformacion │   IC   │ Robusto  │ Estable │ P_Value  │ │
│    ├────────────────┼────────┼──────────┼─────────┼──────────┤ │
│    │ R_1_C          │ 0.016  │ Sí       │ Sí      │ 0.0001   │ │
│    │ Pos_20_C       │-0.028  │ Sí       │ Sí      │ 0.0005   │ │
│    │ R_24-R_96      │ 0.024  │ Sí       │ Sí      │ 0.0010   │ │
│    │ σ₁₀/σ₅₀        │-0.021  │ Sí       │ Sí      │ 0.0020   │ │
│    │ hour×R_4       │ 0.019  │ Sí       │ Sí      │ 0.0030   │ │
│    └────────────────┴────────┴──────────┴─────────┴──────────┘ │
│                                                                  │
│    ┌────────────────────────────────────────────────┐           │
│    │ InterpretacionPostHoc:                         │           │
│    │ 1. Detectar patrones (mean reversion, etc.)    │           │
│    │ 2. Interpretar en lenguaje natural             │           │
│    │ 3. Asignar pesos basados en |IC|               │           │
│    │ 4. Generar señales combinadas                  │           │
│    │                                                 │           │
│    │ FormulacionReglas:                             │           │
│    │ 1. Crear reglas ejecutables (if-then)          │           │
│    │ 2. Calcular position sizing                    │           │
│    │ 3. Definir stop loss / take profit             │           │
│    │ 4. Generar código Python ejecutable            │           │
│    └────────────────────────────────────────────────┘           │
│                                                                  │
│    Output: estrategia_emergente_codigo.py ← CÓDIGO FINAL        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 RESUMEN DE DEPENDENCIAS

### **Módulos que alimentan `estrategia_emergente`:**

| # | Módulo | Qué aporta | Archivo clave |
|---|--------|------------|---------------|
| 1 | **generacion_de_transformaciones** | Genera ~1,700 transformaciones por par | `datos/features/*.parquet` |
| 2 | **analisis_multi_metodo** | IC, MI, RF importance, métricas ML | `datos/analisis_multimetodo/*.csv` |
| 3 | **consenso_metodos** | Features con consenso ≥5 métodos | `datos/consenso_metodos/features_aprobados/*.csv` |
| 4 | **validacion_rigurosa** | Features que pasan walk-forward, bootstrap, permutation, robustez | `datos/validacion_rigurosa/features_validados/*.csv` ⭐ |

### **Datos mínimos requeridos:**

```python
# OPCIÓN 1: Usar features validados (RECOMENDADO)
df_validados = pd.read_csv('datos/validacion_rigurosa/features_validados/EUR_USD_M15_features_validados.csv')

# OPCIÓN 2: Usar features aprobados por consenso
df_aprobados = pd.read_csv('datos/consenso_metodos/features_aprobados/EUR_USD_M15_features_aprobados.csv')

# OPCIÓN 3: Crear manualmente (para testing)
df_manual = pd.DataFrame({
    'Transformacion': ['R_1_C', 'Pos_20_C', ...],
    'IC': [0.016, -0.028, ...],
    'Robusto': ['Sí', 'Sí', ...],
    'Estable': ['Sí', 'Sí', ...]
})
```

---

## 🔄 FLUJO COMPLETO EN CÓDIGO

```python
# PASO 1: Generar transformaciones
!python ejecutar_generacion_transformaciones.py
# → datos/features/*.parquet

# PASO 2: Análisis multi-método
!python ejecutar_analisis_multimetodo.py
# → datos/analisis_multimetodo/*

# PASO 3: Consenso de métodos
!python ejecutar_consenso_metodos.py
# → datos/consenso_metodos/features_aprobados/*.csv

# PASO 4: Validación rigurosa
!python ejecutar_validacion_rigurosa.py
# → datos/validacion_rigurosa/features_validados/*.csv

# PASO 5: Estrategia emergente
from estrategia_emergente.interpretacion_post_hoc import InterpretacionPostHoc
from estrategia_emergente.formulacion_reglas import FormulacionReglas
import pandas as pd

# Cargar features validados
df_validados = pd.read_csv(
    'datos/validacion_rigurosa/features_validados/EUR_USD_M15_features_validados.csv'
)

# Necesitamos agregar IC (viene del análisis)
df_ic = pd.read_csv(
    'datos/analisis_multimetodo/EUR_USD_M15_analisis_IC.csv'
)

# Merge para tener IC + validación
df_completo = df_validados.merge(df_ic[['feature', 'IC']],
                                   left_on='feature',
                                   right_on='feature')

# Renombrar columna para estrategia_emergente
df_completo = df_completo.rename(columns={'feature': 'Transformacion'})

# Agregar columnas requeridas (si no existen)
df_completo['Robusto'] = 'Sí'
df_completo['Estable'] = 'Sí'

# PASO 5A: Interpretar
interpretador = InterpretacionPostHoc(verbose=True)
df_interpretado = interpretador.interpretar_transformaciones_validadas(df_completo)

# PASO 5B: Generar estrategia combinada
df_estrategia = interpretador.generar_estrategia_combinada(df_interpretado)

# PASO 5C: Formular reglas
formulador = FormulacionReglas(df_estrategia, verbose=True)
reglas_long, reglas_short = formulador.generar_reglas_entrada()

# PASO 5D: Generar código ejecutable
codigo = formulador.generar_codigo_estrategia(
    ruta_salida='estrategia_emergente_EUR_USD.py'
)

print("✓ Estrategia generada en: estrategia_emergente_EUR_USD.py")
```

---

## 🎯 CONCLUSIÓN

**`estrategia_emergente` NO genera datos, los consume:**

- ✅ **Recibe**: Features que ya pasaron TODAS las validaciones
- ✅ **Procesa**: Interpreta + Formula reglas + Genera código
- ✅ **Produce**: Estrategia ejecutable en Python

**Depende de:**
1. `generacion_de_transformaciones` → Crear features
2. `consenso_metodos` → Filtrar por consenso
3. `validacion_rigurosa` → Confirmar robustez
4. Luego → `estrategia_emergente` → Convertir en código

**Es el ÚLTIMO PASO del pipeline, no el primero.**
