# PIPELINE VERIFICATION - FOREX EDGE

## Flujo de Datos Entre Módulos

### 1. GENERACION_TRANSFORMACIONES
**Script:** `ejecutar_generacion_transformaciones.py`
**Input:** 
- datos/ohlc/{PAR}_M15.csv

**Output:**
- datos/features/{PAR}_M15_features.parquet

**Status:** ✓ OK - Ya configurado para EUR_USD

---

### 2. ANALISIS_MULTIMETODO
**Script:** `ejecutar_analisis_multimetodo.py`
**Input:** 
- datos/features/{PAR}_M15_features.parquet

**Output:**
- datos/analisis_multimetodo/{PAR}_M15_analisis_IC.csv
- datos/analisis_multimetodo/{PAR}_M15_analisis_MI.csv
- datos/analisis_multimetodo/{PAR}_M15_analisis_RF_importance.csv
- datos/analisis_multimetodo/{PAR}_M15_analisis_DL.json
- datos/analisis_multimetodo/{PAR}_M15_analisis_completo.json

**Status:** ✓ ACTUALIZADO - EUR_USD únicamente

---

### 3. CONSENSO_METODOS
**Script:** `ejecutar_consenso_metodos.py`
**Input:** 
- datos/features/{PAR}_M15_features.parquet (para X, y en modo recálculo)
- datos/analisis_multimetodo/{PAR}_M15_analisis_IC.csv (modo optimizado)
- datos/analisis_multimetodo/{PAR}_M15_analisis_MI.csv (modo optimizado)
- datos/analisis_multimetodo/{PAR}_M15_analisis_RF_importance.csv (modo optimizado)
- datos/analisis_multimetodo/{PAR}_M15_analisis_DL.json (modo optimizado)

**Output:**
- datos/consenso_metodos/rankings/{PAR}_M15_ranking_*.csv
- datos/consenso_metodos/consenso/{PAR}_M15_tabla_consenso.csv
- datos/consenso_metodos/features_aprobados/{PAR}_M15_features_aprobados.csv

**Status:** ✓ ACTUALIZADO - EUR_USD únicamente
**Modo:** OPTIMIZADO si CSVs existen, RECÁLCULO si no

---

### 4. VALIDACION_RIGUROSA
**Script:** `ejecutar_validacion_rigurosa.py`
**Input:** 
- datos/features/{PAR}_M15_features.parquet (features completos)
- datos/consenso_metodos/features_aprobados/{PAR}_M15_features_aprobados.csv

**Output:**
- datos/validacion_rigurosa/walk_forward/{PAR}_M15_walk_forward.csv
- datos/validacion_rigurosa/bootstrap/{PAR}_M15_bootstrap.csv
- datos/validacion_rigurosa/permutation/{PAR}_M15_permutation.csv
- datos/validacion_rigurosa/robustez/{PAR}_M15_robustez.json
- datos/validacion_rigurosa/features_validados/{PAR}_M15_features_validados.csv

**Status:** ✓ ACTUALIZADO - EUR_USD únicamente

---

### 5. ESTRATEGIA_EMERGENTE
**Script:** `ejecutar_estrategia_emergente.py`
**Input:** 
- datos/validacion_rigurosa/features_validados/{PAR}_M15_features_validados.csv
- datos/analisis_multimetodo/{PAR}_M15_analisis_IC.csv

**Output:**
- datos/estrategia_emergente/{PAR}_M15_estrategia.json
- datos/estrategia_emergente/{PAR}_M15_reglas.txt

**Status:** ✓ ACTUALIZADO - EUR_USD por defecto

---

### 6. BACKTEST
**Script:** `ejecutar_backtest.py`
**Input:** 
- datos/ohlc/{PAR}_M15.csv
- datos/estrategia_emergente/{PAR}_M15_estrategia.json

**Output:**
- datos/backtest/{PAR}_M15_backtest.json
- datos/backtest/{PAR}_M15_operaciones.csv
- datos/backtest/{PAR}_M15_metricas.csv

**Status:** ✓ ACTUALIZADO - EUR_USD por defecto

---

### 7. ESTRUCTURA_MATRICIAL_TENSORIAL (Opcional)
**Script:** `ejecutar_estructura_matricial_tensorial.py`
**Input:** 
- datos/features/{PAR}_M15_features.parquet

**Output:**
- datos/estructura_matricial_tensorial/tensor_3d/{PAR}_M15_tensor_3d_lookback_*.npz
- datos/estructura_matricial_tensorial/tensor_4d/{PAR}_M15_tensor_4d.npz

**Status:** ✓ OK - Ya configurado para EUR_USD

---

### 8. METODOS_ESTADISTICOS_CLASICOS (Opcional)
**Script:** `ejecutar_metodos_estadisticos_clasicos.py`
**Input:** 
- datos/features/{PAR}_M15_features.parquet

**Output:**
- datos/metodos_estadisticos_clasicos/regresion/{PAR}_M15_*.csv
- datos/metodos_estadisticos_clasicos/pca/{PAR}_M15_*.csv
- datos/metodos_estadisticos_clasicos/correlacion/{PAR}_M15_*.csv

**Status:** ✓ ACTUALIZADO - EUR_USD únicamente

---

## ORDEN DE EJECUCIÓN RECOMENDADO

1. `python ejecutar_generacion_transformaciones.py`
2. `python ejecutar_analisis_multimetodo.py`
3. `python ejecutar_consenso_metodos.py`
4. `python ejecutar_validacion_rigurosa.py`
5. `python ejecutar_estrategia_emergente.py`
6. `python ejecutar_backtest.py`

**Opcionales (en paralelo después de paso 1):**
- `python ejecutar_estructura_matricial_tensorial.py`
- `python ejecutar_metodos_estadisticos_clasicos.py`

---

## PROBLEMAS DETECTADOS Y CORREGIDOS

### ✓ Problema 1: Duplicación de cálculos (RESUELTO)
- **Antes:** Consenso recalculaba IC, MI, RF
- **Ahora:** Carga CSVs pre-generados cuando existen (modo optimizado)

### ✓ Problema 2: Deep Learning no integrado (RESUELTO)
- **Antes:** DL generado pero no usado en consenso
- **Ahora:** DL-MLP incluido en votación de consenso

### ✓ Problema 3: Normalización inconsistente (RESUELTO)
- **Antes:** MLP usaba datos sin normalizar
- **Ahora:** MLP usa StandardScaler, CNN/LSTM usan Z-score point-in-time

### ✓ Problema 4: Múltiples pares (RESUELTO)
- **Antes:** Scripts procesaban 6 pares (EUR_USD, GBP_USD, USD_JPY, EUR_JPY, GBP_JPY, AUD_USD)
- **Ahora:** Todos los scripts configurados para EUR_USD únicamente

---

## VERIFICACIÓN DE ENLACES

| Módulo | Lee Correctamente | Genera Correctamente | Status |
|--------|-------------------|---------------------|---------|
| generacion_transformaciones | ✓ OHLC | ✓ features.parquet | OK |
| analisis_multimetodo | ✓ features.parquet | ✓ IC/MI/RF/DL CSVs | OK |
| consenso_metodos | ✓ CSVs + features | ✓ features_aprobados.csv | OK |
| validacion_rigurosa | ✓ features + aprobados | ✓ features_validados.csv | OK |
| estrategia_emergente | ✓ validados + IC | ✓ estrategia.json | OK |
| backtest | ✓ estrategia + OHLC | ✓ resultados backtest | OK |

**CONCLUSIÓN:** ✓ Todos los enlaces entre módulos están correctos
