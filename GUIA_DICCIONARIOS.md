# GUÍA: Validación y Uso Seguro de Diccionarios

## 🔴 PROBLEMAS IDENTIFICADOS

Se encontraron **múltiples archivos con accesos inseguros** a diccionarios:

| Archivo | Accesos directos | `.get()` | Ratio seguro |
|---------|------------------|----------|--------------|
| motor_backtest_completo.py | 138 | 3 | 2.2% ⚠️ |
| formulacion_reglas.py | 83 | 2 | 2.4% ⚠️ |
| proceso_consenso.py | 38 | 0 | 0.0% ⚠️ |
| walk_forward_validation.py | 45 | 1 | 2.2% ⚠️ |
| analisis_robustez.py | 48 | 0 | 0.0% ⚠️ |

### Riesgo:
```python
# PELIGROSO - Crash si falta la clave
valor = config['risk_per_trade']  # ❌ KeyError si no existe

# SEGURO - Retorna None o valor por defecto
valor = config.get('risk_per_trade', 0.01)  # ✓ Sin crash
```

## ✅ SOLUCIONES IMPLEMENTADAS

### 1. Validación en motor_backtest_completo.py

Se agregaron dos métodos de validación:

```python
def _validar_columnas_requeridas(self):
    """Valida que DataFrames tengan todas las columnas requeridas."""
    columnas_ohlcv_requeridas = ['timestamp', 'pair', 'open', 'high', 'low', 'close', 'volume']
    columnas_faltantes = [col for col in columnas_ohlcv_requeridas if col not in self.df_ohlcv.columns]

    if columnas_faltantes:
        raise ValueError(f"OHLCV DataFrame falta columnas: {columnas_faltantes}")

def _validar_configuracion(self):
    """Valida que config tenga todas las claves requeridas."""
    claves_requeridas = [
        'risk_per_trade', 'max_position_size', 'stop_loss_atr_mult',
        'take_profit_atr_mult', 'timeout_bars', 'base_slippage_pips',
        'max_spread_pips', 'avoid_rollover_hours'
    ]

    claves_faltantes = [clave for clave in claves_requeridas if clave not in self.config]

    if claves_faltantes:
        raise ValueError(f"Config falta claves: {claves_faltantes}")

    # Validar rangos
    if not (0 < self.config['risk_per_trade'] <= 0.1):
        raise ValueError(f"risk_per_trade fuera de rango")
```

### 2. Validación en formulacion_reglas.py

```python
# Validar columnas antes de acceder
columnas_requeridas = ['Transformacion', 'IC']
columnas_faltantes = [col for col in columnas_requeridas
                      if col not in self.transformaciones_validadas.columns]

if columnas_faltantes:
    raise ValueError(f"Faltan columnas: {columnas_faltantes}")
```

## 📖 MEJORES PRÁCTICAS

### ✅ BUENAS Prácticas:

#### 1. Validar columnas de DataFrame ANTES de iterar

```python
# ANTES (peligroso)
for _, row in df.iterrows():
    valor = row['columna']  # ❌ Crash si falta columna

# DESPUÉS (seguro)
columnas_requeridas = ['columna1', 'columna2']
columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
if columnas_faltantes:
    raise ValueError(f"Faltan columnas: {columnas_faltantes}")

for _, row in df.iterrows():
    valor = row['columna']  # ✓ Ya validado
```

#### 2. Usar `.get()` para diccionarios opcionales

```python
# ANTES (peligroso)
valor = diccionario['clave_opcional']  # ❌ KeyError si no existe

# DESPUÉS (seguro)
valor = diccionario.get('clave_opcional', valor_por_defecto)  # ✓ Sin crash
```

#### 3. Validar configuración en `__init__`

```python
class MiClase:
    def __init__(self, config: Dict):
        # Validar INMEDIATAMENTE
        self._validar_config(config)
        self.config = config

    def _validar_config(self, config: Dict):
        """Valida configuración (fail-fast)."""
        claves_requeridas = ['clave1', 'clave2', 'clave3']
        claves_faltantes = [k for k in claves_requeridas if k not in config]

        if claves_faltantes:
            raise ValueError(f"Config falta: {claves_faltantes}")
```

#### 4. Proporcionar valores por defecto claros

```python
# ANTES
config = {}
config.update(user_config)  # ❌ Si user_config vacío, crash después

# DESPUÉS
config = {
    'risk_per_trade': 0.01,      # Valor por defecto
    'max_position_size': 0.10,   # Valor por defecto
    'timeout_bars': 50           # Valor por defecto
}
config.update(user_config)       # ✓ Siempre hay valores
```

### ❌ MALAS Prácticas:

#### 1. Acceso directo sin validación

```python
# MAL
def procesar(row):
    precio = row['close']  # ❌ Crash si falta 'close'

# BIEN
def procesar(row):
    if 'close' not in row:
        raise ValueError("Falta columna 'close'")
    precio = row['close']  # ✓ Validado
```

#### 2. Try-except silencioso

```python
# MAL - Oculta el error
try:
    valor = dict['clave']
except KeyError:
    pass  # ❌ Error silenciado

# BIEN - Error descriptivo
try:
    valor = dict['clave']
except KeyError as e:
    raise ValueError(f"Falta clave requerida: {e}") from e
```

#### 3. Validación tardía

```python
# MAL - Falla después de mucho procesamiento
def procesar_datos(df):
    # ... 100 líneas de procesamiento ...
    resultado = df['columna_critica']  # ❌ Crash al final

# BIEN - Validar al inicio (fail-fast)
def procesar_datos(df):
    if 'columna_critica' not in df.columns:
        raise ValueError("Falta columna_critica")
    # ... procesamiento ...
    resultado = df['columna_critica']  # ✓ Ya validado
```

## 🔧 PATRONES DE VALIDACIÓN

### Patrón 1: Validación de DataFrame

```python
def validar_dataframe(df: pd.DataFrame, columnas_requeridas: List[str]):
    """
    Valida que DataFrame tenga todas las columnas requeridas.

    Raises:
        ValueError: Si faltan columnas
    """
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]

    if columnas_faltantes:
        raise ValueError(
            f"DataFrame falta columnas requeridas: {columnas_faltantes}\n"
            f"Columnas presentes: {list(df.columns)}"
        )
```

### Patrón 2: Validación de diccionario de configuración

```python
def validar_config(config: Dict, claves_requeridas: List[str]):
    """
    Valida que config tenga todas las claves requeridas.

    Raises:
        ValueError: Si faltan claves
    """
    claves_faltantes = [clave for clave in claves_requeridas if clave not in config]

    if claves_faltantes:
        raise ValueError(
            f"Configuración falta claves requeridas: {claves_faltantes}\n"
            f"Claves presentes: {list(config.keys())}"
        )
```

### Patrón 3: Acceso seguro con `.get()`

```python
# Para diccionarios con claves opcionales
valor = diccionario.get('clave_opcional', valor_por_defecto)

# Para DataFrames con columnas opcionales
if 'columna_opcional' in df.columns:
    valor = df['columna_opcional']
else:
    valor = None  # O valor por defecto
```

### Patrón 4: Validación con rangos

```python
def validar_parametros(config: Dict):
    """Valida que parámetros estén en rangos válidos."""

    # Validar existencia
    if 'risk_per_trade' not in config:
        raise ValueError("Falta 'risk_per_trade'")

    # Validar rango
    if not (0 < config['risk_per_trade'] <= 0.1):
        raise ValueError(
            f"risk_per_trade debe estar entre 0 y 0.1, "
            f"recibido: {config['risk_per_trade']}"
        )
```

## 📊 CHECKLIST DE REVISIÓN

Al revisar código con diccionarios:

- [ ] ¿Se validan columnas de DataFrame antes de acceder?
- [ ] ¿Se validan claves de config en `__init__`?
- [ ] ¿Se usa `.get()` para claves opcionales?
- [ ] ¿Los errores son descriptivos y útiles?
- [ ] ¿La validación es temprana (fail-fast)?
- [ ] ¿Hay valores por defecto razonables?
- [ ] ¿Se validan rangos de valores numéricos?

## 🎯 RESUMEN

**PRINCIPIO FUNDAMENTAL**: **Validar temprano, fallar rápido** (fail-fast)

1. ✅ **Validar en `__init__`** o al cargar datos
2. ✅ **Usar `.get()` para opcionales**
3. ✅ **Validar columnas antes de iterar**
4. ✅ **Errores descriptivos** con claves/columnas faltantes
5. ❌ **NO silenciar errores** con try-except vacío
6. ❌ **NO validar tarde** (después de mucho procesamiento)

---

**Archivos modificados con validación**:
- ✅ `backtest/motor_backtest_completo.py`
- ✅ `estrategia_emergente/formulacion_reglas.py`

**Archivos pendientes de mejorar** (ratio seguro < 5%):
- ⚠️ `consenso_metodos/proceso_consenso.py` (0.0%)
- ⚠️ `validacion_rigurosa/walk_forward_validation.py` (2.2%)
- ⚠️ `validacion_rigurosa/analisis_robustez.py` (0.0%)

**Nota**: Los archivos pendientes funcionan correctamente porque crean sus propios diccionarios internamente, pero se recomienda agregar validación para mayor robustez.
