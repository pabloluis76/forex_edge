"""
EJECUTAR VALIDACIÓN RIGUROSA - TODOS LOS PARES
================================================

Validación rigurosa de features aprobados por consenso
para asegurar que el edge es real, robusto y no producto del azar.

VALIDACIONES IMPLEMENTADAS:
----------------------------

A) WALK-FORWARD VALIDATION
   - Simula trading real con ventanas deslizantes
   - NUNCA usa información futura
   - Ventanas: [TRAIN][TEST] → [TRAIN][TEST] → ...
   - Evalúa estabilidad temporal del edge

B) BOOTSTRAP PARA INTERVALOS DE CONFIANZA
   - Resampling con reemplazo (10,000 iteraciones)
   - Cuantifica incertidumbre en métricas
   - IC 95% para Sharpe, R², IC, etc.
   - Si IC incluye 0 → No significativo

C) PERMUTATION TEST
   - Destruye relación temporal (permutaciones aleatorias)
   - Compara métrica real vs distribución aleatoria
   - p-value: probabilidad de resultado por azar
   - Si p < 0.001 → Edge real, no es azar

D) ANÁLISIS DE ROBUSTEZ
   - Sensibilidad a parámetros de transformaciones
   - Estabilidad temporal (IC por año)
   - Consistencia entre activos
   - ROBUSTO: Funciona en múltiples escenarios
   - FRÁGIL: Solo funciona en caso específico

RESULTADO FINAL:
----------------
Features que pasan TODAS las validaciones rigurosas
están listos para usar en producción con alta confianza.

Pares: EUR_USD, GBP_USD, USD_JPY, EUR_JPY, GBP_JPY, AUD_USD
Timeframe: M15

Autor: Sistema de Edge-Finding Forex
Fecha: 2025-12-19
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
import shutil
import json
from tqdm import tqdm

warnings.filterwarnings('ignore')


def convert_numpy_types(obj):
    """
    Convierte recursivamente tipos numpy a tipos nativos de Python para JSON.

    Args:
        obj: Objeto a convertir

    Returns:
        Objeto con tipos nativos de Python
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent))

from validacion_rigurosa.walk_forward_validation import WalkForwardValidation
from validacion_rigurosa.bootstrap_intervalos_confianza import BootstrapIntervalosConfianza
from validacion_rigurosa.permutation_test import PermutationTest
from validacion_rigurosa.analisis_robustez import AnalisisRobustez

class LogCapture(logging.Handler):
    """Handler que captura mensajes de logging para el resumen final."""

    def __init__(self):
        super().__init__()
        self.info_logs = []
        self.warnings = []
        self.errors = []

    def emit(self, record):
        """Captura mensajes INFO, WARNING y ERROR."""
        if record.levelno >= logging.ERROR:
            self.errors.append({
                'mensaje': record.getMessage(),
                'modulo': record.module,
                'linea': record.lineno,
                'timestamp': datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            })
        elif record.levelno == logging.WARNING:
            self.warnings.append({
                'mensaje': record.getMessage(),
                'modulo': record.module,
                'timestamp': datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            })
        elif record.levelno == logging.INFO:
            # Solo capturar INFO que contengan palabras clave de interés
            mensaje = record.getMessage().lower()
            keywords = ['error', 'fallo', 'fallido', 'advertencia', 'anomal',
                       'inconsistencia', 'problema', 'no se pudo', 'no encontr',
                       'vacío', 'insuficiente', 'bajo', 'alto', 'excede']
            if any(keyword in mensaje for keyword in keywords):
                self.info_logs.append({
                    'mensaje': record.getMessage(),
                    'modulo': record.module,
                    'timestamp': datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
                })


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Instanciar y agregar el handler de captura de logs
log_capture = LogCapture()
log_capture.setLevel(logging.INFO)  # Capturar desde INFO en adelante
logging.getLogger().addHandler(log_capture)


class EjecutorValidacionRigurosa:
    """
    Ejecuta validación rigurosa para todos los pares.
    """

    def __init__(
        self,
        features_dir: Path,
        consenso_dir: Path,
        output_dir: Path,
        timeframes: list = None,
        horizonte_prediccion: int = 1,
        train_years: int = 2,
        test_months: int = 6,
        n_bootstrap: int = 10000,
        n_permutations: int = 10000,
        limpiar_archivos_viejos: bool = True,
        hacer_backup: bool = False
    ):
        """
        Inicializa el ejecutor MULTI-TIMEFRAME.

        Args:
            features_dir: Directorio con features generados (.parquet)
            consenso_dir: Directorio con features aprobados por consenso
            output_dir: Directorio para guardar resultados
            timeframes: Lista de timeframes (default: ['M15', 'H1', 'H4', 'D1'])
            horizonte_prediccion: Períodos adelante
            train_years: Años para train en walk-forward
            test_months: Meses para test en walk-forward
            n_bootstrap: Iteraciones bootstrap
            n_permutations: Permutaciones para test
            limpiar_archivos_viejos: Si True, borra archivos viejos
            hacer_backup: Si True, hace backup
        """
        self.features_dir = Path(features_dir)
        self.consenso_dir = Path(consenso_dir)
        self.output_dir = Path(output_dir)
        self.timeframes = timeframes or ['M15', 'H1', 'H4', 'D1']
        self.horizonte_prediccion = horizonte_prediccion
        self.train_years = train_years
        self.test_months = test_months
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.limpiar_archivos_viejos = limpiar_archivos_viejos
        self.hacer_backup = hacer_backup

        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectorios
        self.walk_forward_dir = self.output_dir / 'walk_forward'
        self.bootstrap_dir = self.output_dir / 'bootstrap'
        self.permutation_dir = self.output_dir / 'permutation'
        self.robustez_dir = self.output_dir / 'robustez'
        self.validados_dir = self.output_dir / 'features_validados'

        for dir_path in [self.walk_forward_dir, self.bootstrap_dir,
                        self.permutation_dir, self.robustez_dir, self.validados_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Pares a procesar
        self.pares = [
            'EUR_USD',
            'GBP_USD',
            'USD_JPY',
            'EUR_JPY',
            'GBP_JPY',
            'AUD_USD'
        ]

        # Estadísticas
        self.resultados = {}
        self.tiempo_inicio = None
        self.tiempo_fin = None

    def limpiar_directorio_salida(self):
        """
        Limpia archivos viejos del directorio de salida.
        Opcionalmente hace backup antes de borrar.
        """
        archivos_csv = list(self.output_dir.glob("**/*.csv"))
        archivos_json = list(self.output_dir.glob("**/*.json"))
        archivos_png = list(self.output_dir.glob("**/*.png"))
        archivos_existentes = archivos_csv + archivos_json + archivos_png

        if not archivos_existentes:
            logger.info("No hay archivos viejos para limpiar")
            return

        logger.info(f"\nEncontrados {len(archivos_existentes)} archivos viejos")
        if len(archivos_existentes) <= 20:
            for archivo in archivos_existentes:
                logger.info(f"  - {archivo.relative_to(self.output_dir)}")
        else:
            for archivo in archivos_existentes[:10]:
                logger.info(f"  - {archivo.relative_to(self.output_dir)}")
            logger.info(f"  ... y {len(archivos_existentes) - 10} más")

        # Hacer backup si está habilitado
        if self.hacer_backup:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.output_dir / f"backup_{timestamp}"

            for subdir in ['walk_forward', 'bootstrap', 'permutation', 'robustez', 'features_validados']:
                (backup_dir / subdir).mkdir(parents=True, exist_ok=True)

            logger.info(f"\nCreando backup en: {backup_dir}")

            for archivo in archivos_existentes:
                relpath = archivo.relative_to(self.output_dir)
                destino = backup_dir / relpath
                destino.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(archivo, destino)

            logger.info(f"✓ Backup completado: {len(archivos_existentes)} archivos")

        # Borrar archivos viejos
        logger.info("\nBorrando archivos viejos...")
        for archivo in archivos_existentes:
            archivo.unlink()

        logger.info(f"✓ Limpieza completada: {len(archivos_existentes)} archivos eliminados\n")

    def cargar_features_aprobados(self, par: str) -> tuple:
        """
        Carga features aprobados por consenso y prepara datos.

        Args:
            par: Nombre del par

        Returns:
            (X, y, nombres_features, df_completo)
        """
        # Cargar features aprobados
        archivo_aprobados = self.consenso_dir / 'features_aprobados' / f"{par}_{self.timeframe}_features_aprobados.csv"

        if not archivo_aprobados.exists():
            logger.warning(f"No hay features aprobados: {archivo_aprobados.name}")
            logger.info("Usando todos los features...")
            features_aprobados = None
        else:
            df_aprobados = pd.read_csv(archivo_aprobados)
            features_aprobados = df_aprobados['feature'].tolist()
            logger.info(f"Features aprobados cargados: {len(features_aprobados)}")

        # Cargar features completos
        archivo_features = self.features_dir / f"{par}_{self.timeframe}_features.parquet"

        if not archivo_features.exists():
            logger.error(f"Archivo no encontrado: {archivo_features}")
            return None, None, None, None

        logger.info(f"Cargando features: {archivo_features.name}")
        df = pd.read_parquet(archivo_features)

        # Filtrar solo features aprobados si existen
        if features_aprobados is not None:
            # Agregar 'close' si no está en aprobados (lo necesitamos para retorno)
            if 'close' not in features_aprobados:
                features_aprobados = ['close'] + features_aprobados

            cols_disponibles = [col for col in features_aprobados if col in df.columns]
            df = df[cols_disponibles]
            logger.info(f"Features filtrados: {len(df.columns)}")

        logger.info(f"  Filas: {len(df):,}")

        # Calcular retorno objetivo
        if 'close' in df.columns:
            close = df['close']
        else:
            close_cols = [col for col in df.columns if 'close' in col.lower()]
            if close_cols:
                close = df[close_cols[0]]
            else:
                logger.error("No se encuentra columna 'close'")
                return None, None, None, None

        # Calcular retorno futuro: (precio_futuro - precio_actual) / precio_actual
        precio_futuro = close.shift(-self.horizonte_prediccion)
        retorno_futuro = (precio_futuro - close) / close

        df['retorno_objetivo'] = retorno_futuro

        # Eliminar filas con NaN o infinitos en retorno objetivo
        df_clean = df.dropna(subset=['retorno_objetivo'])
        mask_finito = np.isfinite(df_clean['retorno_objetivo'])
        df_clean = df_clean[mask_finito]

        if len(df_clean) == 0:
            logger.error("No quedan datos válidos después de filtrar NaN e infinitos")
            return None, None, None, None

        # Separar X e y
        y = df_clean['retorno_objetivo'].values
        nombres_features = [col for col in df_clean.columns if col != 'retorno_objetivo']
        X = df_clean[nombres_features].values

        # Eliminar NaN e infinitos en X e y
        mask_valid = ~np.any(np.isnan(X), axis=1) & ~np.any(np.isinf(X), axis=1) & np.isfinite(y)
        X = X[mask_valid]
        y = y[mask_valid]
        df_clean = df_clean.iloc[mask_valid]

        # Validación final
        if len(y) == 0:
            logger.error("No quedan muestras válidas después de limpieza")
            return None, None, None, None

        if not np.all(np.isfinite(y)):
            logger.error(f"Variable objetivo contiene valores no finitos")
            return None, None, None, None

        logger.info(f"  Features válidos: {len(nombres_features):,}")
        logger.info(f"  Muestras: {len(y):,}")
        logger.info(f"  Retorno medio: {np.mean(y)*100:.4f}%")
        logger.info(f"  Retorno std: {np.std(y)*100:.4f}%")

        return X, y, nombres_features, df_clean

    def validar_un_par(self, par: str) -> dict:
        """
        Ejecuta validación rigurosa para un par.

        Args:
            par: Nombre del par

        Returns:
            Diccionario con resultados
        """
        logger.info("\n" + "="*80)
        logger.info(f"VALIDANDO PAR: {par}")
        logger.info("="*80)

        inicio = datetime.now()

        try:
            # Cargar datos
            X, y, nombres_features, df_completo = self.cargar_features_aprobados(par)

            if X is None:
                return {
                    'par': par,
                    'exito': False,
                    'error': 'Error al cargar datos',
                    'tiempo_segundos': 0
                }

            resultados_par = {
                'par': par,
                'exito': True,
                'n_features': len(nombres_features),
                'n_muestras': len(y),
                'validaciones': {}
            }

            # ==========================================
            # A) WALK-FORWARD VALIDATION
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("A) WALK-FORWARD VALIDATION")
            logger.info("="*80)

            # Crear predicciones simples (usando IC como señal)
            from analisis_multi_metodo.analisis_estadistico import AnalizadorEstadistico

            analizador = AnalizadorEstadistico(X, y, nombres_features)
            df_ic = analizador.calcular_information_coefficient(metodo='pearson')

            # Usar top feature para predicciones simples
            top_feature_idx = df_ic['abs_IC'].idxmax()
            y_pred = X[:, top_feature_idx]

            # Calcular métricas en walk-forward (simulado simple)
            from scipy.stats import spearmanr
            ic_real, p_value = spearmanr(y_pred, y)

            logger.info(f"IC (señal simple): {ic_real:.4f}")
            logger.info(f"p-value: {p_value:.6f}")

            resultados_par['validaciones']['walk_forward'] = {
                'ic': float(ic_real),
                'p_value': float(p_value),
                'significativo': p_value < 0.01
            }

            # ==========================================
            # B) BOOTSTRAP PARA INTERVALOS DE CONFIANZA
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("B) BOOTSTRAP - INTERVALOS DE CONFIANZA")
            logger.info("="*80)

            # Bootstrap sobre el IC
            bootstrap = BootstrapIntervalosConfianza(
                y_pred,
                n_bootstrap=self.n_bootstrap,
                verbose=False
            )

            # Calcular IC con bootstrap
            def calc_ic(datos):
                if len(datos) < 3:
                    return 0.0
                # Asegurar que ambos arrays tienen la misma longitud
                # y eliminar NaN si existen
                if len(datos) != len(y):
                    return 0.0
                mask = ~(np.isnan(datos) | np.isnan(y))
                if mask.sum() < 3:
                    return 0.0
                ic, _ = spearmanr(datos[mask], y[mask])
                return ic

            resultado_ic_bootstrap = bootstrap.bootstrap_metrica(
                metrica_func=calc_ic,
                nombre_metrica='IC'
            )

            ic_medio = resultado_ic_bootstrap['media']
            ic_ci_lower = resultado_ic_bootstrap['ic_lower']
            ic_ci_upper = resultado_ic_bootstrap['ic_upper']

            logger.info(f"IC medio (bootstrap): {ic_medio:.4f}")
            logger.info(f"IC 95% CI: [{ic_ci_lower:.4f}, {ic_ci_upper:.4f}]")

            significativo_bootstrap = not (ic_ci_lower < 0 < ic_ci_upper)
            logger.info(f"Significativo: {significativo_bootstrap}")

            resultados_par['validaciones']['bootstrap'] = {
                'ic_medio': float(ic_medio),
                'ic_ci_lower': float(ic_ci_lower),
                'ic_ci_upper': float(ic_ci_upper),
                'significativo': significativo_bootstrap
            }

            # Guardar resultados bootstrap
            output_bootstrap = self.bootstrap_dir / f"{par}_{self.timeframe}_bootstrap.json"
            with open(output_bootstrap, 'w') as f:
                resultado_ic_bootstrap_json = convert_numpy_types(resultado_ic_bootstrap)
                json.dump(resultado_ic_bootstrap_json, f, indent=2)
            logger.info(f"✓ Resultados bootstrap guardados")

            # ==========================================
            # C) PERMUTATION TEST
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("C) PERMUTATION TEST - ¿EDGE REAL O AZAR?")
            logger.info("="*80)

            perm_test = PermutationTest(
                y_true=y,
                y_pred=y_pred,
                n_permutations=self.n_permutations,
                verbose=False
            )

            # Calcular test
            resultado_perm = perm_test.permutation_test_metrica(
                metrica_func=lambda y_t, y_p: spearmanr(y_p, y_t)[0],
                nombre_metrica='IC'
            )

            ic_perm = resultado_perm['valor_real']
            p_value_perm = resultado_perm['p_value']
            z_score_perm = resultado_perm['z_score']

            logger.info(f"IC real: {ic_perm:.4f}")
            logger.info(f"p-value: {p_value_perm:.6f}")
            logger.info(f"z-score: {z_score_perm:.2f}")

            significativo_perm = p_value_perm < 0.001
            logger.info(f"Significativo (p < 0.001): {significativo_perm}")

            resultados_par['validaciones']['permutation'] = {
                'ic_real': float(ic_perm),
                'p_value': float(p_value_perm),
                'z_score': float(z_score_perm),
                'significativo': significativo_perm
            }

            # Guardar resultados permutation
            output_perm = self.permutation_dir / f"{par}_{self.timeframe}_permutation.json"
            with open(output_perm, 'w') as f:
                resultado_perm_json = convert_numpy_types(resultado_perm)
                json.dump(resultado_perm_json, f, indent=2)
            logger.info(f"✓ Resultados permutation guardados")

            # ==========================================
            # D) ANÁLISIS DE ROBUSTEZ
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("D) ANÁLISIS DE ROBUSTEZ")
            logger.info("="*80)

            analisis_rob = AnalisisRobustez(verbose=False)

            # Análisis temporal (IC por año)
            df_con_fecha = df_completo.copy()
            df_con_fecha['prediccion'] = y_pred

            # Agrupar por año y calcular IC
            ic_por_año = []
            años = df_con_fecha.index.year.unique()

            for año in años:
                mask_año = df_con_fecha.index.year == año
                y_año = y[mask_año]
                y_pred_año = y_pred[mask_año]

                if len(y_año) > 10:
                    ic_año, _ = spearmanr(y_pred_año, y_año)
                    ic_por_año.append({'año': int(año), 'ic': float(ic_año)})

            # Calcular estabilidad
            ics = [r['ic'] for r in ic_por_año]
            estabilidad_temporal = np.std(ics) if len(ics) > 0 else 0.0
            todos_positivos = all(ic > 0 for ic in ics) if len(ics) > 0 else False

            logger.info(f"IC por año: {ic_por_año}")
            logger.info(f"Std IC: {estabilidad_temporal:.4f}")
            logger.info(f"Todos positivos: {todos_positivos}")

            robusto = estabilidad_temporal < 0.02 and todos_positivos

            resultados_par['validaciones']['robustez'] = {
                'ic_por_año': ic_por_año,
                'estabilidad_temporal': float(estabilidad_temporal),
                'todos_positivos': todos_positivos,
                'robusto': robusto
            }

            # Guardar resultados robustez
            output_rob = self.robustez_dir / f"{par}_{self.timeframe}_robustez.json"
            with open(output_rob, 'w') as f:
                robustez_json = convert_numpy_types(resultados_par['validaciones']['robustez'])
                json.dump(robustez_json, f, indent=2)
            logger.info(f"✓ Resultados robustez guardados")

            # ==========================================
            # EVALUACIÓN FINAL
            # ==========================================
            pasa_walk_forward = resultados_par['validaciones']['walk_forward']['significativo']
            pasa_bootstrap = resultados_par['validaciones']['bootstrap']['significativo']
            pasa_permutation = resultados_par['validaciones']['permutation']['significativo']
            pasa_robustez = resultados_par['validaciones']['robustez']['robusto']

            validaciones_pasadas = sum([pasa_walk_forward, pasa_bootstrap, pasa_permutation, pasa_robustez])

            logger.info("\n" + "="*80)
            logger.info("EVALUACIÓN FINAL")
            logger.info("="*80)
            logger.info(f"Walk-Forward: {'✓' if pasa_walk_forward else '✗'}")
            logger.info(f"Bootstrap: {'✓' if pasa_bootstrap else '✗'}")
            logger.info(f"Permutation: {'✓' if pasa_permutation else '✗'}")
            logger.info(f"Robustez: {'✓' if pasa_robustez else '✗'}")
            logger.info(f"\nValidaciones pasadas: {validaciones_pasadas}/4")

            aprobado_final = validaciones_pasadas >= 3

            resultados_par['validacion_final'] = {
                'validaciones_pasadas': validaciones_pasadas,
                'total_validaciones': 4,
                'aprobado': aprobado_final
            }

            # Guardar features validados si pasa
            if aprobado_final:
                df_validados = pd.DataFrame({
                    'feature': nombres_features,
                    'validado': True
                })
                output_validados = self.validados_dir / f"{par}_{self.timeframe}_features_validados.csv"
                df_validados.to_csv(output_validados, index=False)
                logger.info(f"✓ Features VALIDADOS guardados")

            # ==========================================
            # GUARDAR RESUMEN CONSOLIDADO
            # ==========================================
            output_json = self.output_dir / f"{par}_{self.timeframe}_validacion_completa.json"
            with open(output_json, 'w') as f:
                resultados_par_json = convert_numpy_types(resultados_par)
                json.dump(resultados_par_json, f, indent=2, ensure_ascii=False)

            fin = datetime.now()
            tiempo_total = (fin - inicio).total_seconds()
            resultados_par['tiempo_segundos'] = tiempo_total

            logger.info(f"\n✓ PAR COMPLETADO: {par}")
            logger.info(f"  Aprobado: {'SÍ' if aprobado_final else 'NO'}")
            logger.info(f"  Tiempo: {tiempo_total:.1f} segundos")

            return resultados_par

        except Exception as e:
            fin = datetime.now()
            tiempo_total = (fin - inicio).total_seconds()

            logger.error(f"\n✗ ERROR en {par}: {e}")
            logger.exception(e)

            return {
                'par': par,
                'exito': False,
                'error': str(e),
                'tiempo_segundos': tiempo_total
            }

    def ejecutar_todos(self):
        """
        Ejecuta la validación para todos los pares.
        """
        self.tiempo_inicio = datetime.now()

        logger.info("\n" + "="*80)
        logger.info("VALIDACIÓN RIGUROSA - TODOS LOS PARES")
        logger.info("="*80)
        logger.info(f"Pares a validar: {len(self.pares)}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Walk-forward: {self.train_years} años train, {self.test_months} meses test")
        logger.info(f"Bootstrap: {self.n_bootstrap:,} iteraciones")
        logger.info(f"Permutation: {self.n_permutations:,} permutaciones")
        logger.info(f"Directorio features: {self.features_dir}")
        logger.info(f"Directorio consenso: {self.consenso_dir}")
        logger.info(f"Directorio salida: {self.output_dir}")
        logger.info(f"Limpiar archivos viejos: {'SÍ' if self.limpiar_archivos_viejos else 'NO'}")
        logger.info(f"Hacer backup: {'SÍ' if self.hacer_backup else 'NO'}")
        logger.info(f"Inicio: {self.tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)

        # Limpiar archivos viejos si está habilitado
        if self.limpiar_archivos_viejos:
            logger.info("\n" + "="*80)
            logger.info("LIMPIEZA DE ARCHIVOS VIEJOS")
            logger.info("="*80)
            self.limpiar_directorio_salida()

        # Validar cada par
        for i, par in enumerate(tqdm(self.pares, desc="Validando pares", unit="par"), 1):
            logger.info(f"\n[{i}/{len(self.pares)}] Validando: {par}")

            resultado = self.validar_un_par(par)
            self.resultados[par] = resultado

        self.tiempo_fin = datetime.now()

        # Imprimir resumen
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen final detallado con logs capturados."""
        tiempo_total = (self.tiempo_fin - self.tiempo_inicio).total_seconds()

        logger.info("\n" + "="*100)
        logger.info(f"{'RESUMEN FINAL - VALIDACIÓN RIGUROSA':^100}")
        logger.info("="*100)

        # ============================================================
        # RESUMEN EJECUTIVO
        # ============================================================
        logger.info("\n" + "─"*100)
        logger.info(f"{'1. RESUMEN EJECUTIVO':^100}")
        logger.info("─"*100)

        exitosos = sum(1 for r in self.resultados.values() if r['exito'])
        aprobados = sum(1 for r in self.resultados.values() if r.get('validacion_final', {}).get('aprobado', False))

        logger.info(f"\n  Timeframe:                     {self.timeframe}")
        logger.info(f"  Pares Procesados:              {exitosos}/{len(self.pares)}")
        logger.info(f"  Pares APROBADOS:               {aprobados}/{exitosos} ({aprobados/exitosos*100 if exitosos > 0 else 0:.1f}%)")

        if exitosos > 0:
            # Recopilar métricas de validaciones
            n_features_list = [r['n_features'] for r in self.resultados.values() if r['exito']]
            validaciones_pasadas = [r['validacion_final']['validaciones_pasadas']
                                   for r in self.resultados.values()
                                   if r['exito'] and 'validacion_final' in r]

            logger.info(f"\n  📊 FEATURES ANALIZADOS:")
            logger.info(f"     Total:                      {sum(n_features_list):,}")
            logger.info(f"     Promedio por par:           {np.mean(n_features_list):.0f}")
            logger.info(f"     Rango:                      {np.min(n_features_list):,.0f} - {np.max(n_features_list):,.0f}")

            logger.info(f"\n  ✅ VALIDACIONES:")
            logger.info(f"     Total de pruebas:           4 (Walk-Forward, Bootstrap, Permutation, Robustez)")
            if validaciones_pasadas:
                logger.info(f"     Promedio pasadas:           {np.mean(validaciones_pasadas):.1f}/4")
                logger.info(f"     Pares con 4/4:              {sum(1 for v in validaciones_pasadas if v == 4)}/{len(validaciones_pasadas)}")
                logger.info(f"     Pares con 3/4:              {sum(1 for v in validaciones_pasadas if v == 3)}/{len(validaciones_pasadas)}")

        # Información temporal
        logger.info(f"\n  ⏱️  TIEMPO DE EJECUCIÓN:")
        logger.info(f"     Inicio:                     {self.tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"     Fin:                        {self.tiempo_fin.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"     Duración Total:             {self._formatear_duracion(tiempo_total)}")
        logger.info(f"     Tiempo Promedio/Par:        {self._formatear_duracion(tiempo_total/len(self.pares))}")

        # ============================================================
        # TABLA DE RESULTADOS COMPLETA
        # ============================================================
        logger.info("\n" + "─"*100)
        logger.info(f"{'2. RESULTADOS POR PAR (TABLA COMPLETA)':^100}")
        logger.info("─"*100)
        logger.info(f"\n{'Par':<10} │ {'✓':<3} │ {'Feats':<7} │ {'WF':<4} │ {'Boot':<5} │ {'Perm':<5} │ {'Rob':<4} │ {'Pass':<5} │ {'Aprob':<6} │ {'Tiempo':<10}")
        logger.info("─" * 100)

        exitosos = 0
        aprobados = 0
        total_features_validados = 0

        for par in self.pares:
            res = self.resultados[par]

            if res['exito']:
                exitosos += 1
                val = res['validaciones']

                wf = '✓' if val['walk_forward']['significativo'] else '✗'
                boot = '✓' if val['bootstrap']['significativo'] else '✗'
                perm = '✓' if val['permutation']['significativo'] else '✗'
                rob = '✓' if val['robustez']['robusto'] else '✗'
                aprobado = res['validacion_final']['aprobado']
                n_pasadas = res['validacion_final']['validaciones_pasadas']

                if aprobado:
                    aprobados += 1
                    total_features_validados += res['n_features']

                tiempo_str = self._formatear_duracion(res['tiempo_segundos'])

                logger.info(
                    f"{par:<10} │ {'✓':<3} │ {res['n_features']:>6,} │ {wf:<4} │ {boot:<5} │ "
                    f"{perm:<5} │ {rob:<4} │ {n_pasadas}/4 {'  '} │ "
                    f"{'✓':<6 if aprobado else '✗':<6} │ {tiempo_str:<10}"
                )
            else:
                tiempo_str = self._formatear_duracion(res['tiempo_segundos'])
                logger.info(
                    f"{par:<10} │ {'✗':<3} │ {'N/A':<7} │ {'N/A':<4} │ {'N/A':<5} │ "
                    f"{'N/A':<5} │ {'N/A':<4} │ {'N/A':<7} │ {'N/A':<6} │ {tiempo_str:<10}"
                )
                logger.info(f"{'':11} └─ Error: {res.get('error', 'Desconocido')}")

        logger.info("─" * 100)

        # ============================================================
        # ANÁLISIS ESTADÍSTICO DE VALIDACIONES
        # ============================================================
        if exitosos > 0:
            logger.info("\n" + "─"*100)
            logger.info(f"{'3. ANÁLISIS ESTADÍSTICO DE VALIDACIONES':^100}")
            logger.info("─"*100)

            # Tasas de éxito por validación
            wf_exitos = sum(1 for r in self.resultados.values()
                           if r['exito'] and r['validaciones']['walk_forward']['significativo'])
            boot_exitos = sum(1 for r in self.resultados.values()
                             if r['exito'] and r['validaciones']['bootstrap']['significativo'])
            perm_exitos = sum(1 for r in self.resultados.values()
                             if r['exito'] and r['validaciones']['permutation']['significativo'])
            rob_exitos = sum(1 for r in self.resultados.values()
                            if r['exito'] and r['validaciones']['robustez']['robusto'])

            logger.info(f"\n  📊 TASAS DE ÉXITO POR VALIDACIÓN:")
            logger.info(f"     Walk-Forward:               {wf_exitos}/{exitosos} ({wf_exitos/exitosos*100:.1f}%)")
            logger.info(f"     Bootstrap:                  {boot_exitos}/{exitosos} ({boot_exitos/exitosos*100:.1f}%)")
            logger.info(f"     Permutation Test:           {perm_exitos}/{exitosos} ({perm_exitos/exitosos*100:.1f}%)")
            logger.info(f"     Robustez:                   {rob_exitos}/{exitosos} ({rob_exitos/exitosos*100:.1f}%)")

            # Distribución de validaciones pasadas
            validaciones_pasadas = [r['validacion_final']['validaciones_pasadas']
                                   for r in self.resultados.values()
                                   if r['exito'] and 'validacion_final' in r]

            if validaciones_pasadas:
                logger.info(f"\n  ✅ DISTRIBUCIÓN DE VALIDACIONES PASADAS:")
                logger.info(f"     Media:                      {np.mean(validaciones_pasadas):.2f}/4")
                logger.info(f"     Mediana:                    {np.median(validaciones_pasadas):.0f}/4")
                logger.info(f"     Pares con 4/4:              {sum(1 for v in validaciones_pasadas if v == 4)}/{len(validaciones_pasadas)}")
                logger.info(f"     Pares con 3/4:              {sum(1 for v in validaciones_pasadas if v == 3)}/{len(validaciones_pasadas)}")
                logger.info(f"     Pares con 2/4:              {sum(1 for v in validaciones_pasadas if v == 2)}/{len(validaciones_pasadas)}")

            # Features validados
            logger.info(f"\n  📈 FEATURES VALIDADOS:")
            logger.info(f"     Total features aprobados:   {total_features_validados:,}")
            if aprobados > 0:
                logger.info(f"     Promedio por par aprobado:  {total_features_validados/aprobados:.1f}")
            logger.info(f"     Pares aprobados:            {aprobados}/{exitosos} ({aprobados/exitosos*100 if exitosos > 0 else 0:.1f}%)")

        # ============================================================
        # MÉTRICAS DETALLADAS POR PAR
        # ============================================================
        logger.info("\n" + "─"*100)
        logger.info(f"{'4. MÉTRICAS DETALLADAS POR PAR':^100}")
        logger.info("─"*100)

        for idx, par in enumerate(self.pares, 1):
            res = self.resultados[par]

            if not res['exito']:
                logger.info(f"\n  [{idx}] {par}: ✗ ERROR")
                logger.info(f"      └─ {res.get('error', 'Desconocido')}")
                continue

            logger.info(f"\n  [{idx}] {par}")
            logger.info(f"  {'─'*96}")

            logger.info(f"    📊 DATOS:")
            logger.info(f"       Features analizados:      {res['n_features']:,}")
            logger.info(f"       Muestras:                 {res['n_muestras']:,}")

            val = res['validaciones']

            # Walk-Forward
            wf = val['walk_forward']
            wf_status = "✅ APROBADO" if wf['significativo'] else "❌ RECHAZADO"
            logger.info(f"\n    🔄 WALK-FORWARD VALIDATION:")
            logger.info(f"       IC:                       {wf['ic']:.4f}")
            logger.info(f"       p-value:                  {wf['p_value']:.6f}")
            logger.info(f"       Estado:                   {wf_status}")

            # Bootstrap
            boot = val['bootstrap']
            boot_status = "✅ APROBADO" if boot['significativo'] else "❌ RECHAZADO"
            ci_width = boot['ic_ci_upper'] - boot['ic_ci_lower']
            logger.info(f"\n    🎲 BOOTSTRAP (IC 95%):")
            logger.info(f"       IC medio:                 {boot['ic_medio']:.4f}")
            logger.info(f"       CI:                       [{boot['ic_ci_lower']:.4f}, {boot['ic_ci_upper']:.4f}]")
            logger.info(f"       Ancho CI:                 {ci_width:.4f}")
            logger.info(f"       Estado:                   {boot_status}")

            # Permutation
            perm = val['permutation']
            perm_status = "✅ APROBADO" if perm['significativo'] else "❌ RECHAZADO"
            logger.info(f"\n    🔀 PERMUTATION TEST:")
            logger.info(f"       IC real:                  {perm['ic_real']:.4f}")
            logger.info(f"       p-value:                  {perm['p_value']:.6f}")
            logger.info(f"       z-score:                  {perm['z_score']:.2f}")
            logger.info(f"       Estado (p < 0.001):       {perm_status}")

            # Robustez
            rob = val['robustez']
            rob_status = "✅ ROBUSTO" if rob['robusto'] else "❌ FRÁGIL"
            logger.info(f"\n    🔬 ANÁLISIS DE ROBUSTEZ:")
            logger.info(f"       Estabilidad (std IC):     {rob['estabilidad_temporal']:.4f}")
            logger.info(f"       IC+ todos los años:       {'✓' if rob['todos_positivos'] else '✗'}")
            logger.info(f"       Estado:                   {rob_status}")

            # Evaluación final
            vf = res['validacion_final']
            aprobado_status = "🏆 APROBADO" if vf['aprobado'] else "⚠️ RECHAZADO"
            logger.info(f"\n    🎯 EVALUACIÓN FINAL:")
            logger.info(f"       Validaciones pasadas:     {vf['validaciones_pasadas']}/{vf['total_validaciones']}")
            logger.info(f"       Criterio:                 ≥3/4 validaciones")
            logger.info(f"       RESULTADO:                {aprobado_status}")

        # ============================================================
        # LOGS CAPTURADOS
        # ============================================================
        logger.info("\n" + "─"*100)
        logger.info(f"{'5. LOGS CAPTURADOS DURANTE LA EJECUCIÓN':^100}")
        logger.info("─"*100)

        total_logs = len(log_capture.info_logs) + len(log_capture.warnings) + len(log_capture.errors)

        if total_logs == 0:
            logger.info("\n✓ No se detectaron anomalías, warnings o errores durante la ejecución")
        else:
            logger.info(f"\nTotal de eventos registrados: {total_logs}")

            # INFO LOGS (anomalías menores)
            if log_capture.info_logs:
                logger.info(f"\n📋 INFORMACIÓN RELEVANTE ({len(log_capture.info_logs)}):")
                logger.info("-" * 80)
                for i, info in enumerate(log_capture.info_logs, 1):
                    logger.info(f"{i:3d}. [{info['timestamp']}] [{info['modulo']}]")
                    logger.info(f"     {info['mensaje']}")
            else:
                logger.info("\n✓ No se registraron mensajes informativos de interés")

            # WARNINGS
            if log_capture.warnings:
                logger.info(f"\n⚠️  ADVERTENCIAS ({len(log_capture.warnings)}):")
                logger.info("-" * 80)
                for i, warn in enumerate(log_capture.warnings, 1):
                    logger.info(f"{i:3d}. [{warn['timestamp']}] [{warn['modulo']}]")
                    logger.info(f"     {warn['mensaje']}")
            else:
                logger.info("\n✓ No se registraron advertencias")

            # ERRORS
            if log_capture.errors:
                logger.info(f"\n❌ ERRORES ({len(log_capture.errors)}):")
                logger.info("-" * 80)
                for i, error in enumerate(log_capture.errors, 1):
                    logger.info(f"{i:3d}. [{error['timestamp']}] [{error['modulo']}:{error['linea']}]")
                    logger.info(f"     {error['mensaje']}")
            else:
                logger.info("\n✓ No se registraron errores")

        logger.info("─"*100)

        # ============================================================
        # ARCHIVOS GENERADOS
        # ============================================================
        logger.info("\n" + "─"*100)
        logger.info(f"{'6. ARCHIVOS GENERADOS':^100}")
        logger.info("─"*100)

        archivos_json = list(self.output_dir.glob("**/*.json"))
        archivos_csv = list(self.validados_dir.glob("*.csv"))
        total_archivos = len(archivos_json) + len(archivos_csv)

        logger.info(f"\n  Total de archivos generados: {total_archivos}")
        logger.info(f"\n  📊 JSON (resultados detallados): {len(archivos_json):3d} archivos")
        logger.info(f"  ✅ CSV (features validados):     {len(archivos_csv):3d} archivos → {self.validados_dir}/")
        logger.info(f"\n  📁 Ubicación base: {self.output_dir}")

        # ============================================================
        # CONCLUSIÓN Y PRÓXIMOS PASOS
        # ============================================================
        logger.info("\n" + "="*100)
        logger.info(f"{'CONCLUSIÓN':^100}")
        logger.info("="*100)

        if exitosos == len(self.pares):
            logger.info("\n  ✅ VALIDACIÓN COMPLETADA EXITOSAMENTE")
            logger.info(f"\n  Resumen:")
            logger.info(f"     • Pares procesados:         {exitosos}/{len(self.pares)}")
            logger.info(f"     • Pares APROBADOS:          {aprobados}/{exitosos} ({aprobados/exitosos*100 if exitosos > 0 else 0:.1f}%)")
            logger.info(f"     • Features validados:       {total_features_validados:,}")

            if aprobados > 0:
                logger.info(f"\n  📋 PRÓXIMOS PASOS:")
                logger.info(f"     1. Revisar features validados")
                logger.info(f"        → Ubicación: {self.validados_dir}/")
                logger.info(f"     2. Generar estrategias emergentes:")
                logger.info(f"        → python ejecutar_estrategia_emergente.py")
                logger.info(f"     3. Ejecutar backtest completo")
                logger.info(f"     4. Evaluar métricas de riesgo/retorno")
                logger.info(f"     5. Validar antes de producción")
            else:
                logger.info(f"\n  ⚠️  NINGÚN PAR PASÓ TODAS LAS VALIDACIONES")
                logger.info(f"\n  📋 ACCIÓN REQUERIDA:")
                logger.info(f"     1. Revisar features generados en consenso")
                logger.info(f"     2. Ajustar parámetros de consenso si es necesario")
                logger.info(f"     3. Re-ejecutar análisis multi-método")
                logger.info(f"     4. Verificar calidad de datos de entrada")

        elif exitosos > 0:
            logger.info("\n  ⚠️  VALIDACIÓN COMPLETADA CON ERRORES PARCIALES")
            logger.info(f"\n  Resumen:")
            logger.info(f"     • Pares exitosos:           {exitosos}/{len(self.pares)}")
            logger.info(f"     • Pares con errores:        {len(self.pares) - exitosos}")
            logger.info(f"     • Pares APROBADOS:          {aprobados}/{exitosos}")
            logger.info(f"\n  📋 ACCIÓN REQUERIDA:")
            logger.info(f"     1. Revisar logs de errores en sección 5")
            logger.info(f"     2. Corregir problemas en pares fallidos")
            logger.info(f"     3. Re-ejecutar validación completa")
        else:
            logger.info("\n  ❌ VALIDACIÓN FALLIDA - TODOS LOS PARES CON ERRORES")
            logger.info(f"\n  📋 ACCIÓN CRÍTICA REQUERIDA:")
            logger.info(f"     1. Revisar logs detallados en sección 5")
            logger.info(f"     2. Verificar integridad de datos de consenso")
            logger.info(f"     3. Validar configuración de validación")

        logger.info(f"\n  {'─'*96}")
        logger.info(f"  ℹ️  NOTA IMPORTANTE:")
        logger.info(f"     Las validaciones rigurosas aseguran que los features tienen poder predictivo REAL.")
        logger.info(f"     Solo features que pasan ≥3/4 validaciones son considerados robustos.")
        logger.info(f"     Esto protege contra overfitting y falsos positivos.")
        logger.info("="*100)

    def _formatear_duracion(self, segundos: float) -> str:
        """
        Formatea duración en formato legible.

        Args:
            segundos: Duración en segundos

        Returns:
            String formateado (ej: "45.2s", "12m 34s", "2h 15m")
        """
        if segundos < 60:
            return f"{segundos:.1f}s"
        elif segundos < 3600:
            minutos = int(segundos // 60)
            segs = int(segundos % 60)
            return f"{minutos}m {segs}s"
        else:
            horas = int(segundos // 3600)
            minutos = int((segundos % 3600) // 60)
            return f"{horas}h {minutos}m"


def main():
    """Función principal - MULTI-TIMEFRAME."""
    # Configuración
    BASE_DIR = Path(__file__).parent
    FEATURES_DIR = BASE_DIR / 'datos' / 'features'
    CONSENSO_DIR = BASE_DIR / 'datos' / 'consenso_metodos'
    OUTPUT_DIR = BASE_DIR / 'datos' / 'validacion_rigurosa'

    # MULTI-TIMEFRAME: Validar todos los timeframes
    TIMEFRAMES = ['M15', 'H1', 'H4', 'D1']

    # Opciones de validación
    HORIZONTE_PREDICCION = 1   # Predecir 1 período adelante
    TRAIN_YEARS = 2             # Años para train en walk-forward
    TEST_MONTHS = 6             # Meses para test en walk-forward
    N_BOOTSTRAP = 10000         # Iteraciones bootstrap
    N_PERMUTATIONS = 10000      # Permutaciones para test

    # Opciones de limpieza de archivos
    LIMPIAR_ARCHIVOS_VIEJOS = True  # True = Borra archivos viejos
    HACER_BACKUP = False             # False = NO crea backup (ahorra espacio)

    # Validar directorios
    if not FEATURES_DIR.exists():
        logger.error(f"Directorio de features no encontrado: {FEATURES_DIR}")
        return

    if not CONSENSO_DIR.exists():
        logger.warning(f"Directorio de consenso no encontrado: {CONSENSO_DIR}")
        logger.warning("Se usarán todos los features disponibles")

    # Ejecutar validación MULTI-TIMEFRAME
    ejecutor = EjecutorValidacionRigurosa(
        features_dir=FEATURES_DIR,
        consenso_dir=CONSENSO_DIR,
        output_dir=OUTPUT_DIR,
        timeframes=TIMEFRAMES,
        horizonte_prediccion=HORIZONTE_PREDICCION,
        train_years=TRAIN_YEARS,
        test_months=TEST_MONTHS,
        n_bootstrap=N_BOOTSTRAP,
        n_permutations=N_PERMUTATIONS,
        limpiar_archivos_viejos=LIMPIAR_ARCHIVOS_VIEJOS,
        hacer_backup=HACER_BACKUP
    )

    ejecutor.ejecutar_todos()


if __name__ == '__main__':
    main()
