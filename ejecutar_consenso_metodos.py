"""
EJECUTAR CONSENSO DE MÉTODOS - TODOS LOS PARES
================================================

Ejecuta el sistema de consenso de métodos para identificar
transformaciones con evidencia convergente de múltiples
métodos de análisis independientes.

PROCESO DE CONSENSO EN 3 PASOS:
--------------------------------

PASO 1: GENERAR RANKINGS POR MÉTODO
   - IC (Information Coefficient): Top-N por |IC|
   - MI (Mutual Information): Top-N por MI
   - Lasso: Features con β ≠ 0
   - Random Forest: Top-N por feature importance
   - Gradient Boosting: Top-N por importance
   - XGBoost/LightGBM: Top-N por importance (opcional)

PASO 2: CALCULAR INTERSECCIONES
   - Consenso Fuerte: Features en ≥5 métodos (✓✓✓)
   - Consenso Medio: Features en 3-4 métodos (✓✓)
   - Sin Consenso: Features en ≤2 métodos (✗)

PASO 3: VERIFICACIÓN CRUZADA
   - Estabilidad temporal del IC
   - Concordancia entre métodos de ML
   - Filtrado final de features aprobados

RESULTADO FINAL:
----------------
Lista de transformaciones APROBADAS para usar en producción,
con evidencia convergente de múltiples métodos independientes.

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

from constants import EPSILON
from consenso_metodos.tabla_consenso import TablaConsenso
from consenso_metodos.proceso_consenso import ProcesoConsenso

# Configurar logging con captura de errores
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

# Agregar capturador de logs
log_capture = LogCapture()
log_capture.setLevel(logging.INFO)  # Capturar desde INFO en adelante
logging.getLogger().addHandler(log_capture)


class EjecutorConsensoMetodos:
    """
    Ejecuta el proceso de consenso de métodos para todos los pares.
    """

    def __init__(
        self,
        features_dir: Path,
        output_dir: Path,
        timeframes: list = None,
        horizonte_prediccion: int = 1,
        top_n_por_metodo: int = 100,
        limpiar_archivos_viejos: bool = True,
        hacer_backup: bool = False
    ):
        """
        Inicializa el ejecutor MULTI-TIMEFRAME.

        Args:
            features_dir: Directorio con features generados (.parquet)
            output_dir: Directorio para guardar resultados
            timeframes: Lista de timeframes (default: ['M15', 'H1', 'H4', 'D1'])
            horizonte_prediccion: Períodos adelante para calcular retorno
            top_n_por_metodo: Número de top features por método (default: 100)
            limpiar_archivos_viejos: Si True, borra archivos viejos antes de iniciar
            hacer_backup: Si True, hace backup antes de borrar
        """
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.timeframes = timeframes or ['M15', 'H1', 'H4', 'D']
        self.horizonte_prediccion = horizonte_prediccion
        self.top_n_por_metodo = top_n_por_metodo
        self.limpiar_archivos_viejos = limpiar_archivos_viejos
        self.hacer_backup = hacer_backup

        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectorios
        self.rankings_dir = self.output_dir / 'rankings'
        self.consenso_dir = self.output_dir / 'consenso'
        self.aprobados_dir = self.output_dir / 'features_aprobados'

        for dir_path in [self.rankings_dir, self.consenso_dir, self.aprobados_dir]:
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
        # Buscar archivos existentes
        archivos_csv = list(self.output_dir.glob("**/*.csv"))
        archivos_json = list(self.output_dir.glob("**/*.json"))
        archivos_existentes = archivos_csv + archivos_json

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

            # Crear subdirectorios en backup
            for subdir in ['rankings', 'consenso', 'features_aprobados']:
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

    def cargar_resultados_analisis_pregenerados(self, par: str) -> dict:
        """
        Carga resultados de análisis multi-método pre-generados desde CSVs.

        Evita recálculo de IC, MI, RF que ya fueron calculados en
        ejecutar_analisis_multimetodo.py.

        Args:
            par: Nombre del par

        Returns:
            dict con 'ic', 'mi', 'rf', 'dl', 'lasso' (o None si no existen)
        """
        analisis_dir = Path(__file__).parent / 'datos' / 'analisis_multimetodo'

        resultados = {
            'ic': None,
            'mi': None,
            'rf': None,
            'dl': None,
            'lasso': None,
            'gb': None
        }

        # Intentar cargar IC
        archivo_ic = analisis_dir / f"{par}_{self.timeframe}_analisis_IC.csv"
        if archivo_ic.exists():
            resultados['ic'] = pd.read_csv(archivo_ic)
            logger.info(f"  ✓ IC cargado desde CSV: {len(resultados['ic'])} features")

        # Intentar cargar MI
        archivo_mi = analisis_dir / f"{par}_{self.timeframe}_analisis_MI.csv"
        if archivo_mi.exists():
            resultados['mi'] = pd.read_csv(archivo_mi)
            logger.info(f"  ✓ MI cargado desde CSV: {len(resultados['mi'])} features")

        # Intentar cargar RF
        archivo_rf = analisis_dir / f"{par}_{self.timeframe}_analisis_RF_importance.csv"
        if archivo_rf.exists():
            resultados['rf'] = pd.read_csv(archivo_rf)
            logger.info(f"  ✓ RF cargado desde CSV: {len(resultados['rf'])} features")

        # Intentar cargar DL
        archivo_dl = analisis_dir / f"{par}_{self.timeframe}_analisis_DL.json"
        if archivo_dl.exists():
            with open(archivo_dl, 'r') as f:
                resultados['dl'] = json.load(f)
            logger.info(f"  ✓ DL cargado desde JSON")

        # Intentar cargar análisis completo para Lasso y GB
        archivo_completo = analisis_dir / f"{par}_{self.timeframe}_analisis_completo.json"
        if archivo_completo.exists():
            with open(archivo_completo, 'r') as f:
                analisis_completo = json.load(f)

            # Extraer Lasso
            if 'analisis' in analisis_completo and 'estadistico' in analisis_completo['analisis']:
                resultados['lasso'] = analisis_completo['analisis']['estadistico']
                logger.info(f"  ✓ Lasso cargado desde JSON")

            # Extraer Gradient Boosting (si existe)
            if 'analisis' in analisis_completo and 'machine_learning' in analisis_completo['analisis']:
                resultados['gb'] = analisis_completo['analisis']['machine_learning']
                logger.info(f"  ✓ Gradient Boosting cargado desde JSON")

        return resultados

    def cargar_features_y_preparar_datos(self, par: str) -> tuple:
        """
        Carga features y prepara X, y para análisis.

        Args:
            par: Nombre del par

        Returns:
            (X, y, nombres_features, df_completo)
        """
        archivo_features = self.features_dir / f"{par}_{self.timeframe}_features.parquet"

        if not archivo_features.exists():
            logger.error(f"Archivo no encontrado: {archivo_features}")
            return None, None, None, None

        logger.info(f"Cargando features: {archivo_features.name}")
        df = pd.read_parquet(archivo_features)

        logger.info(f"  Features: {len(df.columns):,}")
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

        # Filtrar features válidos
        valid_features = []
        valid_indices = []

        for i, nombre in enumerate(nombres_features):
            col = X[:, i]
            if not np.all(np.isnan(col)) and np.nanstd(col) > EPSILON:
                valid_features.append(nombre)
                valid_indices.append(i)

        X = X[:, valid_indices]
        nombres_features = valid_features

        # Eliminar filas con NaN o infinito
        mask_valid = ~np.any(np.isnan(X), axis=1) & ~np.any(np.isinf(X), axis=1) & np.isfinite(y)
        X = X[mask_valid]
        y = y[mask_valid]

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

    def generar_consenso_desde_resultados_pregenerados(
        self,
        resultados_analisis: dict,
        par: str,
        umbral_ic: float = 0.01,
        umbral_mi: float = 0.01,
        top_rf: int = 100,
        top_gb: int = 100
    ) -> pd.DataFrame:
        """
        Genera tabla de consenso usando resultados pre-calculados.

        Args:
            resultados_analisis: Dict con 'ic', 'mi', 'rf', 'dl', 'lasso', 'gb'
            par: Nombre del par
            umbral_ic: Umbral para IC
            umbral_mi: Umbral para MI
            top_rf: Top N para Random Forest
            top_gb: Top N para Gradient Boosting

        Returns:
            DataFrame con consenso por feature
        """
        logger.info("\nGenerando consenso desde resultados pre-calculados...")

        # Coleccionar todos los features únicos
        todos_features = set()

        if resultados_analisis['ic'] is not None:
            todos_features.update(resultados_analisis['ic']['Feature'].tolist())
        if resultados_analisis['mi'] is not None:
            todos_features.update(resultados_analisis['mi']['Feature'].tolist())
        if resultados_analisis['rf'] is not None:
            todos_features.update(resultados_analisis['rf']['Feature'].tolist())

        todos_features = sorted(list(todos_features))
        logger.info(f"  Features únicos encontrados: {len(todos_features)}")

        # Crear DataFrame de consenso
        df_consenso = pd.DataFrame({'feature': todos_features})

        # Votos por método
        df_consenso['voto_IC'] = False
        df_consenso['voto_MI'] = False
        df_consenso['voto_Lasso'] = False
        df_consenso['voto_RF'] = False
        df_consenso['voto_GB'] = False
        df_consenso['voto_DL_MLP'] = False

        # Evaluar IC
        if resultados_analisis['ic'] is not None:
            df_ic = resultados_analisis['ic']
            if 'abs_IC' in df_ic.columns:
                ic_col = 'abs_IC'
            elif 'IC' in df_ic.columns:
                ic_col = 'IC'
                df_ic['abs_IC'] = df_ic['IC'].abs()
                ic_col = 'abs_IC'
            else:
                logger.warning("Columna IC no encontrada en resultados")
                ic_col = None

            if ic_col:
                features_ic = df_ic[df_ic[ic_col] > umbral_ic]['Feature'].tolist()
                df_consenso.loc[df_consenso['feature'].isin(features_ic), 'voto_IC'] = True
                logger.info(f"  IC: {len(features_ic)} features aprueban (|IC| > {umbral_ic})")

        # Evaluar MI
        if resultados_analisis['mi'] is not None:
            df_mi = resultados_analisis['mi']
            features_mi = df_mi[df_mi['MI'] > umbral_mi]['Feature'].tolist()
            df_consenso.loc[df_consenso['feature'].isin(features_mi), 'voto_MI'] = True
            logger.info(f"  MI: {len(features_mi)} features aprueban (MI > {umbral_mi})")

        # Evaluar RF
        if resultados_analisis['rf'] is not None:
            df_rf = resultados_analisis['rf']
            features_rf = df_rf.head(top_rf)['Feature'].tolist()
            df_consenso.loc[df_consenso['feature'].isin(features_rf), 'voto_RF'] = True
            logger.info(f"  RF: {len(features_rf)} features aprueban (top {top_rf})")

        # Evaluar Deep Learning MLP
        if resultados_analisis['dl'] is not None:
            try:
                dl_data = resultados_analisis['dl']
                if 'mlp' in dl_data and 'top_features' in dl_data['mlp']:
                    features_mlp = dl_data['mlp']['top_features']
                    if features_mlp and len(features_mlp) > 0:
                        df_consenso.loc[df_consenso['feature'].isin(features_mlp), 'voto_DL_MLP'] = True
                        logger.info(f"  DL-MLP: {len(features_mlp)} features aprueban (top features)")
                    else:
                        logger.info(f"  DL-MLP: No hay top features disponibles")
                else:
                    logger.info(f"  DL-MLP: No disponible en resultados")
            except Exception as e:
                logger.warning(f"  DL-MLP: Error al procesar - {e}")

        # Evaluar Lasso (si está disponible)
        # Lasso requiere recálculo ya que solo tenemos n_features_lasso en JSON
        # Por ahora lo dejamos en False

        # Evaluar Gradient Boosting (si está disponible)
        # GB también requiere recálculo completo
        # Por ahora lo dejamos en False

        # Contar votos totales (incluyendo DL si está disponible)
        votos_cols = ['voto_IC', 'voto_MI', 'voto_RF', 'voto_DL_MLP']
        df_consenso['votos'] = df_consenso[votos_cols].sum(axis=1)

        # Ordenar por número de votos
        df_consenso = df_consenso.sort_values('votos', ascending=False)

        # Estadísticas (con 4 métodos: IC, MI, RF, DL-MLP)
        consenso_fuerte = len(df_consenso[df_consenso['votos'] >= 3])
        consenso_medio = len(df_consenso[df_consenso['votos'] == 2])
        consenso_debil = len(df_consenso[df_consenso['votos'] <= 1])

        logger.info(f"\nResultados de consenso (modo optimizado - 4 métodos):")
        logger.info(f"  Consenso Fuerte (≥3 votos): {consenso_fuerte}")
        logger.info(f"  Consenso Medio (2 votos): {consenso_medio}")
        logger.info(f"  Consenso Débil (≤1 voto): {consenso_debil}")

        return df_consenso

    def analizar_un_par(self, par: str) -> dict:
        """
        Ejecuta proceso de consenso para un par.

        Args:
            par: Nombre del par

        Returns:
            Diccionario con resultados
        """
        logger.info("\n" + "="*80)
        logger.info(f"PROCESANDO CONSENSO: {par}")
        logger.info("="*80)

        inicio = datetime.now()

        try:
            # Cargar datos
            X, y, nombres_features, df_completo = self.cargar_features_y_preparar_datos(par)

            if X is None:
                return {
                    'par': par,
                    'exito': False,
                    'error': 'Error al cargar datos',
                    'tiempo_segundos': 0
                }

            # Limitar features si son demasiados
            if len(nombres_features) > 1000:
                logger.warning(f"Demasiados features ({len(nombres_features)}), limitando a 1000")
                X = X[:, :1000]
                nombres_features = nombres_features[:1000]

            resultados_par = {
                'par': par,
                'exito': True,
                'n_features_totales': len(nombres_features),
                'n_muestras': len(y),
                'consenso': {},
                'modo': 'desconocido'
            }

            # ==========================================
            # INTENTAR CARGAR RESULTADOS PRE-GENERADOS
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("CARGANDO RESULTADOS DE ANÁLISIS MULTI-MÉTODO")
            logger.info("="*80)

            resultados_pregenerados = self.cargar_resultados_analisis_pregenerados(par)

            # Verificar si tenemos suficientes resultados pre-generados
            tiene_ic = resultados_pregenerados['ic'] is not None
            tiene_mi = resultados_pregenerados['mi'] is not None
            tiene_rf = resultados_pregenerados['rf'] is not None
            tiene_dl = resultados_pregenerados['dl'] is not None

            usar_modo_optimizado = tiene_ic and tiene_mi and tiene_rf

            if usar_modo_optimizado:
                logger.info("\n✓ Resultados pre-generados encontrados - MODO OPTIMIZADO")
                logger.info(f"  IC: SÍ | MI: SÍ | RF: SÍ | DL: {'SÍ' if tiene_dl else 'NO'}")
                logger.info(f"  (Evitando recálculo de {'IC, MI, RF, DL' if tiene_dl else 'IC, MI, RF'})")
                resultados_par['modo'] = 'optimizado'
                resultados_par['tiene_dl'] = tiene_dl

                # ==========================================
                # MODO OPTIMIZADO: USAR RESULTADOS PRE-GENERADOS
                # ==========================================
                df_consenso = self.generar_consenso_desde_resultados_pregenerados(
                    resultados_pregenerados,
                    par,
                    umbral_ic=0.01,
                    umbral_mi=0.01,
                    top_rf=self.top_n_por_metodo
                )

                # Guardar tabla de consenso
                output_tabla = self.consenso_dir / f"{par}_{self.timeframe}_tabla_consenso.csv"
                df_consenso.to_csv(output_tabla, index=False)
                logger.info(f"✓ Tabla de consenso guardada: {output_tabla.name}")

                # Analizar consenso (ajustado para 3-4 métodos dependiendo de DL)
                n_metodos = 4 if tiene_dl else 3
                umbral_fuerte = 3 if tiene_dl else 3
                consenso_fuerte = df_consenso[df_consenso['votos'] >= umbral_fuerte]
                consenso_medio = df_consenso[df_consenso['votos'] == 2]
                consenso_debil = df_consenso[df_consenso['votos'] <= 1]

            else:
                logger.info("\n⚠️  Resultados pre-generados NO disponibles - MODO RECÁLCULO")
                logger.info("  (Calculando IC, MI, RF desde cero)")
                logger.info(f"  IC disponible: {'SÍ' if tiene_ic else 'NO'}")
                logger.info(f"  MI disponible: {'SÍ' if tiene_mi else 'NO'}")
                logger.info(f"  RF disponible: {'SÍ' if tiene_rf else 'NO'}")
                resultados_par['modo'] = 'recalculo'

                # ==========================================
                # MODO RECÁLCULO: TABLA DE CONSENSO COMPLETA
                # ==========================================
                logger.info("\nGenerando tabla de consenso (recalculando métodos)...")

                tabla = TablaConsenso(X, y, nombres_features)

                # Generar tabla completa
                df_consenso = tabla.generar_tabla_completa(
                    umbral_ic=0.01,
                    umbral_mi=0.01,
                    top_rf=50,
                    top_gb=50
                )

                # Guardar tabla de consenso
                output_tabla = self.consenso_dir / f"{par}_{self.timeframe}_tabla_consenso.csv"
                tabla.guardar_tabla(output_tabla)
                logger.info(f"✓ Tabla de consenso guardada: {output_tabla.name}")

                # Analizar consenso (5 métodos)
                consenso_fuerte = df_consenso[df_consenso['votos'] >= 5]
                consenso_medio = df_consenso[(df_consenso['votos'] >= 3) & (df_consenso['votos'] < 5)]
                consenso_debil = df_consenso[df_consenso['votos'] < 3]

            # Imprimir resultados
            logger.info(f"\nResultados Tabla de Consenso:")
            logger.info(f"  Consenso Fuerte: {len(consenso_fuerte)}")
            logger.info(f"  Consenso Medio: {len(consenso_medio)}")
            logger.info(f"  Consenso Débil: {len(consenso_debil)}")

            resultados_par['consenso']['tabla'] = {
                'n_fuerte': len(consenso_fuerte),
                'n_medio': len(consenso_medio),
                'n_debil': len(consenso_debil),
                'features_fuerte': consenso_fuerte['feature'].tolist() if len(consenso_fuerte) > 0 else []
            }

            # ==========================================
            # OPCIÓN 2: PROCESO DE CONSENSO (3 PASOS)
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("PROCESO DE CONSENSO EN 3 PASOS")
            logger.info("="*80)

            proceso = ProcesoConsenso(
                X, y, nombres_features,
                top_n=self.top_n_por_metodo
            )

            # PASO 1: Generar rankings
            proceso.paso1_generar_rankings()

            # Guardar rankings
            for metodo, ranking in proceso.rankings.items():
                output_ranking = self.rankings_dir / f"{par}_{self.timeframe}_ranking_{metodo}.csv"

                # Convertir set a DataFrame si es necesario
                if isinstance(ranking, set):
                    df_ranking = pd.DataFrame({'Feature': list(ranking)})
                    df_ranking.to_csv(output_ranking, index=False)
                else:
                    ranking.to_csv(output_ranking, index=False)

                logger.info(f"  ✓ Ranking {metodo} guardado")

            # PASO 2: Calcular intersecciones
            feature_counts, feature_metodos = proceso.paso2_calcular_intersecciones()

            logger.info(f"\nIntersecciones:")
            logger.info(f"  Consenso Fuerte: {len(proceso.consenso_fuerte)} features")
            logger.info(f"  Consenso Medio: {len(proceso.consenso_medio)} features")
            logger.info(f"  Sin Consenso: {len(proceso.sin_consenso)} features")

            # PASO 3: Verificación cruzada
            features_aprobados, features_rechazados = proceso.paso3_verificacion_cruzada(
                feature_counts,
                feature_metodos
            )

            # Guardar features aprobados
            if len(proceso.features_aprobados) > 0:
                # Contar en cuántos métodos aparece cada feature
                n_metodos_list = []
                for feat in proceso.features_aprobados:
                    count = 0
                    for ranking in proceso.rankings.values():
                        # Manejar tanto sets como DataFrames
                        if isinstance(ranking, set):
                            if feat in ranking:
                                count += 1
                        elif hasattr(ranking, 'values'):  # DataFrame
                            if 'Feature' in ranking.columns and feat in ranking['Feature'].values:
                                count += 1
                    n_metodos_list.append(count)

                df_aprobados = pd.DataFrame({
                    'feature': proceso.features_aprobados,
                    'n_metodos_aprueban': n_metodos_list
                })

                output_aprobados = self.aprobados_dir / f"{par}_{self.timeframe}_features_aprobados.csv"
                proceso.guardar_features_aprobados(output_aprobados)
                logger.info(f"✓ Features aprobados guardados: {output_aprobados.name}")

            logger.info(f"\n✓ Features APROBADOS: {len(proceso.features_aprobados)}")

            resultados_par['consenso']['proceso'] = {
                'n_consenso_fuerte': len(proceso.consenso_fuerte),
                'n_consenso_medio': len(proceso.consenso_medio),
                'n_aprobados': len(proceso.features_aprobados),
                'features_aprobados': proceso.features_aprobados[:20]  # Primeros 20
            }

            # ==========================================
            # GUARDAR RESUMEN CONSOLIDADO
            # ==========================================
            output_json = self.output_dir / f"{par}_{self.timeframe}_consenso_completo.json"

            # Convertir tipos numpy a tipos nativos de Python para JSON
            resultados_par_json = convert_numpy_types(resultados_par)

            with open(output_json, 'w') as f:
                json.dump(resultados_par_json, f, indent=2, ensure_ascii=False)

            fin = datetime.now()
            tiempo_total = (fin - inicio).total_seconds()
            resultados_par['tiempo_segundos'] = tiempo_total

            logger.info(f"\n✓ PAR COMPLETADO: {par}")
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
        Ejecuta el consenso MULTI-TIMEFRAME para todos los pares.
        """
        self.tiempo_inicio = datetime.now()

        logger.info("\n" + "="*80)
        logger.info("CONSENSO DE MÉTODOS - MULTI-TIMEFRAME")
        logger.info("="*80)
        logger.info(f"Pares a procesar: {len(self.pares)}")
        logger.info(f"Timeframes: {self.timeframes}")
        logger.info(f"Horizonte predicción: {self.horizonte_prediccion} período(s)")
        logger.info(f"Top-N por método: {self.top_n_por_metodo}")
        logger.info(f"Directorio features: {self.features_dir}")
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

        # LOOP MULTI-TIMEFRAME
        total_combinaciones = len(self.pares) * len(self.timeframes)
        combinacion_actual = 0

        for timeframe in self.timeframes:
            # Definir timeframe actual para uso interno
            self.timeframe = timeframe

            logger.info(f"\n{'='*80}")
            logger.info(f"PROCESANDO TIMEFRAME: {timeframe}")
            logger.info(f"{'='*80}")

            for par in self.pares:
                combinacion_actual += 1
                logger.info(f"\n[{combinacion_actual}/{total_combinaciones}] Procesando: {par} ({timeframe})")

                resultado = self.analizar_un_par(par)
                # Usar clave compuesta par_timeframe
                key = f"{par}_{timeframe}"
                self.resultados[key] = resultado

        self.tiempo_fin = datetime.now()

        # Imprimir resumen
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen final detallado."""
        tiempo_total = (self.tiempo_fin - self.tiempo_inicio).total_seconds()

        print(f"\n{'='*100}")
        print(f"{'RESUMEN FINAL - CONSENSO DE MÉTODOS':^100}")
        print(f"{'='*100}")

        # Recopilar estadísticas (ahora con multi-timeframe)
        exitosos = 0
        total_aprobados = 0
        total_fuerte = 0
        total_medio = 0
        total_sin_consenso = 0
        total_combinaciones = len(self.pares) * len(self.timeframes)

        for res in self.resultados.values():
            if res['exito']:
                exitosos += 1
                total_fuerte += res['consenso']['tabla']['n_fuerte']
                total_medio += res['consenso']['tabla']['n_medio']
                total_sin_consenso += res['consenso']['tabla']['n_sin_consenso']
                total_aprobados += res['consenso']['proceso']['n_aprobados']

        # ============================================================
        # RESUMEN EJECUTIVO
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"{'1. RESUMEN EJECUTIVO':^100}")
        print(f"{'─'*100}")

        print(f"\n  Pares:                         {len(self.pares)}")
        print(f"  Timeframes:                    {len(self.timeframes)} ({', '.join(self.timeframes)})")
        print(f"  Combinaciones exitosas:        {exitosos}/{total_combinaciones}")

        if exitosos > 0:
            # Métricas de consenso
            aprobados_por_par = [res['consenso']['proceso']['n_aprobados']
                                for res in self.resultados.values() if res['exito']]
            fuerte_por_par = [res['consenso']['tabla']['n_fuerte']
                             for res in self.resultados.values() if res['exito']]

            print(f"\n  🎯 CONSENSO FUERTE (≥5 métodos):")
            print(f"     Total features:             {total_fuerte:,}")
            print(f"     Promedio por par:           {np.mean(fuerte_por_par):.1f}")
            print(f"     Rango:                      {np.min(fuerte_por_par):.0f} - {np.max(fuerte_por_par):.0f}")

            print(f"\n  📊 CONSENSO MEDIO (3-4 métodos):")
            medio_por_par = [res['consenso']['tabla']['n_medio']
                           for res in self.resultados.values() if res['exito']]
            print(f"     Total features:             {total_medio:,}")
            print(f"     Promedio por par:           {np.mean(medio_por_par):.1f}")

            print(f"\n  ✅ FEATURES APROBADOS:")
            print(f"     Total:                      {total_aprobados:,}")
            print(f"     Promedio por par:           {np.mean(aprobados_por_par):.1f}")
            print(f"     Tasa aprobación:            {total_aprobados/(total_fuerte+total_medio+total_sin_consenso)*100 if (total_fuerte+total_medio+total_sin_consenso) > 0 else 0:.1f}%")

            # Mejor productor
            if aprobados_por_par:
                mejor_idx = np.argmax(aprobados_por_par)
                peor_idx = np.argmin(aprobados_por_par)
                pares_exitosos = [p for p, r in self.resultados.items() if r['exito']]
                mejor_par = pares_exitosos[mejor_idx] if mejor_idx < len(pares_exitosos) else 'N/A'
                peor_par = pares_exitosos[peor_idx] if peor_idx < len(pares_exitosos) else 'N/A'

                print(f"\n  🏆 MEJOR CONSENSO:             {mejor_par} ({aprobados_por_par[mejor_idx]:.0f} features aprobados)")
                print(f"  📊 MENOR CONSENSO:             {peor_par} ({aprobados_por_par[peor_idx]:.0f} features aprobados)")

        # Información temporal
        print(f"\n  ⏱️  TIEMPO DE EJECUCIÓN:")
        print(f"     Inicio:                     {self.tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     Fin:                        {self.tiempo_fin.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     Duración Total:             {self._formatear_duracion(tiempo_total)}")
        print(f"     Tiempo Promedio:            {self._formatear_duracion(tiempo_total/total_combinaciones)} por combinación")

        # ============================================================
        # TABLA DE RESULTADOS COMPLETA (MULTI-TIMEFRAME)
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"{'2. RESULTADOS POR COMBINACIÓN (MULTI-TIMEFRAME)':^100}")
        print(f"{'─'*100}")
        print(f"\n{'Par_TF':<14} │ {'✓':<3} │ {'Fuerte':<8} │ {'Medio':<8} │ {'Sin Cons.':<11} │ {'Aprobados':<11} │ {'Tasa%':<7} │ {'Tiempo':<10}")
        print("─" * 100)

        for key in sorted(self.resultados.keys()):
            res = self.resultados[key]

            if res['exito']:
                n_fuerte = res['consenso']['tabla']['n_fuerte']
                n_medio = res['consenso']['tabla']['n_medio']
                n_sin = res['consenso']['tabla']['n_sin_consenso']
                n_aprobados = res['consenso']['proceso']['n_aprobados']
                total_comb = n_fuerte + n_medio + n_sin
                tasa = (n_aprobados / total_comb * 100) if total_comb > 0 else 0

                print(
                    f"{key:<14} │ {'✓':<3} │ {n_fuerte:>7,} │ {n_medio:>7,} │ "
                    f"{n_sin:>10,} │ {n_aprobados:>10,} │ {tasa:>6.1f} │ {res['tiempo_segundos']:>9.1f}s"
                )
            else:
                print(
                    f"{key:<14} │ {'✗':<3} │ {'N/A':<8} │ {'N/A':<8} │ {'N/A':<11} │ "
                    f"{'N/A':<11} │ {'N/A':<7} │ {res['tiempo_segundos']:>9.1f}s"
                )
                print(f"{'':15} └─ Error: {res.get('error', 'Desconocido')}")

        print("─" * 100)

        # ============================================================
        # MÉTRICAS DETALLADAS POR COMBINACIÓN
        # ============================================================
        if exitosos > 0:
            print(f"\n{'─'*100}")
            print(f"{'3. MÉTRICAS DETALLADAS POR COMBINACIÓN (MULTI-TIMEFRAME)':^100}")
            print(f"{'─'*100}")

            for idx, key in enumerate(sorted(self.resultados.keys()), 1):
                res = self.resultados[key]

                if not res['exito']:
                    print(f"\n  [{idx}] {key}: ✗ ERROR")
                    print(f"      └─ {res.get('error', 'Desconocido')}")
                    continue

                print(f"\n  [{idx}] {key}")
                print(f"  {'─'*96}")

                # Consenso Tabla
                tabla = res['consenso']['tabla']
                total_features = tabla['n_fuerte'] + tabla['n_medio'] + tabla['n_sin_consenso']

                print(f"    📊 CONSENSO POR NIVEL:")
                print(f"       Fuerte (≥5 métodos):      {tabla['n_fuerte']:,}  ({tabla['n_fuerte']/total_features*100 if total_features > 0 else 0:.1f}%)")
                print(f"       Medio (3-4 métodos):      {tabla['n_medio']:,}  ({tabla['n_medio']/total_features*100 if total_features > 0 else 0:.1f}%)")
                print(f"       Sin consenso (≤2):        {tabla['n_sin_consenso']:,}  ({tabla['n_sin_consenso']/total_features*100 if total_features > 0 else 0:.1f}%)")
                print(f"       Total features:           {total_features:,}")

                # Consenso Proceso
                proceso = res['consenso']['proceso']
                n_verificados = proceso.get('n_verificados', 0)
                n_aprobados = proceso['n_aprobados']
                tasa_aprob = proceso.get('tasa_aprobacion', 0)

                print(f"\n    ✅ VERIFICACIÓN CRUZADA:")
                print(f"       Features verificados:     {n_verificados:,}")
                print(f"       Features aprobados:       {n_aprobados:,}")
                print(f"       Tasa aprobación:          {tasa_aprob*100:.1f}%")

                # Rating de calidad
                if tasa_aprob > 0.8:
                    calidad = "🏆 Excelente"
                elif tasa_aprob > 0.6:
                    calidad = "✅ Buena"
                elif tasa_aprob > 0.4:
                    calidad = "📊 Aceptable"
                else:
                    calidad = "⚠️ Baja"
                print(f"       Calidad consenso:         {calidad}")

                # Rankings por método
                if 'rankings' in res['consenso']['proceso']:
                    rankings = res['consenso']['proceso']['rankings']
                    print(f"\n    🔬 MÉTODOS EJECUTADOS:")
                    for metodo, features in sorted(rankings.items(), key=lambda x: len(x[1]) if isinstance(x[1], list) else 0, reverse=True):
                        if isinstance(features, list):
                            print(f"       • {metodo:<25} {len(features):>4} features")

        # ============================================================
        # LOGS CAPTURADOS
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"{'4. LOGS CAPTURADOS DURANTE LA EJECUCIÓN':^100}")
        print(f"{'─'*100}")

        total_logs = len(log_capture.info_logs) + len(log_capture.warnings) + len(log_capture.errors)

        if total_logs == 0:
            print(f"\n✓ No se detectaron anomalías, warnings o errores durante la ejecución")
        else:
            print(f"\nTotal de eventos registrados: {total_logs}")

            # INFO LOGS (anomalías menores)
            if log_capture.info_logs:
                print(f"\n📋 INFORMACIÓN RELEVANTE ({len(log_capture.info_logs)}):")
                print("-" * 80)
                for i, info in enumerate(log_capture.info_logs, 1):
                    print(f"{i:3d}. [{info['timestamp']}] [{info['modulo']}]")
                    print(f"     {info['mensaje']}")
            else:
                print(f"\n✓ No se registraron mensajes informativos de interés")

            # WARNINGS
            if log_capture.warnings:
                print(f"\n⚠️  ADVERTENCIAS ({len(log_capture.warnings)}):")
                print("-" * 80)
                for i, warn in enumerate(log_capture.warnings, 1):
                    print(f"{i:3d}. [{warn['timestamp']}] [{warn['modulo']}]")
                    print(f"     {warn['mensaje']}")
            else:
                print(f"\n✓ No se registraron advertencias")

            # ERRORS
            if log_capture.errors:
                print(f"\n❌ ERRORES ({len(log_capture.errors)}):")
                print("-" * 80)
                for i, error in enumerate(log_capture.errors, 1):
                    print(f"{i:3d}. [{error['timestamp']}] [{error['modulo']}:{error['linea']}]")
                    print(f"     {error['mensaje']}")
            else:
                print(f"\n✓ No se registraron errores")

        print(f"{'─'*100}")

        # ============================================================
        # ARCHIVOS GENERADOS
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"{'5. ARCHIVOS GENERADOS':^100}")
        print(f"{'─'*100}")

        archivos_csv = list(self.output_dir.glob("*.csv"))
        archivos_json = list(self.output_dir.glob("*.json"))
        total_archivos = len(archivos_csv) + len(archivos_json)

        print(f"\n  Total de archivos generados: {total_archivos}")
        print(f"\n  📊 CSV (consenso final):      {len(archivos_csv):3d} archivos")
        print(f"  📈 JSON (detalles):           {len(archivos_json):3d} archivos")
        print(f"\n  📁 Ubicación base: {self.output_dir}")

        # ============================================================
        # CONCLUSIÓN Y PRÓXIMOS PASOS
        # ============================================================
        print(f"\n{'='*100}")
        print(f"{'CONCLUSIÓN':^100}")
        print(f"{'='*100}")

        if exitosos == total_combinaciones:
            print(f"\n  ✅ CONSENSO MULTI-TIMEFRAME COMPLETADO EXITOSAMENTE")
            print(f"\n  Resumen:")
            print(f"     • Pares:                    {len(self.pares)}")
            print(f"     • Timeframes:               {len(self.timeframes)}")
            print(f"     • Combinaciones exitosas:   {exitosos}/{total_combinaciones}")
            print(f"     • Features aprobados:       {total_aprobados:,}")
            print(f"     • Consenso fuerte:          {total_fuerte:,}")
            print(f"     • Consenso medio:           {total_medio:,}")

            print(f"\n  📋 PRÓXIMOS PASOS:")
            print(f"     1. Revisar features aprobados:")
            print(f"        → {self.output_dir}/*_consenso_final.csv")
            print(f"     2. Ejecutar validación rigurosa:")
            print(f"        → python ejecutar_validacion_rigurosa.py")
            print(f"     3. Generar estrategias emergentes:")
            print(f"        → python ejecutar_estrategia_emergente.py")

        elif exitosos > 0:
            print(f"\n  ⚠️  CONSENSO COMPLETADO CON ERRORES PARCIALES")
            print(f"\n  Resumen:")
            print(f"     • Combinaciones exitosas:   {exitosos}/{total_combinaciones}")
            print(f"     • Combinaciones con errores: {total_combinaciones - exitosos}")

            print(f"\n  📋 ACCIÓN REQUERIDA:")
            print(f"     1. Revisar logs de errores en sección 4")
            print(f"     2. Corregir problemas en pares fallidos")

        else:
            print(f"\n  ❌ CONSENSO FALLIDO")
            print(f"\n  📋 ACCIÓN CRÍTICA REQUERIDA:")
            print(f"     1. Revisar logs detallados en sección 4")
            print(f"     2. Verificar datos de entrada")

        print(f"\n  {'─'*96}")
        print(f"  ℹ️  NOTA:")
        print(f"     El consenso identifica features con evidencia convergente.")
        print(f"     Solo features con ≥3 métodos en acuerdo pasan al siguiente paso.")
        print(f"{'='*100}\n")

    def _formatear_duracion(self, segundos: float) -> str:
        """Formatea duración en formato legible."""
        if segundos < 60:
            return f"{segundos:.1f}s"
        elif segundos < 3600:
            mins = int(segundos // 60)
            secs = int(segundos % 60)
            return f"{mins}m {secs}s"
        else:
            horas = int(segundos // 3600)
            mins = int((segundos % 3600) // 60)
            return f"{horas}h {mins}m"


def main():
    """Función principal - MULTI-TIMEFRAME."""
    # Configuración
    BASE_DIR = Path(__file__).parent
    FEATURES_DIR = BASE_DIR / 'datos' / 'features'
    OUTPUT_DIR = BASE_DIR / 'datos' / 'consenso_metodos'

    # MULTI-TIMEFRAME: Consenso para todos los timeframes
    TIMEFRAMES = ['M15', 'H1', 'H4', 'D']

    # Opciones de consenso
    HORIZONTE_PREDICCION = 1    # Predecir retorno 1 período adelante
    TOP_N_POR_METODO = 100       # Top-N features por cada método

    # Opciones de limpieza de archivos
    LIMPIAR_ARCHIVOS_VIEJOS = True  # True = Borra archivos viejos antes de iniciar
    HACER_BACKUP = False             # False = NO crea backup (ahorra espacio)

    # Validar que existe el directorio de features
    if not FEATURES_DIR.exists():
        logger.error(f"Directorio de features no encontrado: {FEATURES_DIR}")
        logger.error("Ejecuta primero: python ejecutar_generacion_transformaciones.py")
        return

    # Ejecutar consenso MULTI-TIMEFRAME
    ejecutor = EjecutorConsensoMetodos(
        features_dir=FEATURES_DIR,
        output_dir=OUTPUT_DIR,
        timeframes=TIMEFRAMES,
        horizonte_prediccion=HORIZONTE_PREDICCION,
        top_n_por_metodo=TOP_N_POR_METODO,
        limpiar_archivos_viejos=LIMPIAR_ARCHIVOS_VIEJOS,
        hacer_backup=HACER_BACKUP
    )

    ejecutor.ejecutar_todos()


if __name__ == '__main__':
    main()
