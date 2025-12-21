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

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent))

from consenso_metodos.tabla_consenso import TablaConsenso
from consenso_metodos.proceso_consenso import ProcesoConsenso

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EjecutorConsensoMetodos:
    """
    Ejecuta el proceso de consenso de métodos para todos los pares.
    """

    def __init__(
        self,
        features_dir: Path,
        output_dir: Path,
        timeframe: str = 'M15',
        horizonte_prediccion: int = 1,
        top_n_por_metodo: int = 100,
        limpiar_archivos_viejos: bool = True,
        hacer_backup: bool = False
    ):
        """
        Inicializa el ejecutor.

        Args:
            features_dir: Directorio con features generados (.parquet)
            output_dir: Directorio para guardar resultados
            timeframe: Timeframe procesado (default: 'M15')
            horizonte_prediccion: Períodos adelante para calcular retorno
            top_n_por_metodo: Número de top features por método (default: 100)
            limpiar_archivos_viejos: Si True, borra archivos viejos antes de iniciar
            hacer_backup: Si True, hace backup antes de borrar
        """
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.timeframe = timeframe
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
            'EUR_USD'
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
            if not np.all(np.isnan(col)) and np.nanstd(col) > 1e-10:
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
            proceso.paso2_calcular_intersecciones()

            logger.info(f"\nIntersecciones:")
            logger.info(f"  Consenso Fuerte: {len(proceso.consenso_fuerte)} features")
            logger.info(f"  Consenso Medio: {len(proceso.consenso_medio)} features")
            logger.info(f"  Sin Consenso: {len(proceso.sin_consenso)} features")

            # PASO 3: Verificación cruzada
            proceso.paso3_verificacion_cruzada(
                verificar_estabilidad_temporal=True,
                verificar_concordancia_ml=True
            )

            # Guardar features aprobados
            if len(proceso.features_aprobados) > 0:
                df_aprobados = pd.DataFrame({
                    'feature': proceso.features_aprobados,
                    'n_metodos_aprueban': [
                        sum([feat in ranking['Feature'].values
                             for ranking in proceso.rankings.values()])
                        for feat in proceso.features_aprobados
                    ]
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
            with open(output_json, 'w') as f:
                json.dump(resultados_par, f, indent=2, ensure_ascii=False)

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
        Ejecuta el consenso para todos los pares.
        """
        self.tiempo_inicio = datetime.now()

        logger.info("\n" + "="*80)
        logger.info("CONSENSO DE MÉTODOS - TODOS LOS PARES")
        logger.info("="*80)
        logger.info(f"Pares a procesar: {len(self.pares)}")
        logger.info(f"Timeframe: {self.timeframe}")
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

        # Procesar cada par
        for i, par in enumerate(tqdm(self.pares, desc="Procesando pares", unit="par"), 1):
            logger.info(f"\n[{i}/{len(self.pares)}] Procesando: {par}")

            resultado = self.analizar_un_par(par)
            self.resultados[par] = resultado

        self.tiempo_fin = datetime.now()

        # Imprimir resumen
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen final."""
        tiempo_total = (self.tiempo_fin - self.tiempo_inicio).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("RESUMEN FINAL - CONSENSO DE MÉTODOS")
        logger.info("="*80)

        # Tabla de resultados
        logger.info("\nRESULTADOS POR PAR:")
        logger.info("-" * 100)
        logger.info(f"{'Par':<10} │ {'Exito':<6} │ {'Fuerte':<8} │ {'Medio':<8} │ {'Aprobados':<10} │ {'Tiempo (s)':<12}")
        logger.info("-" * 100)

        exitosos = 0
        total_aprobados = 0

        for par in self.pares:
            res = self.resultados[par]

            if res['exito']:
                exitosos += 1
                n_fuerte = res['consenso']['tabla']['n_fuerte']
                n_medio = res['consenso']['tabla']['n_medio']
                n_aprobados = res['consenso']['proceso']['n_aprobados']
                total_aprobados += n_aprobados

                logger.info(
                    f"{par:<10} │ {'✓':<6} │ {n_fuerte:<8} │ {n_medio:<8} │ "
                    f"{n_aprobados:<10} │ {res['tiempo_segundos']:<12.1f}"
                )
            else:
                logger.info(
                    f"{par:<10} │ {'✗':<6} │ {'N/A':<8} │ {'N/A':<8} │ "
                    f"{'N/A':<10} │ {res['tiempo_segundos']:<12.1f}"
                )
                logger.info(f"           Error: {res.get('error', 'Desconocido')}")

        logger.info("-" * 100)

        # Estadísticas globales
        logger.info("\nESTADÍSTICAS GLOBALES:")
        logger.info(f"  Pares procesados exitosamente: {exitosos}/{len(self.pares)}")
        logger.info(f"  Features aprobados totales: {total_aprobados}")
        logger.info(f"  Features promedio por par: {total_aprobados/exitosos:.0f}" if exitosos > 0 else "  N/A")
        logger.info(f"  Tiempo total: {tiempo_total:.1f} segundos ({tiempo_total/60:.1f} minutos)")
        logger.info(f"  Tiempo promedio por par: {tiempo_total/len(self.pares):.1f} segundos")

        # Conclusión
        logger.info("\n" + "="*80)
        if exitosos == len(self.pares):
            logger.info("✓ CONSENSO COMPLETADO EXITOSAMENTE")
            logger.info(f"  Todos los {len(self.pares)} pares procesados correctamente")
            logger.info(f"  Resultados guardados en: {self.output_dir}")
            logger.info("\nPRÓXIMO PASO:")
            logger.info("  → Revisar features aprobados en cada par")
            logger.info("  → Validación rigurosa (walk-forward, permutation test)")
            logger.info("  → Construcción de estrategia final")
        else:
            logger.info(f"⚠️  CONSENSO COMPLETADO CON ERRORES")
            logger.info(f"  {exitosos}/{len(self.pares)} pares exitosos")

        logger.info("="*80)


def main():
    """Función principal."""
    # Configuración
    BASE_DIR = Path(__file__).parent
    FEATURES_DIR = BASE_DIR / 'datos' / 'features'
    OUTPUT_DIR = BASE_DIR / 'datos' / 'consenso_metodos'
    TIMEFRAME = 'M15'

    # Opciones de consenso
    HORIZONTE_PREDICCION = 1    # Predecir retorno 1 período adelante
    TOP_N_POR_METODO = 100       # Top-N features por cada método

    # Opciones de limpieza de archivos
    LIMPIAR_ARCHIVOS_VIEJOS = True  # True = Borra archivos viejos antes de iniciar
    HACER_BACKUP = True              # True = Crea backup antes de borrar

    # Validar que existe el directorio de features
    if not FEATURES_DIR.exists():
        logger.error(f"Directorio de features no encontrado: {FEATURES_DIR}")
        logger.error("Ejecuta primero: python ejecutar_generacion_transformaciones.py")
        return

    # Ejecutar consenso
    ejecutor = EjecutorConsensoMetodos(
        features_dir=FEATURES_DIR,
        output_dir=OUTPUT_DIR,
        timeframe=TIMEFRAME,
        horizonte_prediccion=HORIZONTE_PREDICCION,
        top_n_por_metodo=TOP_N_POR_METODO,
        limpiar_archivos_viejos=LIMPIAR_ARCHIVOS_VIEJOS,
        hacer_backup=HACER_BACKUP
    )

    ejecutor.ejecutar_todos()


if __name__ == '__main__':
    main()
