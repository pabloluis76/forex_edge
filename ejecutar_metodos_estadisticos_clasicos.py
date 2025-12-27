"""
EJECUTAR MÉTODOS ESTADÍSTICOS CLÁSICOS - TODOS LOS PARES
==========================================================

Aplica métodos estadísticos clásicos sobre los features generados
para identificar relaciones lineales, correlaciones, y patrones.

MÉTODOS IMPLEMENTADOS:
----------------------

A) REGRESIÓN LINEAL (OLS)
   - Relación lineal entre features y retornos
   - Coeficientes β (importancia de cada feature)
   - R², t-statistics, F-statistic
   - Diagnósticos de residuos

B) REGRESIÓN REGULARIZADA (Ridge/Lasso)
   - Ridge: Penalización L2 para evitar overfitting
   - Lasso: Penalización L1 + selección automática de features
   - Cross-validation para selección de λ
   - Feature importance

C) PCA (Principal Component Analysis)
   - Reducción de dimensionalidad
   - Identificación de componentes principales
   - Varianza explicada por componente
   - Eliminación de multicolinealidad

D) ANÁLISIS DE CORRELACIÓN
   - Matriz de correlación (Pearson, Spearman)
   - Identificación de features redundantes
   - Clustering de features correlacionadas
   - Network de correlaciones

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

from constants import EPSILON

# Importar módulos estadísticos clásicos
sys.path.insert(0, str(Path(__file__).parent / "Métodos Estadísticos Clásicos"))

from regresion_lineal import RegresionLineal
from regresion_ridge_lasso import RegresionRegularizada, RegularizacionCV
from pca import PCA
from correlacion_causalidad import AnalisisCorrelacion

# Configurar logging con captura de INFO, WARNING, ERROR
class LogCapture(logging.Handler):
    """Handler personalizado para capturar logs de INFO, WARNING y ERROR."""

    def __init__(self):
        super().__init__()
        self.info_logs = []
        self.warnings = []
        self.errors = []

    def emit(self, record):
        """Captura logs según su nivel."""
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
            # Solo capturar INFO con keywords de anomalías
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
log_capture = LogCapture()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        log_capture
    ]
)
logger = logging.getLogger(__name__)


class EjecutorMetodosEstadisticosClasicos:
    """
    Ejecuta métodos estadísticos clásicos para todos los pares.
    """

    def __init__(
        self,
        features_dir: Path,
        output_dir: Path,
        timeframe: str = 'M15',
        horizonte_prediccion: int = 1,
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
            limpiar_archivos_viejos: Si True, borra archivos viejos antes de iniciar
            hacer_backup: Si True, hace backup antes de borrar
        """
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.timeframe = timeframe
        self.horizonte_prediccion = horizonte_prediccion
        self.limpiar_archivos_viejos = limpiar_archivos_viejos
        self.hacer_backup = hacer_backup

        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectorios por método
        self.regresion_dir = self.output_dir / 'regresion_lineal'
        self.regularizada_dir = self.output_dir / 'regresion_regularizada'
        self.pca_dir = self.output_dir / 'pca'
        self.correlacion_dir = self.output_dir / 'correlacion'

        for dir_path in [self.regresion_dir, self.regularizada_dir,
                        self.pca_dir, self.correlacion_dir]:
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
        # Buscar archivos de análisis existentes
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

            # Crear subdirectorios en backup
            for subdir in ['regresion_lineal', 'regresion_regularizada', 'pca', 'correlacion']:
                (backup_dir / subdir).mkdir(parents=True, exist_ok=True)

            logger.info(f"\nCreando backup en: {backup_dir}")

            for archivo in archivos_existentes:
                # Mantener estructura de subdirectorios
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

    def cargar_features_y_preparar_datos(self, par: str) -> tuple:
        """
        Carga features y prepara X, y para análisis.

        Args:
            par: Nombre del par (ej: 'EUR_USD')

        Returns:
            (X, y, nombres_features, df_completo)
        """
        # Cargar archivo de features
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

    def analizar_un_par(self, par: str) -> dict:
        """
        Ejecuta análisis estadísticos clásicos para un par.

        Args:
            par: Nombre del par

        Returns:
            Diccionario con resultados
        """
        logger.info("\n" + "="*80)
        logger.info(f"ANALIZANDO PAR: {par}")
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

            # Limitar número de features para análisis (evitar overflow)
            if len(nombres_features) > 500:
                logger.warning(f"Demasiados features ({len(nombres_features)}), limitando a 500")
                X = X[:, :500]
                nombres_features = nombres_features[:500]

            resultados_par = {
                'par': par,
                'exito': True,
                'n_features': len(nombres_features),
                'n_muestras': len(y),
                'analisis': {}
            }

            # ==========================================
            # A) REGRESIÓN LINEAL (OLS)
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("A) REGRESIÓN LINEAL (OLS)")
            logger.info("="*80)

            modelo_ols = RegresionLineal()
            modelo_ols.fit(X, y, feature_names=nombres_features)

            # Obtener coeficientes significativos
            coef_df = modelo_ols.get_coefficients()
            coef_sig = coef_df[coef_df['p_value'] < 0.05].copy()

            # Guardar coeficientes
            output_coef = self.regresion_dir / f"{par}_{self.timeframe}_coeficientes.csv"
            coef_df.to_csv(output_coef, index=False)
            logger.info(f"✓ Coeficientes guardados: {output_coef.name}")

            resultados_par['analisis']['regresion_lineal'] = {
                'r_squared': float(modelo_ols.r_squared_),
                'r_squared_adj': float(modelo_ols.r_squared_adj_),
                'n_coef_significativos': len(coef_sig),
                'f_statistic': float(modelo_ols.f_statistic_),
                'f_pvalue': float(modelo_ols.f_pvalue_)
            }

            # ==========================================
            # B) REGRESIÓN REGULARIZADA (Ridge/Lasso)
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("B) REGRESIÓN REGULARIZADA (Ridge/Lasso)")
            logger.info("="*80)

            # Lasso con CV
            lasso_cv = RegularizacionCV()
            lasso_cv.fit(X, y, feature_names=nombres_features, metodo='lasso')

            # Obtener features seleccionados
            features_seleccionados = lasso_cv.get_selected_features()

            # Guardar resultados
            output_lasso = self.regularizada_dir / f"{par}_{self.timeframe}_lasso_features.csv"
            features_seleccionados.to_csv(output_lasso, index=False)
            logger.info(f"✓ Features Lasso guardados: {output_lasso.name}")

            resultados_par['analisis']['lasso'] = {
                'n_features_seleccionados': len(features_seleccionados),
                'r_squared': float(lasso_cv.r_squared_),
                'lambda_optimo': float(lasso_cv.best_alpha_)
            }

            # Ridge con CV
            ridge_cv = RegularizacionCV()
            ridge_cv.fit(X, y, feature_names=nombres_features, metodo='ridge')

            resultados_par['analisis']['ridge'] = {
                'r_squared': float(ridge_cv.r_squared_),
                'lambda_optimo': float(ridge_cv.best_alpha_)
            }

            # ==========================================
            # C) PCA (Análisis de Componentes Principales)
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("C) PCA - ANÁLISIS DE COMPONENTES PRINCIPALES")
            logger.info("="*80)

            n_components = min(50, X.shape[1], X.shape[0] - 1)
            pca = PCA(n_components=n_components)
            pca.fit(X, feature_names=nombres_features)

            # Varianza explicada
            var_exp = pca.explained_variance_ratio_cumulative_

            # Guardar resultados
            pca_results = pd.DataFrame({
                'componente': range(1, len(var_exp) + 1),
                'varianza_explicada': pca.explained_variance_ratio_,
                'varianza_acumulada': var_exp
            })

            output_pca = self.pca_dir / f"{par}_{self.timeframe}_pca.csv"
            pca_results.to_csv(output_pca, index=False)
            logger.info(f"✓ Resultados PCA guardados: {output_pca.name}")

            resultados_par['analisis']['pca'] = {
                'n_componentes': n_components,
                'varianza_10_comp': float(var_exp[min(9, len(var_exp)-1)]),
                'varianza_20_comp': float(var_exp[min(19, len(var_exp)-1)]) if len(var_exp) >= 20 else None,
                'varianza_50_comp': float(var_exp[-1])
            }

            # ==========================================
            # D) ANÁLISIS DE CORRELACIÓN
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("D) ANÁLISIS DE CORRELACIÓN")
            logger.info("="*80)

            # Limitar features para matriz de correlación
            max_features_corr = 200
            if X.shape[1] > max_features_corr:
                logger.warning(f"Limitando correlación a {max_features_corr} features")
                X_corr = X[:, :max_features_corr]
                nombres_corr = nombres_features[:max_features_corr]
            else:
                X_corr = X
                nombres_corr = nombres_features

            analisis_corr = AnalisisCorrelacion()
            analisis_corr.fit(X_corr, feature_names=nombres_corr)

            # Identificar features redundantes (alta correlación)
            redundantes = analisis_corr.find_redundant_features(threshold=0.9)

            # Guardar matriz de correlación (solo una muestra)
            if len(nombres_corr) <= 50:
                corr_matrix = analisis_corr.corr_matrix_
                corr_df = pd.DataFrame(
                    corr_matrix,
                    index=nombres_corr,
                    columns=nombres_corr
                )
                output_corr = self.correlacion_dir / f"{par}_{self.timeframe}_correlacion.csv"
                corr_df.to_csv(output_corr)
                logger.info(f"✓ Matriz correlación guardada: {output_corr.name}")

            resultados_par['analisis']['correlacion'] = {
                'n_pares_redundantes': len(redundantes),
                'pares_redundantes_muestra': redundantes[:5] if len(redundantes) > 0 else []
            }

            # ==========================================
            # GUARDAR RESULTADOS CONSOLIDADOS
            # ==========================================
            output_json = self.output_dir / f"{par}_{self.timeframe}_estadisticos_completo.json"
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
        Ejecuta el análisis para todos los pares.
        """
        self.tiempo_inicio = datetime.now()

        logger.info("\n" + "="*80)
        logger.info("MÉTODOS ESTADÍSTICOS CLÁSICOS - TODOS LOS PARES")
        logger.info("="*80)
        logger.info(f"Pares a analizar: {len(self.pares)}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Horizonte predicción: {self.horizonte_prediccion} período(s)")
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

        # Analizar cada par
        for i, par in enumerate(tqdm(self.pares, desc="Analizando pares", unit="par"), 1):
            logger.info(f"\n[{i}/{len(self.pares)}] Analizando: {par}")

            resultado = self.analizar_un_par(par)
            self.resultados[par] = resultado

        self.tiempo_fin = datetime.now()

        # Imprimir resumen
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen final detallado del análisis."""
        tiempo_total = (self.tiempo_fin - self.tiempo_inicio).total_seconds()

        # Recopilar estadísticas
        exitosos = 0
        r2_ols_vals = []
        lasso_sel_vals = []
        tiempos = []

        for par in self.pares:
            res = self.resultados[par]
            if res['exito']:
                exitosos += 1
                r2_ols_vals.append(res['analisis']['regresion_lineal']['r_squared'])
                lasso_sel_vals.append(res['analisis']['lasso']['n_features_seleccionados'])
                tiempos.append(res['tiempo_segundos'])

        logger.info("\n" + "="*100)
        logger.info(f"{'RESUMEN FINAL - MÉTODOS ESTADÍSTICOS CLÁSICOS':^100}")
        logger.info("="*100)

        # ============================================================
        # SECCIÓN 1: RESUMEN EJECUTIVO
        # ============================================================
        logger.info("\n1. RESUMEN EJECUTIVO")
        logger.info("-" * 100)
        logger.info(f"  Pares analizados exitosamente: {exitosos}/{len(self.pares)}")
        logger.info(f"  Tiempo total:                  {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
        logger.info(f"  Tiempo promedio por par:       {tiempo_total/len(self.pares):.1f}s")

        if r2_ols_vals:
            logger.info(f"  R² promedio (OLS):             {np.mean(r2_ols_vals):.4f}")
            logger.info(f"  Features seleccionados (Lasso): {int(np.mean(lasso_sel_vals))} promedio")

        # ============================================================
        # SECCIÓN 2: TABLA COMPLETA DE RESULTADOS
        # ============================================================
        logger.info(f"\n2. TABLA COMPLETA DE RESULTADOS")
        logger.info("-" * 100)
        logger.info(f"{'Par':<12} {'Estado':<8} {'R² OLS':<10} {'R² Lasso':<10} {'Feat.Sel':<10} "
                   f"{'PCA 10C':<10} {'Redund.':<10} {'Tiempo(s)':<10}")
        logger.info("-" * 100)

        for par in self.pares:
            res = self.resultados[par]

            if res['exito']:
                r2_ols = res['analisis']['regresion_lineal']['r_squared']
                r2_lasso = res['analisis']['lasso']['r_squared']
                lasso_sel = res['analisis']['lasso']['n_features_seleccionados']
                pca_var = res['analisis']['pca']['varianza_10_comp']
                n_redund = res['analisis']['correlacion']['n_pares_redundantes']

                logger.info(
                    f"{par:<12} {'✓':<8} {r2_ols:<10.4f} {r2_lasso:<10.4f} {lasso_sel:<10} "
                    f"{pca_var:<10.2%} {n_redund:<10} {res['tiempo_segundos']:<10.1f}"
                )
            else:
                logger.info(f"{par:<12} {'✗':<8} Error: {res.get('error', 'Desconocido')}")

        logger.info("-" * 100)

        # ============================================================
        # SECCIÓN 3: ANÁLISIS ESTADÍSTICO
        # ============================================================
        if r2_ols_vals:
            logger.info(f"\n3. ANÁLISIS ESTADÍSTICO DE RESULTADOS")
            logger.info("-" * 100)

            logger.info(f"\n  R² OLS:")
            logger.info(f"    Mínimo:              {np.min(r2_ols_vals):.4f}")
            logger.info(f"    Máximo:              {np.max(r2_ols_vals):.4f}")
            logger.info(f"    Media:               {np.mean(r2_ols_vals):.4f}")
            logger.info(f"    Mediana:             {np.median(r2_ols_vals):.4f}")
            logger.info(f"    Desviación estándar: {np.std(r2_ols_vals):.4f}")

            logger.info(f"\n  Features Seleccionados (Lasso):")
            logger.info(f"    Mínimo:              {int(np.min(lasso_sel_vals))}")
            logger.info(f"    Máximo:              {int(np.max(lasso_sel_vals))}")
            logger.info(f"    Media:               {int(np.mean(lasso_sel_vals))}")
            logger.info(f"    Mediana:             {int(np.median(lasso_sel_vals))}")

            logger.info(f"\n  Tiempos de Procesamiento:")
            logger.info(f"    Mínimo:              {np.min(tiempos):.1f}s")
            logger.info(f"    Máximo:              {np.max(tiempos):.1f}s")
            logger.info(f"    Media:               {np.mean(tiempos):.1f}s")

        # ============================================================
        # SECCIÓN 4: DETALLES POR PAR Y MÉTODO
        # ============================================================
        logger.info(f"\n4. DETALLES POR PAR Y MÉTODO")
        logger.info("-" * 100)

        for par in self.pares:
            res = self.resultados[par]
            if res['exito']:
                logger.info(f"\n  {par}:")

                # OLS
                ols = res['analisis']['regresion_lineal']
                logger.info(f"    Regresión OLS:")
                logger.info(f"      R²:                   {ols['r_squared']:.4f}")
                logger.info(f"      R² ajustado:          {ols['r_squared_adj']:.4f}")
                logger.info(f"      Coef. significativos: {ols['n_coef_significativos']}")
                logger.info(f"      F-statistic:          {ols['f_statistic']:.2f} (p={ols['f_pvalue']:.4e})")

                # Lasso
                lasso = res['analisis']['lasso']
                logger.info(f"    Regresión Lasso:")
                logger.info(f"      R²:                   {lasso['r_squared']:.4f}")
                logger.info(f"      Features seleccionados: {lasso['n_features_seleccionados']}")
                logger.info(f"      Lambda óptimo:        {lasso['lambda_optimo']:.4e}")

                # Ridge
                ridge = res['analisis']['ridge']
                logger.info(f"    Regresión Ridge:")
                logger.info(f"      R²:                   {ridge['r_squared']:.4f}")
                logger.info(f"      Lambda óptimo:        {ridge['lambda_optimo']:.4e}")

                # PCA
                pca = res['analisis']['pca']
                logger.info(f"    PCA:")
                logger.info(f"      N componentes:        {pca['n_componentes']}")
                logger.info(f"      Varianza 10 comp:     {pca['varianza_10_comp']:.2%}")
                if pca['varianza_20_comp'] is not None:
                    logger.info(f"      Varianza 20 comp:     {pca['varianza_20_comp']:.2%}")
                logger.info(f"      Varianza 50 comp:     {pca['varianza_50_comp']:.2%}")

                # Correlación
                corr = res['analisis']['correlacion']
                logger.info(f"    Correlación:")
                logger.info(f"      Pares redundantes:    {corr['n_pares_redundantes']}")
                if corr['pares_redundantes_muestra']:
                    logger.info(f"      Muestra:              {corr['pares_redundantes_muestra'][:2]}")

        # ============================================================
        # SECCIÓN 5: LOGS CAPTURADOS
        # ============================================================
        logger.info(f"\n5. LOGS CAPTURADOS (ERRORES, WARNINGS, ANOMALÍAS)")
        logger.info("-" * 100)

        total_logs = len(log_capture.errors) + len(log_capture.warnings) + len(log_capture.info_logs)

        if total_logs == 0:
            logger.info("  ✓ No se detectaron errores, warnings ni anomalías")
        else:
            if log_capture.errors:
                logger.info(f"\n  ERRORES ({len(log_capture.errors)}):")
                for i, error in enumerate(log_capture.errors[:10], 1):
                    logger.info(f"    [{error['timestamp']}] {error['mensaje']}")
                    logger.info(f"                   → {error['modulo']}:{error['linea']}")
                if len(log_capture.errors) > 10:
                    logger.info(f"    ... y {len(log_capture.errors) - 10} errores más")

            if log_capture.warnings:
                logger.info(f"\n  WARNINGS ({len(log_capture.warnings)}):")
                for i, warn in enumerate(log_capture.warnings[:10], 1):
                    logger.info(f"    [{warn['timestamp']}] {warn['mensaje']}")
                if len(log_capture.warnings) > 10:
                    logger.info(f"    ... y {len(log_capture.warnings) - 10} warnings más")

            if log_capture.info_logs:
                logger.info(f"\n  ANOMALÍAS DETECTADAS EN INFO ({len(log_capture.info_logs)}):")
                for i, info in enumerate(log_capture.info_logs[:10], 1):
                    logger.info(f"    [{info['timestamp']}] {info['mensaje']}")
                if len(log_capture.info_logs) > 10:
                    logger.info(f"    ... y {len(log_capture.info_logs) - 10} más")

        # ============================================================
        # SECCIÓN 6: ARCHIVOS GENERADOS
        # ============================================================
        logger.info(f"\n6. ARCHIVOS GENERADOS")
        logger.info("-" * 100)
        logger.info(f"  Directorio base:     {self.output_dir}")
        logger.info(f"\n  Subdirectorios y archivos:")

        # Regresión lineal
        logger.info(f"    - Regresión Lineal:     {self.regresion_dir}")
        archivos_reg = list(self.regresion_dir.glob("*.csv"))
        if archivos_reg:
            logger.info(f"                            {len(archivos_reg)} archivos CSV (coeficientes)")

        # Regresión regularizada
        logger.info(f"    - Regresión Regularizada: {self.regularizada_dir}")
        archivos_lasso = list(self.regularizada_dir.glob("*.csv"))
        if archivos_lasso:
            logger.info(f"                            {len(archivos_lasso)} archivos CSV (features seleccionados)")

        # PCA
        logger.info(f"    - PCA:                  {self.pca_dir}")
        archivos_pca = list(self.pca_dir.glob("*.csv"))
        if archivos_pca:
            logger.info(f"                            {len(archivos_pca)} archivos CSV (componentes)")

        # Correlación
        logger.info(f"    - Correlación:          {self.correlacion_dir}")
        archivos_corr = list(self.correlacion_dir.glob("*.csv"))
        if archivos_corr:
            logger.info(f"                            {len(archivos_corr)} archivos CSV (matrices)")

        # Archivos consolidados
        archivos_json = list(self.output_dir.glob("*_estadisticos_completo.json"))
        if archivos_json:
            logger.info(f"\n  Resultados consolidados: {len(archivos_json)} archivos JSON")

        # ============================================================
        # SECCIÓN 7: CONCLUSIÓN
        # ============================================================
        logger.info(f"\n7. CONCLUSIÓN")
        logger.info("-" * 100)

        if exitosos == len(self.pares):
            logger.info(f"  ✓ ANÁLISIS ESTADÍSTICO COMPLETADO EXITOSAMENTE")
            logger.info(f"  Todos los {len(self.pares)} pares analizados correctamente")
            logger.info(f"\n  Métodos aplicados:")
            logger.info(f"    • OLS:         Regresión lineal, coeficientes β, R², F-statistic")
            logger.info(f"    • Lasso:       Regularización L1, selección automática de features")
            logger.info(f"    • Ridge:       Regularización L2, prevención de overfitting")
            logger.info(f"    • PCA:         Reducción de dimensionalidad, varianza explicada")
            logger.info(f"    • Correlación: Identificación de features redundantes")
            logger.info(f"\n  Próximos pasos:")
            logger.info(f"    1. ejecutar_analisis_multimetodo.py     → ML + Deep Learning")
            logger.info(f"    2. ejecutar_consenso_metodos.py         → Consenso entre métodos")
            logger.info(f"    3. ejecutar_validacion_rigurosa.py      → Walk-Forward + Bootstrap")
            logger.info(f"    4. ejecutar_estrategia_emergente.py     → Emergencia de reglas")
            logger.info(f"    5. ejecutar_backtest.py                 → Backtest completo")
        else:
            logger.info(f"  ⚠️  ANÁLISIS COMPLETADO CON ERRORES")
            logger.info(f"  {exitosos}/{len(self.pares)} pares exitosos")
            logger.info(f"  Revisar logs de errores arriba")

        logger.info("="*100 + "\n")


def main():
    """Función principal."""
    # Configuración
    BASE_DIR = Path(__file__).parent
    FEATURES_DIR = BASE_DIR / 'datos' / 'features'
    OUTPUT_DIR = BASE_DIR / 'datos' / 'metodos_estadisticos_clasicos'
    TIMEFRAMES = ["M15", "H1", "H4", "D1"]

    # Opciones de análisis
    HORIZONTE_PREDICCION = 1  # Predecir retorno 1 período adelante

    # Opciones de limpieza de archivos
    LIMPIAR_ARCHIVOS_VIEJOS = True  # True = Borra archivos viejos antes de iniciar
    HACER_BACKUP = True              # True = Crea backup antes de borrar

    # Validar que existe el directorio de features
    if not FEATURES_DIR.exists():
        logger.error(f"Directorio de features no encontrado: {FEATURES_DIR}")
        logger.error("Ejecuta primero: python ejecutar_generacion_transformaciones.py")
        return

    # Ejecutar análisis
    ejecutor = EjecutorMetodosEstadisticosClasicos(
        features_dir=FEATURES_DIR,
        output_dir=OUTPUT_DIR,
        timeframe=TIMEFRAME,
        horizonte_prediccion=HORIZONTE_PREDICCION,
        limpiar_archivos_viejos=LIMPIAR_ARCHIVOS_VIEJOS,
        hacer_backup=HACER_BACKUP
    )

    ejecutor.ejecutar_todos()


if __name__ == '__main__':
    main()
