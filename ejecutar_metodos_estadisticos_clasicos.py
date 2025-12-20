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

# Importar módulos estadísticos clásicos
sys.path.insert(0, str(Path(__file__).parent / "Métodos Estadísticos Clásicos"))

from regresion_lineal import RegresionLineal
from regresion_ridge_lasso import RegresionRegularizada, RegularizacionCV
from pca import PCA
from correlacion_causalidad import AnalisisCorrelacion

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

        # Retorno futuro
        retorno_futuro = close.shift(-self.horizonte_prediccion).pct_change()
        retorno_futuro = retorno_futuro.shift(-1)

        df['retorno_objetivo'] = retorno_futuro
        df_clean = df.dropna(subset=['retorno_objetivo'])

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

        # Eliminar filas con NaN
        mask_valid = ~np.any(np.isnan(X), axis=1)
        X = X[mask_valid]
        y = y[mask_valid]

        logger.info(f"  Features válidos: {len(nombres_features):,}")
        logger.info(f"  Muestras: {len(y):,}")

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
        """Imprime resumen final del análisis."""
        tiempo_total = (self.tiempo_fin - self.tiempo_inicio).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("RESUMEN FINAL - MÉTODOS ESTADÍSTICOS CLÁSICOS")
        logger.info("="*80)

        # Tabla de resultados
        logger.info("\nRESULTADOS POR PAR:")
        logger.info("-" * 100)
        logger.info(f"{'Par':<10} │ {'Exito':<6} │ {'R² OLS':<8} │ {'Lasso Sel.':<12} │ {'PCA 50':<8} │ {'Tiempo (s)':<12}")
        logger.info("-" * 100)

        exitosos = 0

        for par in self.pares:
            res = self.resultados[par]

            if res['exito']:
                exitosos += 1
                r2_ols = res['analisis']['regresion_lineal']['r_squared']
                lasso_sel = res['analisis']['lasso']['n_features_seleccionados']
                pca_var = res['analisis']['pca']['varianza_50_comp']

                logger.info(
                    f"{par:<10} │ {'✓':<6} │ {r2_ols:<8.4f} │ "
                    f"{lasso_sel:<12} │ {pca_var:<8.2%} │ {res['tiempo_segundos']:<12.1f}"
                )
            else:
                logger.info(
                    f"{par:<10} │ {'✗':<6} │ {'N/A':<8} │ {'N/A':<12} │ "
                    f"{'N/A':<8} │ {res['tiempo_segundos']:<12.1f}"
                )
                logger.info(f"           Error: {res.get('error', 'Desconocido')}")

        logger.info("-" * 100)

        # Estadísticas globales
        logger.info("\nESTADÍSTICAS GLOBALES:")
        logger.info(f"  Pares analizados exitosamente: {exitosos}/{len(self.pares)}")
        logger.info(f"  Tiempo total: {tiempo_total:.1f} segundos ({tiempo_total/60:.1f} minutos)")
        logger.info(f"  Tiempo promedio por par: {tiempo_total/len(self.pares):.1f} segundos")

        # Conclusión
        logger.info("\n" + "="*80)
        if exitosos == len(self.pares):
            logger.info("✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
            logger.info(f"  Todos los {len(self.pares)} pares analizados correctamente")
            logger.info(f"  Resultados guardados en: {self.output_dir}")
            logger.info("\nPRÓXIMO PASO:")
            logger.info("  → Comparar resultados entre métodos")
            logger.info("  → Identificar features más robustos")
            logger.info("  → Ejecutar sistema de consenso")
        else:
            logger.info(f"⚠️  ANÁLISIS COMPLETADO CON ERRORES")
            logger.info(f"  {exitosos}/{len(self.pares)} pares exitosos")

        logger.info("="*80)


def main():
    """Función principal."""
    # Configuración
    BASE_DIR = Path(__file__).parent
    FEATURES_DIR = BASE_DIR / 'datos' / 'features'
    OUTPUT_DIR = BASE_DIR / 'datos' / 'metodos_estadisticos_clasicos'
    TIMEFRAME = 'M15'

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
