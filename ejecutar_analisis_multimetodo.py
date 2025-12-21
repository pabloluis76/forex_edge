"""
EJECUTAR ANÁLISIS MULTI-MÉTODO - TODOS LOS PARES
=================================================

Ejecuta análisis multi-método sobre las transformaciones generadas
para identificar features con poder predictivo real.

MÉTODOS IMPLEMENTADOS:
----------------------

A) ANÁLISIS ESTADÍSTICO
   - Information Coefficient (IC)
   - Significancia estadística
   - Información mutua
   - Regresión lineal/Ridge/Lasso
   - PCA

B) MACHINE LEARNING
   - Random Forest
   - Gradient Boosting
   - XGBoost/LightGBM (opcional)
   - SVM
   - Clustering de features

C) MÉTODOS DE FÍSICA
   - Proceso Ornstein-Uhlenbeck
   - Random Walk Test
   - Hurst Exponent
   - Análisis espectral

D) DEEP LEARNING (requiere TensorFlow)
   - MLP (Multilayer Perceptron)
   - CNN (Convolutional Networks)
   - LSTM (Recurrent Networks)

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

from analisis_multi_metodo.analisis_estadistico import AnalizadorEstadistico
from analisis_multi_metodo.machine_learning import AnalizadorML
from analisis_multi_metodo.metodos_fisica import MetodosFisica

# Deep Learning es opcional (requiere TensorFlow)
try:
    from analisis_multi_metodo.deep_learning import ModelosDeepLearning
    DEEP_LEARNING_DISPONIBLE = True
except ImportError:
    DEEP_LEARNING_DISPONIBLE = False
    logging.warning("Deep Learning no disponible (instalar: pip install tensorflow)")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EjecutorAnalisisMultimetodo:
    """
    Ejecuta análisis multi-método para todos los pares.
    """

    def __init__(
        self,
        features_dir: Path,
        output_dir: Path,
        timeframe: str = 'M15',
        horizonte_prediccion: int = 1,
        limpiar_archivos_viejos: bool = True,
        hacer_backup: bool = False,
        usar_deep_learning: bool = False
    ):
        """
        Inicializa el ejecutor.

        Args:
            features_dir: Directorio con features generados (.parquet)
            output_dir: Directorio para guardar resultados del análisis
            timeframe: Timeframe procesado (default: 'M15')
            horizonte_prediccion: Períodos adelante para calcular retorno objetivo
            limpiar_archivos_viejos: Si True, borra archivos viejos antes de iniciar
            hacer_backup: Si True, hace backup antes de borrar
            usar_deep_learning: Si True, ejecuta también Deep Learning (requiere TensorFlow)
        """
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.timeframe = timeframe
        self.horizonte_prediccion = horizonte_prediccion
        self.limpiar_archivos_viejos = limpiar_archivos_viejos
        self.hacer_backup = hacer_backup
        self.usar_deep_learning = usar_deep_learning and DEEP_LEARNING_DISPONIBLE

        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        Limpia archivos de análisis viejos del directorio de salida.
        Opcionalmente hace backup antes de borrar.
        """
        # Buscar archivos de análisis existentes
        archivos_json = list(self.output_dir.glob("*_analisis_*.json"))
        archivos_csv = list(self.output_dir.glob("*_analisis_*.csv"))
        archivos_parquet = list(self.output_dir.glob("*_analisis_*.parquet"))
        archivos_existentes = archivos_json + archivos_csv + archivos_parquet

        if not archivos_existentes:
            logger.info("No hay archivos viejos para limpiar")
            return

        logger.info(f"\nEncontrados {len(archivos_existentes)} archivos viejos:")
        for archivo in archivos_existentes:
            logger.info(f"  - {archivo.name}")

        # Hacer backup si está habilitado
        if self.hacer_backup:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.output_dir / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"\nCreando backup en: {backup_dir}")

            for archivo in archivos_existentes:
                destino = backup_dir / archivo.name
                shutil.copy2(archivo, destino)
                logger.info(f"  ✓ Backup: {archivo.name}")

            logger.info(f"✓ Backup completado: {len(archivos_existentes)} archivos")

        # Borrar archivos viejos
        logger.info("\nBorrando archivos viejos...")
        for archivo in archivos_existentes:
            archivo.unlink()
            logger.info(f"  ✓ Borrado: {archivo.name}")

        logger.info(f"✓ Limpieza completada: {len(archivos_existentes)} archivos eliminados\n")

    def seleccionar_features_robustos(self, X, y, nombres_features,
                                      df_ic, df_mi,
                                      ic_threshold=0.02,
                                      p_value_threshold=0.001,
                                      max_features=100):
        """
        Selecciona features robustos basado en IC y MI.

        CRITERIOS DE SELECCIÓN:
        1. |IC| > ic_threshold (default: 0.02)
        2. p-value < p_value_threshold (default: 0.001)
        3. MI significativo (top percentil)
        4. Máximo max_features features

        Args:
            X: Matriz de features
            y: Target
            nombres_features: Lista de nombres
            df_ic: DataFrame con IC
            df_mi: DataFrame con MI
            ic_threshold: Umbral mínimo de |IC|
            p_value_threshold: Umbral máximo de p-value
            max_features: Número máximo de features a retornar

        Returns:
            X_seleccionado, nombres_seleccionados, df_seleccion
        """
        logger.info("\n" + "="*80)
        logger.info("SELECCIÓN ROBUSTA DE FEATURES")
        logger.info("="*80)
        logger.info(f"Features iniciales: {len(nombres_features):,}")
        logger.info(f"Criterios:")
        logger.info(f"  • |IC| > {ic_threshold}")
        logger.info(f"  • p-value < {p_value_threshold}")
        logger.info(f"  • Máximo: {max_features} features")

        # Filtrar por IC y p-value
        df_ic_filtrado = df_ic[
            (df_ic['abs_IC'] > ic_threshold) &
            (df_ic['p_value_corrected'] < p_value_threshold)
        ].copy()

        features_ic = set(df_ic_filtrado['Feature'].tolist())
        logger.info(f"\n✓ Features con IC significativo: {len(features_ic)}")

        # Filtrar por MI (top percentil)
        mi_percentile_90 = df_mi['MI'].quantile(0.90)
        df_mi_filtrado = df_mi[df_mi['MI'] > mi_percentile_90].copy()
        features_mi = set(df_mi_filtrado['Feature'].tolist())
        logger.info(f"✓ Features con MI alto (>P90): {len(features_mi)}")

        # Consenso: features que aparecen en ambos
        features_consenso = features_ic.intersection(features_mi)
        logger.info(f"✓ Features en CONSENSO (IC + MI): {len(features_consenso)}")

        # Si hay muy pocos features en consenso, usar solo IC
        if len(features_consenso) < 20:
            logger.warning(f"⚠️  Consenso muy bajo ({len(features_consenso)}), usando solo IC")
            features_seleccionados = list(features_ic)[:max_features]
        else:
            features_seleccionados = list(features_consenso)[:max_features]

        # Si aún hay demasiados, ordenar por abs_IC y tomar top
        if len(features_seleccionados) > max_features:
            df_ranking = df_ic[df_ic['Feature'].isin(features_seleccionados)].copy()
            df_ranking = df_ranking.sort_values('abs_IC', ascending=False)
            features_seleccionados = df_ranking.head(max_features)['Feature'].tolist()

        logger.info(f"\n✓ FEATURES FINALES SELECCIONADOS: {len(features_seleccionados)}")
        logger.info(f"  Reducción: {len(nombres_features):,} → {len(features_seleccionados)}")
        logger.info(f"  Ratio: {100 * len(features_seleccionados) / len(nombres_features):.1f}%")

        # Filtrar X para quedarnos solo con los features seleccionados
        indices_seleccionados = [i for i, nombre in enumerate(nombres_features)
                                if nombre in features_seleccionados]

        X_seleccionado = X[:, indices_seleccionados]
        nombres_seleccionados = [nombres_features[i] for i in indices_seleccionados]

        # Crear DataFrame de selección
        df_seleccion = df_ic[df_ic['Feature'].isin(features_seleccionados)].copy()
        df_seleccion = df_seleccion.sort_values('abs_IC', ascending=False)

        logger.info(f"\n📊 Top 10 features seleccionados:")
        for idx, row in df_seleccion.head(10).iterrows():
            logger.info(f"  {row['Feature']}: IC={row['IC']:.4f}, p={row['p_value_corrected']:.2e}")

        return X_seleccionado, nombres_seleccionados, df_seleccion

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

        logger.info(f"Cargando features: {archivo_features}")
        df = pd.read_parquet(archivo_features)

        logger.info(f"  Features cargados: {len(df.columns):,}")
        logger.info(f"  Filas: {len(df):,}")
        logger.info(f"  Período: {df.index[0]} → {df.index[-1]}")

        # Calcular retorno objetivo (variable a predecir)
        # Asumimos que 'close' existe en las features originales
        if 'close' in df.columns:
            close = df['close']
        else:
            # Si no hay 'close', usar la primera columna que contenga 'close'
            close_cols = [col for col in df.columns if 'close' in col.lower()]
            if close_cols:
                logger.warning(f"'close' no encontrado, usando: {close_cols[0]}")
                close = df[close_cols[0]]
            else:
                logger.error("No se puede calcular retorno: falta columna 'close'")
                return None, None, None, None

        # Calcular retorno futuro: (precio_futuro - precio_actual) / precio_actual
        precio_futuro = close.shift(-self.horizonte_prediccion)
        retorno_futuro = (precio_futuro - close) / close

        # Agregar retorno al dataframe
        df['retorno_objetivo'] = retorno_futuro

        # Eliminar filas con NaN o infinitos en retorno objetivo
        df_clean = df.dropna(subset=['retorno_objetivo'])
        # Filtrar infinitos
        mask_finito = np.isfinite(df_clean['retorno_objetivo'])
        df_clean = df_clean[mask_finito]

        if len(df_clean) == 0:
            logger.error("No quedan datos válidos después de filtrar NaN e infinitos")
            return None, None, None, None

        # Separar features (X) y objetivo (y)
        y = df_clean['retorno_objetivo'].values
        nombres_features = [col for col in df_clean.columns if col != 'retorno_objetivo']
        X = df_clean[nombres_features].values

        # Eliminar features con todo NaN o varianza cero
        valid_features = []
        valid_indices = []

        for i, nombre in enumerate(nombres_features):
            col = X[:, i]
            if not np.all(np.isnan(col)) and np.nanstd(col) > 1e-10:
                valid_features.append(nombre)
                valid_indices.append(i)

        X = X[:, valid_indices]
        nombres_features = valid_features

        # Eliminar filas con cualquier NaN o infinito
        mask_valid = ~np.any(np.isnan(X), axis=1) & ~np.any(np.isinf(X), axis=1) & np.isfinite(y)
        X = X[mask_valid]
        y = y[mask_valid]

        # Validación final
        if len(y) == 0:
            logger.error("No quedan muestras válidas después de limpieza")
            return None, None, None, None

        if not np.all(np.isfinite(y)):
            logger.error(f"Variable objetivo contiene valores no finitos: inf={np.sum(np.isinf(y))}, nan={np.sum(np.isnan(y))}")
            return None, None, None, None

        logger.info(f"\nDatos preparados:")
        logger.info(f"  Features válidos: {len(nombres_features):,}")
        logger.info(f"  Muestras válidas: {len(y):,}")
        logger.info(f"  Retorno medio: {np.mean(y)*100:.4f}%")
        logger.info(f"  Retorno std: {np.std(y)*100:.4f}%")
        logger.info(f"  Retorno min: {np.min(y)*100:.4f}%")
        logger.info(f"  Retorno max: {np.max(y)*100:.4f}%")

        return X, y, nombres_features, df_clean

    def cargar_tensor_pregenerado(self, par: str, lookback: int):
        """
        Intenta cargar tensor 3D pre-generado desde archivo .npz.

        Args:
            par: Nombre del par (ej: 'EUR_USD')
            lookback: Lookback usado para generar el tensor

        Returns:
            (X_3d, y_3d, feature_names) si existe, sino (None, None, None)
        """
        # Buscar archivo pre-generado
        tensor_dir = Path(__file__).parent / 'datos' / 'estructura_matricial_tensorial' / 'tensor_3d'
        tensor_file = tensor_dir / f"{par}_{self.timeframe}_tensor_3d_lookback_{lookback}.npz"

        if tensor_file.exists():
            try:
                logger.info(f"📦 Cargando tensor pre-generado: {tensor_file.name}")
                data = np.load(tensor_file)

                X_3d = data['X_3d']
                y = data['y']
                feature_names = data.get('feature_names', None)

                logger.info(f"✓ Tensor cargado desde disco: {X_3d.shape}")
                logger.info(f"  Normalizado: {'normalized' in str(tensor_file)}")

                return X_3d, y, feature_names

            except Exception as e:
                logger.warning(f"Error al cargar tensor pre-generado: {e}")
                logger.warning("Creando tensor en memoria...")
                return None, None, None
        else:
            logger.info(f"Tensor pre-generado no encontrado: {tensor_file.name}")
            logger.info("Creando tensor en memoria...")
            return None, None, None

    def preparar_datos_secuenciales(self, X, y, lookback=20, normalizar=True):
        """
        Prepara datos 3D para modelos secuenciales (CNN, LSTM, Transformer).

        Intenta cargar tensor pre-generado. Si no existe, lo crea en memoria.

        IMPORTANTE: La normalización es ALTAMENTE RECOMENDADA para Deep Learning.
        Los modelos de DL son sensibles a la escala de features y convergen mejor
        con datos normalizados.

        Args:
            X: Matriz 2D (n_samples, n_features)
            y: Vector target (n_samples,)
            lookback: Número de timesteps de historia
            normalizar: Si True, aplica normalización Z-score rolling (RECOMENDADO)

        Returns:
            X_3d, y_3d: Datos en formato 3D (n_sequences, lookback, n_features)
        """
        # Validación: advertir si se desactiva normalización
        if not normalizar:
            logger.warning("⚠️  ADVERTENCIA: Normalización desactivada para Deep Learning")
            logger.warning("⚠️  Los modelos DL convergen mejor con datos normalizados")
            logger.warning("⚠️  Se recomienda usar normalizar=True")

        n_samples, n_features = X.shape
        n_sequences = n_samples - lookback

        if n_sequences <= 0:
            logger.warning(f"No hay suficientes datos para lookback={lookback}")
            return None, None

        # Crear tensor 3D mediante ventanas deslizantes
        logger.info(f"Creando tensor 3D en memoria (lookback={lookback})...")
        X_3d = np.zeros((n_sequences, lookback, n_features), dtype=np.float32)
        y_3d = np.zeros(n_sequences, dtype=np.float32)

        for i in range(n_sequences):
            X_3d[i] = X[i:i+lookback]
            y_3d[i] = y[i+lookback]

        # Normalización point-in-time (RECOMENDADA)
        if normalizar:
            logger.info("✓ Aplicando normalización Z-score point-in-time...")
            X_3d = self._normalizar_tensor_point_in_time(X_3d)
        else:
            logger.warning("⚠️  Sin normalización - los resultados pueden ser subóptimos")

        logger.info(f"✓ Tensor 3D preparado: {X_3d.shape}")
        return X_3d, y_3d

    def _normalizar_tensor_point_in_time(self, X_3d, ventana=252):
        """
        Normalización Z-score rolling sin look-ahead bias.

        Para cada punto en tiempo t:
        - Solo usa información hasta t (inclusive)
        - Calcula media y std de ventana pasada
        - Normaliza: (x - mean) / std

        Args:
            X_3d: Tensor (n_sequences, lookback, n_features)
            ventana: Ventana para calcular estadísticas rolling

        Returns:
            X_3d_norm: Tensor normalizado
        """
        n_sequences, lookback, n_features = X_3d.shape
        X_3d_norm = np.zeros_like(X_3d)

        # Para cada secuencia
        for seq_idx in range(n_sequences):
            # Para cada timestep en la secuencia
            for t in range(lookback):
                # Índice global en la serie temporal completa
                global_idx = seq_idx + t

                # Calcular ventana pasada (solo hasta este punto)
                inicio = max(0, global_idx - ventana + 1)

                # IMPORTANTE: Necesitamos acceso a datos 2D originales
                # Por ahora, normalizar dentro de la secuencia actual
                if t < 5:
                    # Primeros puntos: usar disponibles
                    window = X_3d[seq_idx, :t+1, :]
                else:
                    # Usar ventana de 5 timesteps previos
                    window = X_3d[seq_idx, max(0, t-5):t+1, :]

                # Normalizar
                mean = np.nanmean(window, axis=0)
                std = np.nanstd(window, axis=0) + 1e-8
                X_3d_norm[seq_idx, t, :] = (X_3d[seq_idx, t, :] - mean) / std

        return X_3d_norm

    def analizar_un_par(self, par: str) -> dict:
        """
        Ejecuta análisis multi-método para un par.

        Args:
            par: Nombre del par (ej: 'EUR_USD')

        Returns:
            Diccionario con estadísticas del análisis
        """
        logger.info("\n" + "="*80)
        logger.info(f"ANALIZANDO PAR: {par}")
        logger.info("="*80)

        inicio = datetime.now()

        try:
            # Cargar y preparar datos
            X, y, nombres_features, df_completo = self.cargar_features_y_preparar_datos(par)

            if X is None:
                return {
                    'par': par,
                    'exito': False,
                    'error': 'Error al cargar/preparar datos',
                    'tiempo_segundos': 0
                }

            resultados_par = {
                'par': par,
                'exito': True,
                'n_features': len(nombres_features),
                'n_muestras': len(y),
                'analisis': {}
            }

            # ==========================================
            # A) ANÁLISIS ESTADÍSTICO
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("A) ANÁLISIS ESTADÍSTICO")
            logger.info("="*80)

            analizador_est = AnalizadorEstadistico(X, y, nombres_features)

            # Information Coefficient
            df_ic = analizador_est.calcular_information_coefficient(
                metodo='spearman',
                correccion_multipletests=True
            )

            # Guardar top features por IC
            output_ic = self.output_dir / f"{par}_{self.timeframe}_analisis_IC.csv"
            df_ic.to_csv(output_ic, index=False)
            logger.info(f"✓ IC guardado: {output_ic}")

            # Información mutua
            df_mi = analizador_est.calcular_informacion_mutua()
            output_mi = self.output_dir / f"{par}_{self.timeframe}_analisis_MI.csv"
            df_mi.to_csv(output_mi, index=False)
            logger.info(f"✓ MI guardado: {output_mi}")

            # Regresión Lasso (selección automática)
            resultado_lasso = analizador_est.regresion_lasso()
            features_lasso_seleccionados = resultado_lasso['features_seleccionados']

            # PCA
            resultado_pca = analizador_est.analisis_pca(n_components=50)

            resultados_par['analisis']['estadistico'] = {
                'n_features_significativos_ic': len(df_ic[df_ic['p_value_corrected'] < 0.05]),
                'ic_maximo': float(df_ic['IC'].max()),
                'n_features_lasso': len(features_lasso_seleccionados),
                'r2_lasso': float(resultado_lasso['R2']),
                'pca_varianza_explicada_50': float(resultado_pca['varianza_acumulada'][49])
            }

            # ==========================================
            # SELECCIÓN ROBUSTA DE FEATURES
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("SELECCIÓN ROBUSTA DE FEATURES (ANTI-OVERFITTING)")
            logger.info("="*80)

            # Advertir si señal es muy débil
            ic_max = df_ic['abs_IC'].max()
            r2_lasso = resultado_lasso['R2']

            if ic_max < 0.03:
                logger.warning(f"⚠️  SEÑAL MUY DÉBIL DETECTADA:")
                logger.warning(f"   IC máximo: {ic_max:.4f} (< 0.03)")
                logger.warning(f"   R² Lasso: {r2_lasso:.6f}")
                logger.warning(f"   Esto es NORMAL en forex - mercado muy eficiente")
                logger.warning(f"   Los modelos ML pueden overfittear fácilmente")

            # Seleccionar features robustos
            X_seleccionado, nombres_seleccionados, df_seleccion = self.seleccionar_features_robustos(
                X, y, nombres_features,
                df_ic, df_mi,
                ic_threshold=0.02,
                p_value_threshold=0.001,
                max_features=100
            )

            # Guardar features seleccionados
            output_seleccion = self.output_dir / f"{par}_{self.timeframe}_features_seleccionados.csv"
            df_seleccion.to_csv(output_seleccion, index=False)
            logger.info(f"\n✓ Features seleccionados guardados: {output_seleccion}")

            resultados_par['n_features_seleccionados'] = len(nombres_seleccionados)

            # ==========================================
            # B) MACHINE LEARNING (con features seleccionados)
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("B) MACHINE LEARNING (con features seleccionados)")
            logger.info("="*80)

            analizador_ml = AnalizadorML(X_seleccionado, y, nombres_seleccionados)

            # Random Forest
            resultado_rf = analizador_ml.entrenar_random_forest(
                n_estimators=100,
                max_depth=10,
                tarea='regresion',
                test_size=0.2
            )

            # Guardar feature importance
            df_importance = resultado_rf['feature_importance']
            output_rf = self.output_dir / f"{par}_{self.timeframe}_analisis_RF_importance.csv"
            df_importance.to_csv(output_rf, index=False)
            logger.info(f"✓ RF importance guardado: {output_rf}")

            # Gradient Boosting
            resultado_gb = analizador_ml.entrenar_gradient_boosting(
                n_estimators=100,
                max_depth=5,
                tarea='regresion',
                test_size=0.2
            )

            # Advertir si R² es muy bajo o negativo
            r2_rf = resultado_rf['metricas']['r2_test']
            r2_gb = resultado_gb['metricas']['r2_test']

            if r2_rf < 0.01 or r2_gb < 0.01:
                logger.warning(f"\n⚠️  R² MUY BAJO EN MACHINE LEARNING:")
                logger.warning(f"   Random Forest R² test: {r2_rf:.6f}")
                logger.warning(f"   Gradient Boosting R² test: {r2_gb:.6f}")
                logger.warning(f"   Señal predictiva muy débil - normal en forex eficiente")
                logger.warning(f"   NO USAR estos modelos para trading real sin validación adicional")

            if r2_rf < 0 or r2_gb < 0:
                logger.error(f"\n❌ R² NEGATIVO - OVERFITTING SEVERO DETECTADO:")
                logger.error(f"   Random Forest R² test: {r2_rf:.6f}")
                logger.error(f"   Gradient Boosting R² test: {r2_gb:.6f}")
                logger.error(f"   El modelo es PEOR que predecir la media")
                logger.error(f"   Reduce max_depth o usa menos features")

            resultados_par['analisis']['machine_learning'] = {
                'r2_random_forest': float(r2_rf),
                'r2_gradient_boosting': float(r2_gb),
                'top_10_features_rf': resultado_rf['feature_importance'].head(10)['Feature'].tolist(),
                'overfitting_detectado': bool(r2_rf < 0 or r2_gb < 0),
                'senal_muy_debil': bool(r2_rf < 0.01 or r2_gb < 0.01)
            }

            # ==========================================
            # C) MÉTODOS DE FÍSICA
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("C) MÉTODOS DE FÍSICA")
            logger.info("="*80)

            # Usar precio de cierre para análisis de física
            if 'close' in df_completo.columns:
                serie_precio = df_completo['close'].values
            else:
                serie_precio = df_completo.iloc[:, 0].values

            # Ornstein-Uhlenbeck
            resultado_ou = MetodosFisica.ornstein_uhlenbeck(serie_precio)

            resultados_par['analisis']['fisica'] = {
                'mean_reversion_detectado': resultado_ou['mean_reversion'],
                'theta': float(resultado_ou['theta']),
                'half_life': float(resultado_ou['half_life']) if resultado_ou['half_life'] != np.inf else None,
                'ou_r2': float(resultado_ou['r_squared'])
            }

            # ==========================================
            # D) DEEP LEARNING (OPCIONAL)
            # ==========================================
            if self.usar_deep_learning:
                logger.info("\n" + "="*80)
                logger.info("D) DEEP LEARNING")
                logger.info("="*80)

                try:
                    from analisis_multi_metodo.deep_learning import ModelosDeepLearning

                    # Parámetros
                    LOOKBACK = 10  # Reducido de 20 a 10 para reducir memoria
                    TEST_SIZE = 0.2
                    EPOCHS = 50
                    BATCH_SIZE = 64
                    EARLY_STOPPING_PATIENCE = 10

                    logger.info(f"\n📌 CONFIGURACIÓN DEEP LEARNING:")
                    logger.info(f"   Lookback: {LOOKBACK} (reducido para optimizar memoria)")
                    logger.info(f"   Features: {len(nombres_seleccionados)} (selección robusta)")
                    logger.info(f"   Batch size: {BATCH_SIZE}")
                    logger.info(f"   Epochs máx: {EPOCHS}")

                    # Preparar datos 3D para modelos secuenciales (con features seleccionados)
                    logger.info(f"\nPreparando datos secuenciales (lookback={LOOKBACK})...")
                    # IMPORTANTE: Deep Learning requiere normalización obligatoria
                    logger.info("Preparando datos 3D con normalización point-in-time...")
                    X_3d, y_3d = self.preparar_datos_secuenciales(
                        X_seleccionado, y,  # Usar features seleccionados
                        lookback=LOOKBACK,
                        normalizar=True  # Obligatorio para DL
                    )

                    if X_3d is None:
                        logger.error("No se pudieron preparar datos 3D")
                        resultados_par['analisis']['deep_learning'] = {'error': 'Datos insuficientes'}
                    else:
                        # Split train/test
                        split_idx = int(len(X_seleccionado) * (1 - TEST_SIZE))
                        split_idx_3d = int(len(X_3d) * (1 - TEST_SIZE))

                        # Datos 2D para MLP - NORMALIZAR usando StandardScaler
                        logger.info("Normalizando datos 2D para MLP...")
                        from sklearn.preprocessing import StandardScaler
                        scaler_2d = StandardScaler()

                        X_train_2d = X_seleccionado[:split_idx]
                        X_test_2d = X_seleccionado[split_idx:]
                        y_train_2d = y[:split_idx]
                        y_test_2d = y[split_idx:]

                        # Ajustar scaler solo con train, aplicar a ambos
                        X_train_2d = scaler_2d.fit_transform(X_train_2d)
                        X_test_2d = scaler_2d.transform(X_test_2d)

                        logger.info(f"✓ Datos 2D normalizados (mean≈0, std≈1)")
                        logger.info(f"  Train mean: {X_train_2d.mean():.4f}, std: {X_train_2d.std():.4f}")
                        logger.info(f"  Test mean: {X_test_2d.mean():.4f}, std: {X_test_2d.std():.4f}")

                        # Datos 3D para CNN/LSTM/Transformer (ya normalizados)
                        X_train_3d, X_test_3d = X_3d[:split_idx_3d], X_3d[split_idx_3d:]
                        y_train_3d, y_test_3d = y_3d[:split_idx_3d], y_3d[split_idx_3d:]

                        logger.info(f"\nShapes:")
                        logger.info(f"  Train 2D: {X_train_2d.shape}, Test 2D: {X_test_2d.shape}")
                        logger.info(f"  Train 3D: {X_train_3d.shape}, Test 3D: {X_test_3d.shape}")

                        # Inicializar
                        dl = ModelosDeepLearning()
                        resultados_dl = {}

                        # A) MLP (Multilayer Perceptron)
                        logger.info("\n" + "-"*80)
                        logger.info("A) MLP (Multilayer Perceptron)")
                        logger.info("-"*80)
                        try:
                            modelo_mlp = dl.crear_mlp(
                                n_features=X_train_2d.shape[1],
                                capas_ocultas=[128, 64, 32],
                                dropout=0.3,
                                learning_rate=0.001
                            )

                            resultado_mlp = dl.entrenar_modelo(
                                modelo_mlp,
                                X_train_2d.astype(np.float32),
                                y_train_2d.astype(np.float32),
                                X_test_2d.astype(np.float32),
                                y_test_2d.astype(np.float32),
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                early_stopping_patience=EARLY_STOPPING_PATIENCE
                            )

                            # Calcular R² out-of-sample
                            from sklearn.metrics import r2_score
                            y_pred_test_mlp = modelo_mlp.predict(X_test_2d.astype(np.float32), verbose=0).flatten()
                            r2_test_mlp = r2_score(y_test_2d, y_pred_test_mlp)

                            logger.info(f"   📊 R² TEST: {r2_test_mlp:.6f}")

                            if r2_test_mlp < 0:
                                logger.warning(f"   ⚠️  R² negativo - modelo peor que predecir la media")
                            elif r2_test_mlp < 0.01:
                                logger.warning(f"   ⚠️  R² muy bajo - señal predictiva débil")

                            # Extraer feature importance del MLP
                            # Usar magnitud de pesos de la capa de entrada
                            try:
                                import tensorflow as tf
                                primera_capa = modelo_mlp.layers[0]
                                pesos_entrada = primera_capa.get_weights()[0]  # Shape: (n_features, n_neuronas)

                                # Calcular importancia como magnitud promedio absoluta de pesos
                                importancia_mlp = np.mean(np.abs(pesos_entrada), axis=1)

                                # Crear DataFrame de importancia
                                df_importancia_mlp = pd.DataFrame({
                                    'feature': nombres_features,
                                    'importancia': importancia_mlp
                                })
                                df_importancia_mlp = df_importancia_mlp.sort_values('importancia', ascending=False)

                                # Guardar top features
                                top_100_mlp = df_importancia_mlp.head(100)['feature'].tolist()

                                logger.info(f"✓ Feature importance MLP extraída: {len(top_100_mlp)} top features")

                            except Exception as e_imp:
                                logger.warning(f"No se pudo extraer feature importance de MLP: {e_imp}")
                                top_100_mlp = []

                            resultados_dl['mlp'] = {
                                'val_loss': float(resultado_mlp['val_loss']),
                                'val_mae': float(resultado_mlp['val_mae']),
                                'train_loss': float(resultado_mlp['train_loss']),
                                'r2_test': float(r2_test_mlp),
                                'top_features': top_100_mlp
                            }

                            logger.info(f"✓ MLP completado: Val Loss={resultado_mlp['val_loss']:.6f}, R² Test={r2_test_mlp:.6f}")
                        except Exception as e:
                            logger.error(f"Error en MLP: {e}")
                            resultados_dl['mlp'] = {'error': str(e)}

                        # B) CNN (Convolutional Neural Network)
                        logger.info("\n" + "-"*80)
                        logger.info("B) CNN (Convolutional Neural Network)")
                        logger.info("-"*80)
                        try:
                            modelo_cnn = dl.crear_cnn(
                                lookback=LOOKBACK,
                                n_features=X_train_3d.shape[2],
                                filtros=[32, 64],
                                kernel_sizes=[5, 3],
                                pool_sizes=[2, 2],
                                dropout=0.3,
                                learning_rate=0.001
                            )

                            resultado_cnn = dl.entrenar_modelo(
                                modelo_cnn,
                                X_train_3d,
                                y_train_3d,
                                X_test_3d,
                                y_test_3d,
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                early_stopping_patience=EARLY_STOPPING_PATIENCE
                            )

                            # Calcular R² out-of-sample para CNN
                            y_pred_test_cnn = modelo_cnn.predict(X_test_3d, verbose=0).flatten()
                            r2_test_cnn = r2_score(y_test_3d, y_pred_test_cnn)

                            logger.info(f"   📊 R² TEST: {r2_test_cnn:.6f}")

                            if r2_test_cnn < 0:
                                logger.warning(f"   ⚠️  R² negativo - modelo peor que predecir la media")
                            elif r2_test_cnn < 0.01:
                                logger.warning(f"   ⚠️  R² muy bajo - señal predictiva débil")

                            resultados_dl['cnn'] = {
                                'val_loss': float(resultado_cnn['val_loss']),
                                'val_mae': float(resultado_cnn['val_mae']),
                                'train_loss': float(resultado_cnn['train_loss']),
                                'r2_test': float(r2_test_cnn)
                            }

                            logger.info(f"✓ CNN completado: Val Loss={resultado_cnn['val_loss']:.6f}, R² Test={r2_test_cnn:.6f}")
                        except Exception as e:
                            logger.error(f"Error en CNN: {e}")
                            resultados_dl['cnn'] = {'error': str(e)}

                        # C) LSTM (Long Short-Term Memory)
                        logger.info("\n" + "-"*80)
                        logger.info("C) LSTM (Long Short-Term Memory)")
                        logger.info("-"*80)
                        try:
                            modelo_lstm = dl.crear_lstm(
                                lookback=LOOKBACK,
                                n_features=X_train_3d.shape[2],
                                lstm_units=[64, 32],
                                dropout=0.3,
                                learning_rate=0.001
                            )

                            resultado_lstm = dl.entrenar_modelo(
                                modelo_lstm,
                                X_train_3d,
                                y_train_3d,
                                X_test_3d,
                                y_test_3d,
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                early_stopping_patience=EARLY_STOPPING_PATIENCE
                            )

                            # Calcular R² out-of-sample para LSTM
                            y_pred_test_lstm = modelo_lstm.predict(X_test_3d, verbose=0).flatten()
                            r2_test_lstm = r2_score(y_test_3d, y_pred_test_lstm)

                            logger.info(f"   📊 R² TEST: {r2_test_lstm:.6f}")

                            if r2_test_lstm < 0:
                                logger.warning(f"   ⚠️  R² negativo - modelo peor que predecir la media")
                            elif r2_test_lstm < 0.01:
                                logger.warning(f"   ⚠️  R² muy bajo - señal predictiva débil")

                            resultados_dl['lstm'] = {
                                'val_loss': float(resultado_lstm['val_loss']),
                                'val_mae': float(resultado_lstm['val_mae']),
                                'train_loss': float(resultado_lstm['train_loss']),
                                'r2_test': float(r2_test_lstm)
                            }

                            logger.info(f"✓ LSTM completado: Val Loss={resultado_lstm['val_loss']:.6f}, R² Test={r2_test_lstm:.6f}")
                        except Exception as e:
                            logger.error(f"Error en LSTM: {e}")
                            resultados_dl['lstm'] = {'error': str(e)}

                        # Guardar resultados DL
                        resultados_par['analisis']['deep_learning'] = resultados_dl

                        # Guardar resumen en JSON
                        output_dl = self.output_dir / f"{par}_{self.timeframe}_analisis_DL.json"
                        with open(output_dl, 'w') as f:
                            json.dump(resultados_dl, f, indent=2)
                        logger.info(f"\n✓ Resultados DL guardados: {output_dl}")

                except ImportError as e:
                    logger.error(f"TensorFlow no disponible: {e}")
                    logger.error("Instala con: pip install tensorflow")
                    resultados_par['analisis']['deep_learning'] = {'error': 'TensorFlow no instalado'}
                except Exception as e:
                    logger.error(f"Error en Deep Learning: {e}")
                    logger.exception(e)
                    resultados_par['analisis']['deep_learning'] = {'error': str(e)}
            else:
                logger.info("\n⚠️  Deep Learning deshabilitado")
                resultados_par['analisis']['deep_learning'] = None

            # ==========================================
            # GUARDAR RESULTADOS CONSOLIDADOS
            # ==========================================
            output_json = self.output_dir / f"{par}_{self.timeframe}_analisis_completo.json"
            with open(output_json, 'w') as f:
                json.dump(resultados_par, f, indent=2, ensure_ascii=False)

            fin = datetime.now()
            tiempo_total = (fin - inicio).total_seconds()
            resultados_par['tiempo_segundos'] = tiempo_total

            logger.info(f"\n✓ PAR COMPLETADO: {par}")
            logger.info(f"  Tiempo: {tiempo_total:.1f} segundos")
            logger.info(f"  Resultados guardados en: {output_json}")

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
        logger.info("ANÁLISIS MULTI-MÉTODO - TODOS LOS PARES")
        logger.info("="*80)
        logger.info(f"Pares a analizar: {len(self.pares)}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Horizonte predicción: {self.horizonte_prediccion} período(s)")
        logger.info(f"Directorio features: {self.features_dir}")
        logger.info(f"Directorio salida: {self.output_dir}")
        logger.info(f"Limpiar archivos viejos: {'SÍ' if self.limpiar_archivos_viejos else 'NO'}")
        logger.info(f"Hacer backup: {'SÍ' if self.hacer_backup else 'NO'}")
        logger.info(f"Deep Learning: {'SÍ' if self.usar_deep_learning else 'NO'}")
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
        logger.info("RESUMEN FINAL - ANÁLISIS MULTI-MÉTODO")
        logger.info("="*80)

        # Tabla de resultados
        logger.info("\nRESULTADOS POR PAR:")
        logger.info("-" * 80)
        logger.info(f"{'Par':<10} │ {'Exito':<6} │ {'Features':<10} │ {'IC Sig.':<10} │ {'Tiempo (s)':<12}")
        logger.info("-" * 80)

        exitosos = 0
        total_features = 0

        for par in self.pares:
            res = self.resultados[par]

            if res['exito']:
                exitosos += 1
                total_features += res['n_features']
                ic_sig = res['analisis']['estadistico']['n_features_significativos_ic']

                logger.info(
                    f"{par:<10} │ {'✓':<6} │ {res['n_features']:<10,} │ "
                    f"{ic_sig:<10} │ {res['tiempo_segundos']:<12.1f}"
                )
            else:
                logger.info(
                    f"{par:<10} │ {'✗':<6} │ {'N/A':<10} │ {'N/A':<10} │ "
                    f"{res['tiempo_segundos']:<12.1f}"
                )
                logger.info(f"           Error: {res.get('error', 'Desconocido')}")

        logger.info("-" * 80)

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
            logger.info("  → Revisar features con IC significativo")
            logger.info("  → Ejecutar sistema de consenso")
            logger.info("  → Validación rigurosa (walk-forward, bootstrap)")
        else:
            logger.info(f"⚠️  ANÁLISIS COMPLETADO CON ERRORES")
            logger.info(f"  {exitosos}/{len(self.pares)} pares exitosos")
            logger.info(f"  Revisar errores arriba")

        logger.info("="*80)


def main():
    """Función principal."""
    # Configuración
    BASE_DIR = Path(__file__).parent
    FEATURES_DIR = BASE_DIR / 'datos' / 'features'
    OUTPUT_DIR = BASE_DIR / 'datos' / 'analisis_multimetodo'
    TIMEFRAME = 'M15'

    # Opciones de análisis
    HORIZONTE_PREDICCION = 1  # Predecir retorno 1 período adelante
    USAR_DEEP_LEARNING = True  # True = Incluir análisis con DL (requiere TensorFlow)

    # Opciones de limpieza de archivos
    LIMPIAR_ARCHIVOS_VIEJOS = True  # True = Borra archivos viejos antes de iniciar
    HACER_BACKUP = False             # False = NO crea backup (ahorra espacio)

    # Validar que existe el directorio de features
    if not FEATURES_DIR.exists():
        logger.error(f"Directorio de features no encontrado: {FEATURES_DIR}")
        logger.error("Ejecuta primero: python ejecutar_generacion_transformaciones.py")
        return

    # Ejecutar análisis
    ejecutor = EjecutorAnalisisMultimetodo(
        features_dir=FEATURES_DIR,
        output_dir=OUTPUT_DIR,
        timeframe=TIMEFRAME,
        horizonte_prediccion=HORIZONTE_PREDICCION,
        limpiar_archivos_viejos=LIMPIAR_ARCHIVOS_VIEJOS,
        hacer_backup=HACER_BACKUP,
        usar_deep_learning=USAR_DEEP_LEARNING
    )

    ejecutor.ejecutar_todos()


if __name__ == '__main__':
    main()
