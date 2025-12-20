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

        # Calcular retorno futuro
        retorno_futuro = close.shift(-self.horizonte_prediccion).pct_change()
        retorno_futuro = retorno_futuro.shift(-1)  # Alinear con el período actual

        # Agregar retorno al dataframe
        df['retorno_objetivo'] = retorno_futuro

        # Eliminar filas con NaN en retorno objetivo
        df_clean = df.dropna(subset=['retorno_objetivo'])

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

        # Eliminar filas con cualquier NaN
        mask_valid = ~np.any(np.isnan(X), axis=1)
        X = X[mask_valid]
        y = y[mask_valid]

        logger.info(f"\nDatos preparados:")
        logger.info(f"  Features válidos: {len(nombres_features):,}")
        logger.info(f"  Muestras válidas: {len(y):,}")
        logger.info(f"  Retorno medio: {np.mean(y)*100:.4f}%")
        logger.info(f"  Retorno std: {np.std(y)*100:.4f}%")

        return X, y, nombres_features, df_clean

    def preparar_datos_secuenciales(self, X, y, lookback=20):
        """
        Prepara datos 3D para modelos secuenciales (CNN, LSTM, Transformer).

        Args:
            X: Matriz 2D (n_samples, n_features)
            y: Vector target (n_samples,)
            lookback: Número de timesteps de historia

        Returns:
            X_3d, y_3d: Datos en formato 3D (n_sequences, lookback, n_features)
        """
        n_samples, n_features = X.shape
        n_sequences = n_samples - lookback

        if n_sequences <= 0:
            logger.warning(f"No hay suficientes datos para lookback={lookback}")
            return None, None

        # Crear tensor 3D mediante ventanas deslizantes
        X_3d = np.zeros((n_sequences, lookback, n_features), dtype=np.float32)
        y_3d = np.zeros(n_sequences, dtype=np.float32)

        for i in range(n_sequences):
            X_3d[i] = X[i:i+lookback]
            y_3d[i] = y[i+lookback]

        logger.info(f"Datos 3D preparados: {X_3d.shape}")
        return X_3d, y_3d

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
                correccion_multiple=True
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
                'n_features_significativos_ic': len(df_ic[df_ic['significativo'] == True]),
                'ic_maximo': float(df_ic['IC'].max()),
                'n_features_lasso': len(features_lasso_seleccionados),
                'r2_lasso': float(resultado_lasso['r2']),
                'pca_varianza_explicada_50': float(resultado_pca['varianza_explicada_acumulada'][49])
            }

            # ==========================================
            # B) MACHINE LEARNING
            # ==========================================
            logger.info("\n" + "="*80)
            logger.info("B) MACHINE LEARNING")
            logger.info("="*80)

            analizador_ml = AnalizadorML(X, y, nombres_features)

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

            resultados_par['analisis']['machine_learning'] = {
                'r2_random_forest': float(resultado_rf['r2_test']),
                'r2_gradient_boosting': float(resultado_gb['r2_test']),
                'top_10_features_rf': resultado_rf['feature_importance'].head(10)['feature'].tolist()
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
                    LOOKBACK = 20  # Ventana temporal
                    TEST_SIZE = 0.2
                    EPOCHS = 50
                    BATCH_SIZE = 64
                    EARLY_STOPPING_PATIENCE = 10

                    # Preparar datos 3D para modelos secuenciales
                    logger.info(f"Preparando datos secuenciales (lookback={LOOKBACK})...")
                    X_3d, y_3d = self.preparar_datos_secuenciales(X, y, lookback=LOOKBACK)

                    if X_3d is None:
                        logger.error("No se pudieron preparar datos 3D")
                        resultados_par['analisis']['deep_learning'] = {'error': 'Datos insuficientes'}
                    else:
                        # Split train/test
                        split_idx = int(len(X) * (1 - TEST_SIZE))
                        split_idx_3d = int(len(X_3d) * (1 - TEST_SIZE))

                        # Datos 2D para MLP
                        X_train_2d, X_test_2d = X[:split_idx], X[split_idx:]
                        y_train_2d, y_test_2d = y[:split_idx], y[split_idx:]

                        # Datos 3D para CNN/LSTM/Transformer
                        X_train_3d, X_test_3d = X_3d[:split_idx_3d], X_3d[split_idx_3d:]
                        y_train_3d, y_test_3d = y_3d[:split_idx_3d], y_3d[split_idx_3d:]

                        logger.info(f"Train 2D: {X_train_2d.shape}, Test 2D: {X_test_2d.shape}")
                        logger.info(f"Train 3D: {X_train_3d.shape}, Test 3D: {X_test_3d.shape}")

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

                            resultados_dl['mlp'] = {
                                'val_loss': float(resultado_mlp['val_loss']),
                                'val_mae': float(resultado_mlp['val_mae']),
                                'train_loss': float(resultado_mlp['train_loss'])
                            }

                            logger.info(f"✓ MLP completado: Val Loss={resultado_mlp['val_loss']:.6f}")
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

                            resultados_dl['cnn'] = {
                                'val_loss': float(resultado_cnn['val_loss']),
                                'val_mae': float(resultado_cnn['val_mae']),
                                'train_loss': float(resultado_cnn['train_loss'])
                            }

                            logger.info(f"✓ CNN completado: Val Loss={resultado_cnn['val_loss']:.6f}")
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

                            resultados_dl['lstm'] = {
                                'val_loss': float(resultado_lstm['val_loss']),
                                'val_mae': float(resultado_lstm['val_mae']),
                                'train_loss': float(resultado_lstm['train_loss'])
                            }

                            logger.info(f"✓ LSTM completado: Val Loss={resultado_lstm['val_loss']:.6f}")
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
