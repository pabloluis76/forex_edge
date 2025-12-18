"""
EJECUTAR ESTRUCTURA MATRICIAL Y TENSORIAL
==========================================

Procesa los features generados y crea las estructuras matriciales/tensoriales:

1. MATRIZ 2D: Transformaciones × Tiempo
   - Shape: (n_observaciones, n_features)
   - Para ML tradicional: RF, XGBoost, etc.

2. TENSOR 3D: Secuencias × Features × Tiempo
   - Shape: (n_sequences, seq_length, n_features)
   - Para modelos secuenciales: LSTM, GRU

3. TENSOR 4D: Activos × Secuencias × Features × Tiempo
   - Shape: (n_activos, n_sequences, seq_length, n_features)
   - Para modelos multi-activo

4. NORMALIZACIÓN POINT-IN-TIME
   - Z-score rolling
   - Rank rolling
   - Sin look-ahead bias

Autor: Sistema de Edge-Finding Forex
Fecha: 2025-12-17
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EjecutorEstructuraMatricialTensorial:
    """
    Ejecuta la creación de estructuras matriciales y tensoriales
    para todos los pares procesados.
    """

    def __init__(
        self,
        features_dir: Path,
        output_dir: Path,
        pares: List[str]
    ):
        """
        Inicializa el ejecutor.

        Args:
            features_dir: Directorio con features generados (.parquet)
            output_dir: Directorio para guardar estructuras
            pares: Lista de pares a procesar
        """
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.pares = pares

        # Crear subdirectorios
        self.matriz_2d_dir = self.output_dir / 'matriz_2d'
        self.tensor_3d_dir = self.output_dir / 'tensor_3d'
        self.tensor_4d_dir = self.output_dir / 'tensor_4d'
        self.normalizacion_dir = self.output_dir / 'normalizacion'

        for dir_path in [self.matriz_2d_dir, self.tensor_3d_dir,
                         self.tensor_4d_dir, self.normalizacion_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Estadísticas
        self.resultados = {}
        self.tiempo_inicio = None
        self.tiempo_fin = None

    def cargar_features(self, par: str) -> Optional[pd.DataFrame]:
        """
        Carga features de un par.

        Args:
            par: Nombre del par

        Returns:
            DataFrame con features o None si error
        """
        file_path = self.features_dir / f"{par}_M15_features.parquet"

        if not file_path.exists():
            logger.error(f"Archivo no encontrado: {file_path}")
            return None

        try:
            logger.info(f"Cargando features: {par}")
            df = pd.read_parquet(file_path)
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Período: {df.index[0]} → {df.index[-1]}")
            return df
        except Exception as e:
            logger.error(f"Error cargando {par}: {e}")
            return None

    def crear_matriz_2d(self, par: str, df_features: pd.DataFrame) -> Dict:
        """
        Crea matriz 2D (n_observaciones × n_features).

        Args:
            par: Nombre del par
            df_features: DataFrame con features

        Returns:
            Diccionario con estadísticas
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"MATRIZ 2D - {par}")
        logger.info(f"{'='*70}")

        inicio = datetime.now()

        try:
            # Eliminar NaN
            df_clean = df_features.dropna()
            pct_removed = (1 - len(df_clean) / len(df_features)) * 100

            logger.info(f"  Filas con NaN eliminadas: {pct_removed:.2f}%")
            logger.info(f"  Shape final: {df_clean.shape}")

            # Convertir a numpy array (ML convention: rows=samples, cols=features)
            X = df_clean.values
            timestamps = df_clean.index

            logger.info(f"  Matriz X: {X.shape}")
            logger.info(f"  Tipo: {X.dtype}")
            logger.info(f"  Memoria: {X.nbytes / (1024**2):.1f} MB")

            # Guardar
            output_file = self.matriz_2d_dir / f"{par}_matriz_2d.npz"

            np.savez_compressed(
                output_file,
                X=X,
                feature_names=df_clean.columns.values,
                timestamps=timestamps.values
            )

            fin = datetime.now()
            tiempo = (fin - inicio).total_seconds()

            logger.info(f"  ✓ Guardado: {output_file}")
            logger.info(f"  Tamaño archivo: {output_file.stat().st_size / (1024**2):.1f} MB")
            logger.info(f"  Tiempo: {tiempo:.1f}s")

            return {
                'par': par,
                'exito': True,
                'shape': X.shape,
                'memoria_mb': X.nbytes / (1024**2),
                'archivo_mb': output_file.stat().st_size / (1024**2),
                'tiempo_s': tiempo,
                'pct_nan_removed': pct_removed
            }

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            return {
                'par': par,
                'exito': False,
                'error': str(e)
            }

    def crear_secuencias_3d(
        self,
        par: str,
        df_features: pd.DataFrame,
        seq_length: int = 50
    ) -> Dict:
        """
        Crea tensor 3D para modelos secuenciales (LSTM, GRU).

        Shape: (n_sequences, seq_length, n_features)

        Args:
            par: Nombre del par
            df_features: DataFrame con features
            seq_length: Longitud de cada secuencia (default: 50 velas)

        Returns:
            Diccionario con estadísticas
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"TENSOR 3D SECUENCIAL - {par}")
        logger.info(f"{'='*70}")

        inicio = datetime.now()

        try:
            # Limpiar NaN
            df_clean = df_features.dropna()

            # Convertir a numpy
            X_2d = df_clean.values
            n_obs, n_features = X_2d.shape

            # Calcular número de secuencias
            n_sequences = n_obs - seq_length + 1

            if n_sequences <= 0:
                raise ValueError(f"No hay suficientes observaciones para seq_length={seq_length}")

            logger.info(f"  Observaciones: {n_obs}")
            logger.info(f"  Features: {n_features}")
            logger.info(f"  Longitud secuencia: {seq_length}")
            logger.info(f"  Secuencias a crear: {n_sequences}")

            # Crear tensor 3D
            X_3d = np.zeros((n_sequences, seq_length, n_features), dtype=np.float32)

            for i in range(n_sequences):
                X_3d[i] = X_2d[i:i+seq_length]

            logger.info(f"  Tensor 3D: {X_3d.shape}")
            logger.info(f"  Tipo: {X_3d.dtype}")
            logger.info(f"  Memoria: {X_3d.nbytes / (1024**2):.1f} MB")

            # Guardar
            output_file = self.tensor_3d_dir / f"{par}_tensor_3d_seq{seq_length}.npz"

            np.savez_compressed(
                output_file,
                X_3d=X_3d,
                feature_names=df_clean.columns.values,
                timestamps=df_clean.index[seq_length-1:].values,  # Timestamp de cada secuencia
                seq_length=seq_length
            )

            fin = datetime.now()
            tiempo = (fin - inicio).total_seconds()

            logger.info(f"  ✓ Guardado: {output_file}")
            logger.info(f"  Tamaño archivo: {output_file.stat().st_size / (1024**2):.1f} MB")
            logger.info(f"  Tiempo: {tiempo:.1f}s")

            return {
                'par': par,
                'exito': True,
                'shape': X_3d.shape,
                'memoria_mb': X_3d.nbytes / (1024**2),
                'archivo_mb': output_file.stat().st_size / (1024**2),
                'tiempo_s': tiempo,
                'seq_length': seq_length
            }

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            return {
                'par': par,
                'exito': False,
                'error': str(e)
            }

    def normalizar_point_in_time(
        self,
        par: str,
        df_features: pd.DataFrame,
        window: int = 200
    ) -> Dict:
        """
        Normalización point-in-time (sin look-ahead bias).

        Métodos:
        - Z-score rolling: (x - μ_rolling) / σ_rolling
        - Rank rolling: percentile en ventana móvil

        Args:
            par: Nombre del par
            df_features: DataFrame con features
            window: Ventana para normalización (default: 200)

        Returns:
            Diccionario con estadísticas
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"NORMALIZACIÓN POINT-IN-TIME - {par}")
        logger.info(f"{'='*70}")

        inicio = datetime.now()

        try:
            logger.info(f"  Ventana de normalización: {window}")

            # Z-score rolling
            logger.info(f"  Calculando Z-score rolling...")
            mean_rolling = df_features.rolling(window=window, min_periods=window).mean()
            std_rolling = df_features.rolling(window=window, min_periods=window).std()

            df_zscore = (df_features - mean_rolling) / std_rolling

            # Eliminar primeras filas (no hay suficiente historia)
            df_zscore_clean = df_zscore.iloc[window:]

            logger.info(f"  Z-score shape: {df_zscore_clean.shape}")

            # Rank rolling (percentile)
            logger.info(f"  Calculando Rank rolling...")

            def rolling_rank(series, window):
                """Calcula rank percentile en ventana móvil."""
                ranks = series.rolling(window=window, min_periods=window).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                    raw=False
                )
                return ranks

            df_rank = df_features.apply(lambda col: rolling_rank(col, window))
            df_rank_clean = df_rank.iloc[window:]

            logger.info(f"  Rank shape: {df_rank_clean.shape}")

            # Guardar
            output_zscore = self.normalizacion_dir / f"{par}_zscore_w{window}.parquet"
            output_rank = self.normalizacion_dir / f"{par}_rank_w{window}.parquet"

            df_zscore_clean.to_parquet(output_zscore, compression='snappy')
            df_rank_clean.to_parquet(output_rank, compression='snappy')

            fin = datetime.now()
            tiempo = (fin - inicio).total_seconds()

            logger.info(f"  ✓ Z-score guardado: {output_zscore}")
            logger.info(f"  ✓ Rank guardado: {output_rank}")
            logger.info(f"  Tiempo: {tiempo:.1f}s")

            return {
                'par': par,
                'exito': True,
                'shape': df_zscore_clean.shape,
                'window': window,
                'tiempo_s': tiempo,
                'archivo_zscore': str(output_zscore),
                'archivo_rank': str(output_rank)
            }

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            return {
                'par': par,
                'exito': False,
                'error': str(e)
            }

    def procesar_un_par(self, par: str, seq_length: int = 50, norm_window: int = 200):
        """
        Procesa un par completo: matriz 2D + tensor 3D + normalización.

        Args:
            par: Nombre del par
            seq_length: Longitud de secuencias para tensor 3D
            norm_window: Ventana para normalización
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESANDO: {par}")
        logger.info(f"{'='*80}")

        # Cargar features
        df_features = self.cargar_features(par)
        if df_features is None:
            self.resultados[par] = {'exito': False, 'error': 'No se pudieron cargar features'}
            return

        resultados_par = {}

        # 1. Matriz 2D
        resultados_par['matriz_2d'] = self.crear_matriz_2d(par, df_features)

        # 2. Tensor 3D
        resultados_par['tensor_3d'] = self.crear_secuencias_3d(par, df_features, seq_length)

        # 3. Normalización
        resultados_par['normalizacion'] = self.normalizar_point_in_time(par, df_features, norm_window)

        self.resultados[par] = resultados_par

    def ejecutar_todos(self, seq_length: int = 50, norm_window: int = 200):
        """
        Ejecuta el procesamiento para todos los pares.

        Args:
            seq_length: Longitud de secuencias para tensor 3D
            norm_window: Ventana para normalización
        """
        self.tiempo_inicio = datetime.now()

        logger.info(f"\n{'='*80}")
        logger.info("ESTRUCTURA MATRICIAL Y TENSORIAL - TODOS LOS PARES")
        logger.info(f"{'='*80}")
        logger.info(f"Pares: {len(self.pares)}")
        logger.info(f"Directorio features: {self.features_dir}")
        logger.info(f"Directorio salida: {self.output_dir}")
        logger.info(f"Seq length: {seq_length}")
        logger.info(f"Norm window: {norm_window}")
        logger.info(f"Inicio: {self.tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")

        # Procesar cada par
        for i, par in enumerate(self.pares, 1):
            logger.info(f"\n[{i}/{len(self.pares)}] Procesando: {par}")
            self.procesar_un_par(par, seq_length, norm_window)

        self.tiempo_fin = datetime.now()

        # Imprimir resumen
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen final."""
        tiempo_total = (self.tiempo_fin - self.tiempo_inicio).total_seconds()

        logger.info(f"\n{'='*80}")
        logger.info("RESUMEN FINAL - ESTRUCTURA MATRICIAL Y TENSORIAL")
        logger.info(f"{'='*80}")

        # Tabla de resultados
        logger.info(f"\nRESULTADOS POR PAR:")
        logger.info("-" * 80)

        exitosos = 0

        for par in self.pares:
            res = self.resultados.get(par, {})

            matriz_ok = res.get('matriz_2d', {}).get('exito', False)
            tensor_ok = res.get('tensor_3d', {}).get('exito', False)
            norm_ok = res.get('normalizacion', {}).get('exito', False)

            if matriz_ok and tensor_ok and norm_ok:
                exitosos += 1
                estado = "✓"
            else:
                estado = "✗"

            logger.info(f"{par}: {estado} | Matriz 2D: {matriz_ok} | Tensor 3D: {tensor_ok} | Normalización: {norm_ok}")

        logger.info("-" * 80)

        # Estadísticas globales
        logger.info(f"\nESTADÍSTICAS GLOBALES:")
        logger.info(f"  Pares procesados exitosamente: {exitosos}/{len(self.pares)}")
        logger.info(f"  Tiempo total: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
        logger.info(f"  Tiempo promedio por par: {tiempo_total/len(self.pares):.1f}s")

        # Conclusión
        logger.info(f"\n{'='*80}")
        if exitosos == len(self.pares):
            logger.info("✓ ESTRUCTURA MATRICIAL/TENSORIAL COMPLETADA EXITOSAMENTE")
            logger.info(f"  Todos los {len(self.pares)} pares procesados")
            logger.info(f"\nARCHIVOS GENERADOS:")
            logger.info(f"  Matrices 2D:    {self.matriz_2d_dir}")
            logger.info(f"  Tensores 3D:    {self.tensor_3d_dir}")
            logger.info(f"  Normalización:  {self.normalizacion_dir}")
            logger.info(f"\nPRÓXIMO PASO:")
            logger.info(f"  → Análisis multi-método")
            logger.info(f"  → Sistema de consenso")
            logger.info(f"  → Validación rigurosa")
        else:
            logger.info(f"⚠️  PROCESAMIENTO COMPLETADO CON ERRORES")
            logger.info(f"  {exitosos}/{len(self.pares)} pares exitosos")

        logger.info(f"{'='*80}")


def main():
    """Función principal."""
    # Configuración
    BASE_DIR = Path(__file__).parent
    FEATURES_DIR = BASE_DIR / 'datos' / 'features'
    OUTPUT_DIR = BASE_DIR / 'datos' / 'estructura_matricial_tensorial'

    PARES = [
        'EUR_USD',
        'GBP_USD',
        'USD_JPY',
        'EUR_JPY',
        'GBP_JPY',
        'AUD_USD'
    ]

    SEQ_LENGTH = 50  # Longitud de secuencias para LSTM/GRU
    NORM_WINDOW = 200  # Ventana para normalización rolling

    # Validar
    if not FEATURES_DIR.exists():
        logger.error(f"Directorio de features no encontrado: {FEATURES_DIR}")
        return

    # Ejecutar
    ejecutor = EjecutorEstructuraMatricialTensorial(
        features_dir=FEATURES_DIR,
        output_dir=OUTPUT_DIR,
        pares=PARES
    )

    ejecutor.ejecutar_todos(
        seq_length=SEQ_LENGTH,
        norm_window=NORM_WINDOW
    )


if __name__ == '__main__':
    main()
