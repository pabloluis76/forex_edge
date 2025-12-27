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
import shutil
from tqdm import tqdm

warnings.filterwarnings('ignore')

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


class EjecutorEstructuraMatricialTensorial:
    """
    Ejecuta la creación de estructuras matriciales y tensoriales
    para todos los pares procesados.
    """

    def __init__(
        self,
        features_dir: Path,
        output_dir: Path,
        pares: List[str],
        timeframes: list = None,
        limpiar_archivos_viejos: bool = True,
        hacer_backup: bool = True
    ):
        """
        Inicializa el ejecutor MULTI-TIMEFRAME.

        Args:
            features_dir: Directorio con features generados (.parquet)
            output_dir: Directorio para guardar estructuras
            pares: Lista de pares a procesar
            timeframes: Lista de timeframes a procesar (default: ['M15', 'H1', 'H4', 'D1'])
            limpiar_archivos_viejos: Si True, borra archivos viejos antes de iniciar
            hacer_backup: Si True, hace backup antes de borrar
        """
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.pares = pares
        self.timeframes = timeframes or ['M15', 'H1', 'H4', 'D1']
        self.limpiar_archivos_viejos = limpiar_archivos_viejos
        self.hacer_backup = hacer_backup

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

    def limpiar_directorio_salida(self):
        """
        Limpia archivos .npz viejos del directorio de salida.
        Opcionalmente hace backup antes de borrar.
        """
        # Buscar archivos de estructuras existentes
        archivos_npz = list(self.matriz_2d_dir.glob("*.npz"))
        archivos_npz += list(self.tensor_3d_dir.glob("*.npz"))
        archivos_npz += list(self.tensor_4d_dir.glob("*.npz"))
        archivos_npz += list(self.normalizacion_dir.glob("*.npz"))

        if not archivos_npz:
            logger.info("No hay archivos viejos para limpiar")
            return

        logger.info(f"\nEncontrados {len(archivos_npz)} archivos viejos:")
        for archivo in archivos_npz[:10]:  # Mostrar solo primeros 10
            logger.info(f"  - {archivo.name}")
        if len(archivos_npz) > 10:
            logger.info(f"  ... y {len(archivos_npz) - 10} más")

        # Hacer backup si está habilitado
        if self.hacer_backup:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.output_dir / f"backup_{timestamp}"

            # Crear subdirectorios en backup
            for subdir in ['matriz_2d', 'tensor_3d', 'tensor_4d', 'normalizacion']:
                (backup_dir / subdir).mkdir(parents=True, exist_ok=True)

            logger.info(f"\nCreando backup en: {backup_dir}")

            for archivo in archivos_npz:
                # Mantener estructura de subdirectorios
                relpath = archivo.relative_to(self.output_dir)
                destino = backup_dir / relpath
                destino.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(archivo, destino)

            logger.info(f"✓ Backup completado: {len(archivos_npz)} archivos")

        # Borrar archivos viejos
        logger.info("\nBorrando archivos viejos...")
        for archivo in archivos_npz:
            archivo.unlink()

        logger.info(f"✓ Limpieza completada: {len(archivos_npz)} archivos eliminados\n")

    def cargar_features(self, par: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Carga features de un par para un timeframe específico.

        Args:
            par: Nombre del par
            timeframe: Timeframe (ej: 'M15', 'H1', 'H4', 'D1')

        Returns:
            DataFrame con features o None si error
        """
        file_path = self.features_dir / f"{par}_{timeframe}_features.parquet"

        if not file_path.exists():
            logger.error(f"Archivo no encontrado: {file_path}")
            return None

        try:
            logger.info(f"Cargando features: {par} ({timeframe})")
            df = pd.read_parquet(file_path)
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Período: {df.index[0]} → {df.index[-1]}")
            return df
        except Exception as e:
            logger.error(f"Error cargando {par} ({timeframe}): {e}")
            return None

    def crear_matriz_2d(self, par: str, timeframe: str, df_features: pd.DataFrame) -> Dict:
        """
        Crea matriz 2D (n_observaciones × n_features).

        Args:
            par: Nombre del par
            timeframe: Timeframe (ej: 'M15', 'H1', 'H4', 'D1')
            df_features: DataFrame con features

        Returns:
            Diccionario con estadísticas
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"MATRIZ 2D - {par} ({timeframe})")
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
            output_file = self.matriz_2d_dir / f"{par}_{timeframe}_matriz_2d.npz"

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
        timeframe: str,
        df_features: pd.DataFrame,
        seq_length: int = 50
    ) -> Dict:
        """
        Crea tensor 3D para modelos secuenciales (LSTM, GRU).

        Shape: (n_sequences, seq_length, n_features)

        Args:
            par: Nombre del par
            timeframe: Timeframe (ej: 'M15', 'H1', 'H4', 'D1')
            df_features: DataFrame con features
            seq_length: Longitud de cada secuencia (default: 50 velas)

        Returns:
            Diccionario con estadísticas
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"TENSOR 3D SECUENCIAL - {par} ({timeframe})")
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
            output_file = self.tensor_3d_dir / f"{par}_{timeframe}_tensor_3d_seq{seq_length}.npz"

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
        timeframe: str,
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
            timeframe: Timeframe (ej: 'M15', 'H1', 'H4', 'D1')
            df_features: DataFrame con features
            window: Ventana para normalización (default: 200)

        Returns:
            Diccionario con estadísticas
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"NORMALIZACIÓN POINT-IN-TIME - {par} ({timeframe})")
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
            output_zscore = self.normalizacion_dir / f"{par}_{timeframe}_zscore_w{window}.parquet"
            output_rank = self.normalizacion_dir / f"{par}_{timeframe}_rank_w{window}.parquet"

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

    def procesar_un_par(self, par: str, timeframe: str, seq_length: int = 50, norm_window: int = 200):
        """
        Procesa un par completo: matriz 2D + tensor 3D + normalización.

        Args:
            par: Nombre del par
            timeframe: Timeframe (ej: 'M15', 'H1', 'H4', 'D1')
            seq_length: Longitud de secuencias para tensor 3D
            norm_window: Ventana para normalización
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESANDO: {par} ({timeframe})")
        logger.info(f"{'='*80}")

        # Cargar features
        df_features = self.cargar_features(par, timeframe)
        if df_features is None:
            key = f"{par}_{timeframe}"
            self.resultados[key] = {'exito': False, 'error': 'No se pudieron cargar features'}
            return

        resultados_par = {}

        # 1. Matriz 2D
        resultados_par['matriz_2d'] = self.crear_matriz_2d(par, timeframe, df_features)

        # 2. Tensor 3D
        resultados_par['tensor_3d'] = self.crear_secuencias_3d(par, timeframe, df_features, seq_length)

        # 3. Normalización
        resultados_par['normalizacion'] = self.normalizar_point_in_time(par, timeframe, df_features, norm_window)

        key = f"{par}_{timeframe}"
        self.resultados[key] = resultados_par

    def ejecutar_todos(self, seq_length: int = 50, norm_window: int = 200):
        """
        Ejecuta el procesamiento MULTI-TIMEFRAME para todos los pares.

        Args:
            seq_length: Longitud de secuencias para tensor 3D
            norm_window: Ventana para normalización
        """
        self.tiempo_inicio = datetime.now()

        logger.info(f"\n{'='*80}")
        logger.info("ESTRUCTURA MATRICIAL Y TENSORIAL - MULTI-TIMEFRAME")
        logger.info(f"{'='*80}")
        logger.info(f"Pares: {len(self.pares)}")
        logger.info(f"Timeframes: {self.timeframes}")
        logger.info(f"Directorio features: {self.features_dir}")
        logger.info(f"Directorio salida: {self.output_dir}")
        logger.info(f"Limpiar archivos viejos: {'SÍ' if self.limpiar_archivos_viejos else 'NO'}")
        logger.info(f"Hacer backup: {'SÍ' if self.hacer_backup else 'NO'}")
        logger.info(f"Seq length: {seq_length}")
        logger.info(f"Norm window: {norm_window}")
        logger.info(f"Inicio: {self.tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")

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
            logger.info(f"\n{'='*80}")
            logger.info(f"PROCESANDO TIMEFRAME: {timeframe}")
            logger.info(f"{'='*80}")

            for par in self.pares:
                combinacion_actual += 1
                logger.info(f"\n[{combinacion_actual}/{total_combinaciones}] Procesando: {par} ({timeframe})")
                self.procesar_un_par(par, timeframe, seq_length, norm_window)

        self.tiempo_fin = datetime.now()

        # Imprimir resumen
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen final detallado."""
        tiempo_total = (self.tiempo_fin - self.tiempo_inicio).total_seconds()

        # Recopilar estadísticas (ahora con multi-timeframe)
        exitosos = 0
        total_memoria_mb = 0
        total_archivo_mb = 0
        tiempos = []
        total_combinaciones = len(self.pares) * len(self.timeframes)

        # Iterar sobre todas las combinaciones par_timeframe
        for key, res in self.resultados.items():
            matriz_ok = res.get('matriz_2d', {}).get('exito', False)
            tensor_ok = res.get('tensor_3d', {}).get('exito', False)
            norm_ok = res.get('normalizacion', {}).get('exito', False)

            if matriz_ok and tensor_ok and norm_ok:
                exitosos += 1
                # Sumar memoria de matriz 2D y tensor 3D
                total_memoria_mb += res.get('matriz_2d', {}).get('memoria_mb', 0)
                total_memoria_mb += res.get('tensor_3d', {}).get('memoria_mb', 0)
                total_archivo_mb += res.get('matriz_2d', {}).get('archivo_mb', 0)
                total_archivo_mb += res.get('tensor_3d', {}).get('archivo_mb', 0)

                # Tiempo total por combinación
                tiempo_comb = (res.get('matriz_2d', {}).get('tiempo_s', 0) +
                              res.get('tensor_3d', {}).get('tiempo_s', 0) +
                              res.get('normalizacion', {}).get('tiempo_s', 0))
                tiempos.append(tiempo_comb)

        logger.info("\n" + "="*100)
        logger.info(f"{'RESUMEN FINAL - ESTRUCTURA MATRICIAL Y TENSORIAL':^100}")
        logger.info("="*100)

        # ============================================================
        # SECCIÓN 1: RESUMEN EJECUTIVO
        # ============================================================
        logger.info("\n1. RESUMEN EJECUTIVO")
        logger.info("-" * 100)
        logger.info(f"  Pares:                         {len(self.pares)}")
        logger.info(f"  Timeframes:                    {len(self.timeframes)} ({', '.join(self.timeframes)})")
        logger.info(f"  Combinaciones exitosas:        {exitosos}/{total_combinaciones}")
        logger.info(f"  Tiempo total:                  {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
        logger.info(f"  Tiempo promedio:               {tiempo_total/total_combinaciones:.1f}s por combinación")
        logger.info(f"  Memoria total (en RAM):        {total_memoria_mb:.1f} MB")
        logger.info(f"  Archivos generados:            {total_archivo_mb:.1f} MB")

        # Calcular promedio de compresión
        if total_memoria_mb > 0:
            ratio_compresion = (total_archivo_mb / total_memoria_mb) * 100
            logger.info(f"  Ratio de compresión:           {ratio_compresion:.1f}% (npz comprimido)")

        # ============================================================
        # SECCIÓN 2: TABLA COMPLETA DE RESULTADOS (MULTI-TIMEFRAME)
        # ============================================================
        logger.info(f"\n2. TABLA COMPLETA DE RESULTADOS (MULTI-TIMEFRAME)")
        logger.info("-" * 100)
        logger.info(f"{'Par_TF':<16} {'Estado':<8} {'Shape 2D':<18} {'Shape 3D':<22} {'Mem(MB)':<10} {'Archivo(MB)':<12} {'Tiempo(s)':<10}")
        logger.info("-" * 100)

        for key in sorted(self.resultados.keys()):
            res = self.resultados[key]
            matriz_ok = res.get('matriz_2d', {}).get('exito', False)
            tensor_ok = res.get('tensor_3d', {}).get('exito', False)
            norm_ok = res.get('normalizacion', {}).get('exito', False)

            if matriz_ok and tensor_ok and norm_ok:
                estado = "✓"
                shape_2d = res.get('matriz_2d', {}).get('shape', (0,0))
                shape_3d = res.get('tensor_3d', {}).get('shape', (0,0,0))
                memoria_mb = (res.get('matriz_2d', {}).get('memoria_mb', 0) +
                             res.get('tensor_3d', {}).get('memoria_mb', 0))
                archivo_mb = (res.get('matriz_2d', {}).get('archivo_mb', 0) +
                             res.get('tensor_3d', {}).get('archivo_mb', 0))
                tiempo_comb = (res.get('matriz_2d', {}).get('tiempo_s', 0) +
                              res.get('tensor_3d', {}).get('tiempo_s', 0) +
                              res.get('normalizacion', {}).get('tiempo_s', 0))

                shape_2d_str = f"{shape_2d[0]}×{shape_2d[1]}"
                shape_3d_str = f"{shape_3d[0]}×{shape_3d[1]}×{shape_3d[2]}"

                logger.info(f"{key:<16} {estado:<8} {shape_2d_str:<18} {shape_3d_str:<22} "
                          f"{memoria_mb:<10.1f} {archivo_mb:<12.1f} {tiempo_comb:<10.1f}")
            else:
                estado = "✗"
                error_msg = "Error en procesamiento"
                logger.info(f"{key:<16} {estado:<8} {error_msg}")

        logger.info("-" * 100)

        # ============================================================
        # SECCIÓN 3: ANÁLISIS ESTADÍSTICO DE TIEMPOS
        # ============================================================
        if tiempos:
            logger.info(f"\n3. ANÁLISIS ESTADÍSTICO DE TIEMPOS")
            logger.info("-" * 100)
            logger.info(f"  Mínimo:              {np.min(tiempos):.1f}s")
            logger.info(f"  Máximo:              {np.max(tiempos):.1f}s")
            logger.info(f"  Media:               {np.mean(tiempos):.1f}s")
            logger.info(f"  Mediana:             {np.median(tiempos):.1f}s")
            logger.info(f"  Desviación estándar: {np.std(tiempos):.1f}s")

        # ============================================================
        # SECCIÓN 4: DETALLES POR COMBINACIÓN (MULTI-TIMEFRAME)
        # ============================================================
        logger.info(f"\n4. DETALLES POR COMBINACIÓN (MULTI-TIMEFRAME)")
        logger.info("-" * 100)

        for key in sorted(self.resultados.keys()):
            res = self.resultados[key]
            matriz_ok = res.get('matriz_2d', {}).get('exito', False)
            tensor_ok = res.get('tensor_3d', {}).get('exito', False)
            norm_ok = res.get('normalizacion', {}).get('exito', False)

            if matriz_ok and tensor_ok and norm_ok:
                logger.info(f"\n  {key}:")

                # Matriz 2D
                matriz_2d = res.get('matriz_2d', {})
                shape_2d = matriz_2d.get('shape', (0,0))
                logger.info(f"    Matriz 2D:      {shape_2d[0]:,} obs × {shape_2d[1]:,} features "
                          f"({matriz_2d.get('archivo_mb', 0):.1f} MB)")

                # Tensor 3D
                tensor_3d = res.get('tensor_3d', {})
                shape_3d = tensor_3d.get('shape', (0,0,0))
                seq_len = tensor_3d.get('seq_length', 0)
                logger.info(f"    Tensor 3D:      {shape_3d[0]:,} seq × {seq_len} len × {shape_3d[2]:,} features "
                          f"({tensor_3d.get('archivo_mb', 0):.1f} MB)")

                # Normalización
                norm = res.get('normalizacion', {})
                if norm.get('exito'):
                    window = norm.get('window', 0)
                    logger.info(f"    Normalización:  Z-score + Rank (window={window})")

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
                for i, error in enumerate(log_capture.errors[:10], 1):  # Mostrar máximo 10
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
        logger.info(f"\n  Subdirectorios:")
        logger.info(f"    - Matrices 2D:      {self.matriz_2d_dir}")

        archivos_2d = list(self.matriz_2d_dir.glob("*.npz"))
        if archivos_2d:
            tamaño_2d = sum(f.stat().st_size for f in archivos_2d) / (1024**2)
            logger.info(f"                        {len(archivos_2d)} archivos, {tamaño_2d:.1f} MB")

        logger.info(f"    - Tensores 3D:      {self.tensor_3d_dir}")
        archivos_3d = list(self.tensor_3d_dir.glob("*.npz"))
        if archivos_3d:
            tamaño_3d = sum(f.stat().st_size for f in archivos_3d) / (1024**2)
            logger.info(f"                        {len(archivos_3d)} archivos, {tamaño_3d:.1f} MB")

        logger.info(f"    - Normalización:    {self.normalizacion_dir}")
        archivos_norm = list(self.normalizacion_dir.glob("*.parquet"))
        if archivos_norm:
            tamaño_norm = sum(f.stat().st_size for f in archivos_norm) / (1024**2)
            logger.info(f"                        {len(archivos_norm)} archivos, {tamaño_norm:.1f} MB")

        # ============================================================
        # SECCIÓN 7: CONCLUSIÓN
        # ============================================================
        logger.info(f"\n7. CONCLUSIÓN")
        logger.info("-" * 100)

        if exitosos == total_combinaciones:
            logger.info(f"  ✓ ESTRUCTURA MATRICIAL/TENSORIAL MULTI-TIMEFRAME COMPLETADA EXITOSAMENTE")
            logger.info(f"  Todas las {total_combinaciones} combinaciones procesadas correctamente")
            logger.info(f"    • {len(self.pares)} pares × {len(self.timeframes)} timeframes")
            logger.info(f"\n  Estructuras generadas:")
            logger.info(f"    • Matrices 2D:     Para ML tradicional (RF, XGBoost, etc.)")
            logger.info(f"    • Tensores 3D:     Para modelos secuenciales (LSTM, GRU)")
            logger.info(f"    • Normalización:   Z-score + Rank rolling (point-in-time)")
            logger.info(f"\n  Próximos pasos:")
            logger.info(f"    1. ejecutar_metodos_estadisticos_clasicos.py → Análisis estadístico")
            logger.info(f"    2. ejecutar_analisis_multimetodo.py          → ML + Deep Learning")
            logger.info(f"    3. ejecutar_consenso_metodos.py              → Consenso de features")
            logger.info(f"    4. ejecutar_validacion_rigurosa.py           → Walk-Forward + Bootstrap")
            logger.info(f"    5. ejecutar_estrategia_emergente.py          → Emergencia de reglas")
            logger.info(f"    6. ejecutar_backtest.py                      → Backtest completo")
        else:
            logger.info(f"  ⚠️  PROCESAMIENTO COMPLETADO CON ERRORES")
            logger.info(f"  {exitosos}/{total_combinaciones} combinaciones exitosas")
            logger.info(f"  Revisar logs de errores arriba")

        logger.info("="*100 + "\n")


def main():
    """Función principal MULTI-TIMEFRAME."""
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

    # MULTI-TIMEFRAME: Procesar todos los timeframes
    TIMEFRAMES = ['M15', 'H1', 'H4', 'D1']

    SEQ_LENGTH = 50  # Longitud de secuencias para LSTM/GRU
    NORM_WINDOW = 200  # Ventana para normalización rolling

    # Opciones de limpieza de archivos
    LIMPIAR_ARCHIVOS_VIEJOS = True  # True = Borra archivos viejos antes de iniciar
    HACER_BACKUP = False             # False = NO crea backup (ahorra espacio)

    # Validar
    if not FEATURES_DIR.exists():
        logger.error(f"Directorio de features no encontrado: {FEATURES_DIR}")
        return

    # Ejecutar MULTI-TIMEFRAME
    ejecutor = EjecutorEstructuraMatricialTensorial(
        features_dir=FEATURES_DIR,
        output_dir=OUTPUT_DIR,
        pares=PARES,
        timeframes=TIMEFRAMES,
        limpiar_archivos_viejos=LIMPIAR_ARCHIVOS_VIEJOS,
        hacer_backup=HACER_BACKUP
    )

    ejecutor.ejecutar_todos(
        seq_length=SEQ_LENGTH,
        norm_window=NORM_WINDOW
    )


if __name__ == '__main__':
    main()
