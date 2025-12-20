"""
EJECUTAR GENERACIÓN DE TRANSFORMACIONES - TODOS LOS PARES
==========================================================

Genera transformaciones sistemáticas para los 6 pares principales
usando el timeframe M15 (15 minutos).

Pares: EUR_USD, GBP_USD, USD_JPY, EUR_JPY, GBP_JPY, AUD_USD
Timeframe: M15
Período: ~5 años (2019-2023)

Este script ejecuta la generación completa de ~1,700+ transformaciones
por par sin sesgo humano.

Autor: Sistema de Edge-Finding Forex
Fecha: 2025-12-17
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
import shutil
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent))

from generacion_de_transformaciones.aplicacion_sistematica import (
    GeneradorSistematicoFeatures,
    procesar_par
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EjecutorGeneracionCompleta:
    """
    Ejecuta la generación de transformaciones para todos los pares.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        timeframe: str = 'M15',
        limpiar_archivos_viejos: bool = True,
        hacer_backup: bool = True
    ):
        """
        Inicializa el ejecutor.

        Args:
            data_dir: Directorio con datos OHLC
            output_dir: Directorio para guardar features generados
            timeframe: Timeframe a procesar (default: 'M15')
            limpiar_archivos_viejos: Si True, borra archivos .parquet viejos antes de iniciar
            hacer_backup: Si True, hace backup de archivos existentes antes de borrarlos
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.timeframe = timeframe
        self.limpiar_archivos_viejos = limpiar_archivos_viejos
        self.hacer_backup = hacer_backup

        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pares a procesar (solo EUR_USD para prueba)
        self.pares = [
            'EUR_USD'
        ]

        # Estadísticas
        self.resultados = {}
        self.tiempo_inicio = None
        self.tiempo_fin = None

    def limpiar_directorio_salida(self):
        """
        Limpia archivos .parquet y .csv viejos del directorio de salida.
        Opcionalmente hace backup antes de borrar.
        """
        # Buscar archivos de features existentes
        archivos_parquet = list(self.output_dir.glob("*_features.parquet"))
        archivos_csv = list(self.output_dir.glob("*_features.csv"))
        archivos_existentes = archivos_parquet + archivos_csv

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

    def procesar_un_par(self, par: str) -> dict:
        """
        Procesa un par y retorna estadísticas.

        Args:
            par: Nombre del par (ej: 'EUR_USD')

        Returns:
            Diccionario con estadísticas del procesamiento
        """
        logger.info("\n" + "="*80)
        logger.info(f"PROCESANDO PAR: {par}")
        logger.info("="*80)

        inicio = datetime.now()

        try:
            # Directorio y archivo del par
            par_dir = self.data_dir / par
            file_path = par_dir / f"{self.timeframe}.csv"

            if not file_path.exists():
                logger.error(f"✗ Archivo no encontrado: {file_path}")
                return {
                    'par': par,
                    'exito': False,
                    'error': 'Archivo no encontrado',
                    'tiempo_segundos': 0
                }

            # Cargar datos
            logger.info(f"Cargando datos: {file_path}")
            df = pd.read_csv(file_path, index_col='time', parse_dates=True)

            logger.info(f"  Velas cargadas: {len(df):,}")
            logger.info(f"  Período: {df.index[0]} → {df.index[-1]}")
            logger.info(f"  Columnas: {', '.join(df.columns)}")

            # Generar transformaciones
            generador = GeneradorSistematicoFeatures(
                df,
                nombre_par=f"{par}_{self.timeframe}"
            )

            df_features = generador.generar_todas_las_transformaciones()

            # Guardar resultado
            output_file = self.output_dir / f"{par}_{self.timeframe}_features.parquet"

            logger.info(f"\nGuardando features...")
            df_features.to_parquet(output_file, compression='snappy')

            # Estadísticas
            tamaño_mb = output_file.stat().st_size / (1024 ** 2)
            n_features = len(df_features.columns)
            n_filas = len(df_features)
            pct_nan = df_features.isna().mean().mean() * 100

            fin = datetime.now()
            tiempo_total = (fin - inicio).total_seconds()

            logger.info(f"\n✓ PAR COMPLETADO: {par}")
            logger.info(f"  Features generados: {n_features:,}")
            logger.info(f"  Filas: {n_filas:,}")
            logger.info(f"  Tamaño archivo: {tamaño_mb:.1f} MB")
            logger.info(f"  % NaN promedio: {pct_nan:.1f}%")
            logger.info(f"  Tiempo: {tiempo_total:.1f} segundos")
            logger.info(f"  Archivo: {output_file}")

            return {
                'par': par,
                'exito': True,
                'n_features': n_features,
                'n_filas': n_filas,
                'tamaño_mb': tamaño_mb,
                'pct_nan': pct_nan,
                'tiempo_segundos': tiempo_total,
                'archivo': str(output_file)
            }

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
        Ejecuta la generación para todos los pares.
        """
        self.tiempo_inicio = datetime.now()

        logger.info("\n" + "="*80)
        logger.info("GENERACIÓN DE TRANSFORMACIONES - TODOS LOS PARES")
        logger.info("="*80)
        logger.info(f"Pares a procesar: {len(self.pares)}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Directorio OHLC: {self.data_dir}")
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

            resultado = self.procesar_un_par(par)
            self.resultados[par] = resultado

        self.tiempo_fin = datetime.now()

        # Imprimir resumen
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen final de la generación."""
        tiempo_total = (self.tiempo_fin - self.tiempo_inicio).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("RESUMEN FINAL - GENERACIÓN DE TRANSFORMACIONES")
        logger.info("="*80)

        # Tabla de resultados
        logger.info("\nRESULTADOS POR PAR:")
        logger.info("-" * 80)
        logger.info(f"{'Par':<10} │ {'Exito':<6} │ {'Features':<10} │ {'Tamaño (MB)':<12} │ {'Tiempo (s)':<12}")
        logger.info("-" * 80)

        exitosos = 0
        total_features = 0
        total_tamaño = 0

        for par in self.pares:
            res = self.resultados[par]

            if res['exito']:
                exitosos += 1
                total_features += res['n_features']
                total_tamaño += res['tamaño_mb']

                logger.info(
                    f"{par:<10} │ {'✓':<6} │ {res['n_features']:<10,} │ "
                    f"{res['tamaño_mb']:<12.1f} │ {res['tiempo_segundos']:<12.1f}"
                )
            else:
                logger.info(
                    f"{par:<10} │ {'✗':<6} │ {'N/A':<10} │ {'N/A':<12} │ "
                    f"{res['tiempo_segundos']:<12.1f}"
                )
                logger.info(f"           Error: {res.get('error', 'Desconocido')}")

        logger.info("-" * 80)

        # Estadísticas globales
        logger.info("\nESTADÍSTICAS GLOBALES:")
        logger.info(f"  Pares procesados exitosamente: {exitosos}/{len(self.pares)}")
        logger.info(f"  Features promedio por par: {total_features/exitosos:,.0f}" if exitosos > 0 else "  N/A")
        logger.info(f"  Tamaño total: {total_tamaño:.1f} MB")
        logger.info(f"  Tiempo total: {tiempo_total:.1f} segundos ({tiempo_total/60:.1f} minutos)")
        logger.info(f"  Tiempo promedio por par: {tiempo_total/len(self.pares):.1f} segundos")

        # Conclusión
        logger.info("\n" + "="*80)
        if exitosos == len(self.pares):
            logger.info("✓ GENERACIÓN COMPLETADA EXITOSAMENTE")
            logger.info(f"  Todos los {len(self.pares)} pares procesados correctamente")
            logger.info(f"  Archivos guardados en: {self.output_dir}")
            logger.info("\nPRÓXIMO PASO:")
            logger.info("  → Los features están listos para:")
            logger.info("     - Estructura matricial/tensorial")
            logger.info("     - Normalización point-in-time")
            logger.info("     - Análisis multi-método")
            logger.info("     - Sistema de consenso")
            logger.info("     - Validación rigurosa")
        else:
            logger.info(f"⚠️  GENERACIÓN COMPLETADA CON ERRORES")
            logger.info(f"  {exitosos}/{len(self.pares)} pares exitosos")
            logger.info(f"  Revisar errores arriba")

        logger.info("="*80)


def main():
    """Función principal."""
    # Configuración
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'datos' / 'ohlc'
    OUTPUT_DIR = BASE_DIR / 'datos' / 'features'
    TIMEFRAME = 'M15'

    # Opciones de limpieza de archivos
    LIMPIAR_ARCHIVOS_VIEJOS = True  # True = Borra archivos viejos antes de iniciar
    HACER_BACKUP = True              # True = Crea backup antes de borrar

    # Validar que existe el directorio de datos
    if not DATA_DIR.exists():
        logger.error(f"Directorio de datos no encontrado: {DATA_DIR}")
        return

    # Ejecutar generación
    ejecutor = EjecutorGeneracionCompleta(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        timeframe=TIMEFRAME,
        limpiar_archivos_viejos=LIMPIAR_ARCHIVOS_VIEJOS,
        hacer_backup=HACER_BACKUP
    )

    ejecutor.ejecutar_todos()


if __name__ == '__main__':
    main()
