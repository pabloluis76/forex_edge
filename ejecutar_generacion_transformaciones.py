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
        timeframes: list = None,
        limpiar_archivos_viejos: bool = True,
        hacer_backup: bool = True
    ):
        """
        Inicializa el ejecutor MULTI-TIMEFRAME.

        Args:
            data_dir: Directorio con datos OHLC
            output_dir: Directorio para guardar features generados
            timeframes: Lista de timeframes a procesar (default: ['M15', 'H1', 'H4', 'D'])
            limpiar_archivos_viejos: Si True, borra archivos .parquet viejos antes de iniciar
            hacer_backup: Si True, hace backup de archivos existentes antes de borrarlos
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.timeframes = timeframes or ['M15', 'H1', 'H4', 'D']
        self.limpiar_archivos_viejos = limpiar_archivos_viejos
        self.hacer_backup = hacer_backup

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

    def procesar_un_par(self, par: str, timeframe: str) -> dict:
        """
        Procesa un par en un timeframe específico y retorna estadísticas.

        Args:
            par: Nombre del par (ej: 'EUR_USD')
            timeframe: Timeframe a procesar (ej: 'M15', 'H1', etc.)

        Returns:
            Diccionario con estadísticas del procesamiento
        """
        logger.info("\n" + "="*80)
        logger.info(f"PROCESANDO: {par} - {timeframe}")
        logger.info("="*80)

        inicio = datetime.now()

        try:
            # Directorio y archivo del par
            par_dir = self.data_dir / par
            file_path = par_dir / f"{timeframe}.csv"

            if not file_path.exists():
                logger.error(f"✗ Archivo no encontrado: {file_path}")
                return {
                    'par': par,
                    'timeframe': timeframe,
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
                nombre_par=f"{par}_{timeframe}"
            )

            df_features = generador.generar_todas_las_transformaciones()

            # Guardar resultado
            output_file = self.output_dir / f"{par}_{timeframe}_features.parquet"

            logger.info(f"\nGuardando features...")
            df_features.to_parquet(output_file, compression='snappy')

            # Estadísticas
            tamaño_mb = output_file.stat().st_size / (1024 ** 2)
            n_features = len(df_features.columns)
            n_filas = len(df_features)
            pct_nan = df_features.isna().mean().mean() * 100

            fin = datetime.now()
            tiempo_total = (fin - inicio).total_seconds()

            logger.info(f"\n✓ COMPLETADO: {par} - {timeframe}")
            logger.info(f"  Features generados: {n_features:,}")
            logger.info(f"  Filas: {n_filas:,}")
            logger.info(f"  Tamaño archivo: {tamaño_mb:.1f} MB")
            logger.info(f"  % NaN promedio: {pct_nan:.1f}%")
            logger.info(f"  Tiempo: {tiempo_total:.1f} segundos")
            logger.info(f"  Archivo: {output_file}")

            return {
                'par': par,
                'timeframe': timeframe,
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

            logger.error(f"\n✗ ERROR en {par} - {timeframe}: {e}")
            logger.exception(e)

            return {
                'par': par,
                'timeframe': timeframe,
                'exito': False,
                'error': str(e),
                'tiempo_segundos': tiempo_total
            }

    def ejecutar_todos(self):
        """
        Ejecuta la generación MULTI-TIMEFRAME para todos los pares.
        """
        self.tiempo_inicio = datetime.now()

        logger.info("\n" + "="*80)
        logger.info("GENERACIÓN DE TRANSFORMACIONES - MULTI-TIMEFRAME")
        logger.info("="*80)
        logger.info(f"Pares a procesar: {len(self.pares)}")
        logger.info(f"Timeframes: {', '.join(self.timeframes)}")
        logger.info(f"Total combinaciones: {len(self.pares) * len(self.timeframes)}")
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

        # Procesar cada par en cada timeframe
        total_combinaciones = len(self.pares) * len(self.timeframes)
        contador = 0

        for par in self.pares:
            for timeframe in self.timeframes:
                contador += 1
                logger.info(f"\n[{contador}/{total_combinaciones}] {par} - {timeframe}")

                resultado = self.procesar_un_par(par, timeframe)

                # Guardar resultado con clave compuesta
                key = f"{par}_{timeframe}"
                self.resultados[key] = resultado

        self.tiempo_fin = datetime.now()

        # Imprimir resumen
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen final de la generación."""
        tiempo_total = (self.tiempo_fin - self.tiempo_inicio).total_seconds()

        # Recopilar estadísticas
        exitosos = 0
        total_features = 0
        total_tamaño = 0
        total_filas = 0

        for res in self.resultados.values():
            if res['exito']:
                exitosos += 1
                total_features += res['n_features']
                total_tamaño += res['tamaño_mb']
                total_filas += res['n_filas']

        logger.info("\n" + "="*100)
        logger.info(f"{'RESUMEN FINAL - GENERACIÓN DE TRANSFORMACIONES':^100}")
        logger.info("="*100)

        # ============================================================
        # RESUMEN EJECUTIVO
        # ============================================================
        logger.info("\n" + "─"*100)
        logger.info(f"{'1. RESUMEN EJECUTIVO':^100}")
        logger.info("─"*100)

        total_combinaciones = len(self.pares) * len(self.timeframes)
        logger.info(f"\n  Timeframes:                    {', '.join(self.timeframes)}")
        logger.info(f"  Pares:                         {len(self.pares)}")
        logger.info(f"  Combinaciones procesadas:      {exitosos}/{total_combinaciones}")

        if exitosos > 0:
            features_por_par = [r['n_features'] for r in self.resultados.values() if r['exito']]
            filas_por_par = [r['n_filas'] for r in self.resultados.values() if r['exito']]
            tamaño_por_par = [r['tamaño_mb'] for r in self.resultados.values() if r['exito']]

            logger.info(f"\n  📊 TRANSFORMACIONES GENERADAS:")
            logger.info(f"     Total features:             {total_features:,}")
            logger.info(f"     Promedio por par:           {np.mean(features_por_par):.0f}")
            logger.info(f"     Rango:                      {np.min(features_por_par):,.0f} - {np.max(features_por_par):,.0f}")

            logger.info(f"\n  💾 DATOS GENERADOS:")
            logger.info(f"     Total filas:                {total_filas:,}")
            logger.info(f"     Promedio filas/par:         {np.mean(filas_por_par):,.0f}")
            logger.info(f"     Tamaño total:               {total_tamaño:.1f} MB")
            logger.info(f"     Tamaño promedio/par:        {np.mean(tamaño_por_par):.1f} MB")

            # Mejor y Peor productor
            mejor_idx = np.argmax(features_por_par)
            peor_idx = np.argmin(features_por_par)
            pares_exitosos = [p for p, r in self.resultados.items() if r['exito']]
            mejor_par = pares_exitosos[mejor_idx] if mejor_idx < len(pares_exitosos) else 'N/A'
            peor_par = pares_exitosos[peor_idx] if peor_idx < len(pares_exitosos) else 'N/A'

            logger.info(f"\n  🏆 MÁS FEATURES:               {mejor_par} ({features_por_par[mejor_idx]:,.0f} features)")
            logger.info(f"  📊 MENOS FEATURES:             {peor_par} ({features_por_par[peor_idx]:,.0f} features)")

        # Información temporal
        logger.info(f"\n  ⏱️  TIEMPO DE EJECUCIÓN:")
        logger.info(f"     Inicio:                     {self.tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"     Fin:                        {self.tiempo_fin.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"     Duración Total:             {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
        logger.info(f"     Tiempo Promedio/Comb:       {tiempo_total/total_combinaciones:.1f}s")

        # ============================================================
        # TABLA DE RESULTADOS COMPLETA
        # ============================================================
        logger.info("\n" + "─"*100)
        logger.info(f"{'2. RESULTADOS POR PAR Y TIMEFRAME':^100}")
        logger.info("─"*100)
        logger.info(f"\n{'Par':<10} │ {'TF':<4} │ {'✓':<3} │ {'Features':<10} │ {'Filas':<10} │ {'Tamaño MB':<11} │ {'%NaN':<7} │ {'Tiempo s':<10}")
        logger.info("─" * 100)

        for par in self.pares:
            for timeframe in self.timeframes:
                key = f"{par}_{timeframe}"
                res = self.resultados[key]

                if res['exito']:
                    logger.info(
                        f"{par:<10} │ {timeframe:<4} │ {'✓':<3} │ {res['n_features']:>9,} │ "
                        f"{res['n_filas']:>9,} │ {res['tamaño_mb']:>10.1f} │ "
                        f"{res['pct_nan']:>6.1f} │ {res['tiempo_segundos']:>9.1f}"
                    )
                else:
                    logger.info(
                        f"{par:<10} │ {timeframe:<4} │ {'✗':<3} │ {'N/A':<10} │ {'N/A':<10} │ "
                        f"{'N/A':<11} │ {'N/A':<7} │ {res['tiempo_segundos']:>9.1f}"
                    )
                    logger.info(f"{'':15} └─ Error: {res.get('error', 'Desconocido')}")

        logger.info("─" * 100)

        # ============================================================
        # ARCHIVOS GENERADOS
        # ============================================================
        logger.info("\n" + "─"*100)
        logger.info(f"{'3. ARCHIVOS GENERADOS':^100}")
        logger.info("─"*100)

        logger.info(f"\n  📁 Ubicación: {self.output_dir}")
        logger.info(f"\n  Archivos generados:")
        for key in sorted(self.resultados.keys()):
            res = self.resultados[key]
            if res['exito']:
                archivo = Path(res['archivo'])
                logger.info(f"     • {archivo.name:<35} {res['tamaño_mb']:>6.1f} MB")

        # ============================================================
        # CONCLUSIÓN Y PRÓXIMOS PASOS
        # ============================================================
        logger.info("\n" + "="*100)
        logger.info(f"{'CONCLUSIÓN':^100}")
        logger.info("="*100)

        if exitosos == total_combinaciones:
            logger.info(f"\n  ✅ GENERACIÓN MULTI-TIMEFRAME COMPLETADA EXITOSAMENTE")
            logger.info(f"\n  Resumen:")
            logger.info(f"     • Pares:                    {len(self.pares)}")
            logger.info(f"     • Timeframes:               {len(self.timeframes)}")
            logger.info(f"     • Combinaciones exitosas:   {exitosos}/{total_combinaciones}")
            logger.info(f"     • Total features:           {total_features:,}")
            logger.info(f"     • Tamaño total:             {total_tamaño:.1f} MB")

            logger.info(f"\n  📋 PRÓXIMOS PASOS:")
            logger.info(f"     1. Estructura matricial/tensorial:")
            logger.info(f"        → python ejecutar_estructura_matricial_tensorial.py")
            logger.info(f"     2. Normalización point-in-time")
            logger.info(f"     3. Análisis multi-método")
            logger.info(f"     4. Sistema de consenso")
            logger.info(f"     5. Validación rigurosa")

        elif exitosos > 0:
            logger.info(f"\n  ⚠️  GENERACIÓN COMPLETADA CON ERRORES PARCIALES")
            logger.info(f"\n  Resumen:")
            logger.info(f"     • Combinaciones exitosas:   {exitosos}/{total_combinaciones}")
            logger.info(f"     • Combinaciones con errores: {total_combinaciones - exitosos}")

            logger.info(f"\n  📋 ACCIÓN REQUERIDA:")
            logger.info(f"     1. Revisar errores arriba")
            logger.info(f"     2. Corregir problemas")
            logger.info(f"     3. Re-ejecutar generación")

        else:
            logger.info(f"\n  ❌ GENERACIÓN FALLIDA")
            logger.info(f"\n  📋 ACCIÓN CRÍTICA:")
            logger.info(f"     1. Verificar datos OHLC en {self.data_dir}")
            logger.info(f"     2. Revisar logs detallados")

        logger.info(f"\n  {'─'*96}")
        logger.info(f"  ℹ️  NOTA:")
        logger.info(f"     ~1,700+ transformaciones sistemáticas generadas SIN sesgo humano.")
        logger.info(f"     Cada feature será evaluado por múltiples métodos independientes.")
        logger.info("="*100)


def main():
    """Función principal - MULTI-TIMEFRAME."""
    # Configuración
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'datos' / 'ohlc'
    OUTPUT_DIR = BASE_DIR / 'datos' / 'features'

    # MULTI-TIMEFRAME: Generar transformaciones para todos los timeframes
    TIMEFRAMES = ['M15', 'H1', 'H4', 'D']

    # Opciones de limpieza de archivos
    LIMPIAR_ARCHIVOS_VIEJOS = True  # True = Borra archivos viejos antes de iniciar
    HACER_BACKUP = False             # False = NO crea backup (ahorra espacio)

    # Validar que existe el directorio de datos
    if not DATA_DIR.exists():
        logger.error(f"Directorio de datos no encontrado: {DATA_DIR}")
        return

    # Ejecutar generación MULTI-TIMEFRAME
    ejecutor = EjecutorGeneracionCompleta(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        timeframes=TIMEFRAMES,
        limpiar_archivos_viejos=LIMPIAR_ARCHIVOS_VIEJOS,
        hacer_backup=HACER_BACKUP
    )

    ejecutor.ejecutar_todos()


if __name__ == '__main__':
    main()
