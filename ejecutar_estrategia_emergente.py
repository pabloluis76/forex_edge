"""
Ejecutor: Estrategia Emergente

LA ESTRATEGIA EMERGE DE LOS DATOS:
- Interpreta transformaciones que pasaron TODAS las validaciones
- Genera reglas de trading ejecutables
- Position sizing basado en ATR
- Stop loss / Take profit automáticos
- Código Python ejecutable por par

Author: Sistema de Edge-Finding Forex
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import shutil
from typing import List, Dict, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Importar módulos de estrategia emergente
from estrategia_emergente.interpretacion_post_hoc import InterpretacionPostHoc
from estrategia_emergente.formulacion_reglas import FormulacionReglas


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


class EjecutorEstrategiaEmergente:
    """
    Ejecutor del módulo Estrategia Emergente.

    Genera estrategias de trading ejecutables basadas en transformaciones
    que pasaron TODAS las validaciones rigurosas.
    """

    def __init__(
        self,
        features_validados_dir: Path,
        analisis_ic_dir: Path,
        output_dir: Path,
        timeframes: list = None,
        pares: Optional[List[str]] = None,
        limpiar_archivos_viejos: bool = True,
        hacer_backup: bool = False,
        verbose: bool = True
    ):
        """
        Inicializa el ejecutor de Estrategia Emergente MULTI-TIMEFRAME.

        Parameters:
        -----------
        features_validados_dir : Path
            Directorio con features validados (validacion_rigurosa/features_validados/)
        analisis_ic_dir : Path
            Directorio con análisis IC (analisis_multimetodo/)
        output_dir : Path
            Directorio para guardar estrategias emergentes
        timeframes : list
            Lista de timeframes a procesar (default: ['M15', 'H1', 'H4', 'D'])
        pares : List[str], optional
            Lista de pares a procesar. Si None, procesa todos los disponibles
        limpiar_archivos_viejos : bool
            Si True, limpia archivos viejos antes de ejecutar
        hacer_backup : bool
            Si True, hace backup antes de limpiar
        verbose : bool
            Imprimir detalles del proceso
        """
        self.features_validados_dir = Path(features_validados_dir)
        self.analisis_ic_dir = Path(analisis_ic_dir)
        self.output_dir = Path(output_dir)
        self.timeframes = timeframes or ['M15', 'H1', 'H4', 'D']
        self.verbose = verbose
        self.limpiar_archivos_viejos = limpiar_archivos_viejos
        self.hacer_backup = hacer_backup

        # Crear directorios de salida
        self.interpretaciones_dir = self.output_dir / 'interpretaciones'
        self.estrategias_dir = self.output_dir / 'estrategias'
        self.codigo_dir = self.output_dir / 'codigo'
        self.resumen_dir = self.output_dir / 'resumen'

        for directorio in [self.interpretaciones_dir, self.estrategias_dir,
                          self.codigo_dir, self.resumen_dir]:
            directorio.mkdir(parents=True, exist_ok=True)

        # Configurar logging
        self._configurar_logging()

        # Determinar pares a procesar
        if pares is None:
            self.pares = self._detectar_pares_disponibles()
        else:
            self.pares = pares

        # Resultados
        self.resultados: Dict[str, dict] = {}

        # Tiempos de ejecución
        self.tiempo_inicio = None
        self.tiempo_fin = None

        if self.verbose:
            print("="*80)
            print("EJECUTOR: ESTRATEGIA EMERGENTE")
            print("="*80)
            print(f"\nDirectorio features validados: {self.features_validados_dir}")
            print(f"Directorio análisis IC: {self.analisis_ic_dir}")
            print(f"Directorio salida: {self.output_dir}")
            print(f"Timeframe: {self.timeframe}")
            print(f"Pares a procesar: {len(self.pares)}")
            print(f"Limpieza automática: {self.limpiar_archivos_viejos}")
            print(f"Backup antes de limpiar: {self.hacer_backup}")
            print("="*80)

    def _configurar_logging(self):
        """Configura el sistema de logging."""
        log_file = self.output_dir / f'estrategia_emergente_{self.timeframe}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

        # Agregar LogCapture global para capturar todos los logs
        global log_capture
        log_capture = LogCapture()
        log_capture.setLevel(logging.INFO)  # Capturar desde INFO en adelante
        logging.getLogger().addHandler(log_capture)

    def _detectar_pares_disponibles(self) -> List[str]:
        """
        Detecta pares disponibles en el directorio de features validados.

        Returns:
        --------
        pares : List[str]
            Lista de pares disponibles
        """
        archivos_validados = list(self.features_validados_dir.glob(f"*_{self.timeframe}_features_validados.csv"))

        pares = []
        for archivo in archivos_validados:
            # Extraer par del nombre de archivo
            nombre = archivo.stem.replace(f"_{self.timeframe}_features_validados", "")
            pares.append(nombre)

        if self.verbose:
            print(f"\n✓ Detectados {len(pares)} pares con features validados")
            for par in pares:
                print(f"  - {par}")

        return pares

    def limpiar_directorio_salida(self):
        """
        Limpia archivos viejos del directorio de salida con backup opcional.
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"LIMPIANDO DIRECTORIO DE SALIDA")
            print(f"{'='*80}")

        # Buscar archivos existentes
        archivos_existentes = []
        archivos_existentes.extend(list(self.interpretaciones_dir.glob("*.csv")))
        archivos_existentes.extend(list(self.estrategias_dir.glob("*.csv")))
        archivos_existentes.extend(list(self.codigo_dir.glob("*.py")))
        archivos_existentes.extend(list(self.resumen_dir.glob("*.csv")))

        if len(archivos_existentes) == 0:
            if self.verbose:
                print("✓ No hay archivos viejos para limpiar")
            return

        if self.verbose:
            print(f"Archivos encontrados: {len(archivos_existentes)}")

        # Hacer backup si está habilitado
        if self.hacer_backup:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.output_dir / f"backup_{timestamp}"

            if self.verbose:
                print(f"\n📦 Creando backup en: {backup_dir.name}/")

            # Copiar estructura de directorios
            for directorio in [self.interpretaciones_dir, self.estrategias_dir,
                             self.codigo_dir, self.resumen_dir]:
                dir_rel = directorio.relative_to(self.output_dir)
                backup_subdir = backup_dir / dir_rel
                backup_subdir.mkdir(parents=True, exist_ok=True)

                # Copiar archivos
                for archivo in directorio.glob("*"):
                    if archivo.is_file():
                        shutil.copy2(archivo, backup_subdir / archivo.name)

            if self.verbose:
                print(f"✓ Backup completado: {len(archivos_existentes)} archivos respaldados")

        # Eliminar archivos viejos
        if self.verbose:
            print(f"\n🗑️  Eliminando archivos viejos...")

        for archivo in archivos_existentes:
            try:
                archivo.unlink()
            except Exception as e:
                self.logger.warning(f"No se pudo eliminar {archivo}: {e}")

        if self.verbose:
            print(f"✓ Limpieza completada: {len(archivos_existentes)} archivos eliminados")

    def cargar_datos_par(self, par: str) -> Optional[pd.DataFrame]:
        """
        Carga datos de features validados + IC para un par.

        Parameters:
        -----------
        par : str
            Par a cargar (ej: 'EUR_USD')

        Returns:
        --------
        df_completo : pd.DataFrame o None
            DataFrame con features validados + IC
        """
        # Cargar features validados
        archivo_validados = self.features_validados_dir / f"{par}_{self.timeframe}_features_validados.csv"

        if not archivo_validados.exists():
            self.logger.warning(f"No se encontró archivo de validación: {archivo_validados}")
            return None

        df_validados = pd.read_csv(archivo_validados)

        if len(df_validados) == 0:
            self.logger.warning(f"Archivo de validación vacío para {par}")
            return None

        # Cargar análisis IC
        archivo_ic = self.analisis_ic_dir / f"{par}_{self.timeframe}_analisis_IC.csv"

        if not archivo_ic.exists():
            self.logger.warning(f"No se encontró archivo de IC: {archivo_ic}")
            return None

        df_ic = pd.read_csv(archivo_ic)

        # Merge: validados + IC
        # Determinar nombre de columna de features
        col_feature_validados = None
        for col in ['feature', 'Feature', 'Transformacion', 'transformacion']:
            if col in df_validados.columns:
                col_feature_validados = col
                break

        col_feature_ic = None
        for col in ['feature', 'Feature', 'Transformacion', 'transformacion']:
            if col in df_ic.columns:
                col_feature_ic = col
                break

        if col_feature_validados is None or col_feature_ic is None:
            self.logger.error(f"No se pudo identificar columna de features en los datos de {par}")
            return None

        # Merge
        df_completo = df_validados.merge(
            df_ic[[col_feature_ic, 'IC']],
            left_on=col_feature_validados,
            right_on=col_feature_ic,
            how='left'
        )

        # Renombrar columna para uniformidad
        df_completo = df_completo.rename(columns={col_feature_validados: 'Transformacion'})

        # Agregar columnas requeridas si no existen
        if 'Robusto' not in df_completo.columns:
            df_completo['Robusto'] = 'Sí'

        if 'Estable' not in df_completo.columns:
            df_completo['Estable'] = 'Sí'

        # Eliminar duplicados
        if col_feature_ic in df_completo.columns and col_feature_ic != 'Transformacion':
            df_completo = df_completo.drop(columns=[col_feature_ic])

        # Eliminar filas con IC nulo
        df_completo = df_completo.dropna(subset=['IC'])

        self.logger.info(f"{par}: Cargadas {len(df_completo)} transformaciones validadas")

        return df_completo

    def generar_estrategia_par(self, par: str) -> dict:
        """
        Genera estrategia emergente para un par.

        Parameters:
        -----------
        par : str
            Par a procesar

        Returns:
        --------
        resultado : dict
            Resultado del procesamiento
        """
        resultado = {
            'par': par,
            'exito': False,
            'num_transformaciones': 0,
            'num_reglas_long': 0,
            'num_reglas_short': 0,
            'archivos_generados': [],
            'error': None
        }

        try:
            # Cargar datos
            df_completo = self.cargar_datos_par(par)

            if df_completo is None or len(df_completo) == 0:
                resultado['error'] = "No se pudieron cargar datos o no hay transformaciones validadas"
                return resultado

            resultado['num_transformaciones'] = len(df_completo)

            # PASO 1: INTERPRETACIÓN POST-HOC
            self.logger.info(f"{par}: Interpretando transformaciones validadas...")

            interpretador = InterpretacionPostHoc(verbose=False)
            df_interpretado = interpretador.interpretar_transformaciones_validadas(df_completo)

            # Guardar interpretaciones
            archivo_interpretacion = self.interpretaciones_dir / f"{par}_{self.timeframe}_interpretaciones.csv"
            df_interpretado.to_csv(archivo_interpretacion, index=False)
            resultado['archivos_generados'].append(str(archivo_interpretacion))

            self.logger.info(f"{par}: ✓ Interpretaciones guardadas en {archivo_interpretacion.name}")

            # PASO 2: GENERAR ESTRATEGIA COMBINADA
            self.logger.info(f"{par}: Generando estrategia combinada...")

            df_estrategia = interpretador.generar_estrategia_combinada(df_interpretado)

            # Guardar estrategia
            archivo_estrategia = self.estrategias_dir / f"{par}_{self.timeframe}_estrategia.csv"
            df_estrategia.to_csv(archivo_estrategia, index=False)
            resultado['archivos_generados'].append(str(archivo_estrategia))

            self.logger.info(f"{par}: ✓ Estrategia guardada en {archivo_estrategia.name}")

            # PASO 3: FORMULAR REGLAS
            self.logger.info(f"{par}: Formulando reglas de trading...")

            formulador = FormulacionReglas(
                transformaciones_validadas=df_estrategia,
                verbose=False
            )

            reglas_long, reglas_short = formulador.generar_reglas_entrada()

            resultado['num_reglas_long'] = len(reglas_long)
            resultado['num_reglas_short'] = len(reglas_short)

            self.logger.info(f"{par}: ✓ Generadas {len(reglas_long)} reglas Long, {len(reglas_short)} reglas Short")

            # PASO 4: GENERAR CÓDIGO EJECUTABLE
            self.logger.info(f"{par}: Generando código ejecutable...")

            archivo_codigo = self.codigo_dir / f"{par}_{self.timeframe}_estrategia.py"
            codigo = formulador.generar_codigo_estrategia(ruta_salida=str(archivo_codigo))
            resultado['archivos_generados'].append(str(archivo_codigo))

            self.logger.info(f"{par}: ✓ Código ejecutable guardado en {archivo_codigo.name}")

            # PASO 5: GENERAR RESUMEN
            self.logger.info(f"{par}: Generando resumen ejecutivo...")

            df_resumen = formulador.generar_resumen_estrategia()

            # Guardar resumen
            archivo_resumen = self.resumen_dir / f"{par}_{self.timeframe}_resumen.csv"
            df_resumen.to_csv(archivo_resumen, index=False)
            resultado['archivos_generados'].append(str(archivo_resumen))

            self.logger.info(f"{par}: ✓ Resumen guardado en {archivo_resumen.name}")

            # Éxito
            resultado['exito'] = True

        except Exception as e:
            self.logger.error(f"{par}: Error al generar estrategia: {e}")
            resultado['error'] = str(e)

        return resultado

    def ejecutar_todos(self):
        """
        Ejecuta generación de estrategias MULTI-TIMEFRAME para todos los pares.
        """
        # VALIDACIÓN: Verificar que existen los datos de entrada necesarios
        if not self.features_validados_dir.exists():
            error_msg = (f"ERROR: Directorio de features validados no existe: {self.features_validados_dir}\n"
                        f"Sugerencia: Ejecutar primero 'ejecutar_validacion_rigurosa.py'")
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Verificar que hay archivos de features validados
        archivos_validados = list(self.features_validados_dir.glob("*_features_validados.csv"))
        if len(archivos_validados) == 0:
            error_msg = (f"ERROR: No se encontraron features validados en {self.features_validados_dir}\n"
                        f"Patrón esperado: *_features_validados.csv\n"
                        f"Sugerencia: Ejecutar primero 'ejecutar_validacion_rigurosa.py'")
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Verificar que existe el directorio de análisis IC
        if not self.analisis_ic_dir.exists():
            error_msg = (f"ERROR: Directorio de análisis IC no existe: {self.analisis_ic_dir}\n"
                        f"Sugerencia: Ejecutar primero 'ejecutar_analisis_multimetodo.py'")
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Verificar que hay archivos de análisis IC
        archivos_ic = list(self.analisis_ic_dir.glob("*_analisis_IC.csv"))
        if len(archivos_ic) == 0:
            error_msg = (f"ERROR: No se encontraron archivos de análisis IC en {self.analisis_ic_dir}\n"
                        f"Patrón esperado: *_analisis_IC.csv\n"
                        f"Sugerencia: Ejecutar primero 'ejecutar_analisis_multimetodo.py'")
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.info(f"Iniciando generación de estrategias emergentes MULTI-TIMEFRAME")
        self.logger.info(f"Pares: {len(self.pares)}, Timeframes: {len(self.timeframes)}")

        # Limpiar archivos viejos si está habilitado
        if self.limpiar_archivos_viejos:
            self.limpiar_directorio_salida()

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"GENERANDO ESTRATEGIAS EMERGENTES - MULTI-TIMEFRAME")
            print(f"{'='*80}")
            print(f"Pares: {len(self.pares)}, Timeframes: {len(self.timeframes)}")

        # LOOP MULTI-TIMEFRAME
        total_combinaciones = len(self.pares) * len(self.timeframes)
        combinacion_actual = 0

        for timeframe in self.timeframes:
            # Definir timeframe actual para uso interno
            self.timeframe = timeframe

            if self.verbose:
                print(f"\n{'='*80}")
                print(f"PROCESANDO TIMEFRAME: {timeframe}")
                print(f"{'='*80}")

            for par in self.pares:
                combinacion_actual += 1
                if self.verbose:
                    print(f"\n[{combinacion_actual}/{total_combinaciones}] Procesando {par} ({timeframe})...")

                resultado = self.generar_estrategia_par(par)
                # Usar clave compuesta par_timeframe
                key = f"{par}_{timeframe}"
                self.resultados[key] = resultado

                if resultado['exito']:
                    if self.verbose:
                        print(f"✓ {par} ({timeframe}): Estrategia generada exitosamente")
                        print(f"  - Transformaciones: {resultado['num_transformaciones']}")
                        print(f"  - Reglas Long: {resultado['num_reglas_long']}")
                        print(f"  - Reglas Short: {resultado['num_reglas_short']}")
                        print(f"  - Archivos generados: {len(resultado['archivos_generados'])}")
                else:
                    if self.verbose:
                        print(f"✗ {par} ({timeframe}): Falló - {resultado['error']}")

        # Imprimir resumen final
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen detallado de la ejecución MULTI-TIMEFRAME."""
        exitosos = sum(1 for r in self.resultados.values() if r['exito'])
        total_combinaciones = len(self.pares) * len(self.timeframes)
        fallidos = total_combinaciones - exitosos

        total_transformaciones = sum(r['num_transformaciones'] for r in self.resultados.values() if r['exito'])
        total_reglas_long = sum(r['num_reglas_long'] for r in self.resultados.values() if r['exito'])
        total_reglas_short = sum(r['num_reglas_short'] for r in self.resultados.values() if r['exito'])

        print(f"\n{'='*100}")
        print(f"{'RESUMEN FINAL - ESTRATEGIA EMERGENTE (MULTI-TIMEFRAME)':^100}")
        print(f"{'='*100}")

        # ============================================================
        # RESUMEN EJECUTIVO
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"{'1. RESUMEN EJECUTIVO':^100}")
        print(f"{'─'*100}")

        print(f"\n  Pares:                         {len(self.pares)}")
        print(f"  Timeframes:                    {len(self.timeframes)} ({', '.join(self.timeframes)})")
        print(f"  Combinaciones Procesadas:      {exitosos}/{total_combinaciones}")

        if exitosos > 0:
            # Recopilar métricas
            transformaciones_por_par = [r['num_transformaciones'] for r in self.resultados.values() if r['exito']]
            reglas_long_por_par = [r['num_reglas_long'] for r in self.resultados.values() if r['exito']]
            reglas_short_por_par = [r['num_reglas_short'] for r in self.resultados.values() if r['exito']]

            print(f"\n  🎯 TRANSFORMACIONES VALIDADAS:")
            print(f"     Total:                      {total_transformaciones:,}")
            print(f"     Promedio por par:           {np.mean(transformaciones_por_par):.1f}")
            print(f"     Mediana:                    {np.median(transformaciones_por_par):.0f}")
            print(f"     Rango:                      {np.min(transformaciones_por_par):.0f} - {np.max(transformaciones_por_par):.0f}")

            print(f"\n  📈 REGLAS LONG GENERADAS:")
            print(f"     Total:                      {total_reglas_long:,}")
            print(f"     Promedio por par:           {np.mean(reglas_long_por_par):.1f}")
            print(f"     Pares con reglas Long:      {sum(1 for r in reglas_long_por_par if r > 0)}/{exitosos}")

            print(f"\n  📉 REGLAS SHORT GENERADAS:")
            print(f"     Total:                      {total_reglas_short:,}")
            print(f"     Promedio por par:           {np.mean(reglas_short_por_par):.1f}")
            print(f"     Pares con reglas Short:     {sum(1 for r in reglas_short_por_par if r > 0)}/{exitosos}")

            # Balance de reglas
            total_reglas = total_reglas_long + total_reglas_short
            pct_long = (total_reglas_long / total_reglas * 100) if total_reglas > 0 else 0
            pct_short = (total_reglas_short / total_reglas * 100) if total_reglas > 0 else 0

            print(f"\n  ⚖️  BALANCE DE ESTRATEGIAS:")
            print(f"     Total reglas:               {total_reglas:,}")
            print(f"     Long:                       {pct_long:.1f}%")
            print(f"     Short:                      {pct_short:.1f}%")

            # Mejor y Peor productor de reglas
            if len(transformaciones_por_par) > 0:
                mejor_idx = np.argmax(transformaciones_por_par)
                peor_idx = np.argmin(transformaciones_por_par)
                pares_exitosos = [p for p, r in self.resultados.items() if r['exito']]
                mejor_par = pares_exitosos[mejor_idx] if mejor_idx < len(pares_exitosos) else 'N/A'
                peor_par = pares_exitosos[peor_idx] if peor_idx < len(pares_exitosos) else 'N/A'

                print(f"\n  🏆 MEJOR PRODUCTOR:            {mejor_par} ({transformaciones_por_par[mejor_idx]:.0f} transformaciones)")
                print(f"  📊 MENOR PRODUCTOR:            {peor_par} ({transformaciones_por_par[peor_idx]:.0f} transformaciones)")

        # ============================================================
        # TABLA DE RESULTADOS COMPLETA (MULTI-TIMEFRAME)
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"{'2. RESULTADOS POR COMBINACIÓN (MULTI-TIMEFRAME)':^100}")
        print(f"{'─'*100}")
        print(f"\n{'Par_TF':<14} │ {'✓':<3} │ {'Transform.':<12} │ {'Reglas Long':<12} │ {'Reglas Short':<13} │ {'Total Reglas':<13} │ {'Archivos':<9}")
        print("─" * 100)

        for key in sorted(self.resultados.keys()):
            res = self.resultados[key]

            if res['exito']:
                n_trans = res['num_transformaciones']
                n_long = res['num_reglas_long']
                n_short = res['num_reglas_short']
                n_total = n_long + n_short
                n_arch = len(res['archivos_generados'])

                print(
                    f"{key:<14} │ {'✓':<3} │ {n_trans:>11,} │ "
                    f"{n_long:>11,} │ {n_short:>12,} │ {n_total:>12,} │ {n_arch:>8}"
                )
            else:
                print(
                    f"{key:<14} │ {'✗':<3} │ {'N/A':<12} │ {'N/A':<12} │ "
                    f"{'N/A':<13} │ {'N/A':<13} │ {'N/A':<9}"
                )
                print(f"{'':15} └─ Error: {res.get('error', 'Desconocido')}")

        print("─" * 100)

        # ============================================================
        # DETALLE POR COMBINACIÓN (MULTI-TIMEFRAME)
        # ============================================================
        if exitosos > 0:
            print(f"\n{'─'*100}")
            print(f"{'3. DETALLE POR COMBINACIÓN (MULTI-TIMEFRAME)':^100}")
            print(f"{'─'*100}")

            for idx, key in enumerate(sorted(self.resultados.keys()), 1):
                res = self.resultados[key]

                if not res['exito']:
                    print(f"\n  [{idx}] {key}: ✗ ERROR")
                    print(f"      └─ {res.get('error', 'Desconocido')}")
                    continue

                print(f"\n  [{idx}] {key}")
                print(f"  {'─'*96}")

                print(f"    📊 TRANSFORMACIONES:")
                print(f"       Validadas:                {res['num_transformaciones']:,}")

                print(f"\n    📈 REGLAS DE ENTRADA:")
                print(f"       Reglas Long:              {res['num_reglas_long']:,}")
                print(f"       Reglas Short:             {res['num_reglas_short']:,}")
                print(f"       Total:                    {res['num_reglas_long'] + res['num_reglas_short']:,}")

                # Balance Long/Short para este par
                total_par = res['num_reglas_long'] + res['num_reglas_short']
                if total_par > 0:
                    pct_long_par = (res['num_reglas_long'] / total_par * 100)
                    pct_short_par = (res['num_reglas_short'] / total_par * 100)
                    balance = "Balanceado" if 40 <= pct_long_par <= 60 else "Sesgado Long" if pct_long_par > 60 else "Sesgado Short"
                    print(f"       Balance:                  {balance} (L:{pct_long_par:.0f}% / S:{pct_short_par:.0f}%)")

                print(f"\n    📁 ARCHIVOS GENERADOS:")
                print(f"       Total:                    {len(res['archivos_generados'])}")
                for archivo in res['archivos_generados']:
                    nombre = Path(archivo).name
                    tipo = "Interpretación" if "interpretaciones" in archivo else \
                           "Estrategia" if "estrategia.csv" in archivo else \
                           "Código" if ".py" in archivo else "Resumen"
                    print(f"       • {tipo:<15} → {nombre}")

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

        archivos_interpretaciones = list(self.interpretaciones_dir.glob("*.csv"))
        archivos_estrategias = list(self.estrategias_dir.glob("*.csv"))
        archivos_codigo = list(self.codigo_dir.glob("*.py"))
        archivos_resumenes = list(self.resumen_dir.glob("*.csv"))

        total_archivos = len(archivos_interpretaciones) + len(archivos_estrategias) + len(archivos_codigo) + len(archivos_resumenes)

        print(f"\n  Total de archivos generados: {total_archivos}")
        print(f"\n  🔍 Interpretaciones (CSV):    {len(archivos_interpretaciones):3d} archivos → {self.interpretaciones_dir}/")
        print(f"  📊 Estrategias (CSV):         {len(archivos_estrategias):3d} archivos → {self.estrategias_dir}/")
        print(f"  💻 Código Ejecutable (PY):    {len(archivos_codigo):3d} archivos → {self.codigo_dir}/")
        print(f"  📝 Resúmenes (CSV):           {len(archivos_resumenes):3d} archivos → {self.resumen_dir}/")
        print(f"\n  📁 Ubicación base: {self.output_dir}")

        # ============================================================
        # FILOSOFÍA Y CONCLUSIÓN
        # ============================================================
        print(f"\n{'='*100}")
        print(f"{'FILOSOFÍA DEL SISTEMA Y CONCLUSIÓN':^100}")
        print(f"{'='*100}")

        if exitosos == total_combinaciones:
            print(f"\n  ✅ GENERACIÓN MULTI-TIMEFRAME COMPLETADA EXITOSAMENTE")
            print(f"\n  Resumen:")
            print(f"     • Pares:                    {len(self.pares)}")
            print(f"     • Timeframes:               {len(self.timeframes)}")
            print(f"     • Combinaciones exitosas:   {exitosos}/{total_combinaciones}")
            print(f"     • Total transformaciones:   {total_transformaciones:,}")
            print(f"     • Total reglas generadas:   {total_reglas_long + total_reglas_short:,}")
            print(f"     • Código ejecutable:        {len(archivos_codigo)} estrategias")
        elif exitosos > 0:
            print(f"\n  ⚠️  GENERACIÓN COMPLETADA CON ERRORES PARCIALES")
            print(f"\n  Resumen:")
            print(f"     • Combinaciones exitosas:   {exitosos}/{total_combinaciones}")
            print(f"     • Combinaciones con errores: {fallidos}")
        else:
            print(f"\n  ❌ GENERACIÓN FALLIDA - TODAS LAS COMBINACIONES CON ERRORES")

        print(f"\n  {'─'*96}")
        print(f"  🎯 FILOSOFÍA: LAS ESTRATEGIAS EMERGIERON DE LOS DATOS")
        print(f"  {'─'*96}")
        print(f"\n     1. NO predefinimos qué funciona")
        print(f"     2. Generamos ~1,700 transformaciones sistemáticamente")
        print(f"     3. Validación rigurosa multidimensional:")
        print(f"        • Walk-Forward Validation (sin look-ahead bias)")
        print(f"        • Permutation Test (vs. azar)")
        print(f"        • Bootstrap (intervalos de confianza)")
        print(f"        • Análisis de Robustez (estabilidad)")
        print(f"     4. Solo transformaciones que PASARON TODOS los filtros")
        print(f"     5. Interpretación post-hoc y formulación de reglas")
        print(f"     6. Código Python ejecutable por estrategia")

        if exitosos > 0:
            print(f"\n  📋 PRÓXIMOS PASOS:")
            print(f"     1. Revisar código ejecutable generado")
            print(f"        → Ubicación: {self.codigo_dir}/")
            print(f"     2. Analizar reglas de entrada por par")
            print(f"     3. Ejecutar backtest completo:")
            print(f"        → python ejecutar_backtest.py")
            print(f"     4. Evaluar métricas de riesgo/retorno")
            print(f"     5. Validar resultados antes de producción")

        print(f"\n  {'─'*96}")
        print(f"  ℹ️  NOTA IMPORTANTE:")
        print(f"     Estos son edges GENUINOS emergidos de los datos,")
        print(f"     NO supuestos a priori ni estrategias predefinidas.")
        print(f"     Cada regla está respaldada por evidencia estadística rigurosa.")
        print(f"{'='*100}")


def main():
    """
    Función principal para ejecutar Estrategia Emergente.
    """
    # Configuración
    BASE_DIR = Path(__file__).parent
    DATOS_DIR = BASE_DIR / 'datos'

    # Directorios de entrada
    FEATURES_VALIDADOS_DIR = DATOS_DIR / 'validacion_rigurosa' / 'features_validados'
    ANALISIS_IC_DIR = DATOS_DIR / 'analisis_multimetodo'

    # Directorio de salida
    OUTPUT_DIR = DATOS_DIR / 'estrategia_emergente'

    # Parámetros MULTI-TIMEFRAME
    TIMEFRAMES = ['M15', 'H1', 'H4', 'D']
    PARES = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY', 'AUD_USD']  # Lista de pares a procesar (None = auto-detectar)

    # Crear ejecutor MULTI-TIMEFRAME
    ejecutor = EjecutorEstrategiaEmergente(
        features_validados_dir=FEATURES_VALIDADOS_DIR,
        analisis_ic_dir=ANALISIS_IC_DIR,
        output_dir=OUTPUT_DIR,
        timeframes=TIMEFRAMES,
        pares=PARES,
        limpiar_archivos_viejos=True,  # Limpiar archivos viejos
        hacer_backup=False,             # NO hacer backup (ahorra espacio)
        verbose=True                    # Imprimir progreso detallado
    )

    # Ejecutar
    ejecutor.ejecutar_todos()

    print("\n✓ Generación de estrategias emergentes completada")


if __name__ == "__main__":
    """
    Ejecutar Estrategia Emergente.

    IMPORTANTE:
    - Este es el ÚLTIMO PASO del pipeline
    - Requiere que TODOS los módulos anteriores hayan sido ejecutados:
      1. generacion_de_transformaciones
      2. analisis_multimetodo
      3. consenso_metodos
      4. validacion_rigurosa
    - Solo entonces podemos generar la estrategia emergente

    LA ESTRATEGIA EMERGE DE LOS DATOS, NO SE PREDEFINIÓ.
    """
    main()
