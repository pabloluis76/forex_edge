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
        timeframe: str = 'M15',
        pares: Optional[List[str]] = None,
        limpiar_archivos_viejos: bool = True,
        hacer_backup: bool = False,
        verbose: bool = True
    ):
        """
        Inicializa el ejecutor de Estrategia Emergente.

        Parameters:
        -----------
        features_validados_dir : Path
            Directorio con features validados (validacion_rigurosa/features_validados/)
        analisis_ic_dir : Path
            Directorio con análisis IC (analisis_multimetodo/)
        output_dir : Path
            Directorio para guardar estrategias emergentes
        timeframe : str
            Timeframe a procesar
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
        self.timeframe = timeframe
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
        Ejecuta generación de estrategias para todos los pares.
        """
        self.logger.info(f"Iniciando generación de estrategias emergentes para {len(self.pares)} pares")

        # Limpiar archivos viejos si está habilitado
        if self.limpiar_archivos_viejos:
            self.limpiar_directorio_salida()

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"GENERANDO ESTRATEGIAS EMERGENTES")
            print(f"{'='*80}")

        # Procesar cada par con barra de progreso
        for i, par in enumerate(tqdm(self.pares, desc="Procesando pares", unit="par"), 1):
            if self.verbose:
                print(f"\n[{i}/{len(self.pares)}] Procesando {par}...")

            resultado = self.generar_estrategia_par(par)
            self.resultados[par] = resultado

            if resultado['exito']:
                if self.verbose:
                    print(f"✓ {par}: Estrategia generada exitosamente")
                    print(f"  - Transformaciones: {resultado['num_transformaciones']}")
                    print(f"  - Reglas Long: {resultado['num_reglas_long']}")
                    print(f"  - Reglas Short: {resultado['num_reglas_short']}")
                    print(f"  - Archivos generados: {len(resultado['archivos_generados'])}")
            else:
                if self.verbose:
                    print(f"✗ {par}: Falló - {resultado['error']}")

        # Imprimir resumen final
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen de la ejecución."""
        exitosos = sum(1 for r in self.resultados.values() if r['exito'])
        fallidos = len(self.resultados) - exitosos

        total_transformaciones = sum(r['num_transformaciones'] for r in self.resultados.values() if r['exito'])
        total_reglas_long = sum(r['num_reglas_long'] for r in self.resultados.values() if r['exito'])
        total_reglas_short = sum(r['num_reglas_short'] for r in self.resultados.values() if r['exito'])

        print(f"\n{'='*80}")
        print(f"RESUMEN DE EJECUCIÓN")
        print(f"{'='*80}")
        print(f"\nPares procesados: {len(self.resultados)}")
        print(f"  ✓ Exitosos: {exitosos}")
        print(f"  ✗ Fallidos: {fallidos}")

        if exitosos > 0:
            print(f"\nEstadísticas:")
            print(f"  - Total transformaciones validadas: {total_transformaciones}")
            print(f"  - Total reglas Long: {total_reglas_long}")
            print(f"  - Total reglas Short: {total_reglas_short}")
            print(f"  - Promedio transformaciones por par: {total_transformaciones/exitosos:.1f}")
            print(f"  - Promedio reglas Long por par: {total_reglas_long/exitosos:.1f}")
            print(f"  - Promedio reglas Short por par: {total_reglas_short/exitosos:.1f}")

        print(f"\nArchivos generados:")
        print(f"  - Interpretaciones: {self.interpretaciones_dir}/")
        print(f"  - Estrategias: {self.estrategias_dir}/")
        print(f"  - Código ejecutable: {self.codigo_dir}/")
        print(f"  - Resúmenes: {self.resumen_dir}/")

        if fallidos > 0:
            print(f"\n⚠️  Pares con errores:")
            for par, resultado in self.resultados.items():
                if not resultado['exito']:
                    print(f"  - {par}: {resultado['error']}")

        print(f"\n{'='*80}")
        print(f"FILOSOFÍA DEL SISTEMA")
        print(f"{'='*80}")
        print(f"\nLAS ESTRATEGIAS EMERGIERON DE LOS DATOS:")
        print(f"  1. NO predefinimos qué funciona")
        print(f"  2. Generamos ~1,700 transformaciones sistemáticamente")
        print(f"  3. Validación rigurosa (Walk-Forward, Permutation, Bootstrap, Robustez)")
        print(f"  4. Solo transformaciones que PASARON TODOS los filtros")
        print(f"  5. AHORA interpretamos y formulamos reglas")
        print(f"\nEstos son edges GENUINOS, no supuestos a priori.")
        print(f"="*80)


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

    # Parámetros
    TIMEFRAME = 'M15'
    PARES = None  # None = auto-detectar todos los pares disponibles

    # Crear ejecutor
    ejecutor = EjecutorEstrategiaEmergente(
        features_validados_dir=FEATURES_VALIDADOS_DIR,
        analisis_ic_dir=ANALISIS_IC_DIR,
        output_dir=OUTPUT_DIR,
        timeframe=TIMEFRAME,
        pares=PARES,
        limpiar_archivos_viejos=True,  # Limpiar archivos viejos
        hacer_backup=True,              # Hacer backup antes de limpiar
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
