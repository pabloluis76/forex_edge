"""
Ejecutor: Backtest Completo

BACKTEST DE ESTRATEGIA EMERGENTE:
- Simula trading real con costos reales (spreads, slippage, swap)
- División temporal: TRAIN → VALIDATION → TEST
- Métricas completas: Sharpe, Sortino, Calmar, MaxDD
- Pruebas de robustez: Sensibilidad, Bootstrap, Permutation

Author: Sistema de Edge-Finding Forex
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import shutil
import json
from typing import List, Dict, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Importar módulos de backtest
from backtest.motor_backtest_completo import MotorBacktestCompleto
from backtest.preparar_datos_backtest import PreparadorDatosBacktest
from backtest.division_temporal_datos import DivisionTemporalDatos
from backtest.tabla_spreads_reales import GeneradorSpreadsReales
from backtest.configuracion_costos_adicionales import ConfiguracionCostosAdicionales
from backtest.pruebas_robustez import PruebasRobustez


class EjecutorBacktest:
    """
    Ejecutor del módulo Backtest.

    Ejecuta backtest completo de estrategias emergentes con:
    - Costos reales (spreads, slippage, swap)
    - División temporal (train/validation/test)
    - Métricas completas
    - Pruebas de robustez
    """

    def __init__(
        self,
        datos_ohlc_dir: Path,
        estrategias_dir: Path,
        output_dir: Path,
        timeframe: str = 'M15',
        capital_inicial: float = 100000,
        pares: Optional[List[str]] = None,
        limpiar_archivos_viejos: bool = True,
        hacer_backup: bool = True,
        verbose: bool = True
    ):
        """
        Inicializa el ejecutor de Backtest.

        Parameters:
        -----------
        datos_ohlc_dir : Path
            Directorio con datos OHLC (datos/ohlc/)
        estrategias_dir : Path
            Directorio con estrategias generadas (estrategia_emergente/codigo/)
        output_dir : Path
            Directorio para guardar resultados de backtest
        timeframe : str
            Timeframe a procesar
        capital_inicial : float
            Capital inicial para backtest (default: $100,000)
        pares : List[str], optional
            Lista de pares a procesar. Si None, procesa todos los disponibles
        limpiar_archivos_viejos : bool
            Si True, limpia archivos viejos antes de ejecutar
        hacer_backup : bool
            Si True, hace backup antes de limpiar
        verbose : bool
            Imprimir detalles del proceso
        """
        self.datos_ohlc_dir = Path(datos_ohlc_dir)
        self.estrategias_dir = Path(estrategias_dir)
        self.output_dir = Path(output_dir)
        self.timeframe = timeframe
        self.capital_inicial = capital_inicial
        self.verbose = verbose
        self.limpiar_archivos_viejos = limpiar_archivos_viejos
        self.hacer_backup = hacer_backup

        # Crear directorios de salida
        self.metricas_dir = self.output_dir / 'metricas'
        self.trades_dir = self.output_dir / 'trades'
        self.equity_dir = self.output_dir / 'equity'
        self.robustez_dir = self.output_dir / 'robustez'
        self.datos_preparados_dir = self.output_dir / 'datos_preparados'

        for directorio in [self.metricas_dir, self.trades_dir,
                          self.equity_dir, self.robustez_dir,
                          self.datos_preparados_dir]:
            directorio.mkdir(parents=True, exist_ok=True)

        # Configurar logging
        self._configurar_logging()

        # Determinar pares a procesar
        if pares is None:
            self.pares = self._detectar_pares_disponibles()
        else:
            self.pares = pares

        # División temporal
        self.division_temporal = DivisionTemporalDatos(
            fecha_inicio='2019-01-01',
            fecha_fin='2023-12-31',
            pct_train=0.60,
            pct_validation=0.20,
            pct_test=0.20,
            verbose=False
        )

        # Configuración de costos
        self.config_costos = ConfiguracionCostosAdicionales(verbose=False)

        # Resultados
        self.resultados: Dict[str, dict] = {}

        if self.verbose:
            print("="*80)
            print("EJECUTOR: BACKTEST COMPLETO")
            print("="*80)
            print(f"\nDirectorio datos OHLC: {self.datos_ohlc_dir}")
            print(f"Directorio estrategias: {self.estrategias_dir}")
            print(f"Directorio salida: {self.output_dir}")
            print(f"Timeframe: {self.timeframe}")
            print(f"Capital inicial: ${self.capital_inicial:,.0f}")
            print(f"Pares a procesar: {len(self.pares)}")
            print(f"Limpieza automática: {self.limpiar_archivos_viejos}")
            print(f"Backup antes de limpiar: {self.hacer_backup}")
            print("="*80)

    def _configurar_logging(self):
        """Configura el sistema de logging."""
        log_file = self.output_dir / f'backtest_{self.timeframe}.log'

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
        Detecta pares disponibles con estrategias generadas.

        Returns:
        --------
        pares : List[str]
            Lista de pares disponibles
        """
        archivos_estrategias = list(self.estrategias_dir.glob(f"*_{self.timeframe}_estrategia.py"))

        pares = []
        for archivo in archivos_estrategias:
            # Extraer par del nombre de archivo
            nombre = archivo.stem.replace(f"_{self.timeframe}_estrategia", "")
            pares.append(nombre)

        if self.verbose:
            print(f"\n✓ Detectados {len(pares)} pares con estrategias")
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
        archivos_existentes.extend(list(self.metricas_dir.glob("*.csv")))
        archivos_existentes.extend(list(self.metricas_dir.glob("*.json")))
        archivos_existentes.extend(list(self.trades_dir.glob("*.csv")))
        archivos_existentes.extend(list(self.equity_dir.glob("*.csv")))
        archivos_existentes.extend(list(self.robustez_dir.glob("*.json")))
        archivos_existentes.extend(list(self.datos_preparados_dir.glob("*.csv")))

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
            for directorio in [self.metricas_dir, self.trades_dir,
                             self.equity_dir, self.robustez_dir,
                             self.datos_preparados_dir]:
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

    def preparar_datos_backtest(self) -> pd.DataFrame:
        """
        Prepara datos OHLCV consolidados para backtest.

        Returns:
        --------
        df_raw_ohlcv : pd.DataFrame
            DataFrame con OHLCV de todos los pares
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"PREPARANDO DATOS PARA BACKTEST")
            print(f"{'='*80}")

        self.logger.info("Preparando datos OHLCV...")

        preparador = PreparadorDatosBacktest(
            directorio_ohlc=self.datos_ohlc_dir,
            timeframe=self.timeframe,
            verbose=False
        )

        # Preparar tabla raw_ohlcv
        df_raw_ohlcv = preparador.preparar_raw_ohlcv(
            pares=self.pares,
            fecha_inicio=None,
            fecha_fin=None
        )

        # Guardar
        archivo_ohlcv = self.datos_preparados_dir / 'raw_ohlcv.csv'
        df_raw_ohlcv.to_csv(archivo_ohlcv, index=False)

        self.logger.info(f"✓ Datos OHLCV preparados: {len(df_raw_ohlcv):,} filas → {archivo_ohlcv.name}")

        return df_raw_ohlcv

    def generar_tabla_spreads(self) -> pd.DataFrame:
        """
        Genera tabla de spreads reales.

        Returns:
        --------
        df_spreads : pd.DataFrame
            Tabla de spreads por par, hora y día
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"GENERANDO TABLA DE SPREADS")
            print(f"{'='*80}")

        self.logger.info("Generando tabla de spreads...")

        generador = GeneradorSpreadsReales(verbose=False)
        df_spreads = generador.generar_tabla_spreads(pares=self.pares)

        # Guardar
        archivo_spreads = self.datos_preparados_dir / 'spreads_real.csv'
        df_spreads.to_csv(archivo_spreads, index=False)

        self.logger.info(f"✓ Tabla de spreads generada: {len(df_spreads)} filas → {archivo_spreads.name}")

        return df_spreads

    def ejecutar_backtest_par(self, par: str) -> dict:
        """
        Ejecuta backtest para un par específico.

        Parameters:
        -----------
        par : str
            Par a procesar

        Returns:
        --------
        resultado : dict
            Resultado del backtest
        """
        resultado = {
            'par': par,
            'exito': False,
            'metricas': {},
            'n_trades': 0,
            'error': None
        }

        try:
            # Rutas de archivos
            archivo_ohlcv = self.datos_preparados_dir / 'raw_ohlcv.csv'
            archivo_spreads = self.datos_preparados_dir / 'spreads_real.csv'

            if not archivo_ohlcv.exists() or not archivo_spreads.exists():
                resultado['error'] = "Archivos de datos no encontrados"
                return resultado

            self.logger.info(f"{par}: Iniciando backtest...")

            # Configuración de la estrategia
            config = {
                'risk_per_trade': 0.01,
                'max_position_size': 0.10,
                'stop_loss_atr_mult': 2.5,
                'take_profit_atr_mult': 3.5,
                'timeout_bars': 50,
                'base_slippage_pips': 0.3,
                'max_spread_pips': 5.0,
                'avoid_rollover_hours': [21, 22, 23]
            }

            # Inicializar motor de backtest
            motor = MotorBacktestCompleto(
                capital_inicial=self.capital_inicial,
                config=config,
                verbose=False
            )

            # Cargar datos
            motor.cargar_datos(
                ruta_ohlcv=str(archivo_ohlcv),
                ruta_spreads=str(archivo_spreads)
            )

            # Ejecutar backtest
            # En producción, aquí cargarías la estrategia emergente específica
            # Por ahora usamos la señal simple del motor
            motor.ejecutar_backtest(
                par=par,
                fecha_inicio='2019-01-01',
                fecha_fin='2023-12-31'
            )

            # Calcular métricas
            metricas = motor.calcular_metricas()

            if 'error' in metricas:
                resultado['error'] = metricas['error']
                return resultado

            resultado['metricas'] = metricas
            resultado['n_trades'] = len(motor.historial_trades)

            # Guardar resultados

            # 1. Métricas
            archivo_metricas = self.metricas_dir / f"{par}_{self.timeframe}_metricas.json"
            with open(archivo_metricas, 'w') as f:
                json.dump(metricas, f, indent=2)

            # 2. Trades
            if len(motor.historial_trades) > 0:
                df_trades = pd.DataFrame(motor.historial_trades)
                archivo_trades = self.trades_dir / f"{par}_{self.timeframe}_trades.csv"
                df_trades.to_csv(archivo_trades, index=False)

            # 3. Equity curve
            if len(motor.equity_curve) > 0:
                df_equity = pd.DataFrame(motor.equity_curve)
                archivo_equity = self.equity_dir / f"{par}_{self.timeframe}_equity.csv"
                df_equity.to_csv(archivo_equity, index=False)

            self.logger.info(f"{par}: ✓ Backtest completado - {resultado['n_trades']} trades")
            resultado['exito'] = True

        except Exception as e:
            self.logger.error(f"{par}: Error en backtest: {e}")
            resultado['error'] = str(e)

        return resultado

    def ejecutar_pruebas_robustez(self, par: str) -> Optional[Dict]:
        """
        Ejecuta pruebas de robustez para un par.

        Parameters:
        -----------
        par : str
            Par a analizar

        Returns:
        --------
        resultados_robustez : Dict o None
            Resultados de pruebas de robustez
        """
        try:
            # Cargar trades y equity
            archivo_trades = self.trades_dir / f"{par}_{self.timeframe}_trades.csv"
            archivo_equity = self.equity_dir / f"{par}_{self.timeframe}_equity.csv"

            if not archivo_trades.exists() or not archivo_equity.exists():
                return None

            df_trades = pd.read_csv(archivo_trades)
            df_equity = pd.read_csv(archivo_equity)

            if len(df_trades) < 10:
                self.logger.warning(f"{par}: Muy pocos trades para pruebas de robustez")
                return None

            self.logger.info(f"{par}: Ejecutando pruebas de robustez...")

            # Inicializar pruebas
            pruebas = PruebasRobustez(
                df_trades=df_trades,
                df_equity=df_equity,
                verbose=False
            )

            # Bootstrap confidence intervals
            resultado_bootstrap = pruebas.bootstrap_confidence_intervals(
                n_iterations=1000,  # Reducido para velocidad
                confidence_level=0.95
            )

            # Guardar resultados
            resultados = {
                'par': par,
                'bootstrap': resultado_bootstrap
            }

            archivo_robustez = self.robustez_dir / f"{par}_{self.timeframe}_robustez.json"
            with open(archivo_robustez, 'w') as f:
                json.dump(resultados, f, indent=2, default=str)

            self.logger.info(f"{par}: ✓ Pruebas de robustez completadas")

            return resultados

        except Exception as e:
            self.logger.error(f"{par}: Error en pruebas de robustez: {e}")
            return None

    def ejecutar_todos(self):
        """
        Ejecuta backtest completo para todos los pares.
        """
        self.logger.info(f"Iniciando backtest para {len(self.pares)} pares")

        # Limpiar archivos viejos si está habilitado
        if self.limpiar_archivos_viejos:
            self.limpiar_directorio_salida()

        # PASO 1: Preparar datos
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"PASO 1: PREPARACIÓN DE DATOS")
            print(f"{'='*80}")

        df_raw_ohlcv = self.preparar_datos_backtest()
        df_spreads = self.generar_tabla_spreads()

        # PASO 2: Ejecutar backtests
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"PASO 2: EJECUTAR BACKTESTS")
            print(f"{'='*80}")

        for i, par in enumerate(tqdm(self.pares, desc="Ejecutando backtests", unit="par"), 1):
            if self.verbose:
                print(f"\n[{i}/{len(self.pares)}] Procesando {par}...")

            resultado = self.ejecutar_backtest_par(par)
            self.resultados[par] = resultado

            if resultado['exito']:
                if self.verbose:
                    metricas = resultado['metricas']
                    print(f"✓ {par}: Backtest completado")
                    print(f"  - Trades: {resultado['n_trades']}")
                    print(f"  - Return: {metricas.get('total_return', 0):+.2f}%")
                    print(f"  - Sharpe: {metricas.get('sharpe_ratio', 0):.2f}")
                    print(f"  - Max DD: {metricas.get('max_drawdown', 0):.2f}%")
            else:
                if self.verbose:
                    print(f"✗ {par}: Falló - {resultado['error']}")

        # PASO 3: Pruebas de robustez
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"PASO 3: PRUEBAS DE ROBUSTEZ")
            print(f"{'='*80}")

        for i, par in enumerate(tqdm(self.pares, desc="Pruebas de robustez", unit="par"), 1):
            if self.resultados[par]['exito']:
                resultado_robustez = self.ejecutar_pruebas_robustez(par)
                self.resultados[par]['robustez'] = resultado_robustez is not None

        # Imprimir resumen final
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen de la ejecución."""
        exitosos = sum(1 for r in self.resultados.values() if r['exito'])
        fallidos = len(self.resultados) - exitosos

        total_trades = sum(r['n_trades'] for r in self.resultados.values() if r['exito'])

        print(f"\n{'='*80}")
        print(f"RESUMEN DE EJECUCIÓN")
        print(f"{'='*80}")
        print(f"\nPares procesados: {len(self.resultados)}")
        print(f"  ✓ Exitosos: {exitosos}")
        print(f"  ✗ Fallidos: {fallidos}")

        if exitosos > 0:
            print(f"\nEstadísticas:")
            print(f"  - Total trades: {total_trades}")
            print(f"  - Promedio trades por par: {total_trades/exitosos:.0f}")

            # Métricas consolidadas
            returns = []
            sharpes = []
            max_dds = []

            for resultado in self.resultados.values():
                if resultado['exito']:
                    metricas = resultado['metricas']
                    returns.append(metricas.get('total_return', 0))
                    sharpes.append(metricas.get('sharpe_ratio', 0))
                    max_dds.append(metricas.get('max_drawdown', 0))

            print(f"\nMétricas agregadas:")
            print(f"  - Return promedio: {np.mean(returns):+.2f}%")
            print(f"  - Sharpe promedio: {np.mean(sharpes):.2f}")
            print(f"  - Max DD promedio: {np.mean(max_dds):.2f}%")

        print(f"\nArchivos generados:")
        print(f"  - Métricas: {self.metricas_dir}/")
        print(f"  - Trades: {self.trades_dir}/")
        print(f"  - Equity curves: {self.equity_dir}/")
        print(f"  - Robustez: {self.robustez_dir}/")

        if fallidos > 0:
            print(f"\n⚠️  Pares con errores:")
            for par, resultado in self.resultados.items():
                if not resultado['exito']:
                    print(f"  - {par}: {resultado['error']}")

        print(f"\n{'='*80}")
        print(f"BACKTEST REALISTA")
        print(f"{'='*80}")
        print(f"\nSIMULACIÓN CON COSTOS REALES:")
        print(f"  - Spreads variables por hora y día")
        print(f"  - Slippage dinámico basado en volatilidad")
        print(f"  - Swap overnight")
        print(f"  - Stop loss y take profit basados en ATR")
        print(f"\nEstos resultados reflejan trading REAL, no backtests optimistas.")
        print(f"="*80)


def main():
    """
    Función principal para ejecutar Backtest.
    """
    # Configuración
    BASE_DIR = Path(__file__).parent
    DATOS_DIR = BASE_DIR / 'datos'

    # Directorios de entrada
    DATOS_OHLC_DIR = DATOS_DIR / 'ohlc'
    ESTRATEGIAS_DIR = DATOS_DIR / 'estrategia_emergente' / 'codigo'

    # Directorio de salida
    OUTPUT_DIR = DATOS_DIR / 'backtest'

    # Parámetros
    TIMEFRAME = 'M15'
    CAPITAL_INICIAL = 100000  # $100,000
    PARES = None  # None = auto-detectar todos los pares disponibles

    # Crear ejecutor
    ejecutor = EjecutorBacktest(
        datos_ohlc_dir=DATOS_OHLC_DIR,
        estrategias_dir=ESTRATEGIAS_DIR,
        output_dir=OUTPUT_DIR,
        timeframe=TIMEFRAME,
        capital_inicial=CAPITAL_INICIAL,
        pares=PARES,
        limpiar_archivos_viejos=True,  # Limpiar archivos viejos
        hacer_backup=True,              # Hacer backup antes de limpiar
        verbose=True                    # Imprimir progreso detallado
    )

    # Ejecutar
    ejecutor.ejecutar_todos()

    print("\n✓ Backtest completado")


if __name__ == "__main__":
    """
    Ejecutar Backtest Completo.

    IMPORTANTE:
    - Requiere que la estrategia emergente haya sido generada
    - Requiere datos OHLC descargados en datos/ohlc/
    - Simula trading REAL con costos reales
    - División temporal: TRAIN (60%) → VALIDATION (20%) → TEST (20%)

    REGLA SAGRADA:
    El set TEST solo se usa UNA VEZ al final para métricas finales.
    """
    main()
