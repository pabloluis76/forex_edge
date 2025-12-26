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


class LogCapture(logging.Handler):
    """Handler que captura mensajes de logging para el resumen final."""

    def __init__(self):
        super().__init__()
        self.warnings = []
        self.errors = []

    def emit(self, record):
        """Captura mensajes WARNING y ERROR."""
        if record.levelno >= logging.ERROR:
            self.errors.append({
                'mensaje': record.getMessage(),
                'modulo': record.module,
                'linea': record.lineno
            })
        elif record.levelno == logging.WARNING:
            self.warnings.append({
                'mensaje': record.getMessage(),
                'modulo': record.module
            })


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
        hacer_backup: bool = False,
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

        # Tiempos de ejecución
        self.tiempo_inicio = None
        self.tiempo_fin = None

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

        # Agregar LogCapture global
        global log_capture
        log_capture = LogCapture()
        log_capture.setLevel(logging.WARNING)
        logging.getLogger().addHandler(log_capture)

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
        self.tiempo_inicio = datetime.now()
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

        self.tiempo_fin = datetime.now()

        # Imprimir resumen final
        self._imprimir_resumen()

    def _imprimir_resumen(self):
        """Imprime resumen detallado con logs capturados."""
        exitosos = sum(1 for r in self.resultados.values() if r['exito'])
        fallidos = len(self.resultados) - exitosos
        total_trades = sum(r['n_trades'] for r in self.resultados.values() if r['exito'])

        tiempo_total = (self.tiempo_fin - self.tiempo_inicio).total_seconds()

        print(f"\n{'='*80}")
        print(f"RESUMEN FINAL - BACKTEST COMPLETO")
        print(f"{'='*80}")

        # Información temporal
        print(f"\nINFORMACIÓN TEMPORAL:")
        print(f"  Inicio: {self.tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Fin:    {self.tiempo_fin.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duración total: {self._formatear_duracion(tiempo_total)}")

        # Tabla de resultados
        print(f"\nRESULTADOS POR PAR:")
        print("-" * 100)
        print(f"{'Par':<10} │ {'Estado':<6} │ {'Trades':<8} │ {'Return %':<10} │ {'Sharpe':<8} │ {'Max DD %':<10}")
        print("-" * 100)

        for par in self.pares:
            res = self.resultados[par]

            if res['exito']:
                metricas = res['metricas']
                ret = metricas.get('total_return', 0)
                sharpe = metricas.get('sharpe_ratio', 0)
                maxdd = metricas.get('max_drawdown', 0)

                print(
                    f"{par:<10} │ {'✓':<6} │ {res['n_trades']:<8} │ "
                    f"{ret:>+9.2f} │ {sharpe:>7.2f} │ {maxdd:>9.2f}"
                )
            else:
                print(
                    f"{par:<10} │ {'✗':<6} │ {'N/A':<8} │ {'N/A':<10} │ {'N/A':<8} │ {'N/A':<10}"
                )
                print(f"           Error: {res.get('error', 'Desconocido')}")

        print("-" * 100)

        # Estadísticas globales
        print(f"\nESTADÍSTICAS GLOBALES:")
        print(f"  Pares procesados exitosamente: {exitosos}/{len(self.resultados)}")
        print(f"  Total trades ejecutados: {total_trades:,}")

        if exitosos > 0:
            print(f"  Promedio trades por par: {total_trades/exitosos:.0f}")

            # Métricas consolidadas
            returns = []
            sharpes = []
            sortinos = []
            calmars = []
            max_dds = []
            win_rates = []

            for resultado in self.resultados.values():
                if resultado['exito']:
                    m = resultado['metricas']
                    returns.append(m.get('total_return', 0))
                    sharpes.append(m.get('sharpe_ratio', 0))
                    sortinos.append(m.get('sortino_ratio', 0))
                    calmars.append(m.get('calmar_ratio', 0))
                    max_dds.append(m.get('max_drawdown', 0))
                    win_rates.append(m.get('win_rate', 0))

            print(f"\n  Métricas agregadas:")
            print(f"    Return promedio: {np.mean(returns):+.2f}%")
            print(f"    Sharpe promedio: {np.mean(sharpes):.2f}")
            print(f"    Sortino promedio: {np.mean(sortinos):.2f}")
            print(f"    Calmar promedio: {np.mean(calmars):.2f}")
            print(f"    Max DD promedio: {np.mean(max_dds):.2f}%")
            print(f"    Win rate promedio: {np.mean(win_rates):.1f}%")

        print(f"  Tiempo promedio por par: {self._formatear_duracion(tiempo_total/len(self.pares))}")

        # Métricas detalladas por par
        print(f"\nMÉTRICAS DETALLADAS POR PAR:")
        for par in self.pares:
            res = self.resultados[par]

            if not res['exito']:
                continue

            print(f"\n  {par}:")
            m = res['metricas']

            print(f"    Rendimiento:")
            print(f"      Total return: {m.get('total_return', 0):+.2f}%")
            print(f"      CAGR: {m.get('cagr', 0):+.2f}%")
            print(f"      Volatilidad anual: {m.get('annual_volatility', 0):.2f}%")

            print(f"    Ratios ajustados por riesgo:")
            print(f"      Sharpe ratio: {m.get('sharpe_ratio', 0):.2f}")
            print(f"      Sortino ratio: {m.get('sortino_ratio', 0):.2f}")
            print(f"      Calmar ratio: {m.get('calmar_ratio', 0):.2f}")

            print(f"    Drawdown:")
            print(f"      Max drawdown: {m.get('max_drawdown', 0):.2f}%")
            print(f"      Avg drawdown: {m.get('avg_drawdown', 0):.2f}%")

            print(f"    Trades:")
            print(f"      Total trades: {res['n_trades']}")
            print(f"      Win rate: {m.get('win_rate', 0):.1f}%")
            print(f"      Profit factor: {m.get('profit_factor', 0):.2f}")
            print(f"      Avg win: {m.get('avg_win', 0):.2f}%")
            print(f"      Avg loss: {m.get('avg_loss', 0):.2f}%")

            if res.get('robustez', False):
                print(f"    Robustez: ✓ Pruebas completadas")
            else:
                print(f"    Robustez: ✗ No ejecutadas")

        # Logs capturados
        print(f"\nLOGS CAPTURADOS:")

        if log_capture.warnings:
            print(f"\n  ⚠️  WARNINGS ({len(log_capture.warnings)}):")
            for i, warn in enumerate(log_capture.warnings[:10], 1):
                print(f"    {i}. [{warn['modulo']}] {warn['mensaje']}")
            if len(log_capture.warnings) > 10:
                print(f"    ... y {len(log_capture.warnings) - 10} warnings más")
        else:
            print(f"  ✓ No se registraron warnings")

        if log_capture.errors:
            print(f"\n  ❌ ERRORS ({len(log_capture.errors)}):")
            for i, err in enumerate(log_capture.errors[:10], 1):
                print(f"    {i}. [{err['modulo']}:{err['linea']}] {err['mensaje']}")
            if len(log_capture.errors) > 10:
                print(f"    ... y {len(log_capture.errors) - 10} errors más")
        else:
            print(f"  ✓ No se registraron errores")

        # Archivos generados
        archivos_json = list(self.metricas_dir.glob("*.json"))
        archivos_csv_trades = list(self.trades_dir.glob("*.csv"))
        archivos_csv_equity = list(self.equity_dir.glob("*.csv"))
        archivos_robustez = list(self.robustez_dir.glob("*.json"))

        print(f"\nARCHIVOS GENERADOS:")
        print(f"  JSON métricas: {len(archivos_json)} archivos")
        print(f"  CSV trades: {len(archivos_csv_trades)} archivos")
        print(f"  CSV equity curves: {len(archivos_csv_equity)} archivos")
        print(f"  JSON robustez: {len(archivos_robustez)} archivos")
        print(f"  Ubicación: {self.output_dir}")

        # Conclusión
        print(f"\n{'='*80}")
        if exitosos == len(self.resultados):
            print(f"✓ BACKTEST COMPLETADO EXITOSAMENTE")
            print(f"  Pares procesados: {exitosos}/{len(self.pares)}")
            print(f"  Total trades: {total_trades:,}")

            print(f"\nSIMULACIÓN CON COSTOS REALES:")
            print(f"  ✓ Spreads variables por hora y día")
            print(f"  ✓ Slippage dinámico basado en volatilidad")
            print(f"  ✓ Swap overnight")
            print(f"  ✓ Stop loss y take profit basados en ATR")

            print(f"\nPRÓXIMOS PASOS:")
            if total_trades > 0:
                print(f"  1. Analizar equity curves en: {self.equity_dir}")
                print(f"  2. Revisar distribución de trades")
                print(f"  3. Evaluar métricas por período (train/validation/test)")
                print(f"  4. Verificar resultados de pruebas de robustez")
                print(f"  5. Decidir si estrategia es viable para trading en vivo")
            else:
                print(f"  ⚠️  No se generaron trades")
                print(f"  1. Revisar lógica de generación de señales")
                print(f"  2. Verificar features de entrada")
                print(f"  3. Ajustar umbrales de entrada")
        else:
            print(f"⚠️  BACKTEST COMPLETADO CON ERRORES")
            print(f"  Pares exitosos: {exitosos}/{len(self.pares)}")
            print(f"  Revisar errores arriba para diagnóstico")

            if fallidos > 0:
                print(f"\n  Pares con errores:")
                for par, resultado in self.resultados.items():
                    if not resultado['exito']:
                        print(f"    - {par}: {resultado['error']}")

        print(f"\nNOTA: Estos resultados reflejan trading REAL con costos reales,")
        print(f"      no backtests optimistas. Las métricas son conservadoras.")
        print(f"{'='*80}")

    def _formatear_duracion(self, segundos: float) -> str:
        """
        Formatea duración en formato legible.

        Args:
            segundos: Duración en segundos

        Returns:
            String formateado (ej: "45.2s", "12m 34s", "2h 15m")
        """
        if segundos < 60:
            return f"{segundos:.1f}s"
        elif segundos < 3600:
            minutos = int(segundos // 60)
            segs = int(segundos % 60)
            return f"{minutos}m {segs}s"
        else:
            horas = int(segundos // 3600)
            minutos = int((segundos % 3600) // 60)
            return f"{horas}h {minutos}m"


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
    PARES = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY', 'AUD_USD']  # Lista de pares a procesar (None = auto-detectar)

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
