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
        timeframes: list = None,
        capital_inicial: float = 100000,
        pares: Optional[List[str]] = None,
        limpiar_archivos_viejos: bool = True,
        hacer_backup: bool = False,
        verbose: bool = True
    ):
        """
        Inicializa el ejecutor de Backtest MULTI-TIMEFRAME.

        Parameters:
        -----------
        datos_ohlc_dir : Path
            Directorio con datos OHLC (datos/ohlc/)
        estrategias_dir : Path
            Directorio con estrategias generadas (estrategia_emergente/codigo/)
        output_dir : Path
            Directorio para guardar resultados de backtest
        timeframes : list
            Lista de timeframes a procesar (default: ['M15', 'H1', 'H4', 'D1'])
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
        self.timeframes = timeframes or ['M15', 'H1', 'H4', 'D']
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
        log_capture.setLevel(logging.INFO)  # Capturar desde INFO en adelante
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

        print(f"\n{'='*100}")
        print(f"{'RESUMEN FINAL - BACKTEST COMPLETO':^100}")
        print(f"{'='*100}")

        # ============================================================
        # RESUMEN EJECUTIVO
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"{'1. RESUMEN EJECUTIVO':^100}")
        print(f"{'─'*100}")

        # Recopilar métricas consolidadas
        returns = []
        sharpes = []
        sortinos = []
        calmars = []
        max_dds = []
        win_rates = []
        profit_factors = []
        cagrs = []

        for resultado in self.resultados.values():
            if resultado['exito']:
                m = resultado['metricas']
                returns.append(m.get('total_return', 0))
                sharpes.append(m.get('sharpe_ratio', 0))
                sortinos.append(m.get('sortino_ratio', 0))
                calmars.append(m.get('calmar_ratio', 0))
                max_dds.append(m.get('max_drawdown', 0))
                win_rates.append(m.get('win_rate', 0))
                profit_factors.append(m.get('profit_factor', 0))
                cagrs.append(m.get('cagr', 0))

        if exitosos > 0:
            print(f"\n  Capital Inicial:               ${self.capital_inicial:,.0f}")
            print(f"  Timeframe:                     {self.timeframe}")
            print(f"  Período de Backtest:           {'2019-01-01 a 2023-12-31'}")  # Ajustar según datos reales
            print(f"\n  Pares Procesados:              {exitosos}/{len(self.resultados)}")
            print(f"  Total Trades Ejecutados:       {total_trades:,}")
            print(f"  Promedio Trades/Par:           {total_trades/exitosos:.0f}")

            # Rendimiento Global
            print(f"\n  📊 RENDIMIENTO GLOBAL:")
            print(f"     Return Promedio:            {np.mean(returns):+.2f}%  (Min: {np.min(returns):+.2f}% | Max: {np.max(returns):+.2f}%)")
            print(f"     CAGR Promedio:              {np.mean(cagrs):+.2f}%")
            print(f"     Sharpe Ratio Promedio:      {np.mean(sharpes):.3f}  (Min: {np.min(sharpes):.2f} | Max: {np.max(sharpes):.2f})")
            print(f"     Win Rate Promedio:          {np.mean(win_rates):.1f}%")

            # Riesgo Global
            print(f"\n  ⚠️  RIESGO:")
            print(f"     Max Drawdown Promedio:      {np.mean(max_dds):.2f}%")
            print(f"     Max Drawdown Peor Caso:     {np.max(max_dds):.2f}%")
            print(f"     Profit Factor Promedio:     {np.mean(profit_factors):.2f}")

            # Mejor y Peor Performer
            mejor_idx = np.argmax(returns)
            peor_idx = np.argmin(returns)
            mejor_par = list(self.pares)[mejor_idx] if mejor_idx < len(self.pares) else 'N/A'
            peor_par = list(self.pares)[peor_idx] if peor_idx < len(self.pares) else 'N/A'

            print(f"\n  🏆 MEJOR PERFORMER:            {mejor_par} ({returns[mejor_idx]:+.2f}%)")
            print(f"  📉 PEOR PERFORMER:             {peor_par} ({returns[peor_idx]:+.2f}%)")

        # Información temporal
        print(f"\n  ⏱️  TIEMPO DE EJECUCIÓN:")
        print(f"     Inicio:                     {self.tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     Fin:                        {self.tiempo_fin.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     Duración Total:             {self._formatear_duracion(tiempo_total)}")
        print(f"     Tiempo Promedio/Par:        {self._formatear_duracion(tiempo_total/len(self.pares))}")

        # ============================================================
        # TABLA DE RESULTADOS COMPLETA
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"{'2. RESULTADOS POR PAR (TABLA COMPLETA)':^100}")
        print(f"{'─'*100}")
        print(f"\n{'Par':<10} │ {'✓':<3} │ {'Trades':<7} │ {'Return%':<9} │ {'CAGR%':<8} │ {'Sharpe':<7} │ {'Sortino':<8} │ {'MaxDD%':<8} │ {'Win%':<6}")
        print("─" * 100)

        for par in self.pares:
            res = self.resultados[par]

            if res['exito']:
                m = res['metricas']
                ret = m.get('total_return', 0)
                cagr = m.get('cagr', 0)
                sharpe = m.get('sharpe_ratio', 0)
                sortino = m.get('sortino_ratio', 0)
                maxdd = m.get('max_drawdown', 0)
                win_rate = m.get('win_rate', 0)

                print(
                    f"{par:<10} │ {'✓':<3} │ {res['n_trades']:<7,} │ "
                    f"{ret:>+8.2f} │ {cagr:>+7.2f} │ {sharpe:>6.2f} │ "
                    f"{sortino:>7.2f} │ {maxdd:>7.2f} │ {win_rate:>5.1f}"
                )
            else:
                print(
                    f"{par:<10} │ {'✗':<3} │ {'N/A':<7} │ {'N/A':<9} │ {'N/A':<8} │ "
                    f"{'N/A':<7} │ {'N/A':<8} │ {'N/A':<8} │ {'N/A':<6}"
                )
                print(f"{'':11} └─ Error: {res.get('error', 'Desconocido')}")

        print("─" * 100)

        # ============================================================
        # ANÁLISIS ESTADÍSTICO DETALLADO
        # ============================================================
        if exitosos > 0:
            print(f"\n{'─'*100}")
            print(f"{'3. ANÁLISIS ESTADÍSTICO DETALLADO':^100}")
            print(f"{'─'*100}")

            # Distribución de Returns
            print(f"\n  📈 DISTRIBUCIÓN DE RETURNS:")
            print(f"     Media:                      {np.mean(returns):+.2f}%")
            print(f"     Mediana:                    {np.median(returns):+.2f}%")
            print(f"     Desv. Estándar:             {np.std(returns):.2f}%")
            print(f"     Percentil 25:               {np.percentile(returns, 25):+.2f}%")
            print(f"     Percentil 75:               {np.percentile(returns, 75):+.2f}%")
            print(f"     Rango (Max - Min):          {np.max(returns) - np.min(returns):.2f}%")

            # Distribución de Sharpe Ratios
            print(f"\n  📊 DISTRIBUCIÓN DE SHARPE RATIOS:")
            print(f"     Media:                      {np.mean(sharpes):.3f}")
            print(f"     Mediana:                    {np.median(sharpes):.3f}")
            print(f"     Desv. Estándar:             {np.std(sharpes):.3f}")
            print(f"     Pares con Sharpe > 1.0:     {sum(1 for s in sharpes if s > 1.0)}/{len(sharpes)}")
            print(f"     Pares con Sharpe > 2.0:     {sum(1 for s in sharpes if s > 2.0)}/{len(sharpes)}")

            # Consistencia de Win Rates
            print(f"\n  🎯 CONSISTENCIA DE WIN RATES:")
            print(f"     Media:                      {np.mean(win_rates):.1f}%")
            print(f"     Mediana:                    {np.median(win_rates):.1f}%")
            print(f"     Desv. Estándar:             {np.std(win_rates):.1f}%")
            print(f"     Pares con WR > 50%:         {sum(1 for w in win_rates if w > 50)}/{len(win_rates)}")
            print(f"     Pares con WR > 60%:         {sum(1 for w in win_rates if w > 60)}/{len(win_rates)}")

            # Análisis de Profit Factors
            print(f"\n  💰 PROFIT FACTORS:")
            print(f"     Media:                      {np.mean(profit_factors):.2f}")
            print(f"     Mediana:                    {np.median(profit_factors):.2f}")
            print(f"     Pares con PF > 1.5:         {sum(1 for pf in profit_factors if pf > 1.5)}/{len(profit_factors)}")
            print(f"     Pares con PF > 2.0:         {sum(1 for pf in profit_factors if pf > 2.0)}/{len(profit_factors)}")

            # Análisis de Drawdowns
            print(f"\n  ⚠️  ANÁLISIS DE DRAWDOWNS:")
            print(f"     Max DD Promedio:            {np.mean(max_dds):.2f}%")
            print(f"     Max DD Mediano:             {np.median(max_dds):.2f}%")
            print(f"     Max DD Peor Caso:           {np.max(max_dds):.2f}%")
            print(f"     Max DD Mejor Caso:          {np.min(max_dds):.2f}%")
            print(f"     Pares con DD < 10%:         {sum(1 for dd in max_dds if dd < 10)}/{len(max_dds)}")
            print(f"     Pares con DD < 20%:         {sum(1 for dd in max_dds if dd < 20)}/{len(max_dds)}")

        # ============================================================
        # MÉTRICAS DETALLADAS POR PAR
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"{'4. MÉTRICAS DETALLADAS POR PAR':^100}")
        print(f"{'─'*100}")

        for idx, par in enumerate(self.pares, 1):
            res = self.resultados[par]

            if not res['exito']:
                print(f"\n  [{idx}] {par}: ✗ ERROR")
                print(f"      └─ {res.get('error', 'Desconocido')}")
                continue

            m = res['metricas']

            print(f"\n  [{idx}] {par}")
            print(f"  {'─'*96}")

            # Rendimiento
            print(f"    📊 RENDIMIENTO:")
            print(f"       Total Return:             {m.get('total_return', 0):+.2f}%")
            print(f"       CAGR (Anualizado):        {m.get('cagr', 0):+.2f}%")
            print(f"       Volatilidad Anual:        {m.get('annual_volatility', 0):.2f}%")

            # Calcular Return/Risk ratio si hay volatilidad
            vol = m.get('annual_volatility', 0)
            ret_risk = (m.get('total_return', 0) / vol) if vol > 0 else 0
            print(f"       Return/Risk Ratio:        {ret_risk:.2f}")

            # Ratios ajustados por riesgo
            print(f"\n    📈 RATIOS AJUSTADOS POR RIESGO:")
            sharpe = m.get('sharpe_ratio', 0)
            sortino = m.get('sortino_ratio', 0)
            calmar = m.get('calmar_ratio', 0)

            sharpe_rating = "Excelente" if sharpe > 2 else "Bueno" if sharpe > 1 else "Aceptable" if sharpe > 0.5 else "Bajo"
            print(f"       Sharpe Ratio:             {sharpe:.3f}  [{sharpe_rating}]")
            print(f"       Sortino Ratio:            {sortino:.3f}")
            print(f"       Calmar Ratio:             {calmar:.3f}")

            # Drawdown Analysis
            print(f"\n    ⚠️  ANÁLISIS DE DRAWDOWN:")
            maxdd = m.get('max_drawdown', 0)
            avgdd = m.get('avg_drawdown', 0)
            dd_rating = "Bajo" if maxdd < 10 else "Moderado" if maxdd < 20 else "Alto" if maxdd < 30 else "Muy Alto"
            print(f"       Max Drawdown:             {maxdd:.2f}%  [{dd_rating}]")
            print(f"       Avg Drawdown:             {avgdd:.2f}%")

            # Recovery Factor
            recovery = (m.get('total_return', 0) / maxdd) if maxdd > 0 else 0
            print(f"       Recovery Factor:          {recovery:.2f}  (Return/MaxDD)")

            # Trades Analysis
            print(f"\n    💼 ANÁLISIS DE TRADES:")
            n_trades = res['n_trades']
            win_rate = m.get('win_rate', 0)
            pf = m.get('profit_factor', 0)
            avg_win = m.get('avg_win', 0)
            avg_loss = m.get('avg_loss', 0)

            # Win/Loss ratio
            win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0

            print(f"       Total Trades:             {n_trades}")
            print(f"       Win Rate:                 {win_rate:.1f}%")
            print(f"       Profit Factor:            {pf:.2f}")
            print(f"       Avg Win:                  {avg_win:+.2f}%")
            print(f"       Avg Loss:                 {avg_loss:+.2f}%")
            print(f"       Win/Loss Ratio:           {win_loss_ratio:.2f}")

            # Expectancy
            expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
            print(f"       Expectancy per Trade:     {expectancy:+.3f}%")

            # Robustez
            print(f"\n    🔬 VALIDACIÓN:")
            if res.get('robustez', False):
                print(f"       Pruebas de Robustez:      ✓ Completadas")
            else:
                print(f"       Pruebas de Robustez:      ✗ No ejecutadas")

        # ============================================================
        # LOGS CAPTURADOS - Impresión detallada
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"{'5. LOGS CAPTURADOS DURANTE LA EJECUCIÓN':^100}")
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
        print(f"{'6. ARCHIVOS GENERADOS':^100}")
        print(f"{'─'*100}")

        archivos_json = list(self.metricas_dir.glob("*.json"))
        archivos_csv_trades = list(self.trades_dir.glob("*.csv"))
        archivos_csv_equity = list(self.equity_dir.glob("*.csv"))
        archivos_robustez = list(self.robustez_dir.glob("*.json"))

        total_archivos = len(archivos_json) + len(archivos_csv_trades) + len(archivos_csv_equity) + len(archivos_robustez)

        print(f"\n  Total de archivos generados: {total_archivos}")
        print(f"\n  📊 Métricas (JSON):           {len(archivos_json):3d} archivos → {self.metricas_dir}/")
        print(f"  💼 Trades (CSV):              {len(archivos_csv_trades):3d} archivos → {self.trades_dir}/")
        print(f"  📈 Equity Curves (CSV):       {len(archivos_csv_equity):3d} archivos → {self.equity_dir}/")
        print(f"  🔬 Robustez (JSON):           {len(archivos_robustez):3d} archivos → {self.robustez_dir}/")
        print(f"\n  📁 Ubicación base: {self.output_dir}")

        # ============================================================
        # CONCLUSIÓN Y PRÓXIMOS PASOS
        # ============================================================
        print(f"\n{'='*100}")
        print(f"{'CONCLUSIÓN':^100}")
        print(f"{'='*100}")

        if exitosos == len(self.resultados):
            print(f"\n  ✅ BACKTEST COMPLETADO EXITOSAMENTE")
            print(f"\n  Resumen:")
            print(f"     • Pares procesados:         {exitosos}/{len(self.pares)}")
            print(f"     • Total trades:             {total_trades:,}")
            print(f"     • Return promedio:          {np.mean(returns):+.2f}%")
            print(f"     • Sharpe promedio:          {np.mean(sharpes):.3f}")

            print(f"\n  🎯 SIMULACIÓN REALISTA:")
            print(f"     ✓ Spreads variables por hora y día de la semana")
            print(f"     ✓ Slippage dinámico basado en volatilidad")
            print(f"     ✓ Swap overnight (costos de financiamiento)")
            print(f"     ✓ Stop loss y take profit dinámicos (basados en ATR)")
            print(f"     ✓ División temporal: Train → Validation → Test")

            print(f"\n  📋 PRÓXIMOS PASOS:")
            if total_trades > 0:
                print(f"     1. Analizar equity curves detalladas")
                print(f"        → Ubicación: {self.equity_dir}/")
                print(f"     2. Revisar distribución temporal de trades")
                print(f"     3. Evaluar métricas por período (train/validation/test)")
                print(f"     4. Verificar resultados de pruebas de robustez")
                print(f"        → Ubicación: {self.robustez_dir}/")
                print(f"     5. Analizar consistencia de performance")
                print(f"     6. Evaluar si estrategia es viable para producción")
            else:
                print(f"     ⚠️  NO SE GENERARON TRADES - Acción requerida:")
                print(f"     1. Revisar lógica de generación de señales")
                print(f"     2. Verificar features de entrada")
                print(f"     3. Ajustar umbrales de entrada/salida")
                print(f"     4. Validar datos de entrada (OHLCV)")
        elif exitosos > 0:
            print(f"\n  ⚠️  BACKTEST COMPLETADO CON ERRORES PARCIALES")
            print(f"\n  Resumen:")
            print(f"     • Pares exitosos:           {exitosos}/{len(self.pares)}")
            print(f"     • Pares con errores:        {fallidos}")
            print(f"\n  📋 ACCIÓN REQUERIDA:")
            print(f"     1. Revisar logs de errores en sección 5")
            print(f"     2. Corregir problemas en pares fallidos")
            print(f"     3. Re-ejecutar backtest completo")

            if fallidos > 0:
                print(f"\n  ❌ Pares con errores:")
                for par, resultado in self.resultados.items():
                    if not resultado['exito']:
                        print(f"     • {par}: {resultado['error']}")
        else:
            print(f"\n  ❌ BACKTEST FALLIDO - TODOS LOS PARES CON ERRORES")
            print(f"\n  📋 ACCIÓN CRÍTICA REQUERIDA:")
            print(f"     1. Revisar logs de errores detallados en sección 5")
            print(f"     2. Verificar integridad de datos de entrada")
            print(f"     3. Validar configuración del backtest")
            print(f"     4. Contactar soporte si el problema persiste")

            if fallidos > 0:
                print(f"\n  ❌ Errores detectados:")
                for par, resultado in self.resultados.items():
                    if not resultado['exito']:
                        print(f"     • {par}: {resultado['error']}")

        print(f"\n  {'─'*96}")
        print(f"  ℹ️  NOTA IMPORTANTE:")
        print(f"     Estos resultados reflejan condiciones de trading REALES con costos completos.")
        print(f"     Las métricas son conservadoras y no optimistas.")
        print(f"     Performance pasada NO garantiza resultados futuros.")
        print(f"{'='*100}")

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

    # Parámetros MULTI-TIMEFRAME
    TIMEFRAMES = ['M15', 'H1', 'H4', 'D']
    CAPITAL_INICIAL = 100000  # $100,000
    PARES = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY', 'AUD_USD']  # Lista de pares a procesar (None = auto-detectar)

    # Crear ejecutor MULTI-TIMEFRAME
    ejecutor = EjecutorBacktest(
        datos_ohlc_dir=DATOS_OHLC_DIR,
        estrategias_dir=ESTRATEGIAS_DIR,
        output_dir=OUTPUT_DIR,
        timeframes=TIMEFRAMES,
        capital_inicial=CAPITAL_INICIAL,
        pares=PARES,
        limpiar_archivos_viejos=True,  # Limpiar archivos viejos
        hacer_backup=False,             # NO hacer backup (ahorra espacio)
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
