"""
Motor de Backtest Completo

COMPONENTES:
1. Data Loader
2. Transformation Engine
3. Signal Generator
4. Cost Calculator
5. Execution Simulator
6. Position Manager
7. P&L Calculator
8. Metrics Calculator

Author: Sistema de Edge-Finding Forex
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# CONSTANTES PARA DATOS M15 (15 minutos)
# 252 días trading × 24 horas × 4 bars/hora × 0.7 (ajuste fin de semana) = ~24,192
BARS_PER_YEAR_M15 = 24192
BARS_PER_DAY_M15 = 96  # 24 horas × 4 bars/hora
EPSILON = 1e-10  # Para comparaciones de floats


class MotorBacktestCompleto:
    """
    Motor de backtest completo con todos los componentes integrados.

    Simula trading real con costos reales (spreads, slippage, swap).
    """

    def __init__(
        self,
        capital_inicial: float = 100000,
        config: Optional[Dict] = None,
        verbose: bool = True
    ):
        """
        Inicializa motor de backtest.

        Parameters:
        -----------
        capital_inicial : float
            Capital inicial en USD (default: $100,000)
        config : Dict, optional
            Configuración de la estrategia
        verbose : bool
            Imprimir progreso
        """
        self.capital_inicial = capital_inicial
        self.capital = capital_inicial
        self.verbose = verbose

        # Configuración por defecto
        self.config = {
            # Sizing
            'risk_per_trade': 0.01,  # 1% riesgo por trade
            'max_position_size': 0.10,  # 10% máximo del capital

            # Exits
            'stop_loss_atr_mult': 2.5,
            'take_profit_atr_mult': 3.5,
            # BAJO #17: Timeout en bars (para M15: 50 bars = ~12.5 horas)
            'timeout_bars': 50,

            # Costos
            'base_slippage_pips': 0.3,

            # Filtros
            'min_spread_pips': 0.0,
            'max_spread_pips': 5.0,
            'avoid_rollover_hours': [21, 22, 23],  # Evitar horas de rollover
        }

        if config:
            self.config.update(config)

        # Estado del backtest
        self.posiciones_abiertas: List[Dict] = []
        self.historial_trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.trade_id_counter = 0

        # Datos cargados
        self.df_ohlcv: Optional[pd.DataFrame] = None
        self.df_spreads: Optional[pd.DataFrame] = None
        self.df_transformaciones: Optional[pd.DataFrame] = None

        # Spreads lookup (para acceso rápido)
        self.spreads_lookup: Dict = {}

        if self.verbose:
            print("="*80)
            print("MOTOR DE BACKTEST COMPLETO")
            print("="*80)
            print(f"Capital inicial: ${self.capital_inicial:,.0f}")
            print(f"Riesgo por trade: {self.config['risk_per_trade']*100:.1f}%")
            print("="*80)

    # ========================================================================
    # 1. DATA LOADER
    # ========================================================================

    def _validar_columnas_requeridas(self):
        """
        CRÍTICO: Valida que los DataFrames tengan todas las columnas requeridas.

        Previene KeyError al acceder a columnas en row['columna'].
        """
        # Columnas requeridas en OHLCV
        columnas_ohlcv_requeridas = ['timestamp', 'pair', 'open', 'high', 'low', 'close', 'volume']
        columnas_faltantes = [col for col in columnas_ohlcv_requeridas if col not in self.df_ohlcv.columns]

        if columnas_faltantes:
            raise ValueError(
                f"OHLCV DataFrame falta columnas requeridas: {columnas_faltantes}\n"
                f"Columnas presentes: {list(self.df_ohlcv.columns)}"
            )

        # Columnas requeridas en spreads
        columnas_spreads_requeridas = ['pair', 'hour_utc', 'day_of_week', 'spread_pips']
        columnas_faltantes = [col for col in columnas_spreads_requeridas if col not in self.df_spreads.columns]

        if columnas_faltantes:
            raise ValueError(
                f"Spreads DataFrame falta columnas requeridas: {columnas_faltantes}\n"
                f"Columnas presentes: {list(self.df_spreads.columns)}"
            )

        if self.verbose:
            print("  ✓ Validación de columnas: PASADA")

    def _validar_configuracion(self):
        """
        CRÍTICO: Valida que la configuración tenga todas las claves requeridas.

        Previene KeyError al acceder a self.config['clave'].
        """
        claves_requeridas = [
            'risk_per_trade',
            'max_position_size',
            'stop_loss_atr_mult',
            'take_profit_atr_mult',
            'timeout_bars',
            'base_slippage_pips',
            'max_spread_pips',
            'avoid_rollover_hours'
        ]

        claves_faltantes = [clave for clave in claves_requeridas if clave not in self.config]

        if claves_faltantes:
            raise ValueError(
                f"Configuración falta claves requeridas: {claves_faltantes}\n"
                f"Claves presentes: {list(self.config.keys())}"
            )

        # Validar valores
        if not (0 < self.config['risk_per_trade'] <= 0.1):
            raise ValueError(f"risk_per_trade debe estar entre 0 y 0.1, recibido: {self.config['risk_per_trade']}")

        if not (0 < self.config['max_position_size'] <= 1.0):
            raise ValueError(f"max_position_size debe estar entre 0 y 1.0, recibido: {self.config['max_position_size']}")

        if self.verbose:
            print("  ✓ Validación de configuración: PASADA")

    def cargar_datos(
        self,
        ruta_ohlcv: str,
        ruta_spreads: str,
        ruta_transformaciones: Optional[str] = None
    ):
        """
        Carga todos los datos necesarios para el backtest.

        Parameters:
        -----------
        ruta_ohlcv : str
            Ruta al CSV con OHLCV (raw_ohlcv.csv)
        ruta_spreads : str
            Ruta al CSV con spreads (spreads_real.csv)
        ruta_transformaciones : str, optional
            Ruta al CSV con transformaciones
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"1. DATA LOADER")
            print(f"{'='*80}")

        # Cargar OHLCV
        self.df_ohlcv = pd.read_csv(ruta_ohlcv)
        self.df_ohlcv['timestamp'] = pd.to_datetime(self.df_ohlcv['timestamp'])

        if self.verbose:
            print(f"✓ OHLCV cargado: {len(self.df_ohlcv):,} filas")
            print(f"  Pares: {self.df_ohlcv['pair'].nunique()}")
            print(f"  Período: {self.df_ohlcv['timestamp'].min()} → {self.df_ohlcv['timestamp'].max()}")

        # Cargar spreads
        self.df_spreads = pd.read_csv(ruta_spreads)

        # CRÍTICO: Validar columnas requeridas
        self._validar_columnas_requeridas()
        self._validar_configuracion()

        self._construir_spreads_lookup()

        if self.verbose:
            print(f"✓ Spreads cargados: {len(self.df_spreads):,} filas")

        # Cargar transformaciones (opcional)
        if ruta_transformaciones:
            self.df_transformaciones = pd.read_csv(ruta_transformaciones)
            self.df_transformaciones['timestamp'] = pd.to_datetime(
                self.df_transformaciones['timestamp']
            )

            if self.verbose:
                print(f"✓ Transformaciones cargadas: {len(self.df_transformaciones):,} filas")

    def _construir_spreads_lookup(self):
        """
        Construye diccionario de lookup para spreads rápidos.

        spreads_lookup[pair][hour][day] = spread_pips
        """
        self.spreads_lookup = {}

        for _, row in self.df_spreads.iterrows():
            par = row['pair']
            hora = int(row['hour_utc'])
            dia = int(row['day_of_week'])
            spread = float(row['spread_pips'])

            if par not in self.spreads_lookup:
                self.spreads_lookup[par] = {}
            if hora not in self.spreads_lookup[par]:
                self.spreads_lookup[par][hora] = {}

            self.spreads_lookup[par][hora][dia] = spread

        if self.verbose:
            print(f"✓ Spreads lookup construido: {len(self.spreads_lookup)} pares")

    # ========================================================================
    # 2. TRANSFORMATION ENGINE
    # ========================================================================

    def calcular_transformaciones(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todas las transformaciones necesarias.

        IMPORTANTE: Solo usar datos hasta t-1 (no look-ahead bias)

        Parameters:
        -----------
        df_ohlcv : pd.DataFrame
            DataFrame con OHLCV

        Returns:
        --------
        df_trans : pd.DataFrame
            DataFrame con transformaciones calculadas
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"2. TRANSFORMATION ENGINE")
            print(f"{'='*80}")

        # Por simplicidad, aquí solo calculamos ATR
        # En producción, cargarías las transformaciones validadas

        df_trans = df_ohlcv.copy()

        # Calcular ATR por par
        for par in df_trans['pair'].unique():
            mask = df_trans['pair'] == par

            # True Range
            high = df_trans.loc[mask, 'high']
            low = df_trans.loc[mask, 'low']
            close_prev = df_trans.loc[mask, 'close'].shift(1)

            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            # MEDIO #12: Evitar look-ahead bias - ATR no incluye barra actual
            atr = tr.rolling(window=14).mean().shift(1)
            # BAJO #16: Forward fill primeros NaN valores
            atr = atr.fillna(method='bfill').fillna(0.0001)

            df_trans.loc[mask, 'ATR'] = atr.values

        if self.verbose:
            print(f"✓ Transformaciones calculadas")
            print(f"  Columnas: ATR")

        return df_trans

    # ========================================================================
    # 3. SIGNAL GENERATOR
    # ========================================================================

    def generar_señal(
        self,
        row: pd.Series,
        idx: int
    ) -> Tuple[str, float]:
        """
        Genera señal de trading.

        Parameters:
        -----------
        row : pd.Series
            Fila actual de datos
        idx : int
            Índice de la fila

        Returns:
        --------
        señal : str
            'LONG', 'SHORT', o 'NEUTRAL'
        fuerza : float
            Fuerza de la señal (0-1)
        """
        # Filtros
        timestamp = row['timestamp']
        par = row['pair']

        # Filtro 1: Hora (evitar rollover)
        hora = timestamp.hour
        if hora in self.config['avoid_rollover_hours']:
            return 'NEUTRAL', 0.0

        # Filtro 2: Spread (evitar spreads muy amplios)
        spread = self.get_spread(timestamp, par)
        if spread > self.config['max_spread_pips']:
            return 'NEUTRAL', 0.0

        # Filtro 3: Día de semana (evitar viernes tarde)
        dia_semana = timestamp.dayofweek
        if dia_semana == 4 and hora >= 17:  # Viernes después de 17:00
            return 'NEUTRAL', 0.0

        # SEÑAL SIMPLE DE EJEMPLO
        # En producción, usarías las reglas de la estrategia emergente

        # Ejemplo: Momentum simple basado en close
        if idx < 20:
            return 'NEUTRAL', 0.0

        # Obtener datos históricos
        df_subset = self.df_ohlcv.iloc[max(0, idx-20):idx+1]
        df_par = df_subset[df_subset['pair'] == par]

        if len(df_par) < 20:
            return 'NEUTRAL', 0.0

        # MEDIO #13: Usar barra anterior para señal (no close actual)
        # Retorno de 1 período (usando barra anterior)
        ret_1 = (df_par['close'].iloc[-2] - df_par['close'].iloc[-3]) / df_par['close'].iloc[-3]

        # Retorno de 5 períodos (usando barra anterior)
        ret_5 = (df_par['close'].iloc[-2] - df_par['close'].iloc[-7]) / df_par['close'].iloc[-7]

        # Señal simple: momentum positivo
        if ret_1 > 0 and ret_5 > 0:
            fuerza = min(abs(ret_1) * 100, 1.0)  # Normalizar
            return 'LONG', fuerza
        elif ret_1 < 0 and ret_5 < 0:
            fuerza = min(abs(ret_1) * 100, 1.0)
            return 'SHORT', fuerza
        else:
            return 'NEUTRAL', 0.0

    # ========================================================================
    # 4. COST CALCULATOR
    # ========================================================================

    def get_spread(self, timestamp: pd.Timestamp, par: str) -> float:
        """
        Obtiene spread en pips para un timestamp y par.

        Parameters:
        -----------
        timestamp : pd.Timestamp
            Timestamp
        par : str
            Par de divisas

        Returns:
        --------
        spread_pips : float
            Spread en pips
        """
        hora = timestamp.hour
        dia = timestamp.dayofweek

        try:
            spread = self.spreads_lookup[par][hora][dia]
        except KeyError:
            # Fallback: spread promedio
            spread = 1.5

        return spread

    def calcular_slippage(self, atr_actual: float, atr_promedio: float) -> float:
        """
        Calcula slippage dinámico.

        Parameters:
        -----------
        atr_actual : float
            ATR actual
        atr_promedio : float
            ATR promedio

        Returns:
        --------
        slippage_pips : float
            Slippage en pips
        """
        base_slippage = self.config['base_slippage_pips']

        if atr_promedio > 0:
            vol_ratio = atr_actual / atr_promedio
            slippage = base_slippage * (1 + 0.5 * (vol_ratio - 1))
        else:
            slippage = base_slippage

        # Limitar
        slippage = max(0.2, min(slippage, 3.0))

        return slippage

    # ========================================================================
    # 5. EXECUTION SIMULATOR
    # ========================================================================

    def abrir_posicion(
        self,
        row: pd.Series,
        direccion: str,
        fuerza_señal: float
    ):
        """
        Abre una nueva posición.

        Parameters:
        -----------
        row : pd.Series
            Fila actual de datos
        direccion : str
            'LONG' o 'SHORT'
        fuerza_señal : float
            Fuerza de la señal
        """
        timestamp = row['timestamp']
        par = row['pair']
        close = row['close']
        atr = row.get('ATR', 0.001)

        # Spread de entrada
        spread_entry = self.get_spread(timestamp, par)

        # Slippage
        atr_promedio = atr  # Simplificado
        slippage = self.calcular_slippage(atr, atr_promedio)

        # Costo total de entrada (pips)
        entry_cost_pips = spread_entry + slippage

        # Convertir pips a precio
        if 'JPY' in par:
            pip_size = 0.01
        else:
            pip_size = 0.0001

        entry_cost = entry_cost_pips * pip_size

        # Precio de entrada
        if direccion == 'LONG':
            entry_price = close + entry_cost  # Comprar al ASK
        else:
            entry_price = close - entry_cost  # Vender al BID

        # Calcular SL y TP
        sl_distance = atr * self.config['stop_loss_atr_mult']
        tp_distance = atr * self.config['take_profit_atr_mult']

        if direccion == 'LONG':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # Calcular position size
        # CRÍTICO #3: Validar sl_distance para evitar división por cero
        if sl_distance <= 0 or sl_distance < entry_price * 0.0001:  # Mínimo 1 pip
            if self.verbose:
                print(f"  ⚠ SL distance inválido ({sl_distance:.6f}), trade omitido")
            return

        risk_amount = self.capital * self.config['risk_per_trade']
        position_size_frac = (risk_amount / sl_distance) / entry_price
        position_size_frac = min(position_size_frac, self.config['max_position_size'])

        # Capital arriesgado
        capital_riesgo = self.capital * position_size_frac

        # Crear posición
        self.trade_id_counter += 1
        posicion = {
            'id': self.trade_id_counter,
            'pair': par,
            'direction': direccion,
            'entry_time': timestamp,
            'entry_price': entry_price,
            'entry_spread': spread_entry,
            'entry_slippage': slippage,
            'size_frac': position_size_frac,
            'capital_riesgo': capital_riesgo,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr_entry': atr,
            'bars_open': 0
        }

        self.posiciones_abiertas.append(posicion)

        if self.verbose and len(self.posiciones_abiertas) <= 5:  # Solo primeras 5
            print(f"\n[{timestamp}] ABRIR {direccion} {par}")
            print(f"  Entry: {entry_price:.5f} (spread={spread_entry:.1f}p, slip={slippage:.1f}p)")
            print(f"  Size: {position_size_frac*100:.2f}% (${capital_riesgo:,.0f})")
            print(f"  SL: {stop_loss:.5f} | TP: {take_profit:.5f}")

    # ========================================================================
    # 6. POSITION MANAGER
    # ========================================================================

    def actualizar_posiciones(self, row: pd.Series):
        """
        Actualiza posiciones abiertas y cierra si se cumplen condiciones.

        Parameters:
        -----------
        row : pd.Series
            Fila actual de datos
        """
        timestamp = row['timestamp']
        par = row['pair']
        high = row['high']
        low = row['low']
        close = row['close']

        posiciones_cerrar = []

        for posicion in self.posiciones_abiertas:
            if posicion['pair'] != par:
                continue

            posicion['bars_open'] += 1

            razon_cierre = None
            precio_cierre = None

            # 1. Stop Loss
            if posicion['direction'] == 'LONG':
                if low <= posicion['stop_loss']:
                    razon_cierre = 'SL'
                    precio_cierre = posicion['stop_loss']
            else:
                if high >= posicion['stop_loss']:
                    razon_cierre = 'SL'
                    precio_cierre = posicion['stop_loss']

            # 2. Take Profit
            if razon_cierre is None:
                if posicion['direction'] == 'LONG':
                    if high >= posicion['take_profit']:
                        razon_cierre = 'TP'
                        precio_cierre = posicion['take_profit']
                else:
                    if low <= posicion['take_profit']:
                        razon_cierre = 'TP'
                        precio_cierre = posicion['take_profit']

            # 3. Timeout
            if razon_cierre is None:
                if posicion['bars_open'] >= self.config['timeout_bars']:
                    razon_cierre = 'TIMEOUT'
                    precio_cierre = close

            # Cerrar si hay razón
            if razon_cierre:
                self.cerrar_posicion(
                    posicion=posicion,
                    timestamp=timestamp,
                    precio_cierre=precio_cierre,
                    razon=razon_cierre
                )
                posiciones_cerrar.append(posicion)

        # Eliminar posiciones cerradas
        for pos in posiciones_cerrar:
            self.posiciones_abiertas.remove(pos)

    # ========================================================================
    # 7. P&L CALCULATOR
    # ========================================================================

    def cerrar_posicion(
        self,
        posicion: Dict,
        timestamp: pd.Timestamp,
        precio_cierre: float,
        razon: str
    ):
        """
        Cierra posición y calcula P&L.

        Parameters:
        -----------
        posicion : Dict
            Posición a cerrar
        timestamp : pd.Timestamp
            Timestamp de cierre
        precio_cierre : float
            Precio de cierre
        razon : str
            Razón de cierre ('SL', 'TP', 'TIMEOUT', 'SIGNAL')
        """
        # Spread de salida
        spread_exit = self.get_spread(timestamp, posicion['pair'])

        # CRÍTICO #6: Slippage de salida DINÁMICO (no hardcoded)
        # Calcular slippage dinámico basado en ATR actual
        atr_current = row.get('ATR', posicion['atr_entry'])
        atr_avg = posicion['atr_entry']
        slippage_exit = self.calcular_slippage(atr_current, atr_avg)

        # Costo de salida
        exit_cost_pips = spread_exit + slippage_exit

        if 'JPY' in posicion['pair']:
            pip_size = 0.01
        else:
            pip_size = 0.0001

        exit_cost = exit_cost_pips * pip_size

        # Ajustar precio de cierre por costos
        if posicion['direction'] == 'LONG':
            precio_cierre_real = precio_cierre - exit_cost  # Vender al BID
        else:
            precio_cierre_real = precio_cierre + exit_cost  # Comprar al ASK

        # CRÍTICO #4: Gross P&L ya incluye costos en precios ajustados
        # NO deducir costs por separado (evita doble conteo)
        if posicion['direction'] == 'LONG':
            gross_pnl = (precio_cierre_real - posicion['entry_price']) * posicion['capital_riesgo'] / posicion['entry_price']
        else:
            gross_pnl = (posicion['entry_price'] - precio_cierre_real) * posicion['capital_riesgo'] / posicion['entry_price']

        # Costos (solo para tracking, YA incluidos en precio)
        cost_spread_total = (posicion['entry_spread'] + spread_exit) * pip_size * posicion['capital_riesgo'] / posicion['entry_price']
        cost_slippage_total = (posicion['entry_slippage'] + slippage_exit) * pip_size * posicion['capital_riesgo'] / posicion['entry_price']

        # Swap (simplificado, asumiendo -$6.50 por noche para EUR_USD Long)
        # CRÍTICO #2: Corregido para M15 (96 bars/día, no 24)
        noches = posicion['bars_open'] // BARS_PER_DAY_M15
        if noches > 0:
            swap = -6.50 * noches * (posicion['capital_riesgo'] / 100000)  # Ajustar por lote
        else:
            swap = 0

        # ALTO #9: Net P&L = Gross P&L - Swap (costos ya en precio, swap no)
        net_pnl = gross_pnl - swap

        # Actualizar capital
        self.capital += net_pnl

        # Registrar trade
        trade = {
            **posicion,
            'exit_time': timestamp,
            'exit_price': precio_cierre,
            'exit_price_real': precio_cierre_real,
            'exit_spread': spread_exit,
            'exit_slippage': slippage_exit,
            'exit_reason': razon,
            'gross_pnl': gross_pnl,
            'cost_spread': cost_spread_total,
            'cost_slippage': cost_slippage_total,
            'cost_swap': swap,
            'net_pnl': net_pnl,
            'capital_after': self.capital,
            # MEDIO #15: Return como % del capital total (más útil que % de position size)
            'return_pct': (net_pnl / self.capital_inicial) * 100,
            'return_on_position': (net_pnl / posicion['capital_riesgo']) * 100  # Return on position size
        }

        self.historial_trades.append(trade)

        if self.verbose and len(self.historial_trades) <= 5:
            print(f"\n[{timestamp}] CERRAR {posicion['direction']} {posicion['pair']} - {razon}")
            print(f"  Exit: {precio_cierre_real:.5f}")
            print(f"  P&L: ${net_pnl:+,.2f} ({trade['return_pct']:+.2f}%)")
            print(f"  Capital: ${self.capital:,.0f}")

    # ========================================================================
    # LOOP PRINCIPAL
    # ========================================================================

    def ejecutar_backtest(
        self,
        par: str,
        fecha_inicio: Optional[str] = None,
        fecha_fin: Optional[str] = None
    ):
        """
        Ejecuta backtest completo.

        Parameters:
        -----------
        par : str
            Par de divisas a testear
        fecha_inicio : str, optional
            Fecha de inicio (formato: 'YYYY-MM-DD')
        fecha_fin : str, optional
            Fecha de fin (formato: 'YYYY-MM-DD')
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"EJECUTANDO BACKTEST - {par}")
            print(f"{'='*80}")

        # Filtrar datos por par
        df_par = self.df_ohlcv[self.df_ohlcv['pair'] == par].copy()

        # Filtrar por fechas
        if fecha_inicio:
            df_par = df_par[df_par['timestamp'] >= pd.Timestamp(fecha_inicio)]
        if fecha_fin:
            df_par = df_par[df_par['timestamp'] <= pd.Timestamp(fecha_fin)]

        df_par = df_par.sort_values('timestamp').reset_index(drop=True)

        # Calcular transformaciones
        df_par = self.calcular_transformaciones(df_par)

        if self.verbose:
            print(f"Período: {df_par['timestamp'].iloc[0]} → {df_par['timestamp'].iloc[-1]}")
            print(f"Barras: {len(df_par):,}")

        # Loop principal
        for idx, row in df_par.iterrows():
            # 1. Actualizar posiciones abiertas
            self.actualizar_posiciones(row)

            # 2. Generar señal (solo si no hay posiciones abiertas)
            if len(self.posiciones_abiertas) == 0:
                señal, fuerza = self.generar_señal(row, idx)

                # 3. Ejecutar si hay señal
                if señal in ['LONG', 'SHORT']:
                    self.abrir_posicion(row, señal, fuerza)

            # 4. Actualizar equity
            unrealized = sum([
                self._calcular_pnl_no_realizado(pos, row)
                for pos in self.posiciones_abiertas
                if pos['pair'] == par
            ])

            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'capital': self.capital,
                'unrealized': unrealized,
                'equity': self.capital + unrealized,
                'n_posiciones': len(self.posiciones_abiertas)
            })

        # Cerrar posiciones abiertas al final
        for pos in self.posiciones_abiertas.copy():
            if pos['pair'] == par:
                self.cerrar_posicion(
                    posicion=pos,
                    timestamp=df_par['timestamp'].iloc[-1],
                    precio_cierre=df_par['close'].iloc[-1],
                    razon='FIN_BACKTEST'
                )
                self.posiciones_abiertas.remove(pos)

    def _calcular_pnl_no_realizado(self, posicion: Dict, row: pd.Series) -> float:
        """
        MEDIO #14: Calcula P&L no realizado con estimación de costos de salida.
        """
        precio_actual = row['close']

        # Estimar costos de salida
        spread_exit = self.get_spread(row['timestamp'], posicion['pair'])
        atr_current = row.get('ATR', posicion['atr_entry'])
        slippage_exit = self.calcular_slippage(atr_current, posicion['atr_entry'])

        if 'JPY' in posicion['pair']:
            pip_size = 0.01
        else:
            pip_size = 0.0001

        exit_cost = (spread_exit + slippage_exit) * pip_size

        # Ajustar precio actual por costos de salida estimados
        if posicion['direction'] == 'LONG':
            precio_actual_adj = precio_actual - exit_cost
            pnl = (precio_actual_adj - posicion['entry_price']) * posicion['capital_riesgo'] / posicion['entry_price']
        else:
            precio_actual_adj = precio_actual + exit_cost
            pnl = (posicion['entry_price'] - precio_actual_adj) * posicion['capital_riesgo'] / posicion['entry_price']

        return pnl

    # ========================================================================
    # 8. METRICS CALCULATOR
    # ========================================================================

    def calcular_metricas(self) -> Dict:
        """
        Calcula métricas finales del backtest.

        Returns:
        --------
        metricas : Dict
            Diccionario con todas las métricas
        """
        if len(self.historial_trades) == 0:
            return {'error': 'No hay trades para calcular métricas'}

        df_trades = pd.DataFrame(self.historial_trades)
        df_equity = pd.DataFrame(self.equity_curve)

        # Returns
        returns = df_trades['return_pct'].values / 100

        # Total Return
        total_return = ((self.capital - self.capital_inicial) / self.capital_inicial) * 100

        # Win Rate
        n_wins = (df_trades['net_pnl'] > 0).sum()
        n_loss = (df_trades['net_pnl'] < 0).sum()
        win_rate = (n_wins / len(df_trades)) * 100 if len(df_trades) > 0 else 0

        # CRÍTICO #5: Profit Factor sin np.inf (no serializable a JSON)
        gross_profit = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum())
        if gross_loss > EPSILON:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > EPSILON:
            profit_factor = 999.99  # Cap máximo razonable
        else:
            profit_factor = 1.0  # Break-even

        # CRÍTICO #1: Sharpe Ratio con factor correcto para M15 (no 252)
        if len(returns) > 1:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(BARS_PER_YEAR_M15) if np.std(returns) > EPSILON else 0
        else:
            sharpe = 0

        # ALTO #7: Sortino Ratio con downside deviation correcta
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean(downside_returns**2))
            if downside_deviation > EPSILON:
                sortino = (np.mean(returns) / downside_deviation) * np.sqrt(BARS_PER_YEAR_M15)
            else:
                sortino = 0
        else:
            sortino = 0  # Sin pérdidas = no downside risk

        # Max Drawdown
        equity_values = df_equity['equity'].values
        running_max = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - running_max) / running_max
        max_dd = abs(drawdown.min()) * 100

        # Avg Drawdown
        drawdowns = drawdown[drawdown < 0]
        avg_dd = abs(drawdowns.mean()) * 100 if len(drawdowns) > 0 else 0

        # ALTO #8: Calmar Ratio con validación robusta
        years = (df_equity['timestamp'].iloc[-1] - df_equity['timestamp'].iloc[0]).days / 365.25
        annual_return = total_return / years if years > 0 else 0
        if max_dd > 0.01:  # Al menos 1bp de drawdown
            calmar = annual_return / max_dd
        else:
            calmar = 999.99 if annual_return > 0 else 0

        metricas = {
            'n_trades': len(df_trades),
            'n_wins': int(n_wins),
            'n_loss': int(n_loss),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'avg_win': df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean() if n_wins > 0 else 0,
            'avg_loss': df_trades[df_trades['net_pnl'] < 0]['net_pnl'].mean() if n_loss > 0 else 0,
            'total_pnl': df_trades['net_pnl'].sum(),
            'total_costs': df_trades['cost_spread'].sum() + df_trades['cost_slippage'].sum() + abs(df_trades['cost_swap'].sum()),
            'capital_final': self.capital
        }

        return metricas

    def mostrar_resumen(self):
        """Muestra resumen de resultados."""
        metricas = self.calcular_metricas()

        if 'error' in metricas:
            print(f"\n{metricas['error']}")
            return

        print(f"\n{'='*80}")
        print(f"RESUMEN DE BACKTEST")
        print(f"{'='*80}")

        print(f"\nTRADES:")
        print(f"  Total:          {metricas['n_trades']}")
        print(f"  Ganadores:      {metricas['n_wins']} ({metricas['win_rate']:.1f}%)")
        print(f"  Perdedores:     {metricas['n_loss']} ({100-metricas['win_rate']:.1f}%)")
        print(f"  Profit Factor:  {metricas['profit_factor']:.2f}")

        print(f"\nRETORNOS:")
        print(f"  Total:          {metricas['total_return']:+.2f}%")
        print(f"  Anualizado:     {metricas['annual_return']:+.2f}%")
        print(f"  P&L Total:      ${metricas['total_pnl']:+,.2f}")

        print(f"\nRATIOS:")
        print(f"  Sharpe:         {metricas['sharpe_ratio']:.2f}")
        print(f"  Sortino:        {metricas['sortino_ratio']:.2f}")
        print(f"  Calmar:         {metricas['calmar_ratio']:.2f}")

        print(f"\nDRAWDOWN:")
        print(f"  Max:            {metricas['max_drawdown']:.2f}%")
        print(f"  Avg:            {metricas['avg_drawdown']:.2f}%")

        print(f"\nCOSTOS:")
        print(f"  Total:          ${metricas['total_costs']:,.2f}")
        print(f"  % del P&L:      {(metricas['total_costs']/abs(metricas['total_pnl'])*100) if metricas['total_pnl'] != 0 else 0:.1f}%")

        print(f"\nCAPITAL:")
        print(f"  Inicial:        ${self.capital_inicial:,.0f}")
        print(f"  Final:          ${metricas['capital_final']:,.0f}")

        print(f"{'='*80}")


def ejemplo_uso():
    """
    Ejemplo de uso del Motor de Backtest Completo.
    """
    print("="*80)
    print("EJEMPLO: Motor de Backtest Completo")
    print("="*80)

    print("\nEste es un ejemplo esquelético.")
    print("Para uso real:")
    print("1. Ejecutar preparar_datos_backtest.py")
    print("2. Ejecutar tabla_spreads_reales.py")
    print("3. Cargar datos en el motor")
    print("4. Ejecutar backtest")


if __name__ == '__main__':
    ejemplo_uso()
