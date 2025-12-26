"""
Configuración de Costos Adicionales

COSTOS ADICIONALES:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  SLIPPAGE:                                                                  │
│  ─────────                                                                  │
│  Condición normal:     0.2-0.3 pips                                        │
│  Alta volatilidad:     0.5-1.0 pips                                        │
│  Noticias:             1.0-3.0 pips                                        │
│                                                                             │
│  MODELO:                                                                    │
│  slippage = base_slippage × (1 + 0.5 × (ATR_actual / ATR_promedio - 1))   │
│  base_slippage = 0.3 pips                                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SWAP (si posición overnight):                                              │
│  ─────────────────────────────                                              │
│                                                                             │
│  Par      │ Swap Long │ Swap Short │ (USD per lot per night)               │
│  ─────────┼───────────┼────────────┤                                       │
│  EUR_USD  │ -$6.50    │ +$1.20     │                                       │
│  GBP_USD  │ -$5.80    │ +$0.80     │                                       │
│  USD_JPY  │ +$4.20    │ -$8.50     │                                       │
│  EUR_JPY  │ -$2.30    │ -$4.10     │                                       │
│  GBP_JPY  │ +$1.50    │ -$9.20     │                                       │
│  AUD_USD  │ -$3.40    │ -$1.80     │                                       │
│                                                                             │
│  Triple swap: Miércoles → Jueves (por el weekend)                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  COMISIÓN:                                                                  │
│  ─────────                                                                  │
│  OANDA: $0 (incluido en spread)                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Author: Sistema de Edge-Finding Forex
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class ConfiguracionCostosAdicionales:
    """
    Configuración y cálculo de costos adicionales de trading.

    Incluye:
    - Slippage dinámico basado en volatilidad
    - Swap overnight por par y dirección
    - Comisiones (si aplican)
    """

    def __init__(self, verbose: bool = True):
        """
        Inicializa configuración de costos adicionales.

        Parameters:
        -----------
        verbose : bool
            Imprimir información detallada
        """
        self.verbose = verbose

        # ====================================================================
        # CONFIGURACIÓN DE SLIPPAGE
        # ====================================================================
        self.slippage_config = {
            'base_slippage_pips': 0.3,      # Slippage base en condiciones normales
            'min_slippage_pips': 0.2,       # Slippage mínimo
            'max_slippage_pips': 3.0,       # Slippage máximo (eventos de noticias)
            'volatility_multiplier': 0.5,   # Factor de ajuste por volatilidad
            'atr_lookback': 20              # Período para ATR promedio
        }

        # ====================================================================
        # CONFIGURACIÓN DE SWAP (USD por lote estándar por noche)
        # ====================================================================
        self.swap_rates = {
            'EUR_USD': {
                'long': -6.50,
                'short': +1.20
            },
            'GBP_USD': {
                'long': -5.80,
                'short': +0.80
            },
            'USD_JPY': {
                'long': +4.20,
                'short': -8.50
            },
            'EUR_JPY': {
                'long': -2.30,
                'short': -4.10
            },
            'GBP_JPY': {
                'long': +1.50,
                'short': -9.20
            },
            'AUD_USD': {
                'long': -3.40,
                'short': -1.80
            }
        }

        # Triple swap (Miércoles → Jueves, cubre weekend)
        self.triple_swap_day = 3  # 3 = Jueves (0=Lun, 1=Mar, 2=Mié, 3=Jue, 4=Vie)

        # ====================================================================
        # CONFIGURACIÓN DE COMISIONES
        # ====================================================================
        self.commission_config = {
            'broker': 'OANDA',
            'commission_per_lot': 0.0,  # $0 (incluido en spread)
            'commission_currency': 'USD'
        }

        # ====================================================================
        # PIP LOCATIONS (para conversión)
        # ====================================================================
        self.pip_locations = {
            'EUR_USD': -4,  # 0.0001
            'GBP_USD': -4,
            'USD_JPY': -2,  # 0.01
            'EUR_JPY': -2,
            'GBP_JPY': -2,
            'AUD_USD': -4
        }

        if self.verbose:
            print("="*80)
            print("CONFIGURACIÓN DE COSTOS ADICIONALES")
            print("="*80)
            self._mostrar_configuracion()

    def _mostrar_configuracion(self):
        """Muestra configuración actual de costos."""
        print("\n1. SLIPPAGE:")
        print("   ──────────")
        print(f"   Base:         {self.slippage_config['base_slippage_pips']} pips")
        print(f"   Rango:        {self.slippage_config['min_slippage_pips']}-{self.slippage_config['max_slippage_pips']} pips")
        print(f"   Modelo:       slippage = base × (1 + 0.5 × (ATR_actual/ATR_avg - 1))")

        print("\n2. SWAP (USD per lot per night):")
        print("   ────────────────────────────────")
        print("   Par      │ Long    │ Short   │")
        print("   ─────────┼─────────┼─────────┤")
        for par, rates in self.swap_rates.items():
            long_rate = rates['long']
            short_rate = rates['short']
            print(f"   {par:8s} │ ${long_rate:+6.2f} │ ${short_rate:+6.2f} │")
        print(f"\n   Triple swap: Miércoles → Jueves (cubre weekend)")

        print("\n3. COMISIÓN:")
        print("   ──────────")
        print(f"   Broker:       {self.commission_config['broker']}")
        print(f"   Comisión:     ${self.commission_config['commission_per_lot']:.2f} per lot")
        print(f"   Nota:         Incluido en spread")

    def calcular_slippage(
        self,
        par: str,
        atr_actual: float,
        atr_promedio: float,
        es_noticia: bool = False
    ) -> float:
        """
        Calcula slippage dinámico basado en volatilidad.

        MODELO:
        slippage = base_slippage × (1 + 0.5 × (ATR_actual / ATR_promedio - 1))

        Parameters:
        -----------
        par : str
            Par de divisas
        atr_actual : float
            ATR actual
        atr_promedio : float
            ATR promedio del período lookback
        es_noticia : bool
            Si es momento de noticia económica (default: False)

        Returns:
        --------
        slippage_pips : float
            Slippage en pips
        """
        base_slippage = self.slippage_config['base_slippage_pips']
        vol_mult = self.slippage_config['volatility_multiplier']

        # Si es noticia, usar slippage máximo
        if es_noticia:
            slippage_pips = self.slippage_config['max_slippage_pips']
        else:
            # Calcular ratio de volatilidad
            if atr_promedio > 0:
                vol_ratio = atr_actual / atr_promedio
            else:
                vol_ratio = 1.0

            # Aplicar modelo
            slippage_pips = base_slippage * (1 + vol_mult * (vol_ratio - 1))

            # Aplicar límites
            slippage_pips = max(
                self.slippage_config['min_slippage_pips'],
                min(slippage_pips, self.slippage_config['max_slippage_pips'])
            )

        return round(slippage_pips, 2)

    def calcular_slippage_usd(
        self,
        par: str,
        slippage_pips: float,
        lotes: float = 1.0
    ) -> float:
        """
        Convierte slippage en pips a USD.

        Parameters:
        -----------
        par : str
            Par de divisas
        slippage_pips : float
            Slippage en pips
        lotes : float
            Número de lotes (default: 1.0 = 100,000 unidades)

        Returns:
        --------
        slippage_usd : float
            Slippage en USD
        """
        # Valor de 1 pip por lote estándar
        if 'JPY' in par:
            usd_per_pip = 9.0  # Aproximado
        else:
            usd_per_pip = 10.0

        slippage_usd = slippage_pips * usd_per_pip * lotes

        return round(slippage_usd, 2)

    def calcular_swap(
        self,
        par: str,
        direccion: str,
        lotes: float = 1.0,
        noches: int = 1,
        dia_apertura: int = 0
    ) -> float:
        """
        Calcula swap overnight.

        SWAP si posición se mantiene después del rollover (17:00 NY = 22:00 UTC).

        Parameters:
        -----------
        par : str
            Par de divisas
        direccion : str
            'long' o 'short'
        lotes : float
            Número de lotes (default: 1.0)
        noches : int
            Número de noches (default: 1)
        dia_apertura : int
            Día de semana de apertura (0=Lun, ..., 4=Vie)
            Para detectar triple swap

        Returns:
        --------
        swap_usd : float
            Swap en USD (positivo = ganancia, negativo = costo)
        """
        if par not in self.swap_rates:
            if self.verbose:
                print(f"⚠ Par {par} no tiene swap configurado, usando 0")
            return 0.0

        # Obtener tasa de swap
        direccion_lower = direccion.lower()
        swap_rate = self.swap_rates[par].get(direccion_lower, 0.0)

        # Calcular swap base
        swap_total = swap_rate * lotes * noches

        # ALTO #10: Triple swap - verificar si se mantiene a través del cierre del miércoles
        # Triple swap se aplica el miércoles para cubrir sábado y domingo
        # Verificar si alguna noche entre entry y exit es miércoles (día 2)
        triple_swap_aplicado = False
        for day_offset in range(noches):
            current_day = (dia_apertura + day_offset) % 5  # Módulo 5 para días laborales
            if current_day == 2:  # Miércoles
                swap_total += swap_rate * lotes * 2  # +2 noches por fin de semana
                triple_swap_aplicado = True
                break  # Solo aplicar una vez

        return round(swap_total, 2)

    def calcular_comision(
        self,
        par: str,
        lotes: float = 1.0,
        direccion: str = 'long'
    ) -> float:
        """
        Calcula comisión por operación.

        Para OANDA: $0 (incluido en spread)

        Parameters:
        -----------
        par : str
            Par de divisas
        lotes : float
            Número de lotes
        direccion : str
            'long' o 'short' (no afecta en OANDA)

        Returns:
        --------
        comision_usd : float
            Comisión en USD
        """
        comision = self.commission_config['commission_per_lot'] * lotes
        return round(comision, 2)

    def calcular_costos_totales(
        self,
        par: str,
        direccion: str,
        lotes: float,
        atr_actual: float,
        atr_promedio: float,
        noches: int = 0,
        dia_apertura: int = 0,
        es_noticia: bool = False
    ) -> Dict[str, float]:
        """
        Calcula todos los costos de una operación.

        Parameters:
        -----------
        par : str
            Par de divisas
        direccion : str
            'long' o 'short'
        lotes : float
            Número de lotes
        atr_actual : float
            ATR actual
        atr_promedio : float
            ATR promedio
        noches : int
            Número de noches que se mantiene la posición
        dia_apertura : int
            Día de semana de apertura (0-4)
        es_noticia : bool
            Si es momento de noticia

        Returns:
        --------
        costos : Dict[str, float]
            Diccionario con todos los costos
        """
        # Slippage
        slippage_pips = self.calcular_slippage(
            par=par,
            atr_actual=atr_actual,
            atr_promedio=atr_promedio,
            es_noticia=es_noticia
        )
        slippage_usd = self.calcular_slippage_usd(par, slippage_pips, lotes)

        # Swap (solo si hay noches)
        if noches > 0:
            swap_usd = self.calcular_swap(
                par=par,
                direccion=direccion,
                lotes=lotes,
                noches=noches,
                dia_apertura=dia_apertura
            )
        else:
            swap_usd = 0.0

        # Comisión
        comision_usd = self.calcular_comision(par, lotes, direccion)

        # Total
        costo_total = slippage_usd + abs(swap_usd if swap_usd < 0 else 0) + comision_usd

        return {
            'slippage_pips': slippage_pips,
            'slippage_usd': slippage_usd,
            'swap_usd': swap_usd,
            'comision_usd': comision_usd,
            'costo_total_usd': costo_total
        }

    def guardar_configuracion(self, ruta_salida: Optional[str] = None):
        """
        Guarda configuración en JSON.

        Parameters:
        -----------
        ruta_salida : str, optional
            Ruta de salida (default: backtest/costos_config.json)
        """
        if ruta_salida is None:
            ruta_salida = Path(__file__).parent / 'costos_config.json'
        else:
            ruta_salida = Path(ruta_salida)

        config_completa = {
            'slippage': self.slippage_config,
            'swap_rates': self.swap_rates,
            'commission': self.commission_config,
            'pip_locations': self.pip_locations
        }

        with open(ruta_salida, 'w', encoding='utf-8') as f:
            json.dump(config_completa, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"\n✓ Configuración guardada: {ruta_salida}")

    def generar_tabla_swap(self) -> pd.DataFrame:
        """
        Genera tabla de swap rates.

        Returns:
        --------
        df_swap : pd.DataFrame
            Tabla con swap rates por par
        """
        registros = []

        for par, rates in self.swap_rates.items():
            registros.append({
                'pair': par,
                'swap_long_usd': rates['long'],
                'swap_short_usd': rates['short'],
                'triple_swap_day': 'Wednesday → Thursday'
            })

        df_swap = pd.DataFrame(registros)
        return df_swap


def ejemplo_uso():
    """
    Ejemplo de uso de ConfiguracionCostosAdicionales.
    """
    print("="*80)
    print("EJEMPLO: Configuración de Costos Adicionales")
    print("="*80)

    # Inicializar configuración
    config_costos = ConfiguracionCostosAdicionales(verbose=True)

    # Ejemplo 1: Calcular slippage en condición normal
    print(f"\n{'='*80}")
    print(f"EJEMPLO 1: Slippage en Condición Normal")
    print(f"{'='*80}")

    atr_actual = 0.0015
    atr_promedio = 0.0015
    slippage = config_costos.calcular_slippage(
        par='EUR_USD',
        atr_actual=atr_actual,
        atr_promedio=atr_promedio,
        es_noticia=False
    )
    slippage_usd = config_costos.calcular_slippage_usd('EUR_USD', slippage, lotes=1.0)

    print(f"\nATR actual = ATR promedio")
    print(f"Slippage: {slippage} pips = ${slippage_usd} por lote")

    # Ejemplo 2: Slippage con alta volatilidad
    print(f"\n{'='*80}")
    print(f"EJEMPLO 2: Slippage con Alta Volatilidad")
    print(f"{'='*80}")

    atr_actual = 0.0025  # 66% más alto que promedio
    atr_promedio = 0.0015
    slippage = config_costos.calcular_slippage(
        par='EUR_USD',
        atr_actual=atr_actual,
        atr_promedio=atr_promedio,
        es_noticia=False
    )
    slippage_usd = config_costos.calcular_slippage_usd('EUR_USD', slippage, lotes=1.0)

    print(f"\nATR actual = {atr_actual:.4f} (66% mayor)")
    print(f"Slippage: {slippage} pips = ${slippage_usd} por lote")

    # Ejemplo 3: Swap overnight
    print(f"\n{'='*80}")
    print(f"EJEMPLO 3: Swap Overnight")
    print(f"{'='*80}")

    swap_long = config_costos.calcular_swap(
        par='EUR_USD',
        direccion='long',
        lotes=1.0,
        noches=1,
        dia_apertura=1  # Martes
    )

    swap_short = config_costos.calcular_swap(
        par='EUR_USD',
        direccion='short',
        lotes=1.0,
        noches=1,
        dia_apertura=1
    )

    print(f"\nEUR_USD - 1 noche (Martes):")
    print(f"  Long:  ${swap_long:+.2f}")
    print(f"  Short: ${swap_short:+.2f}")

    # Ejemplo 4: Triple swap
    print(f"\n{'='*80}")
    print(f"EJEMPLO 4: Triple Swap (Miércoles → Jueves)")
    print(f"{'='*80}")

    swap_triple = config_costos.calcular_swap(
        par='EUR_USD',
        direccion='long',
        lotes=1.0,
        noches=1,
        dia_apertura=2  # Miércoles
    )

    print(f"\nEUR_USD Long - 1 noche (Miércoles → Jueves):")
    print(f"  Swap: ${swap_triple:+.2f} (triple por weekend)")

    # Ejemplo 5: Costos totales de operación
    print(f"\n{'='*80}")
    print(f"EJEMPLO 5: Costos Totales de Operación")
    print(f"{'='*80}")

    costos = config_costos.calcular_costos_totales(
        par='EUR_USD',
        direccion='long',
        lotes=1.0,
        atr_actual=0.0018,
        atr_promedio=0.0015,
        noches=2,
        dia_apertura=1,  # Martes
        es_noticia=False
    )

    print(f"\nEUR_USD Long - 1 lote - 2 noches:")
    print(f"  Slippage:  {costos['slippage_pips']:.2f} pips = ${costos['slippage_usd']:.2f}")
    print(f"  Swap:      ${costos['swap_usd']:+.2f}")
    print(f"  Comisión:  ${costos['comision_usd']:.2f}")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL:     ${costos['costo_total_usd']:.2f}")

    # Guardar configuración
    config_costos.guardar_configuracion()

    # Generar tabla de swap
    df_swap = config_costos.generar_tabla_swap()
    print(f"\n{'='*80}")
    print(f"TABLA DE SWAP RATES")
    print(f"{'='*80}")
    print(df_swap.to_string(index=False))

    print("\n✓ Ejemplo completado")


if __name__ == '__main__':
    ejemplo_uso()
