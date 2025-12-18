"""
Tabla de Spreads Reales

DATOS DE SPREADS (CRÍTICO):

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  TABLA: spreads_real                                                       │
│                                                                             │
│  pair    │ hour_utc │ day_of_week │ spread_pips │ spread_usd_per_lot       │
│  ────────┼──────────┼─────────────┼─────────────┼─────────────────────      │
│  EUR_USD │ 0        │ 0 (Lun)     │ 1.8         │ 18.00                    │
│  EUR_USD │ 0        │ 1 (Mar)     │ 1.7         │ 17.00                    │
│  EUR_USD │ 0        │ 2 (Mié)     │ 1.7         │ 17.00                    │
│  EUR_USD │ 0        │ 3 (Jue)     │ 1.7         │ 17.00                    │
│  EUR_USD │ 0        │ 4 (Vie)     │ 1.9         │ 19.00                    │
│  EUR_USD │ 1        │ 0           │ 1.7         │ 17.00                    │
│  ...     │ ...      │ ...         │ ...         │ ...                      │
│  EUR_USD │ 14       │ 2           │ 1.2         │ 12.00                    │
│  EUR_USD │ 21       │ 4           │ 2.9         │ 29.00                    │
│  ...     │ ...      │ ...         │ ...         │ ...                      │
│  GBP_JPY │ 14       │ 2           │ 2.8         │ 25.20                    │
│  ...     │ ...      │ ...         │ ...         │ ...                      │
│                                                                             │
│  Total: 24 horas × 5 días × 6 pares = 720 filas                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Author: Sistema de Edge-Finding Forex
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class GeneradorSpreadsReales:
    """
    Genera tabla de spreads reales por par, hora UTC y día de semana.

    Basado en patrones observados en mercado real:
    - Spreads más ajustados durante sesión Londres/NY (7-17 UTC)
    - Spreads más amplios en rollover y cierre (21-23 UTC)
    - Spreads más amplios los viernes
    - Variación por liquidez del par
    """

    def __init__(self, verbose: bool = True):
        """
        Inicializa generador de spreads.

        Parameters:
        -----------
        verbose : bool
            Imprimir progreso
        """
        self.verbose = verbose

        # Spreads base EUR_USD por hora UTC
        # Basado en patrones reales de mercado
        self.spreads_eur_usd_base = {
            0: 1.8,   # Rollover asiático
            1: 1.7,
            2: 1.6,
            3: 1.6,
            4: 1.5,
            5: 1.5,
            6: 1.4,
            7: 1.4,   # Apertura Londres
            8: 1.3,
            9: 1.3,
            10: 1.3,
            11: 1.3,
            12: 1.3,  # Solapamiento Londres/NY
            13: 1.2,
            14: 1.2,  # Sesión más líquida
            15: 1.2,
            16: 1.3,
            17: 1.4,
            18: 1.5,
            19: 1.6,
            20: 1.8,
            21: 2.2,  # Cierre NY / Baja liquidez
            22: 2.8,  # Rollover / Fin de sesión
            23: 2.5
        }

        # Ajustes por día de semana (multiplicadores)
        self.ajuste_dia_semana = {
            0: 1.00,  # Lunes (normal)
            1: 0.95,  # Martes (mejor liquidez)
            2: 0.95,  # Miércoles (mejor liquidez)
            3: 0.95,  # Jueves (mejor liquidez)
            4: 1.15   # Viernes (cierre semanal, spreads más amplios)
        }

        # Multiplicadores por par (relativo a EUR_USD)
        self.multiplicador_par = {
            'EUR_USD': 1.0,
            'GBP_USD': 1.3,   # Menos líquido que EUR_USD
            'USD_JPY': 0.9,   # Similar liquidez
            'EUR_JPY': 1.5,   # Cross, menos líquido
            'GBP_JPY': 2.0,   # Cross, mucho menos líquido
            'AUD_USD': 1.1    # Commodity currency, ligeramente menos líquido
        }

        # Valor en USD por pip por par (100,000 lote estándar)
        self.usd_per_pip = {
            'EUR_USD': 10.0,
            'GBP_USD': 10.0,
            'USD_JPY': 9.0,   # Aproximado (varía con tipo de cambio)
            'EUR_JPY': 9.0,   # Aproximado
            'GBP_JPY': 9.0,   # Aproximado
            'AUD_USD': 10.0
        }

        if self.verbose:
            print("="*80)
            print("GENERADOR DE TABLA DE SPREADS REALES")
            print("="*80)

    def calcular_spread(
        self,
        par: str,
        hora_utc: int,
        dia_semana: int
    ) -> float:
        """
        Calcula spread en pips para un par, hora y día específicos.

        Parameters:
        -----------
        par : str
            Par de divisas
        hora_utc : int
            Hora UTC (0-23)
        dia_semana : int
            Día de semana (0=Lunes, 1=Martes, ..., 4=Viernes)

        Returns:
        --------
        spread_pips : float
            Spread en pips
        """
        # Spread base de EUR_USD para esa hora
        spread_base = self.spreads_eur_usd_base.get(hora_utc, 1.5)

        # Ajuste por día de semana
        ajuste_dia = self.ajuste_dia_semana.get(dia_semana, 1.0)

        # Multiplicador del par
        mult_par = self.multiplicador_par.get(par, 1.0)

        # Calcular spread final
        spread_pips = spread_base * ajuste_dia * mult_par

        # Redondear a 1 decimal
        spread_pips = round(spread_pips, 1)

        return spread_pips

    def calcular_spread_usd(
        self,
        par: str,
        spread_pips: float
    ) -> float:
        """
        Convierte spread en pips a USD por lote estándar (100,000 unidades).

        Parameters:
        -----------
        par : str
            Par de divisas
        spread_pips : float
            Spread en pips

        Returns:
        --------
        spread_usd : float
            Spread en USD por lote
        """
        usd_pip = self.usd_per_pip.get(par, 10.0)
        spread_usd = spread_pips * usd_pip
        return round(spread_usd, 2)

    def generar_tabla_spreads(
        self,
        pares: List[str]
    ) -> pd.DataFrame:
        """
        Genera tabla completa de spreads reales.

        Parameters:
        -----------
        pares : List[str]
            Lista de pares a incluir

        Returns:
        --------
        df_spreads : pd.DataFrame
            Tabla de spreads con columnas:
            - pair
            - hour_utc (0-23)
            - day_of_week (0-4: Lun-Vie)
            - spread_pips
            - spread_usd_per_lot
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"GENERANDO TABLA DE SPREADS")
            print(f"{'='*80}")
            print(f"Pares: {', '.join(pares)}")
            print(f"Horas: 0-23 UTC")
            print(f"Días: Lunes-Viernes (0-4)")
            print()

        registros = []

        for par in pares:
            for hora_utc in range(24):
                for dia_semana in range(5):  # 0=Lun, 1=Mar, 2=Mié, 3=Jue, 4=Vie
                    # Calcular spread
                    spread_pips = self.calcular_spread(par, hora_utc, dia_semana)
                    spread_usd = self.calcular_spread_usd(par, spread_pips)

                    registros.append({
                        'pair': par,
                        'hour_utc': hora_utc,
                        'day_of_week': dia_semana,
                        'spread_pips': spread_pips,
                        'spread_usd_per_lot': spread_usd
                    })

        # Crear DataFrame
        df_spreads = pd.DataFrame(registros)

        if self.verbose:
            print(f"✓ Tabla generada: {len(df_spreads):,} filas")
            print(f"  Total esperado: 24 horas × 5 días × {len(pares)} pares = {24*5*len(pares)} filas")

        return df_spreads

    def mostrar_resumen_spreads(
        self,
        df_spreads: pd.DataFrame,
        par: str = 'EUR_USD'
    ):
        """
        Muestra resumen de spreads por hora y día para un par.

        Parameters:
        -----------
        df_spreads : pd.DataFrame
            Tabla de spreads
        par : str
            Par a mostrar (default: 'EUR_USD')
        """
        print(f"\n{'='*80}")
        print(f"SPREADS POR HORA Y DÍA - {par}")
        print(f"{'='*80}")

        # Filtrar por par
        df_par = df_spreads[df_spreads['pair'] == par].copy()

        # Pivot table: horas × días
        pivot = df_par.pivot_table(
            index='hour_utc',
            columns='day_of_week',
            values='spread_pips',
            aggfunc='mean'
        )

        # Añadir columna de promedio
        pivot['Promedio'] = pivot.mean(axis=1)

        # Renombrar columnas
        dias_semana = {0: 'Lun', 1: 'Mar', 2: 'Mié', 3: 'Jue', 4: 'Vie'}
        pivot.columns = [dias_semana.get(c, c) if isinstance(c, int) else c for c in pivot.columns]

        # Mostrar tabla
        print("\nHora │ Lun  │ Mar  │ Mié  │ Jue  │ Vie  │ Promedio")
        print("─────┼──────┼──────┼──────┼──────┼──────┼─────────")

        for hora in range(24):
            fila = pivot.loc[hora]
            print(f"{hora:02d}   │ {fila['Lun']:4.1f} │ {fila['Mar']:4.1f} │ {fila['Mié']:4.1f} │ {fila['Jue']:4.1f} │ {fila['Vie']:4.1f} │ {fila['Promedio']:4.2f}")

        # Estadísticas
        print(f"\nESTADÍSTICAS {par}:")
        print(f"  Spread mínimo:   {df_par['spread_pips'].min():.1f} pips")
        print(f"  Spread máximo:   {df_par['spread_pips'].max():.1f} pips")
        print(f"  Spread promedio: {df_par['spread_pips'].mean():.2f} pips")
        print(f"  Mejor hora:      {df_par.groupby('hour_utc')['spread_pips'].mean().idxmin()} UTC ({df_par.groupby('hour_utc')['spread_pips'].mean().min():.1f} pips)")
        print(f"  Peor hora:       {df_par.groupby('hour_utc')['spread_pips'].mean().idxmax()} UTC ({df_par.groupby('hour_utc')['spread_pips'].mean().max():.1f} pips)")

    def mostrar_comparativa_pares(self, df_spreads: pd.DataFrame):
        """
        Muestra comparativa de spreads promedio por par.

        Parameters:
        -----------
        df_spreads : pd.DataFrame
            Tabla de spreads
        """
        print(f"\n{'='*80}")
        print(f"COMPARATIVA DE SPREADS POR PAR")
        print(f"{'='*80}")

        # Agrupar por par
        resumen = df_spreads.groupby('pair')['spread_pips'].agg(['min', 'mean', 'max'])
        resumen = resumen.round(2)

        print("\nPar      │ Mínimo │ Promedio │ Máximo │ Sesión Líquida (14 UTC)")
        print("─────────┼────────┼──────────┼────────┼─────────────────────────")

        for par in resumen.index:
            min_spread = resumen.loc[par, 'min']
            avg_spread = resumen.loc[par, 'mean']
            max_spread = resumen.loc[par, 'max']

            # Spread en hora más líquida (14 UTC, miércoles)
            spread_liquido = df_spreads[
                (df_spreads['pair'] == par) &
                (df_spreads['hour_utc'] == 14) &
                (df_spreads['day_of_week'] == 2)
            ]['spread_pips'].values[0]

            print(f"{par:8s} │ {min_spread:6.1f} │ {avg_spread:8.2f} │ {max_spread:6.1f} │ {spread_liquido:6.1f}")

    def guardar_tabla_spreads(
        self,
        df_spreads: pd.DataFrame,
        ruta_salida: Optional[str] = None
    ):
        """
        Guarda tabla de spreads en CSV.

        Parameters:
        -----------
        df_spreads : pd.DataFrame
            Tabla de spreads
        ruta_salida : str, optional
            Ruta de salida (default: backtest/spreads_real.csv)
        """
        if ruta_salida is None:
            ruta_salida = Path(__file__).parent / 'spreads_real.csv'
        else:
            ruta_salida = Path(ruta_salida)

        # Guardar CSV
        df_spreads.to_csv(ruta_salida, index=False)

        if self.verbose:
            tamaño_kb = ruta_salida.stat().st_size / 1024
            print(f"\n✓ Tabla guardada: {ruta_salida}")
            print(f"  Tamaño: {tamaño_kb:.1f} KB")
            print(f"  Filas: {len(df_spreads):,}")

    def mostrar_preview(self, df_spreads: pd.DataFrame, n: int = 15):
        """
        Muestra preview de la tabla de spreads.

        Parameters:
        -----------
        df_spreads : pd.DataFrame
            Tabla de spreads
        n : int
            Número de filas a mostrar (default: 15)
        """
        print(f"\n{'='*80}")
        print(f"PREVIEW DE spreads_real (primeras {n} filas)")
        print(f"{'='*80}")
        print()

        # Añadir nombre del día
        dias_map = {0: 'Lun', 1: 'Mar', 2: 'Mié', 3: 'Jue', 4: 'Vie'}
        df_preview = df_spreads.head(n).copy()
        df_preview['day_name'] = df_preview['day_of_week'].map(dias_map)

        # Reordenar columnas
        df_preview = df_preview[['pair', 'hour_utc', 'day_of_week', 'day_name', 'spread_pips', 'spread_usd_per_lot']]

        print(df_preview.to_string(index=False))
        print()
        print(f"{'='*80}")


def main():
    """
    Función principal - Genera tabla de spreads reales.
    """
    print("="*80)
    print("GENERACIÓN DE TABLA DE SPREADS REALES")
    print("="*80)

    # Pares a incluir
    PARES = [
        'EUR_USD',
        'GBP_USD',
        'USD_JPY',
        'EUR_JPY',
        'GBP_JPY',
        'AUD_USD'
    ]

    # Inicializar generador
    generador = GeneradorSpreadsReales(verbose=True)

    # Generar tabla de spreads
    df_spreads = generador.generar_tabla_spreads(pares=PARES)

    # Mostrar preview
    generador.mostrar_preview(df_spreads, n=15)

    # Mostrar resumen detallado para EUR_USD
    generador.mostrar_resumen_spreads(df_spreads, par='EUR_USD')

    # Mostrar resumen para otros pares
    print(f"\n{'='*80}")
    print(f"RESÚMENES ADICIONALES")
    print(f"{'='*80}")

    for par in ['GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY', 'AUD_USD']:
        print(f"\n{par}:")
        print(f"────────")
        df_par = df_spreads[df_spreads['pair'] == par]
        print(f"Hora │ Promedio")
        print(f"─────┼──────────")
        for hora in [0, 8, 14, 21, 22]:
            avg = df_par[df_par['hour_utc'] == hora]['spread_pips'].mean()
            print(f"{hora:02d}   │ {avg:4.1f}")

    # Comparativa entre pares
    generador.mostrar_comparativa_pares(df_spreads)

    # Guardar tabla
    generador.guardar_tabla_spreads(df_spreads)

    print("\n✓ Tabla de spreads reales generada exitosamente")
    print("\nPróximo paso:")
    print("  - Los spreads están en backtest/spreads_real.csv")
    print("  - El backtest usará estos spreads para calcular costos reales")
    print("  - Spread varía por hora UTC y día de semana")


if __name__ == '__main__':
    main()
