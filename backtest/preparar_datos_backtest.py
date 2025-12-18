"""
Preparar Datos para Backtest

Convierte datos OHLC descargados al formato estándar para backtest:

TABLA: raw_ohlcv
────────────────────────────────────────────────────────────────────
timestamp           │ pair    │ open    │ high    │ low     │ close   │ volume
────────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼───────
2019-01-01 00:00:00 │ EUR_USD │ 1.14510 │ 1.14525 │ 1.14480 │ 1.14505 │ 1250
2019-01-01 00:15:00 │ EUR_USD │ 1.14505 │ 1.14520 │ 1.14490 │ 1.14515 │ 980
2019-01-01 00:30:00 │ EUR_USD │ 1.14515 │ 1.14530 │ 1.14500 │ 1.14510 │ 1100
...                 │ ...     │ ...     │ ...     │ ...     │ ...     │ ...

Timeframe: M15 (velas de 15 minutos)
Período: 5 años (2019-2023)
Pares: EUR_USD, GBP_USD, USD_JPY, EUR_JPY, GBP_JPY, AUD_USD
Total filas: ~350,000 por par × 6 pares = ~2.1 millones

Author: Sistema de Edge-Finding Forex
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class PreparadorDatosBacktest:
    """
    Prepara datos OHLC descargados al formato estándar para backtest.

    Convierte CSVs individuales por par a tabla única raw_ohlcv.
    """

    def __init__(
        self,
        directorio_ohlc: str,
        timeframe: str = 'M15',
        verbose: bool = True
    ):
        """
        Inicializa preparador de datos.

        Parameters:
        -----------
        directorio_ohlc : str
            Directorio con datos OHLC descargados (datos/ohlc/)
        timeframe : str
            Timeframe a procesar (default: 'M15')
        verbose : bool
            Imprimir progreso
        """
        self.directorio_ohlc = Path(directorio_ohlc)
        self.timeframe = timeframe
        self.verbose = verbose

        if self.verbose:
            print("="*80)
            print("PREPARADOR DE DATOS PARA BACKTEST")
            print("="*80)
            print(f"Directorio OHLC: {self.directorio_ohlc}")
            print(f"Timeframe: {self.timeframe}")
            print("="*80)

    def cargar_par(self, par: str) -> Optional[pd.DataFrame]:
        """
        Carga datos de un par específico.

        Parameters:
        -----------
        par : str
            Nombre del par (ej: 'EUR_USD')

        Returns:
        --------
        df : pd.DataFrame or None
            DataFrame con datos del par
        """
        archivo = self.directorio_ohlc / par / f"{self.timeframe}.csv"

        if not archivo.exists():
            if self.verbose:
                print(f"  ✗ {par}: Archivo no encontrado: {archivo}")
            return None

        try:
            # Cargar CSV
            df = pd.read_csv(archivo, index_col=0, parse_dates=True)

            # Extraer solo columnas necesarias (mid prices)
            df_simple = pd.DataFrame({
                'timestamp': df.index,
                'pair': par,
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'close': df['close'],
                'volume': df['volume']
            })

            # Resetear índice
            df_simple.reset_index(drop=True, inplace=True)

            if self.verbose:
                fecha_inicio = df.index[0].strftime('%Y-%m-%d')
                fecha_fin = df.index[-1].strftime('%Y-%m-%d')
                print(f"  ✓ {par}: {len(df_simple):,} velas ({fecha_inicio} → {fecha_fin})")

            return df_simple

        except Exception as e:
            if self.verbose:
                print(f"  ✗ {par}: Error al cargar: {e}")
            return None

    def preparar_raw_ohlcv(
        self,
        pares: List[str],
        fecha_inicio: Optional[str] = None,
        fecha_fin: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepara tabla raw_ohlcv con múltiples pares.

        Parameters:
        -----------
        pares : List[str]
            Lista de pares a incluir
        fecha_inicio : str, optional
            Fecha de inicio (formato: 'YYYY-MM-DD')
        fecha_fin : str, optional
            Fecha de fin (formato: 'YYYY-MM-DD')

        Returns:
        --------
        df_raw_ohlcv : pd.DataFrame
            Tabla raw_ohlcv consolidada
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"PREPARANDO TABLA raw_ohlcv")
            print(f"{'='*80}")
            print(f"Pares: {', '.join(pares)}")
            if fecha_inicio:
                print(f"Fecha inicio: {fecha_inicio}")
            if fecha_fin:
                print(f"Fecha fin: {fecha_fin}")
            print()

        # Cargar cada par
        dfs_pares = []

        for par in pares:
            df_par = self.cargar_par(par)

            if df_par is not None:
                # Filtrar por fechas si se especifican
                if fecha_inicio or fecha_fin:
                    mask = pd.Series([True] * len(df_par))

                    if fecha_inicio:
                        mask &= (df_par['timestamp'] >= pd.Timestamp(fecha_inicio))

                    if fecha_fin:
                        mask &= (df_par['timestamp'] <= pd.Timestamp(fecha_fin))

                    df_par = df_par[mask].copy()

                dfs_pares.append(df_par)

        if len(dfs_pares) == 0:
            raise ValueError("No se pudieron cargar datos de ningún par")

        # Concatenar todos los pares
        df_raw_ohlcv = pd.concat(dfs_pares, ignore_index=True)

        # Ordenar por timestamp y par
        df_raw_ohlcv.sort_values(['timestamp', 'pair'], inplace=True)
        df_raw_ohlcv.reset_index(drop=True, inplace=True)

        # Verificar estructura
        columnas_requeridas = ['timestamp', 'pair', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in df_raw_ohlcv.columns for col in columnas_requeridas), \
            "Faltan columnas requeridas"

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"TABLA raw_ohlcv CREADA")
            print(f"{'='*80}")
            print(f"Total filas: {len(df_raw_ohlcv):,}")
            print(f"Pares únicos: {df_raw_ohlcv['pair'].nunique()}")
            print(f"Período: {df_raw_ohlcv['timestamp'].min()} → {df_raw_ohlcv['timestamp'].max()}")
            print(f"Columnas: {', '.join(df_raw_ohlcv.columns)}")

            # Estadísticas por par
            print(f"\nFILAS POR PAR:")
            for par in pares:
                n_filas = (df_raw_ohlcv['pair'] == par).sum()
                print(f"  {par}: {n_filas:,} filas")

            print(f"{'='*80}")

        return df_raw_ohlcv

    def guardar_raw_ohlcv(
        self,
        df_raw_ohlcv: pd.DataFrame,
        ruta_salida: Optional[str] = None
    ):
        """
        Guarda tabla raw_ohlcv en CSV.

        Parameters:
        -----------
        df_raw_ohlcv : pd.DataFrame
            Tabla raw_ohlcv
        ruta_salida : str, optional
            Ruta de salida (default: backtest/raw_ohlcv.csv)
        """
        if ruta_salida is None:
            ruta_salida = Path(__file__).parent / 'raw_ohlcv.csv'
        else:
            ruta_salida = Path(ruta_salida)

        # Guardar CSV
        df_raw_ohlcv.to_csv(ruta_salida, index=False)

        if self.verbose:
            tamaño_mb = ruta_salida.stat().st_size / (1024 * 1024)
            print(f"\n✓ Tabla guardada: {ruta_salida}")
            print(f"  Tamaño: {tamaño_mb:.1f} MB")

    def mostrar_preview(self, df_raw_ohlcv: pd.DataFrame, n: int = 10):
        """
        Muestra preview de la tabla raw_ohlcv.

        Parameters:
        -----------
        df_raw_ohlcv : pd.DataFrame
            Tabla raw_ohlcv
        n : int
            Número de filas a mostrar (default: 10)
        """
        print(f"\n{'='*80}")
        print(f"PREVIEW DE raw_ohlcv (primeras {n} filas)")
        print(f"{'='*80}")
        print()

        # Formatear para mostrar
        df_preview = df_raw_ohlcv.head(n).copy()
        df_preview['timestamp'] = df_preview['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Redondear precios
        for col in ['open', 'high', 'low', 'close']:
            df_preview[col] = df_preview[col].round(5)

        print(df_preview.to_string(index=False))
        print()
        print(f"{'='*80}")


def main():
    """
    Función principal - Prepara datos para backtest.
    """
    print("="*80)
    print("PREPARACIÓN DE DATOS PARA BACKTEST")
    print("="*80)

    # Configuración
    DIRECTORIO_OHLC = Path(__file__).parent.parent / 'datos' / 'ohlc'
    TIMEFRAME = 'M15'

    PARES = [
        'EUR_USD',
        'GBP_USD',
        'USD_JPY',
        'EUR_JPY',
        'GBP_JPY',
        'AUD_USD'
    ]

    # Opcional: filtrar por fechas
    # FECHA_INICIO = '2019-01-01'
    # FECHA_FIN = '2023-12-31'
    FECHA_INICIO = None
    FECHA_FIN = None

    # Inicializar preparador
    preparador = PreparadorDatosBacktest(
        directorio_ohlc=DIRECTORIO_OHLC,
        timeframe=TIMEFRAME,
        verbose=True
    )

    # Preparar tabla raw_ohlcv
    df_raw_ohlcv = preparador.preparar_raw_ohlcv(
        pares=PARES,
        fecha_inicio=FECHA_INICIO,
        fecha_fin=FECHA_FIN
    )

    # Mostrar preview
    preparador.mostrar_preview(df_raw_ohlcv, n=10)

    # Guardar
    preparador.guardar_raw_ohlcv(df_raw_ohlcv)

    print("\n✓ Datos preparados exitosamente")
    print("\nPróximo paso:")
    print("  - Los datos están en backtest/raw_ohlcv.csv")
    print("  - Listo para calcular transformaciones")
    print("  - Listo para ejecutar backtest")


if __name__ == '__main__':
    main()
