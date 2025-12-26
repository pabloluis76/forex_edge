"""
Normalización y Sincronización de Timestamps

Alinea timestamps entre múltiples pares forex para evitar problemas
de merge/join y asegurar sincronización perfecta.

PROBLEMA:
- Diferentes pares tienen gaps en timestamps diferentes
- 124,238 timestamps comunes vs 124,367 en el par con más datos
- Causa errores en merge, correlaciones cross-pair, y backtest multi-par

SOLUCIÓN:
1. Crear rango completo de timestamps M15
2. Alinear todos los pares a este rango
3. Forward-fill gaps pequeños (< 1 hora)
4. Dejar NaN en gaps grandes (fines de semana)

Author: Sistema de Edge-Finding Forex
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class NormalizadorTimestamps:
    """
    Normaliza y sincroniza timestamps entre múltiples pares forex.

    Asegura que todos los pares tengan exactamente los mismos timestamps,
    evitando problemas de merge/join.
    """

    def __init__(
        self,
        timeframe: str = 'M15',
        max_gap_fill: str = '1H',
        verbose: bool = True
    ):
        """
        Inicializa el normalizador.

        Parameters:
        -----------
        timeframe : str
            Timeframe de las velas (default: 'M15')
        max_gap_fill : str
            Máximo gap a rellenar con forward-fill (default: '1H')
            Gaps mayores (ej: fines de semana) quedan como NaN
        verbose : bool
            Imprimir progreso
        """
        self.timeframe = timeframe
        self.max_gap_fill = pd.Timedelta(max_gap_fill)
        self.verbose = verbose

        # Determinar frecuencia según timeframe
        self.freq_map = {
            'M1': '1min',
            'M5': '5min',
            'M15': '15min',
            'M30': '30min',
            'H1': '1H',
            'H4': '4H',
            'D': '1D'
        }
        self.freq = self.freq_map.get(timeframe, '15min')

        if self.verbose:
            print("="*80)
            print("NORMALIZADOR DE TIMESTAMPS")
            print("="*80)
            print(f"Timeframe: {self.timeframe}")
            print(f"Frecuencia: {self.freq}")
            print(f"Max gap fill: {self.max_gap_fill}")
            print("="*80)

    def crear_rango_completo(
        self,
        fecha_inicio: pd.Timestamp,
        fecha_fin: pd.Timestamp
    ) -> pd.DatetimeIndex:
        """
        Crea rango completo de timestamps sin gaps.

        Parameters:
        -----------
        fecha_inicio : pd.Timestamp
            Timestamp de inicio
        fecha_fin : pd.Timestamp
            Timestamp de fin

        Returns:
        --------
        rango : pd.DatetimeIndex
            Rango completo de timestamps a la frecuencia especificada
        """
        # Crear rango completo (incluye fines de semana)
        rango = pd.date_range(
            start=fecha_inicio,
            end=fecha_fin,
            freq=self.freq,
            tz='UTC'
        )

        if self.verbose:
            print(f"\nRango completo creado:")
            print(f"  Inicio: {rango[0]}")
            print(f"  Fin: {rango[-1]}")
            print(f"  Total timestamps: {len(rango):,}")

        return rango

    def normalizar_par(
        self,
        df: pd.DataFrame,
        rango_completo: pd.DatetimeIndex,
        par: str
    ) -> pd.DataFrame:
        """
        Normaliza timestamps de un par al rango completo.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con timestamp como index
        rango_completo : pd.DatetimeIndex
            Rango completo de timestamps objetivo
        par : str
            Nombre del par (para logging)

        Returns:
        --------
        df_normalizado : pd.DataFrame
            DataFrame con timestamps normalizados
        """
        # Asegurar que timestamp esté en index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        # Asegurar timezone UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif str(df.index.tz) != 'UTC':
            df.index = df.index.tz_convert('UTC')

        # Reindexar al rango completo
        df_normalizado = df.reindex(rango_completo)

        # Forward-fill gaps pequeños (< max_gap_fill)
        # Esto rellena gaps de minutos/horas pero NO fines de semana
        df_normalizado = df_normalizado.ffill(limit=int(self.max_gap_fill / pd.Timedelta(self.freq)))

        # Estadísticas
        n_original = len(df)
        n_normalizado = len(df_normalizado)
        n_added = n_normalizado - n_original
        n_nan = df_normalizado.isnull().any(axis=1).sum()

        if self.verbose:
            print(f"\n{par}:")
            print(f"  Timestamps originales: {n_original:,}")
            print(f"  Timestamps normalizados: {n_normalizado:,}")
            print(f"  Timestamps añadidos: {n_added:,}")
            print(f"  Timestamps con NaN: {n_nan:,} ({n_nan/n_normalizado*100:.1f}%)")

        return df_normalizado

    def normalizar_multiples_pares(
        self,
        dfs_pares: Dict[str, pd.DataFrame],
        fecha_inicio: Optional[pd.Timestamp] = None,
        fecha_fin: Optional[pd.Timestamp] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Normaliza timestamps de múltiples pares simultáneamente.

        Parameters:
        -----------
        dfs_pares : Dict[str, pd.DataFrame]
            Diccionario {par: DataFrame} con datos de cada par
        fecha_inicio : pd.Timestamp, optional
            Fecha de inicio (si None, usa el mínimo común)
        fecha_fin : pd.Timestamp, optional
            Fecha de fin (si None, usa el máximo común)

        Returns:
        --------
        dfs_normalizados : Dict[str, pd.DataFrame]
            Diccionario con DataFrames normalizados
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("NORMALIZANDO MÚLTIPLES PARES")
            print(f"{'='*80}")
            print(f"Pares: {', '.join(dfs_pares.keys())}")

        # Determinar rango común si no se especifica
        if fecha_inicio is None or fecha_fin is None:
            inicios = []
            fines = []

            for par, df in dfs_pares.items():
                if 'timestamp' in df.columns:
                    inicios.append(df['timestamp'].min())
                    fines.append(df['timestamp'].max())
                else:
                    inicios.append(df.index.min())
                    fines.append(df.index.max())

            if fecha_inicio is None:
                fecha_inicio = max(inicios)  # Último inicio (intersección)
                if self.verbose:
                    print(f"Fecha inicio (último inicio común): {fecha_inicio}")

            if fecha_fin is None:
                fecha_fin = min(fines)  # Primer fin (intersección)
                if self.verbose:
                    print(f"Fecha fin (primer fin común): {fecha_fin}")

        # Crear rango completo
        rango_completo = self.crear_rango_completo(fecha_inicio, fecha_fin)

        # Normalizar cada par
        dfs_normalizados = {}

        for par, df in dfs_pares.items():
            df_norm = self.normalizar_par(df, rango_completo, par)
            dfs_normalizados[par] = df_norm

        # Verificar sincronización
        if self.verbose:
            print(f"\n{'='*80}")
            print("VERIFICACIÓN DE SINCRONIZACIÓN")
            print(f"{'='*80}")

            indices = [df.index for df in dfs_normalizados.values()]
            todos_iguales = all(idx.equals(indices[0]) for idx in indices[1:])

            if todos_iguales:
                print("✓ TODOS LOS PARES TIENEN TIMESTAMPS IDÉNTICOS")
                print(f"  Total timestamps: {len(indices[0]):,}")
            else:
                print("✗ ERROR: Los pares aún tienen timestamps diferentes")
                for par, idx in zip(dfs_normalizados.keys(), indices):
                    print(f"  {par}: {len(idx):,} timestamps")

        return dfs_normalizados

    def eliminar_filas_con_todos_nan(
        self,
        dfs_pares: Dict[str, pd.DataFrame],
        columnas_criticas: List[str] = ['close']
    ) -> Dict[str, pd.DataFrame]:
        """
        Elimina timestamps donde TODOS los pares tienen NaN en columnas críticas.

        Útil para eliminar fines de semana donde no hay datos de ningún par.

        Parameters:
        -----------
        dfs_pares : Dict[str, pd.DataFrame]
            Diccionario con DataFrames normalizados
        columnas_criticas : List[str]
            Columnas a verificar (default: ['close'])

        Returns:
        --------
        dfs_filtrados : Dict[str, pd.DataFrame]
            DataFrames sin timestamps con todos NaN
        """
        if len(dfs_pares) == 0:
            return dfs_pares

        # Identificar timestamps donde TODOS tienen NaN
        indices = list(dfs_pares.values())[0].index

        mask_validos = pd.Series(False, index=indices)

        for par, df in dfs_pares.items():
            # Timestamp es válido si al menos una columna crítica NO es NaN
            es_valido = ~df[columnas_criticas].isnull().all(axis=1)
            mask_validos |= es_valido

        n_original = len(indices)
        n_validos = mask_validos.sum()
        n_eliminados = n_original - n_validos

        if self.verbose:
            print(f"\nEliminando timestamps con todos NaN:")
            print(f"  Timestamps originales: {n_original:,}")
            print(f"  Timestamps válidos: {n_validos:,}")
            print(f"  Timestamps eliminados: {n_eliminados:,} ({n_eliminados/n_original*100:.1f}%)")

        # Filtrar todos los DataFrames
        dfs_filtrados = {}
        for par, df in dfs_pares.items():
            dfs_filtrados[par] = df[mask_validos].copy()

        return dfs_filtrados


def normalizar_datos_desde_csv(
    directorio_ohlc: str = 'datos/ohlc',
    pares: List[str] = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY', 'AUD_USD'],
    timeframe: str = 'M15',
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
    eliminar_fines_semana: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Carga y normaliza datos OHLC desde CSV.

    Parameters:
    -----------
    directorio_ohlc : str
        Directorio con datos (default: 'datos/ohlc')
    pares : List[str]
        Lista de pares a cargar
    timeframe : str
        Timeframe (default: 'M15')
    fecha_inicio : str, optional
        Fecha inicio 'YYYY-MM-DD'
    fecha_fin : str, optional
        Fecha fin 'YYYY-MM-DD'
    eliminar_fines_semana : bool
        Eliminar timestamps sin datos (fines de semana)

    Returns:
    --------
    dfs_normalizados : Dict[str, pd.DataFrame]
        DataFrames normalizados por par
    """
    print("="*80)
    print("CARGA Y NORMALIZACIÓN DE DATOS FOREX")
    print("="*80)

    # Cargar datos
    directorio = Path(directorio_ohlc)
    dfs_pares = {}

    for par in pares:
        archivo = directorio / par / f"{timeframe}.csv"

        if archivo.exists():
            df = pd.read_csv(archivo, index_col=0, parse_dates=True)

            # Filtrar por fechas si se especifican
            if fecha_inicio or fecha_fin:
                mask = pd.Series(True, index=df.index)
                if fecha_inicio:
                    mask &= (df.index >= pd.Timestamp(fecha_inicio, tz='UTC'))
                if fecha_fin:
                    mask &= (df.index <= pd.Timestamp(fecha_fin, tz='UTC'))
                df = df[mask]

            dfs_pares[par] = df
            print(f"✓ {par}: {len(df):,} velas cargadas")
        else:
            print(f"✗ {par}: Archivo no encontrado: {archivo}")

    if len(dfs_pares) == 0:
        raise ValueError("No se pudo cargar ningún par")

    # Normalizar
    normalizador = NormalizadorTimestamps(timeframe=timeframe, verbose=True)

    dfs_normalizados = normalizador.normalizar_multiples_pares(
        dfs_pares,
        fecha_inicio=pd.Timestamp(fecha_inicio, tz='UTC') if fecha_inicio else None,
        fecha_fin=pd.Timestamp(fecha_fin, tz='UTC') if fecha_fin else None
    )

    # Eliminar fines de semana si se solicita
    if eliminar_fines_semana:
        dfs_normalizados = normalizador.eliminar_filas_con_todos_nan(
            dfs_normalizados,
            columnas_criticas=['close']
        )

    print(f"\n{'='*80}")
    print("✓ NORMALIZACIÓN COMPLETADA")
    print(f"{'='*80}")

    return dfs_normalizados


if __name__ == '__main__':
    """
    Ejemplo de uso.
    """
    # Normalizar datos
    dfs = normalizar_datos_desde_csv(
        pares=['EUR_USD', 'GBP_USD', 'USD_JPY'],
        timeframe='M15',
        fecha_inicio='2024-01-01',
        fecha_fin='2024-12-31',
        eliminar_fines_semana=True
    )

    # Verificar sincronización
    print("\nVERIFICACIÓN FINAL:")
    for par, df in dfs.items():
        print(f"{par}:")
        print(f"  Shape: {df.shape}")
        print(f"  Timestamps: {len(df)}")
        print(f"  NaN en close: {df['close'].isna().sum()}")
        print(f"  Primero: {df.index[0]}")
        print(f"  Último: {df.index[-1]}")
