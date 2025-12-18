"""
División Temporal de Datos

DIVISIÓN DE DATOS:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  PERÍODO TOTAL: 2019-01-01 a 2023-12-31 (5 años)                           │
│                                                                             │
│  ├──────────────────────────────────┤├────────────┤├────────────┤          │
│  │                                  ││            ││            │          │
│  │         TRAIN (60%)              ││ VALIDATION ││    TEST    │          │
│  │                                  ││   (20%)    ││   (20%)    │          │
│  │      2019-01 a 2021-12           ││ 2022-01 a  ││ 2023-01 a  │          │
│  │         (3 años)                 ││ 2022-12    ││ 2023-12    │          │
│  │                                  ││ (1 año)    ││ (1 año)    │          │
│  │  ~210,000 velas por par          ││~70,000     ││~70,000     │          │
│  │                                  ││            ││            │          │
│  └──────────────────────────────────┘└────────────┘└────────────┘          │
│                                                                             │
│  REGLA SAGRADA:                                                             │
│  El set TEST solo se toca UNA VEZ, al final de todo el proceso.           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Author: Sistema de Edge-Finding Forex
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class DivisionTemporalDatos:
    """
    División temporal de datos para backtest.

    TRAIN → VALIDATION → TEST

    REGLA SAGRADA:
    El set TEST solo se toca UNA VEZ al final, para obtener
    métricas finales no sesgadas.
    """

    def __init__(
        self,
        fecha_inicio: str = '2019-01-01',
        fecha_fin: str = '2023-12-31',
        pct_train: float = 0.60,
        pct_validation: float = 0.20,
        pct_test: float = 0.20,
        verbose: bool = True
    ):
        """
        Inicializa división temporal de datos.

        Parameters:
        -----------
        fecha_inicio : str
            Fecha de inicio del período total (formato: 'YYYY-MM-DD')
        fecha_fin : str
            Fecha de fin del período total (formato: 'YYYY-MM-DD')
        pct_train : float
            Porcentaje para TRAIN (default: 0.60 = 60%)
        pct_validation : float
            Porcentaje para VALIDATION (default: 0.20 = 20%)
        pct_test : float
            Porcentaje para TEST (default: 0.20 = 20%)
        verbose : bool
            Imprimir información detallada
        """
        self.fecha_inicio = pd.Timestamp(fecha_inicio)
        self.fecha_fin = pd.Timestamp(fecha_fin)
        self.pct_train = pct_train
        self.pct_validation = pct_validation
        self.pct_test = pct_test
        self.verbose = verbose

        # Validar porcentajes
        total_pct = pct_train + pct_validation + pct_test
        assert abs(total_pct - 1.0) < 0.01, f"Los porcentajes deben sumar 1.0, suma actual: {total_pct}"

        # Calcular fechas de división
        self._calcular_fechas_division()

        if self.verbose:
            print("="*80)
            print("DIVISIÓN TEMPORAL DE DATOS")
            print("="*80)
            self._mostrar_division()

    def _calcular_fechas_division(self):
        """Calcula fechas de división entre TRAIN, VALIDATION y TEST."""
        # Duración total
        duracion_total = self.fecha_fin - self.fecha_inicio

        # Duraciones de cada set
        duracion_train = duracion_total * self.pct_train
        duracion_validation = duracion_total * self.pct_validation

        # Fechas de corte
        self.fecha_fin_train = self.fecha_inicio + duracion_train
        self.fecha_inicio_validation = self.fecha_fin_train
        self.fecha_fin_validation = self.fecha_inicio_validation + duracion_validation
        self.fecha_inicio_test = self.fecha_fin_validation
        self.fecha_fin_test = self.fecha_fin

        # Ajustar a inicio/fin de mes para fechas más limpias
        self.fecha_fin_train = pd.Timestamp(
            year=self.fecha_fin_train.year,
            month=self.fecha_fin_train.month,
            day=1
        ) - pd.Timedelta(days=1)  # Último día del mes anterior

        self.fecha_inicio_validation = self.fecha_fin_train + pd.Timedelta(days=1)

        self.fecha_fin_validation = pd.Timestamp(
            year=self.fecha_fin_validation.year,
            month=self.fecha_fin_validation.month,
            day=1
        ) - pd.Timedelta(days=1)

        self.fecha_inicio_test = self.fecha_fin_validation + pd.Timedelta(days=1)

    def _mostrar_division(self):
        """Muestra división temporal de datos."""
        print(f"\nPERÍODO TOTAL: {self.fecha_inicio.date()} a {self.fecha_fin.date()}")
        print(f"Duración: {(self.fecha_fin - self.fecha_inicio).days} días")

        print(f"\n{'='*80}")
        print(f"DIVISIÓN:")
        print(f"{'='*80}")

        # TRAIN
        duracion_train = (self.fecha_fin_train - self.fecha_inicio).days
        print(f"\n1. TRAIN ({self.pct_train*100:.0f}%):")
        print(f"   ─────────────")
        print(f"   Inicio:   {self.fecha_inicio.date()}")
        print(f"   Fin:      {self.fecha_fin_train.date()}")
        print(f"   Duración: {duracion_train} días (~{duracion_train/365:.1f} años)")
        print(f"   Uso:      Entrenamiento de modelos, selección de features")

        # VALIDATION
        duracion_validation = (self.fecha_fin_validation - self.fecha_inicio_validation).days
        print(f"\n2. VALIDATION ({self.pct_validation*100:.0f}%):")
        print(f"   ──────────────────")
        print(f"   Inicio:   {self.fecha_inicio_validation.date()}")
        print(f"   Fin:      {self.fecha_fin_validation.date()}")
        print(f"   Duración: {duracion_validation} días (~{duracion_validation/365:.1f} años)")
        print(f"   Uso:      Tuning de hiperparámetros, validación de estrategia")

        # TEST
        duracion_test = (self.fecha_fin_test - self.fecha_inicio_test).days
        print(f"\n3. TEST ({self.pct_test*100:.0f}%):")
        print(f"   ────────")
        print(f"   Inicio:   {self.fecha_inicio_test.date()}")
        print(f"   Fin:      {self.fecha_fin_test.date()}")
        print(f"   Duración: {duracion_test} días (~{duracion_test/365:.1f} años)")
        print(f"   Uso:      ⚠ SOLO UNA VEZ AL FINAL (métricas finales no sesgadas)")

        print(f"\n{'='*80}")
        print(f"⚠ REGLA SAGRADA:")
        print(f"{'='*80}")
        print(f"El set TEST solo se toca UNA VEZ, al final de todo el proceso.")
        print(f"Esto garantiza métricas finales NO SESGADAS.")
        print(f"{'='*80}")

    def dividir_dataframe(
        self,
        df: pd.DataFrame,
        columna_timestamp: str = 'timestamp'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide un DataFrame en TRAIN, VALIDATION y TEST.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con datos
        columna_timestamp : str
            Nombre de la columna de timestamp (default: 'timestamp')

        Returns:
        --------
        df_train : pd.DataFrame
            Set de entrenamiento
        df_validation : pd.DataFrame
            Set de validación
        df_test : pd.DataFrame
            Set de prueba
        """
        # Asegurar que timestamp es datetime
        if df[columna_timestamp].dtype != 'datetime64[ns]':
            df[columna_timestamp] = pd.to_datetime(df[columna_timestamp])

        # Dividir por fechas
        mask_train = (
            (df[columna_timestamp] >= self.fecha_inicio) &
            (df[columna_timestamp] <= self.fecha_fin_train)
        )

        mask_validation = (
            (df[columna_timestamp] >= self.fecha_inicio_validation) &
            (df[columna_timestamp] <= self.fecha_fin_validation)
        )

        mask_test = (
            (df[columna_timestamp] >= self.fecha_inicio_test) &
            (df[columna_timestamp] <= self.fecha_fin_test)
        )

        df_train = df[mask_train].copy()
        df_validation = df[mask_validation].copy()
        df_test = df[mask_test].copy()

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"DIVISIÓN APLICADA")
            print(f"{'='*80}")
            print(f"Total filas:      {len(df):,}")
            print(f"TRAIN:            {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
            print(f"VALIDATION:       {len(df_validation):,} ({len(df_validation)/len(df)*100:.1f}%)")
            print(f"TEST:             {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")
            print(f"{'='*80}")

        return df_train, df_validation, df_test

    def dividir_por_par(
        self,
        df: pd.DataFrame,
        columna_timestamp: str = 'timestamp',
        columna_par: str = 'pair'
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Divide datos separadamente para cada par.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con datos de múltiples pares
        columna_timestamp : str
            Nombre de la columna de timestamp
        columna_par : str
            Nombre de la columna de par

        Returns:
        --------
        division_por_par : Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
            Diccionario con división por par:
            {'EUR_USD': (df_train, df_validation, df_test), ...}
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"DIVISIÓN POR PAR")
            print(f"{'='*80}")

        division_por_par = {}
        pares_unicos = df[columna_par].unique()

        for par in pares_unicos:
            df_par = df[df[columna_par] == par].copy()

            df_train, df_validation, df_test = self.dividir_dataframe(
                df=df_par,
                columna_timestamp=columna_timestamp
            )

            division_por_par[par] = (df_train, df_validation, df_test)

            if self.verbose:
                print(f"\n{par}:")
                print(f"  TRAIN:      {len(df_train):,} velas")
                print(f"  VALIDATION: {len(df_validation):,} velas")
                print(f"  TEST:       {len(df_test):,} velas")

        return division_por_par

    def guardar_division(
        self,
        df_train: pd.DataFrame,
        df_validation: pd.DataFrame,
        df_test: pd.DataFrame,
        directorio_salida: Optional[str] = None,
        prefijo: str = 'data'
    ):
        """
        Guarda división en archivos CSV separados.

        Parameters:
        -----------
        df_train : pd.DataFrame
            Set de entrenamiento
        df_validation : pd.DataFrame
            Set de validación
        df_test : pd.DataFrame
            Set de prueba
        directorio_salida : str, optional
            Directorio de salida (default: backtest/)
        prefijo : str
            Prefijo para nombres de archivo (default: 'data')
        """
        if directorio_salida is None:
            directorio_salida = Path(__file__).parent
        else:
            directorio_salida = Path(directorio_salida)

        directorio_salida.mkdir(parents=True, exist_ok=True)

        # Guardar CSVs
        ruta_train = directorio_salida / f'{prefijo}_train.csv'
        ruta_validation = directorio_salida / f'{prefijo}_validation.csv'
        ruta_test = directorio_salida / f'{prefijo}_test.csv'

        df_train.to_csv(ruta_train, index=False)
        df_validation.to_csv(ruta_validation, index=False)
        df_test.to_csv(ruta_test, index=False)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"ARCHIVOS GUARDADOS")
            print(f"{'='*80}")
            print(f"TRAIN:      {ruta_train}")
            print(f"            {ruta_train.stat().st_size / (1024*1024):.1f} MB")
            print(f"VALIDATION: {ruta_validation}")
            print(f"            {ruta_validation.stat().st_size / (1024*1024):.1f} MB")
            print(f"TEST:       {ruta_test}")
            print(f"            {ruta_test.stat().st_size / (1024*1024):.1f} MB")
            print(f"{'='*80}")

    def obtener_info_division(self) -> Dict:
        """
        Obtiene información completa de la división.

        Returns:
        --------
        info : Dict
            Diccionario con información de división
        """
        return {
            'periodo_total': {
                'inicio': self.fecha_inicio.isoformat(),
                'fin': self.fecha_fin.isoformat(),
                'dias': (self.fecha_fin - self.fecha_inicio).days
            },
            'train': {
                'inicio': self.fecha_inicio.isoformat(),
                'fin': self.fecha_fin_train.isoformat(),
                'dias': (self.fecha_fin_train - self.fecha_inicio).days,
                'porcentaje': self.pct_train
            },
            'validation': {
                'inicio': self.fecha_inicio_validation.isoformat(),
                'fin': self.fecha_fin_validation.isoformat(),
                'dias': (self.fecha_fin_validation - self.fecha_inicio_validation).days,
                'porcentaje': self.pct_validation
            },
            'test': {
                'inicio': self.fecha_inicio_test.isoformat(),
                'fin': self.fecha_fin_test.isoformat(),
                'dias': (self.fecha_fin_test - self.fecha_inicio_test).days,
                'porcentaje': self.pct_test
            }
        }


def ejemplo_uso():
    """
    Ejemplo de uso de DivisionTemporalDatos.
    """
    print("="*80)
    print("EJEMPLO: División Temporal de Datos")
    print("="*80)

    # Inicializar división
    division = DivisionTemporalDatos(
        fecha_inicio='2019-01-01',
        fecha_fin='2023-12-31',
        pct_train=0.60,
        pct_validation=0.20,
        pct_test=0.20,
        verbose=True
    )

    # Simular DataFrame de ejemplo
    print(f"\n{'='*80}")
    print(f"EJEMPLO CON DATOS SIMULADOS")
    print(f"{'='*80}")

    # Crear datos de ejemplo
    fechas = pd.date_range(start='2019-01-01', end='2023-12-31', freq='15min')
    pares = ['EUR_USD', 'GBP_USD', 'USD_JPY']

    registros = []
    for fecha in fechas[:1000]:  # Solo 1000 registros para ejemplo
        for par in pares:
            registros.append({
                'timestamp': fecha,
                'pair': par,
                'close': 1.1000 + np.random.randn() * 0.001
            })

    df_ejemplo = pd.DataFrame(registros)

    print(f"\nDataFrame de ejemplo creado:")
    print(f"  Filas: {len(df_ejemplo):,}")
    print(f"  Pares: {df_ejemplo['pair'].unique().tolist()}")

    # Dividir datos
    df_train, df_validation, df_test = division.dividir_dataframe(df_ejemplo)

    # División por par
    division_por_par = division.dividir_por_par(df_ejemplo)

    # Obtener info
    info = division.obtener_info_division()

    print(f"\n{'='*80}")
    print(f"INFORMACIÓN DE DIVISIÓN")
    print(f"{'='*80}")
    import json
    print(json.dumps(info, indent=2))

    # Guardar (comentado para no crear archivos en ejemplo)
    # division.guardar_division(df_train, df_validation, df_test, prefijo='ejemplo')

    print(f"\n{'='*80}")
    print(f"⚠ RECORDATORIO IMPORTANTE")
    print(f"{'='*80}")
    print(f"El set TEST solo debe usarse UNA VEZ, al final.")
    print(f"Durante desarrollo:")
    print(f"  1. Entrenar modelos en TRAIN")
    print(f"  2. Validar y ajustar en VALIDATION")
    print(f"  3. Repetir pasos 1-2 hasta satisfacer")
    print(f"  4. SOLO AL FINAL: evaluar en TEST para métricas finales")
    print(f"{'='*80}")


if __name__ == '__main__':
    ejemplo_uso()
