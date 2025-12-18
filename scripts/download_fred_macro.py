"""
Script para descargar datos macroeconómicos de FRED
====================================================

Descarga indicadores económicos relevantes para trading de forex desde
la API de Federal Reserve Economic Data (FRED).

Datos descargados:
- Tasas de interés (Fed Funds Rate, etc.)
- Inflación (CPI, PCE)
- PIB y crecimiento económico
- Desempleo
- Índice del dólar (DXY, Trade Weighted Dollar Index)
- Volatilidad (VIX)
- Datos de otros países (ECB rates, BOJ rates, etc.)

REGLA CRÍTICA: Solo datos que existían EN ESE MOMENTO.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import time
import pandas as pd
from dotenv import load_dotenv
import requests
from typing import List, Dict, Optional
import logging
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_fred.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Configuración FRED
FRED_API_KEY = os.getenv('FRED_API_KEY')
FRED_API_URL = 'https://api.stlouisfed.org/fred'

# Directorio de salida
DATA_DIR = Path(__file__).parent.parent / 'datos' / 'macro'

# Series económicas a descargar
FRED_SERIES = {
    # Tasas de interés USA
    'DFF': {
        'name': 'Fed Funds Rate',
        'description': 'Tasa de fondos federales efectiva',
        'frequency': 'daily',
        'category': 'interest_rates'
    },
    'DTB3': {
        'name': 'US Treasury 3-Month',
        'description': 'Tasa del Tesoro USA a 3 meses',
        'frequency': 'daily',
        'category': 'interest_rates'
    },
    'DGS2': {
        'name': 'US Treasury 2-Year',
        'description': 'Tasa del Tesoro USA a 2 años',
        'frequency': 'daily',
        'category': 'interest_rates'
    },
    'DGS10': {
        'name': 'US Treasury 10-Year',
        'description': 'Tasa del Tesoro USA a 10 años',
        'frequency': 'daily',
        'category': 'interest_rates'
    },

    # Inflación USA
    'CPIAUCSL': {
        'name': 'CPI All Items',
        'description': 'Índice de precios al consumidor (todos los items)',
        'frequency': 'monthly',
        'category': 'inflation'
    },
    'CPILFESL': {
        'name': 'CPI Core',
        'description': 'CPI excluyendo alimentos y energía',
        'frequency': 'monthly',
        'category': 'inflation'
    },
    'PCEPI': {
        'name': 'PCE Price Index',
        'description': 'Índice de precios PCE',
        'frequency': 'monthly',
        'category': 'inflation'
    },
    'PCEPILFE': {
        'name': 'PCE Core',
        'description': 'PCE excluyendo alimentos y energía',
        'frequency': 'monthly',
        'category': 'inflation'
    },

    # PIB y crecimiento USA
    'GDP': {
        'name': 'GDP',
        'description': 'Producto Interno Bruto',
        'frequency': 'quarterly',
        'category': 'gdp'
    },
    'A191RL1Q225SBEA': {
        'name': 'Real GDP Growth',
        'description': 'Crecimiento del PIB real (% anual)',
        'frequency': 'quarterly',
        'category': 'gdp'
    },

    # Empleo USA
    'UNRATE': {
        'name': 'Unemployment Rate',
        'description': 'Tasa de desempleo',
        'frequency': 'monthly',
        'category': 'employment'
    },
    'PAYEMS': {
        'name': 'Nonfarm Payrolls',
        'description': 'Nóminas no agrícolas',
        'frequency': 'monthly',
        'category': 'employment'
    },
    'ICSA': {
        'name': 'Initial Jobless Claims',
        'description': 'Solicitudes iniciales de desempleo',
        'frequency': 'weekly',
        'category': 'employment'
    },

    # Índice del dólar
    'DTWEXBGS': {
        'name': 'Trade Weighted Dollar Index (Broad)',
        'description': 'Índice del dólar ponderado por comercio (amplio)',
        'frequency': 'daily',
        'category': 'dollar_index'
    },
    'DTWEXM': {
        'name': 'Trade Weighted Dollar Index (Major)',
        'description': 'Índice del dólar vs monedas principales',
        'frequency': 'daily',
        'category': 'dollar_index'
    },

    # Volatilidad y riesgo
    'VIXCLS': {
        'name': 'VIX',
        'description': 'Índice de volatilidad CBOE',
        'frequency': 'daily',
        'category': 'volatility'
    },

    # Tasas de interés internacionales
    'ECBDFR': {
        'name': 'ECB Deposit Facility Rate',
        'description': 'Tasa de facilidad de depósito del BCE',
        'frequency': 'daily',
        'category': 'international_rates'
    },
    'IRSTCB01GBM156N': {
        'name': 'UK Official Bank Rate',
        'description': 'Tasa oficial del Banco de Inglaterra',
        'frequency': 'monthly',
        'category': 'international_rates'
    },
    'IRSTCB01JPM156N': {
        'name': 'Japan Policy Rate',
        'description': 'Tasa de política del Banco de Japón',
        'frequency': 'monthly',
        'category': 'international_rates'
    },
    'IRSTCB01CHM156N': {
        'name': 'Switzerland Policy Rate',
        'description': 'Tasa del Banco Nacional Suizo',
        'frequency': 'monthly',
        'category': 'international_rates'
    },
    'IRSTCB01AUM156N': {
        'name': 'Australia Cash Rate',
        'description': 'Tasa del Banco de la Reserva de Australia',
        'frequency': 'monthly',
        'category': 'international_rates'
    },
    'IRSTCB01CAM156N': {
        'name': 'Canada Overnight Rate',
        'description': 'Tasa del Banco de Canadá',
        'frequency': 'monthly',
        'category': 'international_rates'
    },
    'IRSTCB01NZM156N': {
        'name': 'New Zealand OCR',
        'description': 'Tasa oficial de Nueva Zelanda',
        'frequency': 'monthly',
        'category': 'international_rates'
    },

    # PIB Internacional
    'CLVMNACSCAB1GQUK': {
        'name': 'UK Real GDP',
        'description': 'PIB real del Reino Unido',
        'frequency': 'quarterly',
        'category': 'international_gdp'
    },
    'CLVMEURSCAB1GQEA19': {
        'name': 'Euro Area Real GDP',
        'description': 'PIB real de la Eurozona',
        'frequency': 'quarterly',
        'category': 'international_gdp'
    },
    'JPNRGDPEXP': {
        'name': 'Japan Real GDP',
        'description': 'PIB real de Japón',
        'frequency': 'quarterly',
        'category': 'international_gdp'
    },
    'AUSRGDPEXP': {
        'name': 'Australia Real GDP',
        'description': 'PIB real de Australia',
        'frequency': 'quarterly',
        'category': 'international_gdp'
    },
    'CANRGDPEXP': {
        'name': 'Canada Real GDP',
        'description': 'PIB real de Canadá',
        'frequency': 'quarterly',
        'category': 'international_gdp'
    },

    # Inflación Internacional
    'GBRCPIALLMINMEI': {
        'name': 'UK CPI',
        'description': 'IPC del Reino Unido',
        'frequency': 'monthly',
        'category': 'international_inflation'
    },
    'CP0000EZ19M086NEST': {
        'name': 'Euro Area CPI',
        'description': 'IPC de la Eurozona',
        'frequency': 'monthly',
        'category': 'international_inflation'
    },
    'JPNCPIALLMINMEI': {
        'name': 'Japan CPI',
        'description': 'IPC de Japón',
        'frequency': 'monthly',
        'category': 'international_inflation'
    },

    # Desempleo Internacional
    'LRHUTTTTGBM156S': {
        'name': 'UK Unemployment',
        'description': 'Tasa de desempleo del Reino Unido',
        'frequency': 'monthly',
        'category': 'international_employment'
    },
    'LRHUTTTTEZM156S': {
        'name': 'Euro Area Unemployment',
        'description': 'Tasa de desempleo de la Eurozona',
        'frequency': 'monthly',
        'category': 'international_employment'
    },
    'LRHUTTTTJPM156S': {
        'name': 'Japan Unemployment',
        'description': 'Tasa de desempleo de Japón',
        'frequency': 'monthly',
        'category': 'international_employment'
    },

    # Commodities (relevantes para forex)
    'DCOILWTICO': {
        'name': 'WTI Crude Oil',
        'description': 'Precio del petróleo WTI',
        'frequency': 'daily',
        'category': 'commodities'
    },
    'GOLDAMGBD228NLBM': {
        'name': 'Gold Price',
        'description': 'Precio del oro',
        'frequency': 'daily',
        'category': 'commodities'
    },

    # Confianza del consumidor
    'UMCSENT': {
        'name': 'Consumer Sentiment',
        'description': 'Índice de confianza del consumidor (U. Michigan)',
        'frequency': 'monthly',
        'category': 'sentiment'
    },
}


class FREDDownloader:
    """Descargador de datos de FRED"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limit_delay = 0.2  # FRED permite más requests que OANDA

    def get_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Descarga una serie de FRED

        Args:
            series_id: ID de la serie (ej: 'DFF')
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)

        Returns:
            DataFrame con los datos o None si hay error
        """
        url = f'{FRED_API_URL}/series/observations'

        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
        }

        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            observations = data.get('observations', [])

            if not observations:
                return None

            # Convertir a DataFrame
            df = pd.DataFrame(observations)

            # Filtrar valores "." (missing)
            df = df[df['value'] != '.']

            if df.empty:
                return None

            # Convertir tipos
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'])

            # Seleccionar columnas relevantes
            df = df[['date', 'value']]
            df.set_index('date', inplace=True)
            df.columns = [series_id]

            return df

        except requests.exceptions.HTTPError as e:
            logger.error(f"Error HTTP al descargar {series_id}: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Respuesta: {e.response.text}")
            return None

        except Exception as e:
            logger.error(f"Error al descargar {series_id}: {e}")
            return None

    def download_all(
        self,
        series_dict: Dict[str, Dict],
        years: int = 10
    ):
        """
        Descarga todas las series especificadas

        Args:
            series_dict: Diccionario con las series a descargar
            years: Años de histórico
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Descargando {len(series_dict)} series de FRED")
        logger.info(f"Histórico: {years} años")
        logger.info(f"{'='*60}\n")

        # Calcular fechas
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Agrupar por categoría
        by_category = {}
        for series_id, info in series_dict.items():
            category = info['category']
            if category not in by_category:
                by_category[category] = {}
            by_category[category][series_id] = info

        start_time = time.time()
        downloaded = 0
        failed = 0

        # Descargar por categoría
        for category, series in by_category.items():
            logger.info(f"\nCategoría: {category.upper()}")
            logger.info("-" * 40)

            category_data = {}

            for series_id, info in series.items():
                logger.info(f"  Descargando {series_id} ({info['name']})...")

                df = self.get_series(
                    series_id=series_id,
                    start_date=start_str,
                    end_date=end_str
                )

                if df is not None and not df.empty:
                    category_data[series_id] = df
                    logger.info(
                        f"    ✓ {len(df)} observaciones "
                        f"({df.index[0].date()} a {df.index[-1].date()})"
                    )
                    downloaded += 1
                else:
                    logger.warning(f"    ✗ No se obtuvieron datos")
                    failed += 1

            # Guardar datos de la categoría
            if category_data:
                # Combinar todas las series de la categoría
                combined_df = pd.concat(category_data.values(), axis=1)

                # Guardar CSV
                output_file = DATA_DIR / f"{category}.csv"
                combined_df.to_csv(output_file)
                logger.info(f"  → Guardado en: {output_file}")

        # Guardar metadata
        metadata_file = DATA_DIR / 'series_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(series_dict, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time

        logger.info(f"\n{'='*60}")
        logger.info(f"Descarga completada en {elapsed:.1f} segundos")
        logger.info(f"Series descargadas: {downloaded}")
        logger.info(f"Series fallidas: {failed}")
        logger.info(f"Datos guardados en: {DATA_DIR}")
        logger.info(f"Metadata guardada en: {metadata_file}")
        logger.info(f"{'='*60}")


def main():
    """Función principal"""
    # Validar credenciales
    if not FRED_API_KEY:
        logger.error("ERROR: FRED_API_KEY no encontrado en .env")
        logger.error("\nPara obtener una API key GRATIS:")
        logger.error("1. Visita: https://fred.stlouisfed.org/")
        logger.error("2. Crea una cuenta (gratis, toma 1 minuto)")
        logger.error("3. Ve a tu perfil > API Keys")
        logger.error("4. Copia tu API key y agrégala al archivo .env")
        sys.exit(1)

    # Crear directorio de datos
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Inicializar descargador
    downloader = FREDDownloader(api_key=FRED_API_KEY)

    # Descargar datos
    downloader.download_all(
        series_dict=FRED_SERIES,
        years=10
    )


if __name__ == '__main__':
    main()
