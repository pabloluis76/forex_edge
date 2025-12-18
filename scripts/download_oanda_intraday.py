"""
Script para descargar datos intraday de alta frecuencia de OANDA
==================================================================

Descarga datos de M1 (1 minuto) y S5 (5 segundos) de OANDA.

ADVERTENCIA: Estos timeframes generan MUCHOS datos:
- M1: ~2.6 millones de velas por par (5 años)
- S5: ~31 millones de velas por par (5 años)

Se recomienda descargar solo el último año o menos para S5.
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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_oanda_intraday.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Configuración OANDA
OANDA_API_KEY = os.getenv('OANDA_API_KEY')
OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID')
OANDA_ENVIRONMENT = os.getenv('OANDA_ENVIRONMENT', 'practice')

# URLs de OANDA
if OANDA_ENVIRONMENT == 'practice':
    OANDA_API_URL = 'https://api-fxpractice.oanda.com'
else:
    OANDA_API_URL = 'https://api-fxtrade.oanda.com'

# Configuración de descarga
PAIRS = [
    'EUR_USD',
    'GBP_USD',
    'USD_JPY',
    'USD_CHF',
    'AUD_USD',
    'USD_CAD',
    'NZD_USD',
    'EUR_GBP',
    'EUR_JPY',
    'GBP_JPY',
]

# Timeframes intraday
TIMEFRAMES = {
    'M1': 'M1',    # 1 minuto
    'S5': 'S5',    # 5 segundos
}

# Años de histórico (menos para S5 por la cantidad de datos)
YEARS_CONFIG = {
    'M1': 5,   # 5 años para M1 (~2.6M velas por par)
    'S5': 1,   # 1 año para S5 (~6M velas por par) - MUY PESADO
}

# Directorio de salida
DATA_DIR = Path(__file__).parent.parent / 'datos' / 'ohlc'


class OandaIntradayDownloader:
    """Descargador de datos intraday de OANDA"""

    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.rate_limit_delay = 0.5

    def get_candles(
        self,
        instrument: str,
        granularity: str,
        from_time: datetime,
        to_time: datetime
    ) -> Optional[pd.DataFrame]:
        """Descarga velas de OANDA"""
        url = f'{self.api_url}/v3/instruments/{instrument}/candles'

        params = {
            'granularity': granularity,
            'from': from_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z'),
            'to': to_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z'),
            'price': 'MBA',
        }

        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()

            data = response.json()
            candles = data.get('candles', [])

            if not candles:
                return None

            records = []
            for candle in candles:
                if not candle.get('complete'):
                    continue

                record = {
                    'time': candle['time'],
                    'volume': candle['volume'],
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'bid_open': float(candle['bid']['o']),
                    'bid_high': float(candle['bid']['h']),
                    'bid_low': float(candle['bid']['l']),
                    'bid_close': float(candle['bid']['c']),
                    'ask_open': float(candle['ask']['o']),
                    'ask_high': float(candle['ask']['h']),
                    'ask_low': float(candle['ask']['l']),
                    'ask_close': float(candle['ask']['c']),
                }
                records.append(record)

            df = pd.DataFrame(records)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

            return df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limit alcanzado. Esperando 60 segundos...")
                time.sleep(60)
                return self.get_candles(instrument, granularity, from_time, to_time)
            else:
                logger.error(f"Error HTTP al descargar {instrument} {granularity}: {e}")
                return None

        except Exception as e:
            logger.error(f"Error al descargar {instrument} {granularity}: {e}")
            return None

    def download_instrument_timeframe(
        self,
        instrument: str,
        granularity: str,
        years: int = 1
    ) -> bool:
        """Descarga datos intraday para un par e intervalo específico"""
        logger.info(f"Descargando {instrument} {granularity} ({years} años)...")

        # Crear directorio
        instrument_dir = DATA_DIR / instrument
        instrument_dir.mkdir(parents=True, exist_ok=True)

        # Calcular fechas
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=years*365)

        # Chunks más pequeños para alta frecuencia
        chunk_sizes = {
            'S5': timedelta(days=3),    # 3 días de datos de 5 segundos (~52k velas)
            'M1': timedelta(days=15),   # 15 días de datos de 1 minuto (~21k velas)
        }
        chunk_size = chunk_sizes.get(granularity, timedelta(days=7))

        all_data = []
        current_start = start_time
        total_chunks = int((end_time - start_time) / chunk_size) + 1
        current_chunk = 0

        while current_start < end_time:
            current_chunk += 1
            current_end = min(current_start + chunk_size, end_time)

            df = self.get_candles(
                instrument=instrument,
                granularity=granularity,
                from_time=current_start,
                to_time=current_end
            )

            if df is not None and not df.empty:
                all_data.append(df)
                logger.info(
                    f"  [{current_chunk}/{total_chunks}] {instrument} {granularity}: "
                    f"{current_start.date()} a {current_end.date()} "
                    f"({len(df)} velas)"
                )

            current_start = current_end

        if not all_data:
            logger.warning(f"No se descargaron datos para {instrument} {granularity}")
            return False

        # Combinar datos
        final_df = pd.concat(all_data)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        final_df.sort_index(inplace=True)

        # Guardar en CSV
        output_file = instrument_dir / f"{granularity}.csv"
        final_df.to_csv(output_file)

        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(
            f"✓ {instrument} {granularity}: {len(final_df):,} velas guardadas "
            f"({final_df.index[0].date()} a {final_df.index[-1].date()}) "
            f"[{size_mb:.1f} MB]"
        )

        return True

    def download_all(self, pairs: List[str], timeframes: Dict[str, str]):
        """Descarga todos los pares y timeframes intraday"""
        total = len(pairs) * len(timeframes)
        current = 0

        logger.info(f"\n{'='*60}")
        logger.info(f"DESCARGA DE DATOS INTRADAY DE ALTA FRECUENCIA")
        logger.info(f"{'='*60}")
        logger.info(f"Total de combinaciones: {total}")
        logger.info(f"Pares: {', '.join(pairs)}")
        logger.info(f"Timeframes: {', '.join(timeframes.keys())}")
        logger.info(f"\n⚠️  ADVERTENCIA: Esto generará MUCHOS datos")
        logger.info(f"    M1 (1 min): ~2.6M velas × {len(pairs)} pares × 5 años")
        logger.info(f"    S5 (5 seg): ~6M velas × {len(pairs)} pares × 1 año\n")

        start_time = time.time()

        for pair in pairs:
            for tf_name, tf_code in timeframes.items():
                current += 1
                years = YEARS_CONFIG.get(tf_name, 1)

                logger.info(f"\n[{current}/{total}] {pair} - {tf_name} ({years} años)")

                success = self.download_instrument_timeframe(
                    instrument=pair,
                    granularity=tf_code,
                    years=years
                )

                if not success:
                    logger.error(f"✗ Falló descarga de {pair} {tf_name}")

        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Descarga completada en {elapsed/60:.1f} minutos")
        logger.info(f"Datos guardados en: {DATA_DIR}")
        logger.info(f"{'='*60}")


def main():
    """Función principal"""
    if not OANDA_API_KEY:
        logger.error("ERROR: OANDA_API_KEY no encontrado en .env")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloader = OandaIntradayDownloader(
        api_key=OANDA_API_KEY,
        api_url=OANDA_API_URL
    )

    # Advertencia final
    logger.info("\n" + "="*60)
    logger.info("ADVERTENCIA FINAL")
    logger.info("="*60)
    logger.info("Este proceso descargará datos de ALTA FRECUENCIA.")
    logger.info("Puede tomar VARIAS HORAS y generar GIGABYTES de datos.")
    logger.info("")
    logger.info("Configuración:")
    logger.info(f"  - M1 (1 minuto): {YEARS_CONFIG['M1']} años × {len(PAIRS)} pares")
    logger.info(f"  - S5 (5 segundos): {YEARS_CONFIG['S5']} año × {len(PAIRS)} pares")
    logger.info("")
    logger.info("Iniciando descarga en 5 segundos...")
    logger.info("="*60 + "\n")

    time.sleep(5)

    downloader.download_all(
        pairs=PAIRS,
        timeframes=TIMEFRAMES
    )


if __name__ == '__main__':
    main()
