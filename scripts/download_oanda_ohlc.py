"""
Script para descargar datos OHLC de OANDA
==========================================

Descarga datos históricos de múltiples pares y timeframes de OANDA.
También descarga información de instrumentos (spreads, comisiones, etc.)

REGLA CRÍTICA: Solo datos que existían EN ESE MOMENTO.
               Nunca usar datos del futuro.

Características:
- Múltiples timeframes: M15, H1, H4, D1, W1
- Múltiples pares: 6-12 pares principales
- Histórico: 5+ años mínimo
- Volumen: Tick volume incluido
- Información de pares: spreads, comisiones, pip location
- Organización: datos/ohlc/{par}/{timeframe}.csv
"""

import os
import sys
import json
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
        logging.FileHandler('download_oanda.log'),
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
    'EUR_USD',  # Euro / US Dollar
    'GBP_USD',  # British Pound / US Dollar
    'USD_JPY',  # US Dollar / Japanese Yen
    'USD_CHF',  # US Dollar / Swiss Franc
    'AUD_USD',  # Australian Dollar / US Dollar
    'USD_CAD',  # US Dollar / Canadian Dollar
    'NZD_USD',  # New Zealand Dollar / US Dollar
    'EUR_GBP',  # Euro / British Pound
    'EUR_JPY',  # Euro / Japanese Yen
    'GBP_JPY',  # British Pound / Japanese Yen
]

TIMEFRAMES = {
    'M15': 'M15',   # 15 minutos
    'H1': 'H1',     # 1 hora
    'H4': 'H4',     # 4 horas
    'D': 'D',       # Diario
    'W': 'W',       # Semanal
}

# Número de velas por request (máximo de OANDA es 5000)
MAX_CANDLES = 5000

# Directorio de salida
DATA_DIR = Path(__file__).parent.parent / 'datos' / 'ohlc'


class OandaDownloader:
    """Descargador de datos históricos de OANDA"""

    def __init__(self, api_key: str, account_id: str, api_url: str):
        self.api_key = api_key
        self.account_id = account_id
        self.api_url = api_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.rate_limit_delay = 0.5  # 0.5 segundos entre requests

    def get_instrument_info(self, instruments: List[str]) -> Dict:
        """
        Obtiene información detallada de los instrumentos (spreads, comisiones, etc.)

        Args:
            instruments: Lista de pares (ej: ['EUR_USD', 'GBP_USD'])

        Returns:
            Diccionario con información de cada par
        """
        logger.info("Descargando información de instrumentos...")

        instruments_info = {}

        for instrument in instruments:
            try:
                time.sleep(self.rate_limit_delay)

                # Endpoint para información del instrumento
                url = f'{self.api_url}/v3/accounts/{self.account_id}/instruments'
                params = {'instruments': instrument}

                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()

                data = response.json()

                if 'instruments' in data and len(data['instruments']) > 0:
                    inst_data = data['instruments'][0]

                    # Obtener también el pricing actual para spreads
                    pricing_url = f'{self.api_url}/v3/accounts/{self.account_id}/pricing'
                    pricing_params = {'instruments': instrument}

                    time.sleep(self.rate_limit_delay)
                    pricing_response = requests.get(
                        pricing_url,
                        headers=self.headers,
                        params=pricing_params
                    )
                    pricing_response.raise_for_status()
                    pricing_data = pricing_response.json()

                    current_spread = None
                    if 'prices' in pricing_data and len(pricing_data['prices']) > 0:
                        price = pricing_data['prices'][0]
                        if 'asks' in price and 'bids' in price:
                            ask = float(price['asks'][0]['price'])
                            bid = float(price['bids'][0]['price'])
                            current_spread = ask - bid

                    instruments_info[instrument] = {
                        'name': inst_data.get('name'),
                        'type': inst_data.get('type'),
                        'displayName': inst_data.get('displayName'),
                        'pipLocation': int(inst_data.get('pipLocation', -4)),
                        'displayPrecision': int(inst_data.get('displayPrecision', 5)),
                        'tradeUnitsPrecision': int(inst_data.get('tradeUnitsPrecision', 0)),
                        'minimumTradeSize': float(inst_data.get('minimumTradeSize', 1)),
                        'maximumTrailingStopDistance': float(inst_data.get('maximumTrailingStopDistance', 100)),
                        'minimumTrailingStopDistance': float(inst_data.get('minimumTrailingStopDistance', 0.00050)),
                        'maximumPositionSize': float(inst_data.get('maximumPositionSize', 0)),
                        'maximumOrderUnits': float(inst_data.get('maximumOrderUnits', 100000000)),
                        'marginRate': float(inst_data.get('marginRate', 0.02)),
                        'commission': inst_data.get('commission', {
                            'commission': 0,
                            'unitsTraded': 0,
                            'minimumCommission': 0
                        }),
                        'financing': {
                            'longRate': float(inst_data.get('financing', {}).get('longRate', 0)),
                            'shortRate': float(inst_data.get('financing', {}).get('shortRate', 0)),
                        },
                        'current_spread_pips': round(current_spread / (10 ** inst_data.get('pipLocation', -4)), 2) if current_spread else None,
                        'current_spread_raw': current_spread,
                        'timestamp': datetime.utcnow().isoformat()
                    }

                    logger.info(f"  ✓ {instrument}: spread={instruments_info[instrument]['current_spread_pips']} pips, margin={instruments_info[instrument]['marginRate']*100}%")

            except Exception as e:
                logger.error(f"Error obteniendo info de {instrument}: {e}")
                instruments_info[instrument] = {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }

        return instruments_info

    def get_candles(
        self,
        instrument: str,
        granularity: str,
        from_time: datetime,
        to_time: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Descarga velas de OANDA

        Args:
            instrument: Par de divisas (ej: 'EUR_USD')
            granularity: Timeframe (ej: 'H1')
            from_time: Fecha de inicio
            to_time: Fecha de fin

        Returns:
            DataFrame con los datos o None si hay error
        """
        url = f'{self.api_url}/v3/instruments/{instrument}/candles'

        params = {
            'granularity': granularity,
            'from': from_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z'),
            'to': to_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z'),
            'price': 'MBA',  # Mid, Bid, Ask
        }

        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()

            data = response.json()
            candles = data.get('candles', [])

            if not candles:
                return None

            # Procesar datos
            records = []
            for candle in candles:
                if not candle.get('complete'):
                    continue  # Solo velas completas

                record = {
                    'time': candle['time'],
                    'volume': candle['volume'],
                    # Mid prices
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    # Bid prices
                    'bid_open': float(candle['bid']['o']),
                    'bid_high': float(candle['bid']['h']),
                    'bid_low': float(candle['bid']['l']),
                    'bid_close': float(candle['bid']['c']),
                    # Ask prices
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
        years: int = 5
    ) -> bool:
        """
        Descarga datos históricos para un par e intervalo específico

        Args:
            instrument: Par de divisas
            granularity: Timeframe
            years: Años de histórico a descargar

        Returns:
            True si la descarga fue exitosa
        """
        logger.info(f"Descargando {instrument} {granularity} ({years} años)...")

        # Crear directorio para el instrumento
        instrument_dir = DATA_DIR / instrument
        instrument_dir.mkdir(parents=True, exist_ok=True)

        # Calcular fechas
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=years*365)

        # Determinar el tamaño del chunk basado en el timeframe
        chunk_sizes = {
            'M15': timedelta(days=30),    # ~3000 velas/mes
            'H1': timedelta(days=120),     # ~3000 velas/4 meses
            'H4': timedelta(days=500),     # ~3000 velas/~16 meses
            'D': timedelta(days=3000),     # ~3000 velas/~8 años
            'W': timedelta(days=15000),    # ~2000 velas/~40 años
        }
        chunk_size = chunk_sizes.get(granularity, timedelta(days=30))

        all_data = []
        current_start = start_time

        while current_start < end_time:
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
                    f"  {instrument} {granularity}: "
                    f"{current_start.date()} a {current_end.date()} "
                    f"({len(df)} velas)"
                )

            current_start = current_end

        if not all_data:
            logger.warning(f"No se descargaron datos para {instrument} {granularity}")
            return False

        # Combinar todos los datos
        final_df = pd.concat(all_data)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        final_df.sort_index(inplace=True)

        # Guardar en CSV
        output_file = instrument_dir / f"{granularity}.csv"
        final_df.to_csv(output_file)

        logger.info(
            f"✓ {instrument} {granularity}: {len(final_df)} velas guardadas "
            f"({final_df.index[0].date()} a {final_df.index[-1].date()})"
        )

        return True

    def download_all(self, pairs: List[str], timeframes: Dict[str, str], years: int = 5):
        """
        Descarga todos los pares y timeframes

        Args:
            pairs: Lista de pares a descargar
            timeframes: Diccionario de timeframes
            years: Años de histórico
        """
        # Primero, descargar información de los instrumentos
        logger.info("\n" + "="*60)
        logger.info("PASO 1: Descargando información de instrumentos")
        logger.info("="*60 + "\n")

        instruments_info = self.get_instrument_info(pairs)

        # Guardar información de instrumentos
        info_file = DATA_DIR / 'instruments_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(instruments_info, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ Información de instrumentos guardada en: {info_file}\n")

        # Luego, descargar datos OHLC
        total = len(pairs) * len(timeframes)
        current = 0

        logger.info("\n" + "="*60)
        logger.info("PASO 2: Descargando datos OHLC históricos")
        logger.info("="*60)
        logger.info(f"Total de combinaciones: {total}")
        logger.info(f"Pares: {', '.join(pairs)}")
        logger.info(f"Timeframes: {', '.join(timeframes.keys())}")
        logger.info(f"Histórico: {years} años\n")

        start_time = time.time()

        for pair in pairs:
            for tf_name, tf_code in timeframes.items():
                current += 1
                logger.info(f"\n[{current}/{total}] {pair} - {tf_name}")

                success = self.download_instrument_timeframe(
                    instrument=pair,
                    granularity=tf_code,
                    years=years
                )

                if not success:
                    logger.error(f"✗ Falló descarga de {pair} {tf_name}")

        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Descarga completada en {elapsed:.1f} segundos")
        logger.info(f"Datos guardados en: {DATA_DIR}")
        logger.info(f"{'='*60}")


def main():
    """Función principal"""
    # Validar credenciales
    if not OANDA_API_KEY:
        logger.error("ERROR: OANDA_API_KEY no encontrado en .env")
        sys.exit(1)

    if not OANDA_ACCOUNT_ID:
        logger.error("ERROR: OANDA_ACCOUNT_ID no encontrado en .env")
        sys.exit(1)

    # Crear directorio de datos
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Inicializar descargador
    downloader = OandaDownloader(
        api_key=OANDA_API_KEY,
        account_id=OANDA_ACCOUNT_ID,
        api_url=OANDA_API_URL
    )

    # Descargar datos
    downloader.download_all(
        pairs=PAIRS,
        timeframes=TIMEFRAMES,
        years=5
    )


if __name__ == '__main__':
    main()
