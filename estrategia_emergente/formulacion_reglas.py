"""
Formulación de Reglas

LA ESTRATEGIA EMERGE:

Basándose en las transformaciones validadas:


SEÑAL LONG:
───────────
- R₁(C) > 0                    (momentum inmediato positivo)
- R₂₄(C) - R₉₆(C) > threshold  (momentum diferencial positivo)
- Pos₂₀(C) < 0.3               (precio bajo en rango → espacio para subir)
- σ₁₀/σ₅₀ < 0.8                (volatilidad comprimida)
- Hora favorable según T_0403


SEÑAL SHORT:
────────────
Lo opuesto.


SIZING:
───────
Basado en:
- Fuerza combinada de las señales
- Volatilidad actual (ATR)
- Correlación con posiciones existentes


EXIT:
─────
- Stop loss basado en ATR
- Take profit basado en riesgo/beneficio
- Timeout si la señal se debilita

Author: Sistema de Edge-Finding Forex
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FormulacionReglas:
    """
    Formulación de Reglas de Trading basadas en transformaciones validadas.

    La estrategia EMERGE de los datos, no se predefinió.
    """

    def __init__(
        self,
        transformaciones_validadas: pd.DataFrame,
        verbose: bool = True
    ):
        """
        Inicializa el sistema de formulación de reglas.

        Parameters:
        -----------
        transformaciones_validadas : pd.DataFrame
            DataFrame con transformaciones que pasaron validación
            Columnas: 'Transformacion', 'IC', 'Señal', 'Peso', etc.
        verbose : bool
            Imprimir reglas generadas
        """
        self.transformaciones_validadas = transformaciones_validadas
        self.verbose = verbose

        # Reglas de entrada
        self.reglas_long: List[Dict] = []
        self.reglas_short: List[Dict] = []

        # Parámetros de sizing
        self.sizing_params = {
            'base_size': 0.02,  # 2% del capital por posición base
            'max_size': 0.10,   # 10% máximo por posición
            'atr_multiplier': 2.0,  # Multiplicador ATR para stop loss
            'risk_per_trade': 0.01  # 1% de riesgo por trade
        }

        # Parámetros de exit
        self.exit_params = {
            'stop_loss_atr': 2.0,      # Stop loss a 2 ATR
            'take_profit_atr': 3.0,    # Take profit a 3 ATR (R:R = 1.5)
            'timeout_bars': 50,        # Exit después de 50 barras si no hay movimiento
            'trailing_stop_atr': 1.5   # Trailing stop a 1.5 ATR
        }

        if self.verbose:
            print("="*80)
            print("FORMULACIÓN DE REGLAS - LA ESTRATEGIA EMERGE")
            print("="*80)

    def generar_reglas_entrada(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Genera reglas de entrada Long y Short basadas en transformaciones validadas.

        Returns:
        --------
        reglas_long : List[Dict]
            Reglas para señales LONG
        reglas_short : List[Dict]
            Reglas para señales SHORT
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"GENERANDO REGLAS DE ENTRADA")
            print(f"{'='*80}")

        self.reglas_long = []
        self.reglas_short = []

        for idx, row in self.transformaciones_validadas.iterrows():
            transformacion = row['Transformacion']
            ic = row['IC']
            señal_base = row.get('Señal', 'Long' if ic > 0 else 'Short')
            peso = row.get('Peso', 1.0 / len(self.transformaciones_validadas))

            # Generar regla
            regla = self._generar_regla_individual(
                transformacion=transformacion,
                ic=ic,
                señal_base=señal_base,
                peso=peso
            )

            # Añadir a Long o Short
            if señal_base == 'Long':
                self.reglas_long.append(regla)
            elif señal_base == 'Short':
                self.reglas_short.append(regla)

        # Mostrar reglas
        if self.verbose:
            self._mostrar_reglas()

        return self.reglas_long, self.reglas_short

    def _generar_regla_individual(
        self,
        transformacion: str,
        ic: float,
        señal_base: str,
        peso: float
    ) -> Dict:
        """
        Genera una regla individual basada en transformación.

        Parameters:
        -----------
        transformacion : str
            Nombre de la transformación
        ic : float
            Information Coefficient
        señal_base : str
            Señal base (Long/Short)
        peso : float
            Peso de la transformación

        Returns:
        --------
        regla : Dict
            Diccionario con la regla
        """
        # Extraer componentes de la transformación
        componentes = self._parsear_transformacion(transformacion)

        # Determinar operador y threshold
        if ic > 0:
            operador = '>'
            threshold = self._estimar_threshold(transformacion, ic, positivo=True)
        else:
            operador = '<'
            threshold = self._estimar_threshold(transformacion, ic, positivo=False)

        # Invertir si es señal Short
        if señal_base == 'Short':
            operador = '<' if operador == '>' else '>'

        # Construir regla
        regla = {
            'transformacion': transformacion,
            'componentes': componentes,
            'operador': operador,
            'threshold': threshold,
            'peso': peso,
            'ic': ic,
            'señal': señal_base,
            'descripcion': self._generar_descripcion_regla(
                transformacion, operador, threshold, señal_base
            )
        }

        return regla

    def _parsear_transformacion(self, transformacion: str) -> Dict:
        """
        Parsea componentes de una transformación.

        Parameters:
        -----------
        transformacion : str
            Nombre de la transformación

        Returns:
        --------
        componentes : Dict
            Diccionario con componentes
        """
        # Detectar tipo de transformación
        if 'Pos' in transformacion:
            tipo = 'Posicion'
        elif 'minus' in transformacion or '-' in transformacion:
            tipo = 'Diferencial'
        elif 'div' in transformacion or '/' in transformacion:
            tipo = 'Ratio'
        elif 'times' in transformacion or '*' in transformacion:
            tipo = 'Producto'
        elif 'R' in transformacion and 'R_1' in transformacion:
            tipo = 'Momentum_Inmediato'
        elif 'R' in transformacion:
            tipo = 'Momentum'
        elif 'sigma' in transformacion or 'σ' in transformacion:
            tipo = 'Volatilidad'
        elif 'mu' in transformacion or 'μ' in transformacion:
            tipo = 'Media'
        else:
            tipo = 'Complejo'

        return {
            'tipo': tipo,
            'nombre': transformacion
        }

    def _estimar_threshold(
        self,
        transformacion: str,
        ic: float,
        positivo: bool = True
    ) -> float:
        """
        Estima threshold apropiado para una transformación.

        Parameters:
        -----------
        transformacion : str
            Nombre de la transformación
        ic : float
            Information Coefficient
        positivo : bool
            Si True, threshold para IC positivo

        Returns:
        --------
        threshold : float
            Threshold estimado
        """
        # Thresholds basados en tipo de transformación
        if 'Pos' in transformacion:
            # Posición relativa [0, 1]
            return 0.3 if not positivo else 0.7

        elif 'R_1' in transformacion:
            # Momentum inmediato (retorno 1 período)
            return 0.0  # Simplemente > 0 o < 0

        elif 'minus' in transformacion or 'diff' in transformacion:
            # Diferencial
            return 0.0  # Diferencia > 0 o < 0

        elif 'div' in transformacion or 'ratio' in transformacion:
            # Ratio
            if 'sigma' in transformacion:
                # Ratio de volatilidad
                return 0.8 if not positivo else 1.2
            else:
                return 1.0

        elif 'sigma' in transformacion or 'σ' in transformacion:
            # Volatilidad absoluta
            # Difícil sin conocer la escala, usar 0
            return 0.0

        elif 'Z' in transformacion:
            # Z-score
            return -1.0 if not positivo else 1.0

        else:
            # Por defecto
            return 0.0

    def _generar_descripcion_regla(
        self,
        transformacion: str,
        operador: str,
        threshold: float,
        señal: str
    ) -> str:
        """
        Genera descripción en lenguaje natural de la regla.

        Parameters:
        -----------
        transformacion : str
            Nombre de la transformación
        operador : str
            Operador ('>', '<', etc.)
        threshold : float
            Threshold
        señal : str
            Tipo de señal (Long/Short)

        Returns:
        --------
        descripcion : str
            Descripción de la regla
        """
        # Simplificar nombre
        nombre_simple = transformacion.replace('_', ' ')

        descripcion = f"{nombre_simple} {operador} {threshold:.3f}"
        return descripcion

    def _mostrar_reglas(self):
        """
        Muestra las reglas generadas.
        """
        print(f"\n{'='*80}")
        print(f"REGLAS LONG")
        print(f"{'='*80}")

        if len(self.reglas_long) == 0:
            print("  (No hay reglas Long)")
        else:
            for i, regla in enumerate(self.reglas_long, 1):
                print(f"\n{i}. {regla['transformacion']}")
                print(f"   Condición: {regla['descripcion']}")
                print(f"   Peso: {regla['peso']:.3f} | IC: {regla['ic']:.4f}")

        print(f"\n{'='*80}")
        print(f"REGLAS SHORT")
        print(f"{'='*80}")

        if len(self.reglas_short) == 0:
            print("  (No hay reglas Short)")
        else:
            for i, regla in enumerate(self.reglas_short, 1):
                print(f"\n{i}. {regla['transformacion']}")
                print(f"   Condición: {regla['descripcion']}")
                print(f"   Peso: {regla['peso']:.3f} | IC: {regla['ic']:.4f}")

        print(f"\n{'='*80}")
        print(f"LÓGICA DE COMBINACIÓN")
        print(f"{'='*80}")
        print(f"\nSeñal Long Final = Σ(peso_i × condicion_i)")
        print(f"  Si Señal Long > umbral_long → ABRIR LONG")
        print(f"\nSeñal Short Final = Σ(peso_i × condicion_i)")
        print(f"  Si Señal Short > umbral_short → ABRIR SHORT")
        print(f"\nUmbral recomendado: suma_pesos / 2 = {sum(r['peso'] for r in self.reglas_long) / 2:.3f}")

    def calcular_señal(
        self,
        valores_transformaciones: Dict[str, float]
    ) -> Tuple[float, float, str]:
        """
        Calcula señal de trading basada en valores actuales de transformaciones.

        Parameters:
        -----------
        valores_transformaciones : Dict[str, float]
            {transformacion: valor_actual}

        Returns:
        --------
        señal_long : float
            Fuerza de señal Long (0 a 1)
        señal_short : float
            Fuerza de señal Short (0 a 1)
        decision : str
            'LONG', 'SHORT', o 'NEUTRAL'
        """
        señal_long = 0.0
        señal_short = 0.0

        # Evaluar reglas Long
        for regla in self.reglas_long:
            trans = regla['transformacion']
            if trans in valores_transformaciones:
                valor = valores_transformaciones[trans]
                threshold = regla['threshold']
                operador = regla['operador']
                peso = regla['peso']

                # Evaluar condición
                if operador == '>':
                    cumple = valor > threshold
                elif operador == '<':
                    cumple = valor < threshold
                else:
                    cumple = False

                if cumple:
                    señal_long += peso

        # Evaluar reglas Short
        for regla in self.reglas_short:
            trans = regla['transformacion']
            if trans in valores_transformaciones:
                valor = valores_transformaciones[trans]
                threshold = regla['threshold']
                operador = regla['operador']
                peso = regla['peso']

                # Evaluar condición
                if operador == '>':
                    cumple = valor > threshold
                elif operador == '<':
                    cumple = valor < threshold
                else:
                    cumple = False

                if cumple:
                    señal_short += peso

        # Determinar decisión
        umbral_long = sum(r['peso'] for r in self.reglas_long) / 2
        umbral_short = sum(r['peso'] for r in self.reglas_short) / 2

        if señal_long > umbral_long and señal_long > señal_short:
            decision = 'LONG'
        elif señal_short > umbral_short and señal_short > señal_long:
            decision = 'SHORT'
        else:
            decision = 'NEUTRAL'

        return señal_long, señal_short, decision

    def calcular_position_size(
        self,
        capital: float,
        señal_fuerza: float,
        atr: float,
        precio_actual: float,
        correlacion_posiciones: float = 0.0
    ) -> float:
        """
        Calcula tamaño de posición basado en señal y riesgo.

        SIZING:
        ───────
        Basado en:
        - Fuerza combinada de las señales
        - Volatilidad actual (ATR)
        - Correlación con posiciones existentes

        Parameters:
        -----------
        capital : float
            Capital disponible
        señal_fuerza : float
            Fuerza de la señal (0 a 1)
        atr : float
            Average True Range actual
        precio_actual : float
            Precio actual del activo
        correlacion_posiciones : float
            Correlación con posiciones existentes (0 a 1)

        Returns:
        --------
        position_size : float
            Tamaño de posición en unidades de capital (0 a 1)
        """
        # Tamaño base
        base_size = self.sizing_params['base_size']
        max_size = self.sizing_params['max_size']
        risk_per_trade = self.sizing_params['risk_per_trade']
        atr_mult = self.sizing_params['atr_multiplier']

        # 1. Ajustar por fuerza de señal
        size_señal = base_size * señal_fuerza

        # 2. Ajustar por volatilidad (ATR)
        # A mayor volatilidad, menor tamaño para mantener riesgo constante
        stop_loss_dist = atr * atr_mult
        risk_amount = capital * risk_per_trade
        size_atr = (risk_amount / stop_loss_dist) / precio_actual

        # 3. Ajustar por correlación con posiciones existentes
        # Si alta correlación, reducir tamaño
        factor_correlacion = 1.0 - (correlacion_posiciones * 0.5)

        # Combinar factores
        position_size = min(size_señal, size_atr) * factor_correlacion

        # Limitar a max_size
        position_size = min(position_size, max_size)

        return position_size

    def calcular_stop_loss_take_profit(
        self,
        precio_entrada: float,
        direccion: str,
        atr: float
    ) -> Tuple[float, float]:
        """
        Calcula niveles de stop loss y take profit.

        EXIT:
        ─────
        - Stop loss basado en ATR
        - Take profit basado en riesgo/beneficio

        Parameters:
        -----------
        precio_entrada : float
            Precio de entrada
        direccion : str
            'LONG' o 'SHORT'
        atr : float
            Average True Range actual

        Returns:
        --------
        stop_loss : float
            Nivel de stop loss
        take_profit : float
            Nivel de take profit
        """
        sl_atr = self.exit_params['stop_loss_atr']
        tp_atr = self.exit_params['take_profit_atr']

        if direccion == 'LONG':
            stop_loss = precio_entrada - (atr * sl_atr)
            take_profit = precio_entrada + (atr * tp_atr)
        elif direccion == 'SHORT':
            stop_loss = precio_entrada + (atr * sl_atr)
            take_profit = precio_entrada - (atr * tp_atr)
        else:
            stop_loss = precio_entrada
            take_profit = precio_entrada

        return stop_loss, take_profit

    def generar_codigo_estrategia(
        self,
        ruta_salida: Optional[str] = None
    ) -> str:
        """
        Genera código ejecutable de la estrategia.

        Returns:
        --------
        codigo : str
            Código Python de la estrategia
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"GENERANDO CÓDIGO DE ESTRATEGIA")
            print(f"{'='*80}")

        # Construir código
        codigo = self._construir_codigo_estrategia()

        # Guardar si se especifica ruta
        if ruta_salida:
            with open(ruta_salida, 'w') as f:
                f.write(codigo)
            if self.verbose:
                print(f"\n✓ Código guardado en: {ruta_salida}")

        return codigo

    def _construir_codigo_estrategia(self) -> str:
        """
        Construye código ejecutable de la estrategia.

        Returns:
        --------
        codigo : str
            Código Python
        """
        codigo = '''"""
Estrategia de Trading Emergente

GENERADA AUTOMÁTICAMENTE desde transformaciones validadas.

NO predefinida, EMERGIÓ de los datos después de:
- Generación sistemática de ~1,700 transformaciones
- Validación rigurosa (Walk-Forward, Permutation, Bootstrap, Robustez)
- Solo las transformaciones que PASARON TODOS los filtros

Autor: Sistema de Edge-Finding Forex
"""

import numpy as np
import pandas as pd


class EstrategiaEmergente:
    """
    Estrategia de trading basada en transformaciones validadas.
    """

    def __init__(self):
        """Inicializa la estrategia."""
        # Parámetros de sizing
        self.base_size = 0.02
        self.max_size = 0.10
        self.risk_per_trade = 0.01

        # Parámetros de exit
        self.stop_loss_atr = 2.0
        self.take_profit_atr = 3.0
        self.timeout_bars = 50

    def calcular_señal(self, datos: pd.DataFrame) -> str:
        """
        Calcula señal de trading.

        Parameters:
        -----------
        datos : pd.DataFrame
            DataFrame con transformaciones calculadas

        Returns:
        --------
        señal : str
            'LONG', 'SHORT', o 'NEUTRAL'
        """
'''

        # Añadir reglas Long
        codigo += "\n        # Evaluar reglas LONG\n"
        codigo += "        señal_long = 0.0\n"
        for regla in self.reglas_long:
            trans = regla['transformacion']
            oper = regla['operador']
            thresh = regla['threshold']
            peso = regla['peso']
            codigo += f"        if datos['{trans}'].iloc[-1] {oper} {thresh:.4f}:\n"
            codigo += f"            señal_long += {peso:.4f}  # {regla['descripcion']}\n"

        # Añadir reglas Short
        codigo += "\n        # Evaluar reglas SHORT\n"
        codigo += "        señal_short = 0.0\n"
        for regla in self.reglas_short:
            trans = regla['transformacion']
            oper = regla['operador']
            thresh = regla['threshold']
            peso = regla['peso']
            codigo += f"        if datos['{trans}'].iloc[-1] {oper} {thresh:.4f}:\n"
            codigo += f"            señal_short += {peso:.4f}  # {regla['descripcion']}\n"

        # Añadir lógica de decisión
        umbral_long = sum(r['peso'] for r in self.reglas_long) / 2 if self.reglas_long else 0.5
        umbral_short = sum(r['peso'] for r in self.reglas_short) / 2 if self.reglas_short else 0.5

        codigo += f'''
        # Decisión final
        umbral_long = {umbral_long:.4f}
        umbral_short = {umbral_short:.4f}

        if señal_long > umbral_long and señal_long > señal_short:
            return 'LONG'
        elif señal_short > umbral_short and señal_short > señal_long:
            return 'SHORT'
        else:
            return 'NEUTRAL'

    def calcular_position_size(self, capital, señal_fuerza, atr, precio):
        """Calcula tamaño de posición."""
        base_size = self.base_size * señal_fuerza
        risk_amount = capital * self.risk_per_trade
        stop_dist = atr * self.stop_loss_atr
        size_atr = (risk_amount / stop_dist) / precio

        position_size = min(base_size, size_atr, self.max_size)
        return position_size

    def calcular_stops(self, precio_entrada, direccion, atr):
        """Calcula stop loss y take profit."""
        if direccion == 'LONG':
            stop_loss = precio_entrada - (atr * self.stop_loss_atr)
            take_profit = precio_entrada + (atr * self.take_profit_atr)
        else:
            stop_loss = precio_entrada + (atr * self.stop_loss_atr)
            take_profit = precio_entrada - (atr * self.take_profit_atr)

        return stop_loss, take_profit
'''

        return codigo

    def generar_resumen_estrategia(self) -> pd.DataFrame:
        """
        Genera resumen ejecutivo de la estrategia.

        Returns:
        --------
        df_resumen : pd.DataFrame
            Resumen de la estrategia
        """
        resumen = {
            'Componente': [],
            'Descripción': [],
            'Valor': []
        }

        # Entrada
        resumen['Componente'].append('Reglas Long')
        resumen['Descripción'].append(f'Número de condiciones para señal Long')
        resumen['Valor'].append(len(self.reglas_long))

        resumen['Componente'].append('Reglas Short')
        resumen['Descripción'].append(f'Número de condiciones para señal Short')
        resumen['Valor'].append(len(self.reglas_short))

        # Sizing
        resumen['Componente'].append('Base Size')
        resumen['Descripción'].append('Tamaño base de posición (% capital)')
        resumen['Valor'].append(f"{self.sizing_params['base_size']*100:.1f}%")

        resumen['Componente'].append('Max Size')
        resumen['Descripción'].append('Tamaño máximo de posición (% capital)')
        resumen['Valor'].append(f"{self.sizing_params['max_size']*100:.1f}%")

        resumen['Componente'].append('Risk per Trade')
        resumen['Descripción'].append('Riesgo por operación (% capital)')
        resumen['Valor'].append(f"{self.sizing_params['risk_per_trade']*100:.1f}%")

        # Exit
        resumen['Componente'].append('Stop Loss')
        resumen['Descripción'].append('Stop loss en ATR')
        resumen['Valor'].append(f"{self.exit_params['stop_loss_atr']:.1f} ATR")

        resumen['Componente'].append('Take Profit')
        resumen['Descripción'].append('Take profit en ATR')
        resumen['Valor'].append(f"{self.exit_params['take_profit_atr']:.1f} ATR")

        resumen['Componente'].append('Risk/Reward')
        resumen['Descripción'].append('Ratio riesgo/beneficio')
        resumen['Valor'].append(f"1:{self.exit_params['take_profit_atr']/self.exit_params['stop_loss_atr']:.1f}")

        resumen['Componente'].append('Timeout')
        resumen['Descripción'].append('Exit por timeout (barras)')
        resumen['Valor'].append(f"{self.exit_params['timeout_bars']} barras")

        df_resumen = pd.DataFrame(resumen)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RESUMEN EJECUTIVO DE LA ESTRATEGIA")
            print(f"{'='*80}")
            print(df_resumen.to_string(index=False))

        return df_resumen


def ejemplo_uso():
    """
    Ejemplo de uso de Formulación de Reglas.
    """
    print("="*80)
    print("EJEMPLO: Formulación de Reglas")
    print("="*80)

    # Simular transformaciones validadas
    transformaciones_validadas = pd.DataFrame({
        'Transformacion': [
            'R_1_C',
            'R_24_C_minus_R_96_C',
            'Pos_20_C',
            'sigma_10_C_div_sigma_50_C',
            'hour_sin_times_R_4_C'
        ],
        'IC': [0.016, 0.024, -0.028, -0.021, 0.019],
        'Señal': ['Long', 'Long', 'Short', 'Short', 'Long'],
        'Peso': [0.148, 0.222, 0.259, 0.194, 0.176],
        'Robusto': ['Sí']*5,
        'Estable': ['Sí']*5
    })

    print("\nTRANSFORMACIONES VALIDADAS:")
    print(transformaciones_validadas.to_string(index=False))

    # Inicializar formulador
    formulador = FormulacionReglas(
        transformaciones_validadas=transformaciones_validadas,
        verbose=True
    )

    # Generar reglas
    reglas_long, reglas_short = formulador.generar_reglas_entrada()

    # Simular cálculo de señal
    print(f"\n{'='*80}")
    print(f"EJEMPLO DE CÁLCULO DE SEÑAL")
    print(f"{'='*80}")

    valores_actuales = {
        'R_1_C': 0.0005,
        'R_24_C_minus_R_96_C': 0.0012,
        'Pos_20_C': 0.25,
        'sigma_10_C_div_sigma_50_C': 0.75,
        'hour_sin_times_R_4_C': 0.0003
    }

    señal_long, señal_short, decision = formulador.calcular_señal(valores_actuales)

    print(f"\nValores actuales de transformaciones:")
    for trans, valor in valores_actuales.items():
        print(f"  {trans}: {valor:.4f}")

    print(f"\nSeñales calculadas:")
    print(f"  Señal Long:  {señal_long:.3f}")
    print(f"  Señal Short: {señal_short:.3f}")
    print(f"  DECISIÓN: {decision}")

    # Calcular sizing
    if decision != 'NEUTRAL':
        fuerza = señal_long if decision == 'LONG' else señal_short
        position_size = formulador.calcular_position_size(
            capital=100000,
            señal_fuerza=fuerza,
            atr=0.0015,
            precio_actual=1.0850,
            correlacion_posiciones=0.3
        )

        print(f"\nPosition Sizing:")
        print(f"  Tamaño: {position_size*100:.2f}% del capital")

        # Calcular stops
        sl, tp = formulador.calcular_stop_loss_take_profit(
            precio_entrada=1.0850,
            direccion=decision,
            atr=0.0015
        )

        print(f"\nNiveles de Exit:")
        print(f"  Stop Loss: {sl:.4f}")
        print(f"  Take Profit: {tp:.4f}")
        print(f"  Risk/Reward: 1:{formulador.exit_params['take_profit_atr']/formulador.exit_params['stop_loss_atr']:.1f}")

    # Generar código
    codigo = formulador.generar_codigo_estrategia(
        ruta_salida="estrategia_emergente_codigo.py"
    )

    # Resumen
    df_resumen = formulador.generar_resumen_estrategia()

    print("\n✓ Formulación de Reglas completada")


if __name__ == "__main__":
    """
    Ejecutar Formulación de Reglas.

    CONVIERTE transformaciones validadas en reglas ejecutables de trading.
    """
    ejemplo_uso()
