"""
Interpretación Post-Hoc

DESPUÉS DE VALIDACIÓN:

Supongamos que 5 transformaciones pasan todos los filtros:

1. T_0247: Pos₂₀(C) → IC = -0.028, estable, robusto
2. T_0891: R₂₄(C) - R₉₆(C) → IC = +0.024, estable
3. T_0156: σ₁₀(C) / σ₅₀(C) → IC = -0.021, estable
4. T_0403: hour_sin × R₄(C) → IC = +0.019, estable
5. T_0012: R₁(C) → IC = +0.016, estable


AHORA INTERPRETAMOS:

T_0247 (Pos₂₀): IC negativo
→ Cuando precio está alto en el rango, retorno futuro es negativo
→ MEAN REVERSION funciona en escala de 20 períodos

T_0891 (R₂₄ - R₉₆): IC positivo
→ Cuando momentum de 24h supera momentum de 96h, retorno es positivo
→ MOMENTUM DIFERENCIAL funciona

T_0156 (σ₁₀/σ₅₀): IC negativo
→ Cuando volatilidad reciente está baja, retornos son negativos...
→ O más probablemente: predice DIRECCIÓN según contexto
→ VOLATILIDAD COMPRIMIDA precede movimientos

T_0403 (hora × retorno): IC positivo
→ Hay interacción entre hora del día y momentum
→ PATRÓN HORARIO existe

T_0012 (R₁): IC positivo
→ Momentum de muy corto plazo continúa
→ MOMENTUM INMEDIATO funciona


NO PREDEFINIMOS ESTO.
LOS DATOS LO REVELARON.

Author: Sistema de Edge-Finding Forex
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')


class InterpretacionPostHoc:
    """
    Interpretación Post-Hoc de transformaciones validadas.

    FILOSOFÍA:
    - NO predefinimos qué funciona
    - Dejamos que los DATOS revelen los edges
    - DESPUÉS de validación estadística rigurosa
    - Solo entonces interpretamos qué descubrimos
    """

    def __init__(self, verbose: bool = True):
        """
        Inicializa el sistema de Interpretación Post-Hoc.

        Parameters:
        -----------
        verbose : bool
            Imprimir interpretaciones detalladas
        """
        self.verbose = verbose

        # Patrones conocidos (para interpretación, no para búsqueda)
        self.patrones_interpretacion = {
            'mean_reversion': {
                'indicadores': ['Pos', 'Z', 'Rank'],
                'ic_signo': 'negativo',
                'descripcion': 'Mean Reversion (reversión a la media)'
            },
            'momentum': {
                'indicadores': ['R', 'r', 'Delta'],
                'ic_signo': 'positivo',
                'descripcion': 'Momentum (continuación)'
            },
            'volatilidad': {
                'indicadores': ['sigma', 'σ'],
                'ic_signo': 'ambos',
                'descripcion': 'Volatilidad (régimen de mercado)'
            },
            'temporal': {
                'indicadores': ['hour', 'day', 'month'],
                'ic_signo': 'ambos',
                'descripcion': 'Patrón Temporal (estacionalidad)'
            },
            'diferencial': {
                'indicadores': ['minus', '-', 'diff'],
                'ic_signo': 'ambos',
                'descripcion': 'Diferencial (comparación de escalas)'
            },
            'ratio': {
                'indicadores': ['div', '/', 'ratio'],
                'ic_signo': 'ambos',
                'descripcion': 'Ratio (normalización relativa)'
            }
        }

        if self.verbose:
            print("="*80)
            print("INTERPRETACIÓN POST-HOC")
            print("="*80)
            print("\nFILOSOFÍA:")
            print("  1. NO predefinimos qué funciona")
            print("  2. Validación estadística PRIMERO")
            print("  3. Interpretación DESPUÉS")
            print("  4. Los DATOS revelan los edges")
            print("="*80)

    def interpretar_transformaciones_validadas(
        self,
        transformaciones_validadas: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Interpreta transformaciones que pasaron validación.

        Parameters:
        -----------
        transformaciones_validadas : pd.DataFrame
            DataFrame con transformaciones validadas
            Columnas esperadas: 'Transformacion', 'IC', 'Robusto', 'Estable', etc.

        Returns:
        --------
        df_interpretado : pd.DataFrame
            DataFrame con interpretaciones añadidas
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"INTERPRETANDO {len(transformaciones_validadas)} TRANSFORMACIONES VALIDADAS")
            print(f"{'='*80}")

        # Copiar DataFrame
        df_interpretado = transformaciones_validadas.copy()

        # Interpretar cada transformación
        interpretaciones = []
        patrones_detectados = []
        señales = []

        for idx, row in df_interpretado.iterrows():
            transformacion = row['Transformacion']
            ic = row.get('IC', 0)

            # Interpretar
            interpretacion, patron, señal = self._interpretar_transformacion(
                transformacion=transformacion,
                ic=ic
            )

            interpretaciones.append(interpretacion)
            patrones_detectados.append(patron)
            señales.append(señal)

        # Añadir columnas
        df_interpretado['Patron'] = patrones_detectados
        df_interpretado['Señal'] = señales
        df_interpretado['Interpretacion'] = interpretaciones

        # Mostrar resumen
        if self.verbose:
            self._mostrar_resumen_interpretaciones(df_interpretado)

        return df_interpretado

    def _interpretar_transformacion(
        self,
        transformacion: str,
        ic: float
    ) -> Tuple[str, str, str]:
        """
        Interpreta una transformación individual.

        Parameters:
        -----------
        transformacion : str
            Nombre de la transformación
        ic : float
            Information Coefficient

        Returns:
        --------
        interpretacion : str
            Interpretación en lenguaje natural
        patron : str
            Tipo de patrón detectado
        señal : str
            Dirección de la señal (Long/Short)
        """
        # Determinar señal (Long/Short) basado en IC
        if ic > 0:
            señal = "Long"
            relacion = "positiva"
        elif ic < 0:
            señal = "Short"
            relacion = "negativa"
        else:
            señal = "Neutral"
            relacion = "neutral"

        # Detectar patrón
        patron = self._detectar_patron(transformacion, ic)

        # Construir interpretación
        interpretacion = self._construir_interpretacion(
            transformacion=transformacion,
            ic=ic,
            patron=patron,
            señal=señal
        )

        return interpretacion, patron, señal

    def _detectar_patron(self, transformacion: str, ic: float) -> str:
        """
        Detecta el tipo de patrón de la transformación.

        Parameters:
        -----------
        transformacion : str
            Nombre de la transformación
        ic : float
            Information Coefficient

        Returns:
        --------
        patron : str
            Tipo de patrón detectado
        """
        transformacion_lower = transformacion.lower()
        ic_signo = 'positivo' if ic > 0 else 'negativo'

        # Buscar patrones
        for patron_nombre, patron_info in self.patrones_interpretacion.items():
            # Verificar si algún indicador está presente
            for indicador in patron_info['indicadores']:
                if indicador.lower() in transformacion_lower:
                    # Verificar signo de IC si es relevante
                    if patron_info['ic_signo'] == 'ambos' or patron_info['ic_signo'] == ic_signo:
                        return patron_info['descripcion']

        # Si no se detecta patrón específico
        return "Patrón Complejo"

    def _construir_interpretacion(
        self,
        transformacion: str,
        ic: float,
        patron: str,
        señal: str
    ) -> str:
        """
        Construye interpretación en lenguaje natural.

        Parameters:
        -----------
        transformacion : str
            Nombre de la transformación
        ic : float
            Information Coefficient
        patron : str
            Tipo de patrón
        señal : str
            Dirección de señal

        Returns:
        --------
        interpretacion : str
            Interpretación en lenguaje natural
        """
        # Extraer componentes de la transformación
        componentes = self._extraer_componentes(transformacion)

        # Plantillas de interpretación según patrón
        if patron == "Mean Reversion (reversión a la media)":
            if ic < 0:
                interpretacion = (
                    f"Cuando {componentes} está ALTO en el rango, "
                    f"el retorno futuro tiende a ser NEGATIVO. "
                    f"→ Reversión a la media funciona (escala: {self._extraer_periodo(transformacion)} períodos)"
                )
            else:
                interpretacion = (
                    f"Cuando {componentes} está BAJO en el rango, "
                    f"el retorno futuro tiende a ser NEGATIVO. "
                    f"→ Anti-reversión (escala: {self._extraer_periodo(transformacion)} períodos)"
                )

        elif patron == "Momentum (continuación)":
            if ic > 0:
                interpretacion = (
                    f"Cuando {componentes} es POSITIVO, "
                    f"el retorno futuro tiende a ser POSITIVO. "
                    f"→ Momentum continúa (escala: {self._extraer_periodo(transformacion)} períodos)"
                )
            else:
                interpretacion = (
                    f"Cuando {componentes} es POSITIVO, "
                    f"el retorno futuro tiende a ser NEGATIVO. "
                    f"→ Reversión de momentum (escala: {self._extraer_periodo(transformacion)} períodos)"
                )

        elif patron == "Volatilidad (régimen de mercado)":
            if ic < 0:
                interpretacion = (
                    f"Cuando {componentes} está BAJA, "
                    f"los retornos futuros tienden a ser NEGATIVOS. "
                    f"→ Volatilidad comprimida precede movimientos"
                )
            else:
                interpretacion = (
                    f"Cuando {componentes} está ALTA, "
                    f"los retornos futuros tienden a ser POSITIVOS. "
                    f"→ Alta volatilidad favorece continuación"
                )

        elif patron == "Patrón Temporal (estacionalidad)":
            interpretacion = (
                f"Existe interacción entre {componentes} y dirección del mercado. "
                f"→ Patrón horario/temporal detectado"
            )

        elif patron == "Diferencial (comparación de escalas)":
            if ic > 0:
                interpretacion = (
                    f"Cuando {componentes} es POSITIVO, "
                    f"el retorno futuro tiende a ser POSITIVO. "
                    f"→ Momentum diferencial funciona"
                )
            else:
                interpretacion = (
                    f"Cuando {componentes} es POSITIVO, "
                    f"el retorno futuro tiende a ser NEGATIVO. "
                    f"→ Divergencia entre escalas indica reversión"
                )

        elif patron == "Ratio (normalización relativa)":
            if ic < 0:
                interpretacion = (
                    f"Cuando {componentes} es ALTO, "
                    f"el retorno futuro tiende a ser NEGATIVO. "
                    f"→ Normalización relativa indica reversión"
                )
            else:
                interpretacion = (
                    f"Cuando {componentes} es ALTO, "
                    f"el retorno futuro tiende a ser POSITIVO. "
                    f"→ Ratio indica continuación"
                )

        else:
            # Patrón complejo
            if ic > 0:
                interpretacion = (
                    f"Cuando {componentes} aumenta, "
                    f"el retorno futuro tiende a ser POSITIVO. "
                    f"→ Patrón complejo con relación directa"
                )
            else:
                interpretacion = (
                    f"Cuando {componentes} aumenta, "
                    f"el retorno futuro tiende a ser NEGATIVO. "
                    f"→ Patrón complejo con relación inversa"
                )

        return interpretacion

    def _extraer_componentes(self, transformacion: str) -> str:
        """
        Extrae componentes legibles de la transformación.

        Parameters:
        -----------
        transformacion : str
            Nombre de la transformación

        Returns:
        --------
        componentes : str
            Descripción de componentes
        """
        # Simplificar nombre para hacerlo más legible
        # Ejemplo: "Pos_20_C" → "posición relativa en 20 períodos"
        # Ejemplo: "R_24_C_minus_R_96_C" → "momentum 24h vs 96h"

        transformacion_limpia = transformacion.replace('_', ' ')

        # Reemplazos comunes
        reemplazos = {
            'Pos': 'posición relativa',
            'mu': 'media',
            'sigma': 'volatilidad',
            'R': 'retorno',
            'r': 'log-retorno',
            'Delta': 'cambio',
            'Z': 'z-score',
            'Rank': 'ranking',
            'Max': 'máximo',
            'Min': 'mínimo',
            'EMA': 'media exponencial',
            'minus': 'menos',
            'plus': 'más',
            'times': 'por',
            'div': 'dividido',
            'C': 'close',
            'H': 'high',
            'L': 'low',
            'O': 'open',
            'V': 'volume'
        }

        componentes = transformacion_limpia
        for original, reemplazo in reemplazos.items():
            componentes = componentes.replace(original, reemplazo)

        return componentes

    def _extraer_periodo(self, transformacion: str) -> int:
        """
        Extrae el período principal de la transformación.

        Parameters:
        -----------
        transformacion : str
            Nombre de la transformación

        Returns:
        --------
        periodo : int
            Período en número de barras
        """
        # Buscar primer número en el nombre
        numeros = re.findall(r'\d+', transformacion)
        if numeros:
            return int(numeros[0])
        return 0

    def _mostrar_resumen_interpretaciones(self, df_interpretado: pd.DataFrame):
        """
        Muestra resumen de interpretaciones.

        Parameters:
        -----------
        df_interpretado : pd.DataFrame
            DataFrame con interpretaciones
        """
        print(f"\n{'='*80}")
        print(f"RESUMEN DE INTERPRETACIONES")
        print(f"{'='*80}")

        for idx, row in df_interpretado.iterrows():
            print(f"\n{idx + 1}. {row['Transformacion']}")
            print(f"   IC: {row['IC']:.4f} | Patrón: {row['Patron']} | Señal: {row['Señal']}")
            print(f"   → {row['Interpretacion']}")

        # Resumen de patrones detectados
        print(f"\n{'='*80}")
        print(f"PATRONES DETECTADOS")
        print(f"{'='*80}")

        patrones_count = df_interpretado['Patron'].value_counts()
        for patron, count in patrones_count.items():
            print(f"  - {patron}: {count} transformaciones")

        # Resumen de señales
        print(f"\n{'='*80}")
        print(f"DISTRIBUCIÓN DE SEÑALES")
        print(f"{'='*80}")

        señales_count = df_interpretado['Señal'].value_counts()
        for señal, count in señales_count.items():
            print(f"  - {señal}: {count} transformaciones")

        # Conclusión
        print(f"\n{'='*80}")
        print(f"CONCLUSIÓN")
        print(f"{'='*80}")
        print(f"\nLos DATOS revelaron {len(df_interpretado)} edges validados:")
        print(f"  - NO predefinimos estos patrones")
        print(f"  - Pasaron validación estadística RIGUROSA")
        print(f"  - Walk-Forward + Permutation + Bootstrap + Robustez")
        print(f"  - AHORA sabemos qué funciona y por qué")
        print(f"\nEstos son edges GENUINOS, no supuestos a priori.")
        print(f"="*80)

    def generar_estrategia_combinada(
        self,
        df_interpretado: pd.DataFrame,
        pesos: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Genera estrategia combinada de transformaciones validadas.

        Parameters:
        -----------
        df_interpretado : pd.DataFrame
            DataFrame con transformaciones interpretadas
        pesos : Dict[str, float], optional
            Pesos personalizados por transformación
            Si None, usa pesos iguales

        Returns:
        --------
        df_estrategia : pd.DataFrame
            Estrategia combinada con pesos
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"GENERANDO ESTRATEGIA COMBINADA")
            print(f"{'='*80}")

        # Si no hay pesos, usar IC como pesos (normalizado)
        if pesos is None:
            # Pesos proporcionales a |IC|
            ics = df_interpretado['IC'].abs()
            suma_ics = ics.sum()

            # CRÍTICO #3 CORREGIDO: Validar suma_ics antes de división por cero
            if suma_ics == 0 or np.isnan(suma_ics):
                # Si todos los ICs son 0, distribuir peso uniforme
                pesos_calculados = np.ones(len(df_interpretado)) / len(df_interpretado)
            else:
                pesos_calculados = (ics / suma_ics).values
        else:
            # Usar pesos personalizados
            pesos_calculados = [
                pesos.get(t, 1/len(df_interpretado))
                for t in df_interpretado['Transformacion']
            ]

        # Normalizar pesos
        pesos_calculados = np.array(pesos_calculados)
        suma_pesos = pesos_calculados.sum()

        # CRÍTICO #4 CORREGIDO: Validar suma_pesos antes de división por cero
        if suma_pesos == 0 or np.isnan(suma_pesos):
            # Si suma es 0, distribuir peso uniforme
            pesos_calculados = np.ones(len(pesos_calculados)) / len(pesos_calculados)
        else:
            pesos_calculados = pesos_calculados / suma_pesos

        # Crear DataFrame de estrategia
        df_estrategia = df_interpretado.copy()
        df_estrategia['Peso'] = pesos_calculados

        # Ajustar señal por peso
        df_estrategia['Peso_Ajustado'] = df_estrategia.apply(
            lambda row: row['Peso'] if row['Señal'] == 'Long' else -row['Peso'],
            axis=1
        )

        if self.verbose:
            print(f"\nPesos de la estrategia combinada:")
            for idx, row in df_estrategia.iterrows():
                print(f"  {row['Transformacion']}: {row['Peso']:.3f} ({row['Señal']})")

            print(f"\nSeñal combinada final = Σ(transformación × peso × señal)")
            print(f"  - Si señal_combinada > 0 → LONG")
            print(f"  - Si señal_combinada < 0 → SHORT")
            print(f"  - Umbral recomendado: |señal| > {0.1/len(df_estrategia):.4f}")

        return df_estrategia

    def exportar_interpretaciones(
        self,
        df_interpretado: pd.DataFrame,
        ruta_salida: Optional[str] = None
    ):
        """
        Exporta interpretaciones a CSV.

        Parameters:
        -----------
        df_interpretado : pd.DataFrame
            DataFrame con interpretaciones
        ruta_salida : str, optional
            Ruta para guardar CSV
        """
        if ruta_salida is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_salida = f"interpretaciones_post_hoc_{timestamp}.csv"

        df_interpretado.to_csv(ruta_salida, index=False)

        if self.verbose:
            print(f"\n✓ Interpretaciones exportadas a: {ruta_salida}")


def ejemplo_uso():
    """
    Ejemplo de uso de Interpretación Post-Hoc.
    """
    print("="*80)
    print("EJEMPLO: Interpretación Post-Hoc")
    print("="*80)

    # Simular transformaciones que pasaron validación
    transformaciones_validadas = pd.DataFrame({
        'Transformacion': [
            'Pos_20_C',
            'R_24_C_minus_R_96_C',
            'sigma_10_C_div_sigma_50_C',
            'hour_sin_times_R_4_C',
            'R_1_C'
        ],
        'IC': [-0.028, 0.024, -0.021, 0.019, 0.016],
        'Robusto': ['Sí', 'Sí', 'Sí', 'Sí', 'Sí'],
        'Estable': ['Sí', 'Sí', 'Sí', 'Sí', 'Sí'],
        'P_Value': [0.0001, 0.0005, 0.0010, 0.0020, 0.0030]
    })

    print("\nTRANSFORMACIONES VALIDADAS:")
    print(transformaciones_validadas.to_string(index=False))

    # Inicializar interpretador
    interpretador = InterpretacionPostHoc(verbose=True)

    # Interpretar transformaciones
    df_interpretado = interpretador.interpretar_transformaciones_validadas(
        transformaciones_validadas
    )

    # Generar estrategia combinada
    df_estrategia = interpretador.generar_estrategia_combinada(df_interpretado)

    # Exportar
    interpretador.exportar_interpretaciones(df_interpretado)

    print("\n✓ Interpretación Post-Hoc completada")


if __name__ == "__main__":
    """
    Ejecutar Interpretación Post-Hoc.

    IMPORTANTE:
    - Solo interpretar DESPUÉS de validación rigurosa
    - NO usar interpretación para buscar transformaciones
    - Dejar que los DATOS revelen los edges
    """
    ejemplo_uso()
