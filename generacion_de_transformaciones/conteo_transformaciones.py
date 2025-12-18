"""
CONTEO TOTAL DE TRANSFORMACIONES
=================================

Calcula el número total de transformaciones que se generan
en el sistema de feature engineering automático.

PRINCIPIO: Todas las transformaciones son generadas automáticamente.
           Ninguna es privilegiada sobre otra.

Autor: Sistema de Edge Finding
Fecha: 2025-12-16
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class ConfiguracionTransformaciones:
    """Configuración del sistema de transformaciones"""

    # Variables base (OHLCV)
    num_variables_base: int = 5

    # Operadores
    operadores_con_ventana: List[str] = None
    operadores_sin_ventana: List[str] = None

    # Ventanas
    ventanas: List[int] = None

    # Combinaciones de variables
    num_ratios: int = 7
    num_diferencias: int = 4
    num_promedios: int = 3

    # Composiciones
    num_composiciones_basicas: int = 50
    ventanas_composicion: int = 5

    # Comparaciones temporales
    num_comparaciones_temporales: int = 30

    # Información temporal
    num_features_temporales: int = 15

    # Inter-activo (múltiples pares)
    num_pares: int = 1
    num_features_inter_activo_por_par: int = 40

    def __post_init__(self):
        """Inicializa listas por defecto"""
        if self.operadores_con_ventana is None:
            self.operadores_con_ventana = [
                'Delta', 'R', 'r', 'mu', 'sigma',
                'Max', 'Min', 'Z', 'Pos', 'Rank', 'EMA', 'P'
            ]

        if self.operadores_sin_ventana is None:
            self.operadores_sin_ventana = ['D1', 'D2']

        if self.ventanas is None:
            self.ventanas = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]


class ContadorTransformaciones:
    """
    Calcula el número total de transformaciones generadas
    en cada nivel del sistema.
    """

    def __init__(self, config: ConfiguracionTransformaciones = None):
        """
        Inicializa el contador.

        Args:
            config: Configuración del sistema (usa defaults si es None)
        """
        self.config = config or ConfiguracionTransformaciones()
        self.conteos = {}

    def nivel_1_operadores_basicos(self) -> int:
        """
        NIVEL 1: OPERADORES BÁSICOS SOBRE VARIABLES
        ============================================
        Para cada variable V ∈ {O, H, L, C, V}:
        Para cada operador Op ∈ {Δ, R, r, μ, σ, Max, Min, Z, Pos, Rank, EMA, P}:
        Para cada ventana n ∈ {1, 2, 3, 4, 5, 10, 20, 50, 100, 200}:
            Crear transformación: Op_n(V)
        """
        # Operadores con ventana
        con_ventana = (
            self.config.num_variables_base *
            len(self.config.operadores_con_ventana) *
            len(self.config.ventanas)
        )

        # Operadores sin ventana (D1, D2)
        sin_ventana = (
            self.config.num_variables_base *
            len(self.config.operadores_sin_ventana)
        )

        total = con_ventana + sin_ventana

        self.conteos['nivel_1'] = {
            'con_ventana': con_ventana,
            'sin_ventana': sin_ventana,
            'total': total
        }

        return total

    def nivel_2_combinaciones_variables(self) -> int:
        """
        NIVEL 2: COMBINACIONES DE VARIABLES
        ====================================
        Ratios: C/O, H/L, C/H, C/L, (C-L)/(H-L), (H-C)/(H-L), (C-O)/(H-L)
        Diferencias: H-L, C-O, H-C, C-L
        Promedios típicos: HL2, HLC3, OHLC4

        Luego aplicar operadores clave sobre estas combinaciones.
        """
        # Combinaciones básicas
        combinaciones_basicas = (
            self.config.num_ratios +
            self.config.num_diferencias +
            self.config.num_promedios
        )

        # Aplicar operadores sobre combinaciones
        # (solo operadores clave: mu, sigma, Z, Pos, D1)
        # (solo ventanas clave: 5, 20, 50)
        num_operadores_clave = 5
        num_ventanas_clave = 3

        transformaciones_sobre_combinaciones = (
            combinaciones_basicas *
            num_operadores_clave *
            num_ventanas_clave
        )

        total = combinaciones_basicas + transformaciones_sobre_combinaciones

        self.conteos['nivel_2'] = {
            'combinaciones_basicas': combinaciones_basicas,
            'transformaciones': transformaciones_sobre_combinaciones,
            'total': total
        }

        return total

    def nivel_3_composiciones(self) -> int:
        """
        NIVEL 3: COMPOSICIÓN DE OPERADORES
        ===================================
        Aplicar operadores sobre resultados de otros operadores:
        - Z_20(μ_10(C))     → Z-score del promedio
        - Pos_50(σ_20(C))   → Posición de la volatilidad
        - D¹(μ_20(C))       → Velocidad del promedio
        - D²(Z_50(C))       → Aceleración del z-score
        etc.
        """
        composiciones = (
            self.config.num_composiciones_basicas *
            self.config.ventanas_composicion
        )

        self.conteos['nivel_3'] = {
            'total': composiciones
        }

        return composiciones

    def nivel_4_comparaciones_temporales(self) -> int:
        """
        NIVEL 4: COMPARACIONES TEMPORALES
        ==================================
        - μ_m(x) / μ_n(x)   → Ratio de promedios
        - σ_m(x) / σ_n(x)   → Ratio de volatilidades
        - μ_m(x) - μ_n(x)   → Diferencia de promedios
        - EMA_m / EMA_n     → Ratio de exponenciales
        """
        total = self.config.num_comparaciones_temporales

        self.conteos['nivel_4'] = {
            'total': total
        }

        return total

    def nivel_5_informacion_temporal(self) -> int:
        """
        NIVEL 5: INFORMACIÓN TEMPORAL
        ==============================
        De τ extraer:
        - hour_sin, hour_cos
        - day_sin, day_cos
        - es_lunes, es_viernes
        - es_inicio_mes, es_fin_mes
        - month_sin, month_cos
        etc.
        """
        total = self.config.num_features_temporales

        self.conteos['nivel_5'] = {
            'total': total
        }

        return total

    def nivel_6_inter_activo(self) -> int:
        """
        NIVEL 6: INTER-ACTIVO (MÚLTIPLES PARES)
        ========================================
        Si hay múltiples pares:
        - ρ_n(A, B)                    → Correlación entre activos
        - Z_n(A - β×B)                 → Spread normalizado
        - R_n(A) - R_n(B)              → Diferencia de retornos
        - ρ_m(A,B) - ρ_n(A,B)          → Cambio en correlación
        """
        if self.config.num_pares <= 1:
            total = 0
        else:
            # Combinaciones de pares (n choose 2)
            combinaciones_pares = (
                self.config.num_pares * (self.config.num_pares - 1) // 2
            )

            total = (
                combinaciones_pares *
                self.config.num_features_inter_activo_por_par
            )

        self.conteos['nivel_6'] = {
            'combinaciones_pares': self.config.num_pares * (self.config.num_pares - 1) // 2 if self.config.num_pares > 1 else 0,
            'total': total
        }

        return total

    def calcular_totales(self) -> Dict[str, int]:
        """
        Calcula el total de transformaciones en todos los niveles.

        Returns:
            Diccionario con conteos por nivel y total
        """
        print("="*70)
        print("CONTEO TOTAL DE TRANSFORMACIONES")
        print("="*70)
        print()

        # Calcular cada nivel
        n1 = self.nivel_1_operadores_basicos()
        n2 = self.nivel_2_combinaciones_variables()
        n3 = self.nivel_3_composiciones()
        n4 = self.nivel_4_comparaciones_temporales()
        n5 = self.nivel_5_informacion_temporal()
        n6 = self.nivel_6_inter_activo()

        total = n1 + n2 + n3 + n4 + n5 + n6

        # Mostrar resultados
        self._mostrar_reporte(n1, n2, n3, n4, n5, n6, total)

        return {
            'nivel_1': n1,
            'nivel_2': n2,
            'nivel_3': n3,
            'nivel_4': n4,
            'nivel_5': n5,
            'nivel_6': n6,
            'total': total
        }

    def _mostrar_reporte(self, n1, n2, n3, n4, n5, n6, total):
        """Muestra el reporte formateado"""

        print("CONFIGURACIÓN:")
        print("-" * 70)
        print(f"  Variables base (OHLCV):          {self.config.num_variables_base}")
        print(f"  Operadores con ventana:          {len(self.config.operadores_con_ventana)}")
        print(f"  Operadores sin ventana:          {len(self.config.operadores_sin_ventana)}")
        print(f"  Ventanas:                        {len(self.config.ventanas)}")
        print(f"  Número de pares:                 {self.config.num_pares}")
        print()

        print("ESTIMACIÓN POR NIVELES:")
        print("-" * 70)
        print()

        print("Nivel 1 (Operadores básicos sobre variables):")
        print(f"  - Con ventana: {self.config.num_variables_base} vars × "
              f"{len(self.config.operadores_con_ventana)} ops × "
              f"{len(self.config.ventanas)} ventanas = {self.conteos['nivel_1']['con_ventana']}")
        print(f"  - Sin ventana: {self.config.num_variables_base} vars × "
              f"{len(self.config.operadores_sin_ventana)} ops = {self.conteos['nivel_1']['sin_ventana']}")
        print(f"  SUBTOTAL: {n1}")
        print()

        print("Nivel 2 (Combinaciones de variables):")
        print(f"  - Combinaciones básicas:         {self.conteos['nivel_2']['combinaciones_basicas']}")
        print(f"  - Transformaciones sobre ellas:  {self.conteos['nivel_2']['transformaciones']}")
        print(f"  SUBTOTAL: {n2}")
        print()

        print("Nivel 3 (Composición de operadores):")
        print(f"  - ~{self.config.num_composiciones_basicas} composiciones × "
              f"{self.config.ventanas_composicion} ventanas = {n3}")
        print(f"  SUBTOTAL: {n3}")
        print()

        print("Nivel 4 (Comparaciones temporales):")
        print(f"  - ~{n4} comparaciones")
        print(f"  SUBTOTAL: {n4}")
        print()

        print("Nivel 5 (Información temporal):")
        print(f"  - ~{n5} features temporales")
        print(f"  SUBTOTAL: {n5}")
        print()

        if self.config.num_pares > 1:
            print("Nivel 6 (Inter-activo, múltiples pares):")
            print(f"  - {self.conteos['nivel_6']['combinaciones_pares']} combinaciones de pares × "
                  f"{self.config.num_features_inter_activo_por_par} features = {n6}")
            print(f"  SUBTOTAL: {n6}")
        else:
            print("Nivel 6 (Inter-activo):")
            print(f"  - No aplicable (solo 1 par)")
            print(f"  SUBTOTAL: 0")

        print()
        print("=" * 70)
        print(f"TOTAL: ~{total:,} transformaciones por activo")

        if self.config.num_pares > 1:
            total_con_inter = total
            total_sin_inter = total - n6
            print(f"       ~{total_sin_inter:,} sin inter-activo")
            print(f"       ~{total_con_inter:,} con inter-activo ({self.config.num_pares} pares)")
        else:
            print(f"       ~{total:,} transformaciones (1 par)")

        print("=" * 70)
        print()

        print("PRINCIPIOS:")
        print("-" * 70)
        print("✓ Todas las transformaciones se generan AUTOMÁTICAMENTE")
        print("✓ Ninguna transformación es privilegiada sobre otra")
        print("✓ Sin sesgo humano ni conocimiento previo del dominio")
        print("✓ Sin look-ahead bias (solo datos del pasado)")
        print("✓ Los DATOS deciden qué features son útiles")
        print("=" * 70)


def calcular_escenarios():
    """Calcula diferentes escenarios"""

    print("\n" + "="*70)
    print("ESCENARIOS DE TRANSFORMACIONES")
    print("="*70)
    print()

    # Escenario 1: Un solo par (EUR/USD)
    print("\n📊 ESCENARIO 1: Un solo par (EUR/USD)")
    print("-" * 70)
    config1 = ConfiguracionTransformaciones(num_pares=1)
    contador1 = ContadorTransformaciones(config1)
    totales1 = contador1.calcular_totales()

    # Escenario 2: 6 pares principales
    print("\n\n📊 ESCENARIO 2: 6 pares principales")
    print("-" * 70)
    config2 = ConfiguracionTransformaciones(num_pares=6)
    contador2 = ContadorTransformaciones(config2)
    totales2 = contador2.calcular_totales()

    # Escenario 3: 10 pares (todos los descargados)
    print("\n\n📊 ESCENARIO 3: 10 pares (todos descargados)")
    print("-" * 70)
    config3 = ConfiguracionTransformaciones(num_pares=10)
    contador3 = ContadorTransformaciones(config3)
    totales3 = contador3.calcular_totales()

    # Resumen comparativo
    print("\n\n" + "="*70)
    print("RESUMEN COMPARATIVO")
    print("="*70)

    df = pd.DataFrame({
        'Escenario': ['1 par', '6 pares', '10 pares'],
        'Nivel 1': [totales1['nivel_1'], totales2['nivel_1'], totales3['nivel_1']],
        'Nivel 2': [totales1['nivel_2'], totales2['nivel_2'], totales3['nivel_2']],
        'Nivel 3': [totales1['nivel_3'], totales2['nivel_3'], totales3['nivel_3']],
        'Nivel 4': [totales1['nivel_4'], totales2['nivel_4'], totales3['nivel_4']],
        'Nivel 5': [totales1['nivel_5'], totales2['nivel_5'], totales3['nivel_5']],
        'Nivel 6': [totales1['nivel_6'], totales2['nivel_6'], totales3['nivel_6']],
        'TOTAL': [totales1['total'], totales2['total'], totales3['total']]
    })

    print()
    print(df.to_string(index=False))
    print()
    print("="*70)

    # Estimación de tamaño en memoria
    print("\n💾 ESTIMACIÓN DE TAMAÑO EN MEMORIA")
    print("-" * 70)
    print("Asumiendo 31,000 velas (5 años de H1) × 8 bytes (float64):\n")

    for escenario, total in zip(['1 par', '6 pares', '10 pares'],
                                 [totales1['total'], totales2['total'], totales3['total']]):
        size_mb = (total * 31000 * 8) / (1024**2)
        print(f"  {escenario:10s}: ~{total:5,} features × 31k velas = ~{size_mb:6.1f} MB")

    print()
    print("="*70)


def main():
    """Función principal"""
    calcular_escenarios()

    print("\n\n🎯 CONCLUSIÓN:")
    print("-" * 70)
    print("Con este sistema, generamos un espacio MASIVO de features")
    print("sin ningún sesgo humano. El modelo de machine learning")
    print("decidirá cuáles son útiles basándose ÚNICAMENTE en los datos.")
    print()
    print("Esto es EDGE FINDING verdadero: dejar que los datos hablen.")
    print("="*70)


if __name__ == '__main__':
    main()
