"""
Permutation Test

¿EL EDGE ES REAL O POR AZAR?

1. Mezclar aleatoriamente las fechas/retornos
2. Calcular métrica con datos mezclados
3. Repetir 10,000 veces
4. Comparar métrica real vs distribución aleatoria

EJEMPLO:

Sharpe real: 0.85
Sharpe aleatorio (distribución): media 0, std 0.15

Sharpe real está a 5.7 std de la media aleatoria.
p-value < 0.0001

CONCLUSIÓN: El edge NO es por azar.

Author: Sistema de Edge-Finding Forex
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class PermutationTest:
    """
    Permutation Test para determinar si el edge es real o por azar.

    Destruye la relación temporal entre predicciones y retornos
    mediante permutaciones aleatorias.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_permutations: int = 10000,
        seed: Optional[int] = 42,
        verbose: bool = True
    ):
        """
        Inicializa el Permutation Test.

        Parameters:
        -----------
        y_true : np.ndarray
            Retornos reales (verdaderos)
        y_pred : np.ndarray
            Predicciones o señales del modelo
        n_permutations : int
            Número de permutaciones aleatorias (default: 10,000)
        seed : int, optional
            Semilla para reproducibilidad
        verbose : bool
            Imprimir resultados detallados
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.n_permutations = n_permutations
        self.seed = seed
        self.verbose = verbose

        # Eliminar NaN
        mask = ~(np.isnan(self.y_true) | np.isnan(self.y_pred))
        self.y_true = self.y_true[mask]
        self.y_pred = self.y_pred[mask]
        self.n_obs = len(self.y_true)

        if self.seed is not None:
            np.random.seed(self.seed)

        # Resultados
        self.resultados_permutation: Dict[str, Dict] = {}

        if self.verbose:
            print("="*80)
            print("PERMUTATION TEST - ¿EL EDGE ES REAL O POR AZAR?")
            print("="*80)
            print(f"N observaciones: {self.n_obs}")
            print(f"N permutaciones: {self.n_permutations:,}")
            print("="*80)

    def _permute_predictions(self) -> np.ndarray:
        """
        Permuta (mezcla) aleatoriamente las predicciones.

        Esto destruye la relación temporal entre predicciones y retornos.
        Si el modelo tiene edge real, la métrica caerá significativamente.

        Returns:
        --------
        y_pred_permuted : np.ndarray
            Predicciones permutadas aleatoriamente
        """
        indices = np.random.permutation(self.n_obs)
        return self.y_pred[indices]

    def _calcular_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula Information Coefficient (correlación).

        Parameters:
        -----------
        y_true : np.ndarray
            Retornos reales
        y_pred : np.ndarray
            Predicciones

        Returns:
        --------
        ic : float
            Information Coefficient (correlación de Pearson)
        """
        if len(y_true) < 2:
            return np.nan

        # Eliminar NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 2:
            return np.nan

        ic = np.corrcoef(y_true, y_pred)[0, 1]
        return ic

    def _calcular_sharpe(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        periodos_anuales: int = 252
    ) -> float:
        """
        Calcula Sharpe Ratio basado en señales.

        Parameters:
        -----------
        y_true : np.ndarray
            Retornos reales
        y_pred : np.ndarray
            Predicciones (se usa signo como señal)
        periodos_anuales : int
            Períodos de trading por año

        Returns:
        --------
        sharpe : float
            Sharpe Ratio
        """
        señales = np.sign(y_pred)
        retornos_estrategia = señales * y_true

        # Eliminar NaN
        retornos_estrategia = retornos_estrategia[~np.isnan(retornos_estrategia)]

        if len(retornos_estrategia) < 2:
            return np.nan

        media = np.mean(retornos_estrategia)
        std = np.std(retornos_estrategia, ddof=1)

        if std == 0 or np.isnan(std):
            return 0.0

        sharpe = (media / std) * np.sqrt(periodos_anuales)
        return sharpe

    def _calcular_return_total(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calcula Return Total basado en señales.

        Parameters:
        -----------
        y_true : np.ndarray
            Retornos reales
        y_pred : np.ndarray
            Predicciones (se usa signo como señal)

        Returns:
        --------
        ret_total : float
            Return total acumulado (porcentaje)
        """
        señales = np.sign(y_pred)
        retornos_estrategia = señales * y_true

        # Eliminar NaN
        retornos_estrategia = retornos_estrategia[~np.isnan(retornos_estrategia)]

        if len(retornos_estrategia) == 0:
            return np.nan

        ret_total = np.prod(1 + retornos_estrategia) - 1
        return ret_total * 100  # En porcentaje

    def _calcular_profit_factor(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calcula Profit Factor basado en señales.

        Parameters:
        -----------
        y_true : np.ndarray
            Retornos reales
        y_pred : np.ndarray
            Predicciones (se usa signo como señal)

        Returns:
        --------
        pf : float
            Profit Factor
        """
        señales = np.sign(y_pred)
        retornos_estrategia = señales * y_true

        # Eliminar NaN
        retornos_estrategia = retornos_estrategia[~np.isnan(retornos_estrategia)]

        ganancias = retornos_estrategia[retornos_estrategia > 0]
        perdidas = retornos_estrategia[retornos_estrategia < 0]

        suma_ganancias = np.sum(ganancias) if len(ganancias) > 0 else 0
        suma_perdidas = np.abs(np.sum(perdidas)) if len(perdidas) > 0 else 0

        if suma_perdidas == 0:
            return np.inf if suma_ganancias > 0 else 0.0

        pf = suma_ganancias / suma_perdidas
        return pf

    def _calcular_win_rate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calcula Win Rate basado en señales.

        Parameters:
        -----------
        y_true : np.ndarray
            Retornos reales
        y_pred : np.ndarray
            Predicciones (se usa signo como señal)

        Returns:
        --------
        wr : float
            Win Rate (0 a 1)
        """
        señales = np.sign(y_pred)
        retornos_estrategia = señales * y_true

        # Eliminar NaN
        retornos_estrategia = retornos_estrategia[~np.isnan(retornos_estrategia)]

        if len(retornos_estrategia) == 0:
            return np.nan

        n_wins = np.sum(retornos_estrategia > 0)
        wr = n_wins / len(retornos_estrategia)
        return wr

    def permutation_test_metrica(
        self,
        metrica_func: Callable,
        nombre_metrica: str,
        **kwargs
    ) -> Dict:
        """
        Ejecuta permutation test para una métrica específica.

        PROCESO:
        1. Calcular métrica real con datos originales
        2. Para cada permutación:
            - Mezclar aleatoriamente las predicciones
            - Calcular métrica con predicciones mezcladas
        3. Comparar métrica real vs distribución aleatoria
        4. Calcular p-value y z-score

        Parameters:
        -----------
        metrica_func : Callable
            Función que calcula la métrica: f(y_true, y_pred) -> float
        nombre_metrica : str
            Nombre de la métrica
        **kwargs : dict
            Argumentos adicionales para metrica_func

        Returns:
        --------
        resultado : Dict
            - valor_real: Métrica en datos reales
            - valores_permuted: Array con n_permutations valores aleatorios
            - media_permuted: Media de distribución aleatoria
            - std_permuted: Desviación estándar
            - z_score: Cuántas std el real está de la media aleatoria
            - p_value: Probabilidad de obtener métrica real por azar
            - significativo: True si p < 0.05
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Permutation Test: {nombre_metrica}")
            print(f"{'='*80}")

        # Calcular métrica real
        valor_real = metrica_func(self.y_true, self.y_pred, **kwargs)

        if self.verbose:
            print(f"Métrica REAL: {valor_real:.4f}")
            print(f"Ejecutando {self.n_permutations:,} permutaciones aleatorias...")

        # Permutation test
        valores_permuted = np.zeros(self.n_permutations)

        for i in range(self.n_permutations):
            y_pred_permuted = self._permute_predictions()
            valores_permuted[i] = metrica_func(self.y_true, y_pred_permuted, **kwargs)

        # Calcular estadísticas
        media_permuted = np.mean(valores_permuted)
        std_permuted = np.std(valores_permuted, ddof=1)

        # Z-score: cuántas desviaciones estándar está el real de la media aleatoria
        if std_permuted > 0:
            z_score = (valor_real - media_permuted) / std_permuted
        else:
            z_score = np.inf if valor_real != media_permuted else 0.0

        # P-value (two-tailed):
        # ¿Cuántas permutaciones tienen métrica >= |métrica real - media|?
        # Para métricas donde "mayor es mejor" (Sharpe, IC, etc.)
        if valor_real >= media_permuted:
            # One-tailed superior
            n_extremos = np.sum(valores_permuted >= valor_real)
        else:
            # One-tailed inferior
            n_extremos = np.sum(valores_permuted <= valor_real)

        p_value = n_extremos / self.n_permutations

        # Ajustar para two-tailed
        p_value_two_tailed = 2 * min(p_value, 1 - p_value)

        # Significativo si p < 0.05
        significativo = p_value < 0.05

        resultado = {
            'nombre': nombre_metrica,
            'valor_real': valor_real,
            'valores_permuted': valores_permuted,
            'media_permuted': media_permuted,
            'std_permuted': std_permuted,
            'z_score': z_score,
            'p_value': p_value,
            'p_value_two_tailed': p_value_two_tailed,
            'significativo': significativo
        }

        # Guardar resultado
        self.resultados_permutation[nombre_metrica] = resultado

        # Imprimir resultados
        if self.verbose:
            print(f"\nResultados Permutation Test:")
            print(f"  Métrica REAL:              {valor_real:.4f}")
            print(f"  Distribución ALEATORIA:")
            print(f"    Media:                   {media_permuted:.4f}")
            print(f"    Std:                     {std_permuted:.4f}")
            print(f"\n  Z-Score:                   {z_score:.2f}")
            print(f"    (Métrica real está a {abs(z_score):.2f} std de la media aleatoria)")
            print(f"\n  P-Value (one-tailed):      {p_value:.4f}")
            print(f"  P-Value (two-tailed):      {p_value_two_tailed:.4f}")

            print(f"\n{'='*80}")
            print(f"CONCLUSIÓN:")
            print(f"{'='*80}")

            if significativo and valor_real > media_permuted:
                print(f"✓ EL EDGE NO ES POR AZAR")
                print(f"  - Métrica real significativamente MAYOR que aleatoria")
                print(f"  - p-value = {p_value:.4f} < 0.05")
                print(f"  - Hay evidencia estadística de edge genuino")
                if z_score > 3:
                    print(f"  - Z-score = {z_score:.2f} (ALTAMENTE significativo)")
                elif z_score > 2:
                    print(f"  - Z-score = {z_score:.2f} (Significativo)")
                else:
                    print(f"  - Z-score = {z_score:.2f} (Moderadamente significativo)")
            elif significativo and valor_real < media_permuted:
                print(f"✗ EL MODELO ES PEOR QUE ALEATORIO")
                print(f"  - Métrica real significativamente MENOR que aleatoria")
                print(f"  - p-value = {p_value:.4f} < 0.05")
                print(f"  - Hay evidencia de que el modelo destruye valor")
            else:
                print(f"⚠ NO SE PUEDE DISTINGUIR DE AZAR")
                print(f"  - p-value = {p_value:.4f} >= 0.05")
                print(f"  - No hay evidencia estadística suficiente")
                print(f"  - El resultado podría ser simplemente suerte/ruido")
                print(f"  - NO operar este sistema")

        return resultado

    def permutation_test_todas_metricas(
        self,
        periodos_anuales: int = 252
    ) -> pd.DataFrame:
        """
        Ejecuta permutation test para todas las métricas estándar.

        Métricas incluidas:
        - Information Coefficient (IC)
        - Sharpe Ratio
        - Return Total
        - Profit Factor
        - Win Rate

        Parameters:
        -----------
        periodos_anuales : int
            Períodos de trading por año (default: 252)

        Returns:
        --------
        df_resumen : pd.DataFrame
            Resumen de todos los permutation tests
        """
        if self.verbose:
            print("\n" + "="*80)
            print("PERMUTATION TEST PARA TODAS LAS MÉTRICAS")
            print("="*80)

        # 1. Information Coefficient
        self.permutation_test_metrica(
            metrica_func=self._calcular_ic,
            nombre_metrica="Information Coefficient"
        )

        # 2. Sharpe Ratio
        self.permutation_test_metrica(
            metrica_func=self._calcular_sharpe,
            nombre_metrica="Sharpe Ratio",
            periodos_anuales=periodos_anuales
        )

        # 3. Return Total
        self.permutation_test_metrica(
            metrica_func=self._calcular_return_total,
            nombre_metrica="Return Total"
        )

        # 4. Profit Factor
        self.permutation_test_metrica(
            metrica_func=self._calcular_profit_factor,
            nombre_metrica="Profit Factor"
        )

        # 5. Win Rate
        self.permutation_test_metrica(
            metrica_func=self._calcular_win_rate,
            nombre_metrica="Win Rate"
        )

        # Crear DataFrame resumen
        df_resumen = self._crear_resumen()

        if self.verbose:
            print("\n" + "="*80)
            print("RESUMEN DE TODOS LOS PERMUTATION TESTS")
            print("="*80)
            print(df_resumen.to_string(index=False))

        return df_resumen

    def _crear_resumen(self) -> pd.DataFrame:
        """
        Crea DataFrame resumen con todos los permutation tests.

        Returns:
        --------
        df_resumen : pd.DataFrame
            Resumen de permutation tests
        """
        filas = []

        for nombre, resultado in self.resultados_permutation.items():
            # Determinar interpretación
            if resultado['significativo'] and resultado['valor_real'] > resultado['media_permuted']:
                interpretacion = "✓ EDGE REAL"
            elif resultado['significativo'] and resultado['valor_real'] < resultado['media_permuted']:
                interpretacion = "✗ PEOR QUE AZAR"
            else:
                interpretacion = "⚠ NO DISTINGUIBLE"

            fila = {
                'Métrica': nombre,
                'Valor Real': resultado['valor_real'],
                'Media Aleatoria': resultado['media_permuted'],
                'Std Aleatoria': resultado['std_permuted'],
                'Z-Score': resultado['z_score'],
                'P-Value': resultado['p_value'],
                'Significativo': '✓' if resultado['significativo'] else '✗',
                'Interpretación': interpretacion
            }
            filas.append(fila)

        df_resumen = pd.DataFrame(filas)
        return df_resumen

    def visualizar_distribucion(
        self,
        nombre_metrica: str,
        guardar: bool = False,
        ruta_salida: Optional[str] = None
    ):
        """
        Visualiza la distribución permutada vs valor real.

        Parameters:
        -----------
        nombre_metrica : str
            Nombre de la métrica a visualizar
        guardar : bool
            Si True, guarda la figura
        ruta_salida : str, optional
            Ruta para guardar la figura
        """
        if nombre_metrica not in self.resultados_permutation:
            print(f"✗ Métrica '{nombre_metrica}' no encontrada.")
            print(f"  Métricas disponibles: {list(self.resultados_permutation.keys())}")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("✗ matplotlib no instalado. Instalar con: pip install matplotlib")
            return

        resultado = self.resultados_permutation[nombre_metrica]
        valores_permuted = resultado['valores_permuted']
        valor_real = resultado['valor_real']
        media_permuted = resultado['media_permuted']
        std_permuted = resultado['std_permuted']
        z_score = resultado['z_score']
        p_value = resultado['p_value']

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))

        # Histograma de distribución aleatoria
        ax.hist(valores_permuted, bins=50, alpha=0.7, color='gray',
                edgecolor='black', label='Distribución Aleatoria (Permutada)')

        # Línea de valor real
        ax.axvline(valor_real, color='red', linestyle='--', linewidth=3,
                   label=f'Valor REAL: {valor_real:.4f}')

        # Línea de media aleatoria
        ax.axvline(media_permuted, color='blue', linestyle='--', linewidth=2,
                   label=f'Media Aleatoria: {media_permuted:.4f}')

        # Líneas de ±1, ±2, ±3 std
        for i in [1, 2, 3]:
            ax.axvline(media_permuted + i*std_permuted, color='green',
                       linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(media_permuted - i*std_permuted, color='green',
                       linestyle=':', linewidth=1, alpha=0.5)

        # Etiquetas
        ax.set_xlabel(nombre_metrica, fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        ax.set_title(
            f'Permutation Test: {nombre_metrica}\n'
            f'Z-Score = {z_score:.2f}, P-Value = {p_value:.4f}',
            fontsize=14
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Añadir texto con interpretación
        if resultado['significativo'] and valor_real > media_permuted:
            texto = "✓ EDGE REAL\n(No es azar)"
            color_texto = 'green'
        elif resultado['significativo'] and valor_real < media_permuted:
            texto = "✗ PEOR QUE AZAR"
            color_texto = 'red'
        else:
            texto = "⚠ NO DISTINGUIBLE\n(Posible azar)"
            color_texto = 'orange'

        ax.text(
            0.98, 0.98, texto,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=color_texto, alpha=0.3)
        )

        plt.tight_layout()

        if guardar:
            if ruta_salida is None:
                ruta_salida = f"permutation_{nombre_metrica.replace(' ', '_')}.png"
            plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
            print(f"✓ Figura guardada: {ruta_salida}")

        plt.show()


def ejemplo_uso():
    """
    Ejemplo de uso del Permutation Test.
    """
    print("="*80)
    print("EJEMPLO: Permutation Test - ¿El Edge es Real o por Azar?")
    print("="*80)

    # Simular datos
    np.random.seed(42)
    n_obs = 1000

    # Escenario 1: Modelo con edge real (IC ~ 0.05)
    print("\n--- ESCENARIO 1: Modelo con Edge Real ---")

    # Retornos reales
    y_true = np.random.normal(loc=0, scale=0.02, size=n_obs)

    # Predicciones con correlación real (edge)
    ruido = np.random.normal(loc=0, scale=0.02, size=n_obs)
    y_pred = 0.3 * y_true + 0.7 * ruido  # IC esperado ~ 0.3

    # Ejecutar permutation test
    perm_test = PermutationTest(
        y_true=y_true,
        y_pred=y_pred,
        n_permutations=10000,
        verbose=True
    )

    df_resumen = perm_test.permutation_test_todas_metricas(periodos_anuales=252)

    # Visualizar IC (opcional)
    # perm_test.visualizar_distribucion('Information Coefficient', guardar=True)

    print("\n" + "="*80)

    # Escenario 2: Modelo aleatorio (sin edge)
    print("\n--- ESCENARIO 2: Modelo Aleatorio (Sin Edge) ---")

    # Predicciones completamente aleatorias (sin relación con y_true)
    y_pred_random = np.random.normal(loc=0, scale=0.02, size=n_obs)

    perm_test_random = PermutationTest(
        y_true=y_true,
        y_pred=y_pred_random,
        n_permutations=10000,
        verbose=True
    )

    df_resumen_random = perm_test_random.permutation_test_todas_metricas(
        periodos_anuales=252
    )


def permutation_test_desde_csv(
    ruta_csv: str,
    columna_y_true: str = 'y_true',
    columna_y_pred: str = 'y_pred',
    periodos_anuales: int = 252
) -> pd.DataFrame:
    """
    Ejecuta permutation test desde archivo CSV.

    Parameters:
    -----------
    ruta_csv : str
        Ruta al CSV con resultados
    columna_y_true : str
        Nombre de la columna con retornos reales
    columna_y_pred : str
        Nombre de la columna con predicciones
    periodos_anuales : int
        Períodos de trading por año

    Returns:
    --------
    df_resumen : pd.DataFrame
        Resumen de permutation tests
    """
    # Cargar datos
    df = pd.read_csv(ruta_csv)

    if columna_y_true not in df.columns or columna_y_pred not in df.columns:
        print(f"✗ Columnas no encontradas en CSV")
        print(f"  Columnas disponibles: {df.columns.tolist()}")
        return pd.DataFrame()

    y_true = df[columna_y_true].values
    y_pred = df[columna_y_pred].values

    # Ejecutar permutation test
    perm_test = PermutationTest(
        y_true=y_true,
        y_pred=y_pred,
        n_permutations=10000,
        verbose=True
    )

    df_resumen = perm_test.permutation_test_todas_metricas(
        periodos_anuales=periodos_anuales
    )

    return df_resumen


if __name__ == "__main__":
    """
    Ejecutar Permutation Test.

    OBJETIVO:
    Determinar si el edge del modelo es real o simplemente suerte/azar.

    INTERPRETACIÓN:
    - Si p-value < 0.05 y métrica real > aleatoria → Edge real
    - Si p-value >= 0.05 → No distinguible de azar (NO operar)
    """
    ejemplo_uso()
