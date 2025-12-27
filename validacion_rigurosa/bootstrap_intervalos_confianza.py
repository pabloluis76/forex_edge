"""
Bootstrap para Intervalos de Confianza

CUANTIFICAR INCERTIDUMBRE:

1. Resamplear N trades con reemplazo
2. Calcular métrica (Sharpe, PF, etc.)
3. Repetir 10,000 veces
4. Obtener distribución

EJEMPLO:

Sharpe calculado: 0.85

Bootstrap:
- Media: 0.85
- Std: 0.20
- IC 95%: [0.45, 1.25]

INTERPRETACIÓN:

Si IC 95% incluye 0 → No significativo
Si IC 95% es [0.45, 1.25] → Probablemente rentable pero con incertidumbre

Author: Sistema de Edge-Finding Forex
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

# Importar constantes centralizadas
sys.path.append(str(Path(__file__).parent.parent))
from constants import EPSILON


class BootstrapIntervalosConfianza:
    """
    Bootstrap para calcular intervalos de confianza en métricas de trading.

    Usa resampling con reemplazo para cuantificar incertidumbre.
    """

    def __init__(
        self,
        retornos: np.ndarray,
        n_bootstrap: int = 10000,
        nivel_confianza: float = 0.95,
        seed: Optional[int] = 42,
        verbose: bool = True
    ):
        """
        Inicializa el sistema de Bootstrap.

        Parameters:
        -----------
        retornos : np.ndarray
            Array de retornos de trades o períodos
        n_bootstrap : int
            Número de iteraciones bootstrap (default: 10,000)
        nivel_confianza : float
            Nivel de confianza para intervalos (default: 0.95)
        seed : int, optional
            Semilla para reproducibilidad
        verbose : bool
            Imprimir resultados detallados
        """
        self.retornos = np.array(retornos)
        self.n_bootstrap = n_bootstrap
        self.nivel_confianza = nivel_confianza
        self.seed = seed
        self.verbose = verbose

        # Eliminar NaN
        self.retornos = self.retornos[~np.isnan(self.retornos)]
        self.n_trades = len(self.retornos)

        if self.seed is not None:
            np.random.seed(self.seed)

        # Resultados
        self.resultados_bootstrap: Dict[str, Dict] = {}

        if self.verbose:
            print("="*80)
            print("BOOTSTRAP PARA INTERVALOS DE CONFIANZA")
            print("="*80)
            print(f"N trades/períodos: {self.n_trades}")
            print(f"N bootstrap: {self.n_bootstrap:,}")
            print(f"Nivel confianza: {self.nivel_confianza*100:.0f}%")
            print("="*80)

    def _resample(self) -> np.ndarray:
        """
        Resamplea retornos con reemplazo.

        Returns:
        --------
        retornos_resampled : np.ndarray
            Array de retornos resampled (mismo tamaño que original)
        """
        indices = np.random.randint(0, self.n_trades, size=self.n_trades)
        return self.retornos[indices]

    def _calcular_sharpe(self, retornos: np.ndarray, periodos_anuales: int = 252) -> float:
        """
        Calcula Sharpe Ratio.

        Parameters:
        -----------
        retornos : np.ndarray
            Array de retornos
        periodos_anuales : int
            Períodos de trading por año (252 días, 252*24 horas, etc.)

        Returns:
        --------
        sharpe : float
            Sharpe Ratio anualizado
        """
        if len(retornos) == 0:
            return np.nan

        media = np.mean(retornos)
        std = np.std(retornos, ddof=1)

        # Usar np.isclose para comparación de floats
        if np.isclose(std, 0) or np.isnan(std):
            return 0.0

        sharpe = (media / std) * np.sqrt(periodos_anuales)
        return sharpe

    def _calcular_profit_factor(self, retornos: np.ndarray) -> float:
        """
        Calcula Profit Factor.

        PF = Suma(Ganancias) / Suma(|Pérdidas|)

        Parameters:
        -----------
        retornos : np.ndarray
            Array de retornos

        Returns:
        --------
        pf : float
            Profit Factor
        """
        ganancias = retornos[retornos > 0]
        perdidas = retornos[retornos < 0]

        suma_ganancias = np.sum(ganancias) if len(ganancias) > 0 else 0
        suma_perdidas = np.abs(np.sum(perdidas)) if len(perdidas) > 0 else 0

        # CRÍTICO CORREGIDO: No retornar np.inf, usar valor finito alto
        if suma_perdidas == 0:
            return 999.99 if suma_ganancias > 0 else 0.0

        pf = suma_ganancias / suma_perdidas
        return pf

    def _calcular_win_rate(self, retornos: np.ndarray) -> float:
        """
        Calcula Win Rate.

        WR = N(retornos > 0) / N(retornos)

        Parameters:
        -----------
        retornos : np.ndarray
            Array de retornos

        Returns:
        --------
        wr : float
            Win Rate (0 a 1)
        """
        if len(retornos) == 0:
            return np.nan

        n_wins = np.sum(retornos > 0)
        wr = n_wins / len(retornos)
        return wr

    def _calcular_avg_win_loss_ratio(self, retornos: np.ndarray) -> float:
        """
        Calcula Average Win / Average Loss Ratio.

        Parameters:
        -----------
        retornos : np.ndarray
            Array de retornos

        Returns:
        --------
        ratio : float
            Avg Win / Avg Loss
        """
        ganancias = retornos[retornos > 0]
        perdidas = retornos[retornos < 0]

        avg_win = np.mean(ganancias) if len(ganancias) > 0 else 0
        avg_loss = np.abs(np.mean(perdidas)) if len(perdidas) > 0 else 0

        # CRÍTICO CORREGIDO: No retornar np.inf, usar valor finito alto
        if avg_loss == 0:
            return 999.99 if avg_win > 0 else 0.0

        ratio = avg_win / avg_loss
        return ratio

    def _calcular_max_drawdown(self, retornos: np.ndarray) -> float:
        """
        Calcula Maximum Drawdown.

        Parameters:
        -----------
        retornos : np.ndarray
            Array de retornos

        Returns:
        --------
        max_dd : float
            Maximum Drawdown (valor positivo)
        """
        cumret = np.cumprod(1 + retornos)
        running_max = np.maximum.accumulate(cumret)

        # CRÍTICO #2 CORREGIDO: Evitar división por cero si running_max es 0
        drawdown = (cumret - running_max) / np.maximum(running_max, EPSILON)
        max_dd = np.abs(np.min(drawdown))
        return max_dd

    def _calcular_return_total(self, retornos: np.ndarray) -> float:
        """
        Calcula Return Total.

        Parameters:
        -----------
        retornos : np.ndarray
            Array de retornos

        Returns:
        --------
        ret_total : float
            Return total (porcentaje)
        """
        ret_total = np.prod(1 + retornos) - 1
        return ret_total * 100  # En porcentaje

    def _calcular_return_anualizado(
        self,
        retornos: np.ndarray,
        periodos_anuales: int = 252
    ) -> float:
        """
        Calcula Return Anualizado.

        Parameters:
        -----------
        retornos : np.ndarray
            Array de retornos
        periodos_anuales : int
            Períodos de trading por año

        Returns:
        --------
        ret_anual : float
            Return anualizado (porcentaje)
        """
        ret_medio = np.mean(retornos)
        ret_anual = ret_medio * periodos_anuales * 100  # En porcentaje
        return ret_anual

    def bootstrap_metrica(
        self,
        metrica_func: Callable[[np.ndarray], float],
        nombre_metrica: str,
        **kwargs
    ) -> Dict:
        """
        Ejecuta bootstrap para una métrica específica.

        PROCESO:
        1. Resamplear N trades con reemplazo
        2. Calcular métrica
        3. Repetir n_bootstrap veces
        4. Obtener distribución

        Parameters:
        -----------
        metrica_func : Callable
            Función que calcula la métrica: f(retornos) -> float
        nombre_metrica : str
            Nombre de la métrica (para reportes)
        **kwargs : dict
            Argumentos adicionales para metrica_func

        Returns:
        --------
        resultado : Dict
            - valor_original: Métrica calculada en datos originales
            - valores_bootstrap: Array con n_bootstrap valores
            - media: Media de distribución bootstrap
            - std: Desviación estándar
            - ic_lower: Límite inferior IC
            - ic_upper: Límite superior IC
            - significativo: True si IC no incluye 0
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Bootstrap: {nombre_metrica}")
            print(f"{'='*80}")

        # Calcular métrica en datos originales
        valor_original = metrica_func(self.retornos, **kwargs)

        if self.verbose:
            print(f"Valor original: {valor_original:.4f}")
            print(f"Ejecutando {self.n_bootstrap:,} iteraciones bootstrap...")

        # Bootstrap
        valores_bootstrap = np.zeros(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            retornos_resampled = self._resample()
            valores_bootstrap[i] = metrica_func(retornos_resampled, **kwargs)

        # Calcular estadísticas
        media = np.mean(valores_bootstrap)
        std = np.std(valores_bootstrap, ddof=1)

        # Intervalos de confianza (percentiles)
        alpha = 1 - self.nivel_confianza
        ic_lower = np.percentile(valores_bootstrap, alpha/2 * 100)
        ic_upper = np.percentile(valores_bootstrap, (1 - alpha/2) * 100)

        # Verificar significancia
        # Para métricas donde 0 es neutral (Sharpe, IC, etc.)
        # Si el IC incluye 0, no es significativo
        significativo = not (ic_lower <= 0 <= ic_upper)

        resultado = {
            'nombre': nombre_metrica,
            'valor_original': valor_original,
            'valores_bootstrap': valores_bootstrap,
            'media': media,
            'std': std,
            'ic_lower': ic_lower,
            'ic_upper': ic_upper,
            'nivel_confianza': self.nivel_confianza,
            'significativo': significativo
        }

        # Guardar resultado
        self.resultados_bootstrap[nombre_metrica] = resultado

        # Imprimir resultados
        if self.verbose:
            print(f"\nResultados Bootstrap:")
            print(f"  Media:     {media:.4f}")
            print(f"  Std:       {std:.4f}")
            print(f"  IC {self.nivel_confianza*100:.0f}%:   [{ic_lower:.4f}, {ic_upper:.4f}]")
            print(f"\nInterpretación:")
            if significativo:
                if media > 0:
                    print(f"  ✓ SIGNIFICATIVO POSITIVO")
                    print(f"    El IC 95% NO incluye 0")
                    print(f"    Evidencia estadística de rentabilidad")
                else:
                    print(f"  ✗ SIGNIFICATIVO NEGATIVO")
                    print(f"    El IC 95% NO incluye 0")
                    print(f"    Evidencia de pérdidas consistentes")
            else:
                print(f"  ⚠ NO SIGNIFICATIVO")
                print(f"    El IC 95% INCLUYE 0")
                print(f"    No hay evidencia estadística suficiente")
                print(f"    Podría ser ruido aleatorio")

        return resultado

    def bootstrap_todas_metricas(
        self,
        periodos_anuales: int = 252
    ) -> pd.DataFrame:
        """
        Ejecuta bootstrap para todas las métricas estándar.

        Métricas incluidas:
        - Sharpe Ratio
        - Profit Factor
        - Win Rate
        - Avg Win/Loss Ratio
        - Max Drawdown
        - Return Anualizado

        Parameters:
        -----------
        periodos_anuales : int
            Períodos de trading por año (default: 252 para diario)

        Returns:
        --------
        df_resumen : pd.DataFrame
            DataFrame con resumen de todas las métricas
        """
        if self.verbose:
            print("\n" + "="*80)
            print("BOOTSTRAP PARA TODAS LAS MÉTRICAS")
            print("="*80)

        # 1. Sharpe Ratio
        self.bootstrap_metrica(
            metrica_func=self._calcular_sharpe,
            nombre_metrica="Sharpe Ratio",
            periodos_anuales=periodos_anuales
        )

        # 2. Profit Factor
        self.bootstrap_metrica(
            metrica_func=self._calcular_profit_factor,
            nombre_metrica="Profit Factor"
        )

        # 3. Win Rate
        self.bootstrap_metrica(
            metrica_func=self._calcular_win_rate,
            nombre_metrica="Win Rate"
        )

        # 4. Avg Win/Loss Ratio
        self.bootstrap_metrica(
            metrica_func=self._calcular_avg_win_loss_ratio,
            nombre_metrica="Avg Win/Loss Ratio"
        )

        # 5. Max Drawdown
        self.bootstrap_metrica(
            metrica_func=self._calcular_max_drawdown,
            nombre_metrica="Max Drawdown"
        )

        # 6. Return Anualizado
        self.bootstrap_metrica(
            metrica_func=self._calcular_return_anualizado,
            nombre_metrica="Return Anualizado",
            periodos_anuales=periodos_anuales
        )

        # Crear DataFrame resumen
        df_resumen = self._crear_resumen()

        if self.verbose:
            print("\n" + "="*80)
            print("RESUMEN DE TODAS LAS MÉTRICAS")
            print("="*80)
            print(df_resumen.to_string(index=False))

        return df_resumen

    def _crear_resumen(self) -> pd.DataFrame:
        """
        Crea DataFrame resumen con todas las métricas.

        Returns:
        --------
        df_resumen : pd.DataFrame
            Resumen de métricas bootstrap
        """
        filas = []

        for nombre, resultado in self.resultados_bootstrap.items():
            fila = {
                'Métrica': nombre,
                'Valor Original': resultado['valor_original'],
                'Media Bootstrap': resultado['media'],
                'Std Bootstrap': resultado['std'],
                f'IC {resultado["nivel_confianza"]*100:.0f}% Lower': resultado['ic_lower'],
                f'IC {resultado["nivel_confianza"]*100:.0f}% Upper': resultado['ic_upper'],
                'Significativo': '✓' if resultado['significativo'] else '✗'
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
        Visualiza la distribución bootstrap de una métrica.

        Parameters:
        -----------
        nombre_metrica : str
            Nombre de la métrica a visualizar
        guardar : bool
            Si True, guarda la figura
        ruta_salida : str, optional
            Ruta para guardar la figura
        """
        if nombre_metrica not in self.resultados_bootstrap:
            print(f"✗ Métrica '{nombre_metrica}' no encontrada.")
            print(f"  Métricas disponibles: {list(self.resultados_bootstrap.keys())}")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("✗ matplotlib no instalado. Instalar con: pip install matplotlib")
            return

        resultado = self.resultados_bootstrap[nombre_metrica]
        valores = resultado['valores_bootstrap']
        valor_original = resultado['valor_original']
        ic_lower = resultado['ic_lower']
        ic_upper = resultado['ic_upper']

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))

        # Histograma
        ax.hist(valores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

        # Líneas de referencia
        ax.axvline(valor_original, color='red', linestyle='--', linewidth=2,
                   label=f'Valor Original: {valor_original:.4f}')
        ax.axvline(ic_lower, color='green', linestyle='--', linewidth=2,
                   label=f'IC {resultado["nivel_confianza"]*100:.0f}% Lower: {ic_lower:.4f}')
        ax.axvline(ic_upper, color='green', linestyle='--', linewidth=2,
                   label=f'IC {resultado["nivel_confianza"]*100:.0f}% Upper: {ic_upper:.4f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        # Etiquetas
        ax.set_xlabel(nombre_metrica, fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        ax.set_title(f'Distribución Bootstrap: {nombre_metrica}\n'
                     f'(N={self.n_bootstrap:,} iteraciones)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if guardar:
            if ruta_salida is None:
                ruta_salida = f"bootstrap_{nombre_metrica.replace(' ', '_')}.png"
            plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
            print(f"✓ Figura guardada: {ruta_salida}")

        plt.show()


def ejemplo_uso():
    """
    Ejemplo de uso del sistema Bootstrap.
    """
    print("="*80)
    print("EJEMPLO: Bootstrap para Intervalos de Confianza")
    print("="*80)

    # Simular retornos de trading
    np.random.seed(42)

    # Escenario 1: Sistema con edge positivo (Sharpe ~0.85)
    n_trades = 500
    retornos = np.random.normal(loc=0.0015, scale=0.02, size=n_trades)

    print(f"\nSimulando {n_trades} trades...")
    print(f"Media: {np.mean(retornos):.4f}")
    print(f"Std: {np.std(retornos):.4f}")

    # Inicializar Bootstrap
    bootstrap = BootstrapIntervalosConfianza(
        retornos=retornos,
        n_bootstrap=10000,
        nivel_confianza=0.95,
        verbose=True
    )

    # Ejecutar bootstrap para todas las métricas
    df_resumen = bootstrap.bootstrap_todas_metricas(periodos_anuales=252)

    print("\n" + "="*80)
    print("INTERPRETACIÓN FINAL")
    print("="*80)

    # Analizar Sharpe Ratio
    sharpe_resultado = bootstrap.resultados_bootstrap['Sharpe Ratio']
    sharpe_ic = (sharpe_resultado['ic_lower'], sharpe_resultado['ic_upper'])

    print(f"\nSharpe Ratio:")
    print(f"  Valor: {sharpe_resultado['valor_original']:.2f}")
    print(f"  IC 95%: [{sharpe_ic[0]:.2f}, {sharpe_ic[1]:.2f}]")

    if sharpe_resultado['significativo']:
        print(f"\n  ✓ El IC 95% NO incluye 0")
        print(f"  ✓ Sistema probablemente rentable")
        print(f"  ✓ Pero con incertidumbre: ±{sharpe_resultado['std']:.2f}")
    else:
        print(f"\n  ✗ El IC 95% INCLUYE 0")
        print(f"  ✗ No hay evidencia estadística de rentabilidad")
        print(f"  ✗ Podría ser ruido aleatorio")

    # Visualizar (opcional)
    # bootstrap.visualizar_distribucion('Sharpe Ratio', guardar=True)


def bootstrap_desde_csv(
    ruta_csv: str,
    columna_retornos: str = 'retornos',
    periodos_anuales: int = 252
) -> pd.DataFrame:
    """
    Ejecuta bootstrap desde archivo CSV.

    Parameters:
    -----------
    ruta_csv : str
        Ruta al CSV con retornos
    columna_retornos : str
        Nombre de la columna con retornos
    periodos_anuales : int
        Períodos de trading por año

    Returns:
    --------
    df_resumen : pd.DataFrame
        Resumen de métricas bootstrap
    """
    # Cargar datos
    df = pd.read_csv(ruta_csv)

    if columna_retornos not in df.columns:
        print(f"✗ Columna '{columna_retornos}' no encontrada en CSV")
        print(f"  Columnas disponibles: {df.columns.tolist()}")
        return pd.DataFrame()

    retornos = df[columna_retornos].values

    # Ejecutar bootstrap
    bootstrap = BootstrapIntervalosConfianza(
        retornos=retornos,
        n_bootstrap=10000,
        nivel_confianza=0.95,
        verbose=True
    )

    df_resumen = bootstrap.bootstrap_todas_metricas(
        periodos_anuales=periodos_anuales
    )

    return df_resumen


if __name__ == "__main__":
    """
    Ejecutar Bootstrap para Intervalos de Confianza.

    CASOS DE USO:
    1. Cuantificar incertidumbre en métricas de trading
    2. Verificar significancia estadística de Sharpe Ratio
    3. Determinar si resultados son reales o ruido aleatorio
    """
    ejemplo_uso()
