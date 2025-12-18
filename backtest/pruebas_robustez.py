"""
Pruebas de Robustez

Valida la robustez de la estrategia mediante:
1. Análisis de Sensibilidad a Parámetros
2. Bootstrap Confidence Intervals
3. Permutation Test
4. Consistencia Entre Pares

Author: Sistema de Edge-Finding Forex
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


class PruebasRobustez:
    """
    Ejecuta pruebas de robustez sobre resultados de backtest.

    Valida que la estrategia sea robusta a cambios en parámetros,
    estadísticamente significativa, y consistente entre pares.
    """

    def __init__(
        self,
        df_trades: pd.DataFrame,
        df_equity: pd.DataFrame,
        verbose: bool = True
    ):
        """
        Inicializa pruebas de robustez.

        Parameters:
        -----------
        df_trades : pd.DataFrame
            Historial de trades del backtest
        df_equity : pd.DataFrame
            Curva de equity del backtest
        verbose : bool
            Imprimir resultados
        """
        self.df_trades = df_trades.copy()
        self.df_equity = df_equity.copy()
        self.verbose = verbose

        # Calcular retornos de trades
        if 'return_pct' in self.df_trades.columns:
            self.returns = self.df_trades['return_pct'].values / 100
        elif 'net_pnl' in self.df_trades.columns and 'capital_riesgo' in self.df_trades.columns:
            self.returns = (self.df_trades['net_pnl'] / self.df_trades['capital_riesgo']).values
        else:
            raise ValueError("df_trades debe tener 'return_pct' o 'net_pnl'+'capital_riesgo'")

        if self.verbose:
            print("="*80)
            print("PRUEBAS DE ROBUSTEZ")
            print("="*80)
            print(f"Trades analizados: {len(self.df_trades)}")
            print(f"Período: {self.df_trades['entry_time'].min()} → {self.df_trades['exit_time'].max()}")
            print("="*80)

    # ========================================================================
    # 8.1 SENSIBILIDAD A PARÁMETROS
    # ========================================================================

    def analisis_sensibilidad(
        self,
        funcion_backtest: Callable,
        parametro_nombre: str,
        valores_parametro: List[float],
        valor_base: float,
        otros_kwargs: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Ejecuta análisis de sensibilidad variando un parámetro.

        Parameters:
        -----------
        funcion_backtest : Callable
            Función que ejecuta el backtest y retorna métricas
            Debe aceptar: funcion_backtest(**{parametro_nombre: valor, **otros_kwargs})
            Debe retornar: Dict con al menos {'sharpe_train', 'sharpe_wf'}
        parametro_nombre : str
            Nombre del parámetro a variar
        valores_parametro : List[float]
            Valores a probar
        valor_base : float
            Valor base del parámetro
        otros_kwargs : Dict, optional
            Otros argumentos para la función de backtest

        Returns:
        --------
        df_sensibilidad : pd.DataFrame
            Resultados del análisis de sensibilidad
        """
        if otros_kwargs is None:
            otros_kwargs = {}

        resultados = []

        for valor in valores_parametro:
            # Ejecutar backtest con este valor de parámetro
            kwargs = {parametro_nombre: valor, **otros_kwargs}
            metricas = funcion_backtest(**kwargs)

            # Calcular diferencia respecto a WF
            sharpe_train = metricas.get('sharpe_train', 0)
            sharpe_wf = metricas.get('sharpe_wf', 0)

            if sharpe_train != 0:
                diferencia_pct = ((sharpe_wf - sharpe_train) / sharpe_train) * 100
            else:
                diferencia_pct = 0

            # Determinar si es robusto
            # Criterio: diferencia < 30% y curva suave
            es_robusto = abs(diferencia_pct) < 30

            es_base = (valor == valor_base)

            resultados.append({
                'parametro': parametro_nombre,
                'valor': valor,
                'sharpe_train': sharpe_train,
                'sharpe_wf': sharpe_wf,
                'diferencia_pct': diferencia_pct,
                'es_robusto': es_robusto,
                'es_base': es_base
            })

        df_sens = pd.DataFrame(resultados)

        if self.verbose:
            self._imprimir_sensibilidad(df_sens, parametro_nombre, valor_base)

        return df_sens

    def _imprimir_sensibilidad(self, df: pd.DataFrame, param_nombre: str, valor_base: float):
        """Imprime tabla de sensibilidad."""
        print(f"\n{'='*80}")
        print("ANÁLISIS DE SENSIBILIDAD")
        print(f"{'='*80}")
        print(f"\nPARÁMETRO: {param_nombre}")
        print(f"Valor base: {valor_base}")
        print()
        print(f"{'Valor':<10} │ {'Sharpe (Train)':<15} │ {'Sharpe (WF)':<12} │ {'Diferencia':<11} │ {'Robusto?'}")
        print("─"*10 + "┼" + "─"*16 + "┼" + "─"*13 + "┼" + "─"*12 + "┼" + "─"*15)

        for _, row in df.iterrows():
            base_marker = " (base)" if row['es_base'] else ""
            robusto_icon = "✓" if row['es_robusto'] else "⚠️"

            print(
                f"{row['valor']:<10.2f} │ "
                f"{row['sharpe_train']:<15.2f} │ "
                f"{row['sharpe_wf']:<12.2f} │ "
                f"{row['diferencia_pct']:+10.0f}% │ "
                f"{robusto_icon}{base_marker}"
            )

        # Análisis de curva
        sharpes_wf = df['sharpe_wf'].values
        variacion = np.std(sharpes_wf) / np.mean(sharpes_wf) if np.mean(sharpes_wf) != 0 else 0

        print()
        if variacion < 0.15:
            print(f"RESULTADO: Curva suave (CV={variacion:.1%}), robusto a variaciones en {param_nombre}")
        else:
            print(f"RESULTADO: Curva variable (CV={variacion:.1%}), sensible a {param_nombre}")
        print(f"{'='*80}\n")

    # ========================================================================
    # 8.2 BOOTSTRAP CONFIDENCE INTERVALS
    # ========================================================================

    def bootstrap_confidence_intervals(
        self,
        n_iterations: int = 10000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = 42
    ) -> Dict:
        """
        Calcula intervalos de confianza mediante bootstrap.

        Parameters:
        -----------
        n_iterations : int
            Número de iteraciones de bootstrap (default: 10,000)
        confidence_level : float
            Nivel de confianza (default: 0.95 → 95%)
        random_seed : int, optional
            Semilla para reproducibilidad

        Returns:
        --------
        resultados : Dict
            Diccionario con métricas y sus intervalos de confianza
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        n_trades = len(self.returns)

        # Arrays para almacenar métricas bootstrapped
        sharpe_boot = np.zeros(n_iterations)
        profit_factor_boot = np.zeros(n_iterations)
        win_rate_boot = np.zeros(n_iterations)
        max_dd_boot = np.zeros(n_iterations)
        return_anual_boot = np.zeros(n_iterations)

        if self.verbose:
            print(f"\n{'='*80}")
            print("BOOTSTRAP CONFIDENCE INTERVALS")
            print(f"{'='*80}")
            print(f"Trades: {n_trades}")
            print(f"Iteraciones: {n_iterations:,}")
            print(f"Nivel de confianza: {confidence_level:.0%}")
            print("\nResampling...", end="", flush=True)

        # Bootstrap
        for i in range(n_iterations):
            if self.verbose and i % 1000 == 0 and i > 0:
                print(f" {i:,}", end="", flush=True)

            # Resamplear con reemplazo
            indices = np.random.choice(n_trades, size=n_trades, replace=True)
            returns_boot = self.returns[indices]

            # Calcular métricas
            sharpe_boot[i] = self._calcular_sharpe(returns_boot)
            profit_factor_boot[i] = self._calcular_profit_factor(returns_boot)
            win_rate_boot[i] = self._calcular_win_rate(returns_boot)
            max_dd_boot[i] = self._calcular_max_drawdown(returns_boot)
            return_anual_boot[i] = self._calcular_return_anual(returns_boot, n_trades)

        if self.verbose:
            print(" Completado!\n")

        # Calcular valores reales
        sharpe_real = self._calcular_sharpe(self.returns)
        pf_real = self._calcular_profit_factor(self.returns)
        wr_real = self._calcular_win_rate(self.returns)
        dd_real = self._calcular_max_drawdown(self.returns)
        ret_real = self._calcular_return_anual(self.returns, n_trades)

        # Calcular intervalos de confianza
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        resultados = {
            'sharpe': {
                'real': sharpe_real,
                'mean': np.mean(sharpe_boot),
                'std': np.std(sharpe_boot),
                'ci_lower': np.percentile(sharpe_boot, lower_percentile),
                'ci_upper': np.percentile(sharpe_boot, upper_percentile)
            },
            'profit_factor': {
                'real': pf_real,
                'mean': np.mean(profit_factor_boot),
                'std': np.std(profit_factor_boot),
                'ci_lower': np.percentile(profit_factor_boot, lower_percentile),
                'ci_upper': np.percentile(profit_factor_boot, upper_percentile)
            },
            'win_rate': {
                'real': wr_real,
                'mean': np.mean(win_rate_boot),
                'std': np.std(win_rate_boot),
                'ci_lower': np.percentile(win_rate_boot, lower_percentile),
                'ci_upper': np.percentile(win_rate_boot, upper_percentile)
            },
            'max_drawdown': {
                'real': dd_real,
                'mean': np.mean(max_dd_boot),
                'std': np.std(max_dd_boot),
                'ci_lower': np.percentile(max_dd_boot, lower_percentile),
                'ci_upper': np.percentile(max_dd_boot, upper_percentile)
            },
            'return_anual': {
                'real': ret_real,
                'mean': np.mean(return_anual_boot),
                'std': np.std(return_anual_boot),
                'ci_lower': np.percentile(return_anual_boot, lower_percentile),
                'ci_upper': np.percentile(return_anual_boot, upper_percentile)
            }
        }

        if self.verbose:
            self._imprimir_bootstrap(resultados, confidence_level)

        return resultados

    def _imprimir_bootstrap(self, resultados: Dict, confidence_level: float):
        """Imprime resultados de bootstrap."""
        print(f"{'='*80}")
        print("RESULTADOS BOOTSTRAP")
        print(f"{'='*80}")
        print()
        print(f"{'Métrica':<16} │ {'Valor Real':<11} │ {'Media Boot':<11} │ {'Std':<6} │ {'CI ' + f'{confidence_level:.0%}'}")
        print("─"*16 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*7 + "┼" + "─"*20)

        metricas_formato = {
            'sharpe': ('Sharpe', '{:.2f}', 1),
            'profit_factor': ('Profit Factor', '{:.2f}', 1),
            'win_rate': ('Win Rate', '{:.1f}%', 100),
            'max_drawdown': ('Max Drawdown', '{:.1f}%', 100),
            'return_anual': ('Return (anual)', '{:.1f}%', 100)
        }

        for key, (nombre, fmt, mult) in metricas_formato.items():
            r = resultados[key]
            real_str = fmt.format(r['real'] * mult)
            mean_str = fmt.format(r['mean'] * mult)
            std_str = fmt.format(r['std'] * mult)
            ci_str = f"[{fmt.format(r['ci_lower'] * mult)}, {fmt.format(r['ci_upper'] * mult)}]"

            print(f"{nombre:<16} │ {real_str:<11} │ {mean_str:<11} │ {std_str:<6} │ {ci_str}")

        print()
        print("INTERPRETACIÓN:")
        print()

        # Sharpe
        if resultados['sharpe']['ci_lower'] > 0:
            print("  • Sharpe CI no incluye 0 → SIGNIFICATIVO ✓")
        else:
            print("  • Sharpe CI incluye 0 → NO SIGNIFICATIVO ✗")

        # Profit Factor
        if resultados['profit_factor']['ci_lower'] > 1.0:
            print("  • PF CI > 1.0 → RENTABLE ✓")
        elif resultados['profit_factor']['ci_upper'] > 1.0:
            print("  • PF CI incluye 1.0 marginalmente → CUIDADO ⚠️")
        else:
            print("  • PF CI < 1.0 → NO RENTABLE ✗")

        # Return
        if resultados['return_anual']['ci_lower'] > 0:
            print(f"  • Return CI positivo → PROBABLE RENTABLE ✓")
        else:
            print(f"  • Return CI incluye 0 → INCIERTO ⚠️")

        print()

        # Veredicto
        sharpe_pasa = resultados['sharpe']['ci_lower'] > 0
        if sharpe_pasa:
            print("VEREDICTO: Bootstrap PASA ✓ (CI de Sharpe no incluye 0)")
        else:
            print("VEREDICTO: Bootstrap FALLA ✗ (CI de Sharpe incluye 0)")

        print(f"{'='*80}\n")

    # ========================================================================
    # 8.3 PERMUTATION TEST
    # ========================================================================

    def permutation_test(
        self,
        metrica: str = 'sharpe',
        n_permutations: int = 10000,
        random_seed: Optional[int] = 42
    ) -> Dict:
        """
        Ejecuta permutation test para validar significancia estadística.

        Parameters:
        -----------
        metrica : str
            Métrica a testear ('sharpe', 'profit_factor', 'win_rate')
        n_permutations : int
            Número de permutaciones (default: 10,000)
        random_seed : int, optional
            Semilla para reproducibilidad

        Returns:
        --------
        resultados : Dict
            Diccionario con valor real, distribución nula, y p-value
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Calcular métrica real
        if metrica == 'sharpe':
            valor_real = self._calcular_sharpe(self.returns)
            calc_func = self._calcular_sharpe
        elif metrica == 'profit_factor':
            valor_real = self._calcular_profit_factor(self.returns)
            calc_func = self._calcular_profit_factor
        elif metrica == 'win_rate':
            valor_real = self._calcular_win_rate(self.returns)
            calc_func = self._calcular_win_rate
        else:
            raise ValueError(f"Métrica '{metrica}' no soportada")

        if self.verbose:
            print(f"\n{'='*80}")
            print("PERMUTATION TEST")
            print(f"{'='*80}")
            print(f"Métrica: {metrica}")
            print(f"Valor real: {valor_real:.4f}")
            print(f"Permutaciones: {n_permutations:,}")
            print("\nPermutando...", end="", flush=True)

        # Generar distribución nula mediante permutaciones
        valores_permutados = np.zeros(n_permutations)

        for i in range(n_permutations):
            if self.verbose and i % 1000 == 0 and i > 0:
                print(f" {i:,}", end="", flush=True)

            # Mezclar aleatoriamente los retornos
            returns_permutados = np.random.permutation(self.returns)

            # Calcular métrica
            valores_permutados[i] = calc_func(returns_permutados)

        if self.verbose:
            print(" Completado!\n")

        # Calcular p-value
        # p-value = proporción de permutaciones con métrica >= valor real
        n_extreme = np.sum(valores_permutados >= valor_real)
        p_value = n_extreme / n_permutations

        resultados = {
            'metrica': metrica,
            'valor_real': valor_real,
            'distribucion_nula': {
                'mean': np.mean(valores_permutados),
                'std': np.std(valores_permutados),
                'min': np.min(valores_permutados),
                'max': np.max(valores_permutados)
            },
            'n_extreme': n_extreme,
            'p_value': p_value,
            'valores_permutados': valores_permutados  # Para histograma
        }

        if self.verbose:
            self._imprimir_permutation_test(resultados)

        return resultados

    def _imprimir_permutation_test(self, resultados: Dict):
        """Imprime resultados de permutation test."""
        print(f"{'='*80}")
        print("RESULTADOS PERMUTATION TEST")
        print(f"{'='*80}")
        print()

        # Distribución nula
        nula = resultados['distribucion_nula']
        print(f"Distribución Nula (permutaciones aleatorias):")
        print(f"  Media: {nula['mean']:.4f}")
        print(f"  Std:   {nula['std']:.4f}")
        print(f"  Min:   {nula['min']:.4f}")
        print(f"  Max:   {nula['max']:.4f}")
        print()

        # Histograma simple
        print("DISTRIBUCIÓN:")
        print()
        self._imprimir_histograma(resultados['valores_permutados'], resultados['valor_real'])
        print()

        # Resultado
        print(f"Permutaciones con {resultados['metrica']} ≥ {resultados['valor_real']:.4f}: "
              f"{resultados['n_extreme']} de {len(resultados['valores_permutados']):,}")
        print(f"p-value = {resultados['p_value']:.4f} ({resultados['p_value']*100:.2f}%)")
        print()

        # Interpretación
        print("INTERPRETACIÓN:")
        print()
        if resultados['p_value'] < 0.01:
            print(f"  p-value < 0.01 → ALTAMENTE SIGNIFICATIVO ✓")
            print(f"  La probabilidad de obtener este {resultados['metrica']} por azar es < 1%")
        elif resultados['p_value'] < 0.05:
            print(f"  p-value < 0.05 → SIGNIFICATIVO ✓")
            print(f"  La probabilidad de obtener este {resultados['metrica']} por azar es < 5%")
        elif resultados['p_value'] < 0.10:
            print(f"  p-value < 0.10 → MARGINALMENTE SIGNIFICATIVO ⚠️")
        else:
            print(f"  p-value ≥ 0.10 → NO SIGNIFICATIVO ✗")
            print(f"  Posible resultado aleatorio")

        print()

        # Veredicto
        if resultados['p_value'] < 0.05:
            print("VEREDICTO: Permutation test PASA ✓")
        else:
            print("VEREDICTO: Permutation test FALLA ✗")

        print(f"{'='*80}\n")

    def _imprimir_histograma(self, valores: np.ndarray, valor_real: float, n_bins: int = 30):
        """Imprime histograma ASCII."""
        hist, bin_edges = np.histogram(valores, bins=n_bins)
        max_freq = np.max(hist)

        # Escalar a ancho de 60 caracteres
        ancho = 60

        # Encontrar bin del valor real
        bin_real = np.digitize([valor_real], bin_edges)[0] - 1

        print("Frecuencia")
        print("     │")

        for i in range(len(hist) - 1, -1, -1):
            freq = hist[i]
            bar_len = int((freq / max_freq) * ancho) if max_freq > 0 else 0
            bar = "█" * bar_len

            # Marcar si es el bin del valor real
            if i == bin_real:
                bar = bar + " ← Valor Real"

            if i % 5 == 0:
                print(f"{freq:4.0f} │ {bar}")
            else:
                print(f"     │ {bar}")

        print(f"   0 └{'─' * ancho}")

        # Eje X
        x_labels = [f"{bin_edges[0]:.2f}", f"{bin_edges[len(bin_edges)//2]:.2f}", f"{bin_edges[-1]:.2f}"]
        spacing = ancho // 2
        print(f"      {x_labels[0]:<{spacing}}{x_labels[1]:<{spacing//2}}{x_labels[2]}")

    # ========================================================================
    # 8.4 CONSISTENCIA ENTRE PARES
    # ========================================================================

    def consistencia_entre_pares(self) -> pd.DataFrame:
        """
        Analiza consistencia de resultados entre diferentes pares.

        Returns:
        --------
        df_pares : pd.DataFrame
            Métricas por par
        """
        if 'pair' not in self.df_trades.columns:
            raise ValueError("df_trades debe tener columna 'pair'")

        pares = self.df_trades['pair'].unique()

        resultados = []

        for par in pares:
            df_par = self.df_trades[self.df_trades['pair'] == par]

            if len(df_par) < 10:
                # Muy pocos trades para análisis confiable
                continue

            returns_par = (df_par['net_pnl'] / df_par['capital_riesgo']).values

            # Calcular métricas
            n_trades = len(df_par)
            win_rate = self._calcular_win_rate(returns_par) * 100
            profit_factor = self._calcular_profit_factor(returns_par)
            sharpe = self._calcular_sharpe(returns_par)
            total_return = np.sum(returns_par) * 100

            # Determinar si es consistente
            # Criterios: Sharpe > 0.5, PF > 1.0, WR > 45%
            es_consistente = (sharpe > 0.5) and (profit_factor > 1.0) and (win_rate > 45)

            # Clasificar
            if sharpe > 0.5 and profit_factor > 1.15:
                categoria = "✓"
            elif sharpe > 0.3 and profit_factor > 1.0:
                categoria = "⚠️ (marginal)"
            else:
                categoria = "✗"

            resultados.append({
                'par': par,
                'n_trades': n_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe': sharpe,
                'return_pct': total_return,
                'es_consistente': es_consistente,
                'categoria': categoria
            })

        df_pares = pd.DataFrame(resultados)
        df_pares = df_pares.sort_values('sharpe', ascending=False)

        if self.verbose:
            self._imprimir_consistencia_pares(df_pares)

        return df_pares

    def _imprimir_consistencia_pares(self, df_pares: pd.DataFrame):
        """Imprime análisis de consistencia entre pares."""
        print(f"\n{'='*80}")
        print("CONSISTENCIA ENTRE PARES")
        print(f"{'='*80}")
        print()
        print(f"{'Par':<9} │ {'Trades':<6} │ {'WR':<5} │ {'PF':<5} │ {'Sharpe':<7} │ {'Return':<7} │ {'Consistente?'}")
        print("─"*9 + "┼" + "─"*7 + "┼" + "─"*6 + "┼" + "─"*6 + "┼" + "─"*8 + "┼" + "─"*8 + "┼" + "─"*20)

        for _, row in df_pares.iterrows():
            print(
                f"{row['par']:<9} │ "
                f"{row['n_trades']:<6.0f} │ "
                f"{row['win_rate']:>4.0f}% │ "
                f"{row['profit_factor']:>4.2f} │ "
                f"{row['sharpe']:>6.2f} │ "
                f"{row['return_pct']:>+6.1f}% │ "
                f"{row['categoria']}"
            )

        # Totales
        print("─"*9 + "┼" + "─"*7 + "┼" + "─"*6 + "┼" + "─"*6 + "┼" + "─"*8 + "┼" + "─"*8 + "┼" + "─"*20)
        total_trades = df_pares['n_trades'].sum()

        # Calcular métricas totales (ponderado por n_trades)
        total_wr = np.average(df_pares['win_rate'], weights=df_pares['n_trades'])
        total_pf = np.average(df_pares['profit_factor'], weights=df_pares['n_trades'])
        total_sharpe = np.average(df_pares['sharpe'], weights=df_pares['n_trades'])
        total_return = df_pares['return_pct'].sum()

        print(
            f"{'TOTAL':<9} │ "
            f"{total_trades:<6.0f} │ "
            f"{total_wr:>4.0f}% │ "
            f"{total_pf:>4.2f} │ "
            f"{total_sharpe:>6.2f} │ "
            f"{total_return:>+6.1f}% │"
        )

        print()
        print("ANÁLISIS:")
        print()

        # Análisis
        n_consistentes = df_pares['es_consistente'].sum()
        n_total = len(df_pares)

        print(f"  • {n_consistentes} de {n_total} pares con Sharpe > 0.5: "
              f"{'✓' if n_consistentes >= n_total * 0.7 else '⚠️'}")

        n_rentables = (df_pares['return_pct'] > 0).sum()
        print(f"  • {n_rentables} de {n_total} pares rentables: "
              f"{'✓' if n_rentables == n_total else '⚠️'}")

        # Identificar pares problemáticos
        pares_marginales = df_pares[df_pares['categoria'].str.contains('marginal')]
        pares_malos = df_pares[df_pares['categoria'].str.contains('✗')]

        if len(pares_marginales) > 0:
            print(f"  • Pares marginales: {', '.join(pares_marginales['par'].values)}")

        if len(pares_malos) > 0:
            print(f"  • Pares problemáticos: {', '.join(pares_malos['par'].values)}")

        print()
        print("RECOMENDACIÓN:")

        if n_consistentes >= n_total * 0.7:
            if len(pares_marginales) > 0:
                print(f"  • Operar todos los pares excepto: {', '.join(pares_marginales['par'].values)}")
                print(f"  • O reducir sizing en pares marginales")
            else:
                print(f"  • Operar todos los pares ✓")
        else:
            print(f"  • Revisar estrategia - baja consistencia entre pares")

        print()

        # Veredicto
        if n_consistentes >= n_total * 0.7:
            print(f"VEREDICTO: Consistencia entre pares PASA ✓ ({n_consistentes}/{n_total} pares)")
        else:
            print(f"VEREDICTO: Consistencia entre pares FALLA ✗ ({n_consistentes}/{n_total} pares)")

        print(f"{'='*80}\n")

    # ========================================================================
    # MÉTRICAS AUXILIARES
    # ========================================================================

    def _calcular_sharpe(self, returns: np.ndarray) -> float:
        """Calcula Sharpe ratio anualizado."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Asumir ~252 días de trading anuales
        sharpe = (mean_return / std_return) * np.sqrt(252)

        return sharpe

    def _calcular_profit_factor(self, returns: np.ndarray) -> float:
        """Calcula profit factor."""
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 1.0

        return gross_profit / gross_loss

    def _calcular_win_rate(self, returns: np.ndarray) -> float:
        """Calcula win rate (proporción)."""
        if len(returns) == 0:
            return 0.0

        n_wins = np.sum(returns > 0)
        win_rate = n_wins / len(returns)

        return win_rate

    def _calcular_max_drawdown(self, returns: np.ndarray) -> float:
        """Calcula maximum drawdown (proporción)."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(np.min(drawdown))

        return max_dd

    def _calcular_return_anual(self, returns: np.ndarray, n_trades_total: int) -> float:
        """Calcula retorno anual estimado."""
        # Total return
        total_return = np.sum(returns)

        # Estimar años (asumiendo ~250 trades por año)
        years = n_trades_total / 250

        if years > 0:
            annual_return = total_return / years
        else:
            annual_return = 0.0

        return annual_return

    # ========================================================================
    # REPORTE COMPLETO
    # ========================================================================

    def generar_reporte_completo(
        self,
        incluir_bootstrap: bool = True,
        incluir_permutation: bool = True,
        incluir_pares: bool = True
    ) -> Dict:
        """
        Genera reporte completo de robustez.

        Parameters:
        -----------
        incluir_bootstrap : bool
            Ejecutar bootstrap
        incluir_permutation : bool
            Ejecutar permutation test
        incluir_pares : bool
            Analizar consistencia entre pares

        Returns:
        --------
        reporte : Dict
            Diccionario con todos los resultados
        """
        reporte = {}

        # Bootstrap
        if incluir_bootstrap:
            reporte['bootstrap'] = self.bootstrap_confidence_intervals()

        # Permutation Test
        if incluir_permutation:
            reporte['permutation'] = self.permutation_test(metrica='sharpe')

        # Consistencia Entre Pares
        if incluir_pares:
            reporte['pares'] = self.consistencia_entre_pares()

        # Resumen
        if self.verbose:
            self._imprimir_resumen_final(reporte)

        return reporte

    def _imprimir_resumen_final(self, reporte: Dict):
        """Imprime resumen final de todas las pruebas."""
        print(f"\n{'='*80}")
        print("RESUMEN FINAL - PRUEBAS DE ROBUSTEZ")
        print(f"{'='*80}")
        print()

        pasa_total = 0
        total_pruebas = 0

        # Bootstrap
        if 'bootstrap' in reporte:
            total_pruebas += 1
            sharpe_ci_lower = reporte['bootstrap']['sharpe']['ci_lower']
            pasa_bootstrap = sharpe_ci_lower > 0

            if pasa_bootstrap:
                print("✓ Bootstrap: PASA (CI de Sharpe > 0)")
                pasa_total += 1
            else:
                print("✗ Bootstrap: FALLA (CI de Sharpe incluye 0)")

        # Permutation
        if 'permutation' in reporte:
            total_pruebas += 1
            p_value = reporte['permutation']['p_value']
            pasa_permutation = p_value < 0.05

            if pasa_permutation:
                print(f"✓ Permutation Test: PASA (p-value = {p_value:.4f})")
                pasa_total += 1
            else:
                print(f"✗ Permutation Test: FALLA (p-value = {p_value:.4f})")

        # Pares
        if 'pares' in reporte:
            total_pruebas += 1
            df_pares = reporte['pares']
            n_consistentes = df_pares['es_consistente'].sum()
            n_total = len(df_pares)
            pasa_pares = n_consistentes >= n_total * 0.7

            if pasa_pares:
                print(f"✓ Consistencia Entre Pares: PASA ({n_consistentes}/{n_total} pares)")
                pasa_total += 1
            else:
                print(f"✗ Consistencia Entre Pares: FALLA ({n_consistentes}/{n_total} pares)")

        print()
        print(f"RESULTADO GLOBAL: {pasa_total}/{total_pruebas} pruebas pasadas")
        print()

        if pasa_total == total_pruebas:
            print("🎉 ESTRATEGIA ROBUSTA - Todas las pruebas pasadas")
            print("   → Proceder a optimización y validación final")
        elif pasa_total >= total_pruebas * 0.7:
            print("⚠️  ESTRATEGIA PARCIALMENTE ROBUSTA")
            print("   → Revisar pruebas fallidas antes de continuar")
        else:
            print("❌ ESTRATEGIA NO ROBUSTA")
            print("   → Revisar estrategia o metodología")

        print(f"{'='*80}\n")


def main():
    """
    Función principal - Ejemplo de uso.
    """
    print("="*80)
    print("PRUEBAS DE ROBUSTEZ - EJEMPLO")
    print("="*80)

    # Ejemplo: cargar datos de backtest
    # En producción, esto vendría de motor_backtest_completo.py

    # Generar datos de ejemplo
    np.random.seed(42)
    n_trades = 332

    # Simular trades con ligero edge positivo
    returns = np.random.normal(0.002, 0.01, n_trades)  # 0.2% mean, 1% std

    df_trades = pd.DataFrame({
        'entry_time': pd.date_range('2022-01-01', periods=n_trades, freq='12H'),
        'exit_time': pd.date_range('2022-01-01', periods=n_trades, freq='12H') + pd.Timedelta(hours=6),
        'pair': np.random.choice(['EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY', 'AUD_USD'], n_trades),
        'net_pnl': returns * 1000,
        'capital_riesgo': np.ones(n_trades) * 1000,
        'return_pct': returns * 100
    })

    df_equity = pd.DataFrame({
        'timestamp': pd.date_range('2022-01-01', periods=n_trades, freq='12H'),
        'equity': 100000 + np.cumsum(returns * 1000)
    })

    # Inicializar pruebas
    pruebas = PruebasRobustez(
        df_trades=df_trades,
        df_equity=df_equity,
        verbose=True
    )

    # Ejecutar reporte completo
    reporte = pruebas.generar_reporte_completo(
        incluir_bootstrap=True,
        incluir_permutation=True,
        incluir_pares=True
    )

    print("\n✓ Pruebas de robustez completadas")
    print("\nPróximo paso:")
    print("  - Integrar con motor_backtest_completo.py")
    print("  - Ejecutar análisis de sensibilidad en parámetros de estrategia")
    print("  - Validar robustez antes de TEST final")


if __name__ == '__main__':
    main()
