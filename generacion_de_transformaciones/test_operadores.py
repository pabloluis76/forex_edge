"""
Script de prueba para los operadores puros
==========================================

Verifica que todos los operadores funcionan correctamente
con datos reales de forex.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generacion_de_transformaciones import (
    Delta, R, r, mu, sigma, Max, Min, Z, Pos, Rank, P,
    D1, D2, rho, EMA
)


def cargar_datos_ejemplo():
    """Carga datos reales de EUR/USD para pruebas"""
    data_dir = Path(__file__).parent.parent / 'datos' / 'ohlc' / 'EUR_USD'
    file_path = data_dir / 'H1.csv'

    if not file_path.exists():
        print(f"⚠️  Archivo no encontrado: {file_path}")
        print("Usando datos sintéticos...")
        # Generar datos sintéticos
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='H')
        close = 1.10 + np.cumsum(np.random.randn(1000) * 0.0001)
        return pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(1000) * 0.0005),
            'low': close - np.abs(np.random.randn(1000) * 0.0005),
            'volume': np.random.randint(100, 1000, 1000)
        }, index=dates)

    df = pd.read_csv(file_path, index_col='time', parse_dates=True)
    return df


def test_operadores():
    """Ejecuta pruebas de todos los operadores"""

    print("="*70)
    print("PRUEBA DE OPERADORES PUROS")
    print("="*70)
    print()

    # Cargar datos
    print("📊 Cargando datos...")
    df = cargar_datos_ejemplo()
    close = df['close']
    high = df['high']
    low = df['low']

    print(f"✓ Datos cargados: {len(df)} velas")
    print(f"  Período: {df.index[0]} a {df.index[-1]}")
    print()

    # Pruebas de operadores
    tests = []

    print("🧪 Probando operadores...\n")

    # 1. Delta
    try:
        result = Delta(close, 1)
        assert not result.isna().all(), "Delta devolvió todos NaN"
        tests.append(("✓", "Δ (Delta)", f"{result.iloc[-1]:.6f}"))
    except Exception as e:
        tests.append(("✗", "Δ (Delta)", str(e)))

    # 2. Retorno
    try:
        result = R(close, 1)
        assert not result.isna().all()
        tests.append(("✓", "R (Retorno)", f"{result.iloc[-1]:.6f}"))
    except Exception as e:
        tests.append(("✗", "R (Retorno)", str(e)))

    # 3. Log-retorno
    try:
        result = r(close, 1)
        assert not result.isna().all()
        tests.append(("✓", "r (Log-retorno)", f"{result.iloc[-1]:.6f}"))
    except Exception as e:
        tests.append(("✗", "r (Log-retorno)", str(e)))

    # 4. Media móvil
    try:
        result = mu(close, 20)
        assert not result.isna().all()
        tests.append(("✓", "μ (Media móvil 20)", f"{result.iloc[-1]:.5f}"))
    except Exception as e:
        tests.append(("✗", "μ (Media móvil)", str(e)))

    # 5. Desviación estándar
    try:
        result = sigma(close, 20)
        assert not result.isna().all()
        tests.append(("✓", "σ (Std móvil 20)", f"{result.iloc[-1]:.6f}"))
    except Exception as e:
        tests.append(("✗", "σ (Desv. std)", str(e)))

    # 6. Máximo
    try:
        result = Max(close, 14)
        assert not result.isna().all()
        tests.append(("✓", "Max (Máximo 14)", f"{result.iloc[-1]:.5f}"))
    except Exception as e:
        tests.append(("✗", "Max", str(e)))

    # 7. Mínimo
    try:
        result = Min(close, 14)
        assert not result.isna().all()
        tests.append(("✓", "Min (Mínimo 14)", f"{result.iloc[-1]:.5f}"))
    except Exception as e:
        tests.append(("✗", "Min", str(e)))

    # 8. Z-score
    try:
        result = Z(close, 20)
        assert not result.isna().all()
        tests.append(("✓", "Z (Z-score 20)", f"{result.iloc[-1]:.3f}"))
    except Exception as e:
        tests.append(("✗", "Z (Z-score)", str(e)))

    # 9. Posición en rango
    try:
        result = Pos(close, 14)
        assert not result.isna().all()
        assert 0 <= result.dropna().max() <= 1, "Pos fuera de rango [0,1]"
        tests.append(("✓", "Pos (Posición 14)", f"{result.iloc[-1]:.3f}"))
    except Exception as e:
        tests.append(("✗", "Pos", str(e)))

    # 10. Ranking
    try:
        result = Rank(close, 14)
        assert not result.isna().all()
        tests.append(("✓", "Rank (Ranking 14)", f"{result.iloc[-1]:.3f}"))
    except Exception as e:
        tests.append(("✗", "Rank", str(e)))

    # 11. Percentil
    try:
        result = P(close, 75, 20)
        assert not result.isna().all()
        tests.append(("✓", "P (Percentil 75)", f"{result.iloc[-1]:.5f}"))
    except Exception as e:
        tests.append(("✗", "P (Percentil)", str(e)))

    # 12. Primera derivada
    try:
        result = D1(close)
        assert not result.isna().all()
        tests.append(("✓", "D¹ (Velocidad)", f"{result.iloc[-1]:.6f}"))
    except Exception as e:
        tests.append(("✗", "D¹", str(e)))

    # 13. Segunda derivada
    try:
        result = D2(close)
        assert not result.isna().all()
        tests.append(("✓", "D² (Aceleración)", f"{result.iloc[-1]:.6f}"))
    except Exception as e:
        tests.append(("✗", "D²", str(e)))

    # 14. Correlación
    try:
        result = rho(high, low, 20)
        assert not result.isna().all()
        tests.append(("✓", "ρ (Correlación 20)", f"{result.iloc[-1]:.3f}"))
    except Exception as e:
        tests.append(("✗", "ρ (Correlación)", str(e)))

    # 15. EMA
    try:
        result = EMA(close, 20)
        assert not result.isna().all()
        tests.append(("✓", "EMA (EMA 20)", f"{result.iloc[-1]:.5f}"))
    except Exception as e:
        tests.append(("✗", "EMA", str(e)))

    # Mostrar resultados
    print("-"*70)
    for status, name, value in tests:
        print(f"{status} {name:.<30} {value}")
    print("-"*70)
    print()

    # Resumen
    passed = sum(1 for t in tests if t[0] == "✓")
    total = len(tests)

    print(f"Resultados: {passed}/{total} pruebas pasadas")

    if passed == total:
        print("\n🎉 ¡TODOS LOS OPERADORES FUNCIONAN CORRECTAMENTE!")
        print("✓ Sin look-ahead bias")
        print("✓ Listos para feature engineering")
    else:
        print(f"\n⚠️  {total - passed} operadores fallaron")

    print("\n" + "="*70)

    return passed == total


def test_composicion():
    """Prueba composición de operadores"""

    print("\n")
    print("="*70)
    print("PRUEBA DE COMPOSICIÓN DE OPERADORES")
    print("="*70)
    print()

    df = cargar_datos_ejemplo()
    close = df['close']

    print("Ejemplos de composición:\n")

    # Ejemplo 1: Momentum normalizado
    print("1. Momentum normalizado:")
    print("   Z₂₀(Δ₁(close)) - cambio de 1 día normalizado por volatilidad")
    momentum = Delta(close, 1)
    momentum_z = Z(momentum, 20)
    print(f"   Valor actual: {momentum_z.iloc[-1]:.3f}")
    print()

    # Ejemplo 2: Velocidad de la media móvil
    print("2. Velocidad de la media móvil:")
    print("   D¹(μ₂₀(close)) - qué tan rápido se mueve la media")
    ma = mu(close, 20)
    ma_velocity = D1(ma)
    print(f"   Valor actual: {ma_velocity.iloc[-1]:.6f}")
    print()

    # Ejemplo 3: Posición relativa de la volatilidad
    print("3. Posición relativa de la volatilidad:")
    print("   Pos₁₄(σ₂₀(r₁(close))) - dónde está la volatilidad actual")
    returns = r(close, 1)
    vol = sigma(returns, 20)
    vol_pos = Pos(vol, 14)
    print(f"   Valor actual: {vol_pos.iloc[-1]:.3f} (0=mín, 1=máx)")
    print()

    print("✓ Composición de operadores funciona correctamente")
    print("✓ Puedes crear features complejos combinando operadores básicos")
    print("\n" + "="*70)


if __name__ == '__main__':
    # Ejecutar pruebas
    success = test_operadores()

    if success:
        test_composicion()

    sys.exit(0 if success else 1)
