#!/bin/bash
#
# EJECUTAR PIPELINE COMPLETO - FOREX EDGE FINDING
# =================================================
#
# Este script re-ejecuta el pipeline completo después de aplicar
# las correcciones al proceso de consenso.
#
# Pasos:
# 1. Consenso de métodos (con correcciones)
# 2. Validación rigurosa
# 3. Estrategia emergente
#

set -e  # Salir si hay errores

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETO - FOREX EDGE FINDING"
echo "================================================================================"
echo ""
echo "Este script ejecutará:"
echo "  1. Consenso de métodos (MEJORADO - ahora incluye consenso medio)"
echo "  2. Validación rigurosa (walk-forward, bootstrap, permutation, robustez)"
echo "  3. Estrategia emergente (interpretación y reglas de trading)"
echo ""
echo "Tiempo estimado: 4-6 horas (depende del hardware)"
echo ""
read -p "¿Continuar? (s/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "Cancelado por el usuario."
    exit 0
fi

# Directorio base
BASE_DIR=$(dirname "$(readlink -f "$0")")
cd "$BASE_DIR"

echo ""
echo "${GREEN}================================================================================${NC}"
echo "${GREEN}PASO 1: CONSENSO DE MÉTODOS (MEJORADO)${NC}"
echo "${GREEN}================================================================================${NC}"
echo ""
echo "Ejecutando: python ejecutar_consenso_metodos.py"
echo ""

python ejecutar_consenso_metodos.py

if [ $? -eq 0 ]; then
    echo ""
    echo "${GREEN}✓ PASO 1 COMPLETADO EXITOSAMENTE${NC}"
    echo ""

    # Verificar que hay features aprobados
    echo "Verificando features aprobados generados..."
    APROBADOS_DIR="datos/consenso_metodos/features_aprobados"

    if [ -d "$APROBADOS_DIR" ]; then
        N_FILES=$(ls -1 "$APROBADOS_DIR"/*.csv 2>/dev/null | wc -l)
        echo "  Archivos generados: $N_FILES"

        if [ $N_FILES -gt 0 ]; then
            echo "${GREEN}  ✓ Features aprobados encontrados para $N_FILES par(es)${NC}"
        else
            echo "${YELLOW}  ⚠️  No se encontraron archivos de features aprobados${NC}"
            echo "  Revisa los logs de consenso para ver si hubo problemas."
        fi
    fi
else
    echo ""
    echo "${RED}✗ ERROR EN PASO 1 - Consenso de Métodos${NC}"
    echo "Revisa los logs arriba para identificar el problema."
    exit 1
fi

echo ""
read -p "Presiona Enter para continuar con Paso 2 (Validación Rigurosa)..."

echo ""
echo "${GREEN}================================================================================${NC}"
echo "${GREEN}PASO 2: VALIDACIÓN RIGUROSA${NC}"
echo "${GREEN}================================================================================${NC}"
echo ""
echo "Ejecutando: python ejecutar_validacion_rigurosa.py"
echo ""

python ejecutar_validacion_rigurosa.py

if [ $? -eq 0 ]; then
    echo ""
    echo "${GREEN}✓ PASO 2 COMPLETADO EXITOSAMENTE${NC}"
    echo ""

    # Verificar features validados
    echo "Verificando features validados generados..."
    VALIDADOS_DIR="datos/validacion_rigurosa/features_validados"

    if [ -d "$VALIDADOS_DIR" ]; then
        N_FILES=$(ls -1 "$VALIDADOS_DIR"/*.csv 2>/dev/null | wc -l)
        echo "  Archivos generados: $N_FILES"

        if [ $N_FILES -gt 0 ]; then
            echo "${GREEN}  ✓ Features validados encontrados para $N_FILES par(es)${NC}"
        else
            echo "${YELLOW}  ⚠️  No se encontraron archivos de features validados${NC}"
            echo "  Es posible que ningún par haya pasado las 4 validaciones."
        fi
    fi
else
    echo ""
    echo "${RED}✗ ERROR EN PASO 2 - Validación Rigurosa${NC}"
    echo "Revisa los logs arriba para identificar el problema."
    exit 1
fi

echo ""
read -p "Presiona Enter para continuar con Paso 3 (Estrategia Emergente)..."

echo ""
echo "${GREEN}================================================================================${NC}"
echo "${GREEN}PASO 3: ESTRATEGIA EMERGENTE${NC}"
echo "${GREEN}================================================================================${NC}"
echo ""
echo "Ejecutando: python ejecutar_estrategia_emergente.py"
echo ""

python ejecutar_estrategia_emergente.py

if [ $? -eq 0 ]; then
    echo ""
    echo "${GREEN}✓ PASO 3 COMPLETADO EXITOSAMENTE${NC}"
    echo ""

    # Verificar estrategias generadas
    echo "Verificando estrategias generadas..."
    ESTRATEGIA_DIR="datos/estrategia_emergente/estrategias"

    if [ -d "$ESTRATEGIA_DIR" ]; then
        N_FILES=$(ls -1 "$ESTRATEGIA_DIR"/*.csv 2>/dev/null | wc -l)
        echo "  Archivos generados: $N_FILES"

        if [ $N_FILES -gt 0 ]; then
            echo "${GREEN}  ✓ Estrategias generadas para $N_FILES par(es)${NC}"
        fi
    fi
else
    echo ""
    echo "${RED}✗ ERROR EN PASO 3 - Estrategia Emergente${NC}"
    echo "Revisa los logs arriba para identificar el problema."
    exit 1
fi

echo ""
echo "${GREEN}================================================================================${NC}"
echo "${GREEN}✓ PIPELINE COMPLETO EJECUTADO EXITOSAMENTE${NC}"
echo "${GREEN}================================================================================${NC}"
echo ""
echo "Resultados guardados en:"
echo "  - Consenso: datos/consenso_metodos/"
echo "  - Validación: datos/validacion_rigurosa/"
echo "  - Estrategias: datos/estrategia_emergente/"
echo ""
echo "Próximos pasos:"
echo "  1. Revisar features aprobados por par"
echo "  2. Analizar estrategias emergentes generadas"
echo "  3. Backtesting de estrategias"
echo "  4. Paper trading / Live testing"
echo ""
echo "================================================================================"
