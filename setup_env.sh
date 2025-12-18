#!/bin/bash
# ==============================================================================
# Forex Edge - Setup Script
# Script para configurar el entorno de desarrollo
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Forex Edge - Environment Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}Error: Python 3.9+ is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}Python $PYTHON_VERSION detected${NC}"

# Create virtual environment
echo ""
echo -e "${YELLOW}[2/6] Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping...${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}[3/6] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo -e "${YELLOW}[4/6] Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo ""
echo -e "${YELLOW}[5/6] Installing dependencies...${NC}"
echo -e "${YELLOW}This may take a few minutes...${NC}"
pip install -r requirements.txt

# Create data directory
echo ""
echo -e "${YELLOW}[6/6] Creating data directory...${NC}"
mkdir -p datos
echo -e "${GREEN}Data directory created${NC}"

# Check for .env file
echo ""
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo -e "${YELLOW}Copy .env.example to .env and configure your API keys:${NC}"
    echo -e "  cp .env.example .env"
    echo -e "  nano .env  # or your preferred editor"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "To activate the environment in the future, run:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo -e "To run the transformation generation:"
echo -e "  ${YELLOW}python ejecutar_generacion_transformaciones.py${NC}"
echo ""
echo -e "To run the matrix/tensor structure creation:"
echo -e "  ${YELLOW}python ejecutar_estructura_matricial_tensorial.py${NC}"
echo ""
