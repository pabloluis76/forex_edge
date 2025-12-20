#!/usr/bin/env python3
"""
SCANNER DE DEPENDENCIAS - FOREX EDGE
=====================================
Verifica qué dependencias están instaladas y cuáles faltan.
"""

import subprocess
import sys
from pathlib import Path

def get_installed_packages():
    """Obtiene lista de paquetes instalados."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
            capture_output=True,
            text=True,
            check=True
        )
        installed = {}
        for line in result.stdout.strip().split('\n'):
            if '==' in line:
                pkg, ver = line.split('==')
                installed[pkg.lower()] = ver
        return installed
    except Exception as e:
        print(f"Error obteniendo paquetes instalados: {e}")
        return {}

def parse_requirements(req_file):
    """Parse requirements.txt."""
    requirements = {}
    if not req_file.exists():
        return requirements
    
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '==' in line:
                    pkg, ver = line.split('==')
                    requirements[pkg.lower()] = ver
                elif '>=' in line:
                    pkg = line.split('>=')[0]
                    requirements[pkg.lower()] = 'any'
                else:
                    requirements[line.lower()] = 'any'
    return requirements

def scan_imports_in_file(filepath):
    """Escanea imports en un archivo Python."""
    imports = set()
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith('import '):
                    module = line.split()[1].split('.')[0]
                    imports.add(module)
                elif line.startswith('from '):
                    module = line.split()[1].split('.')[0]
                    imports.add(module)
    except Exception:
        pass
    return imports

def main():
    print("=" * 70)
    print("🔍 SCANNER DE DEPENDENCIAS - FOREX EDGE")
    print("=" * 70)
    
    # 1. Leer requirements.txt
    print("\n1. LEYENDO REQUIREMENTS.TXT")
    print("-" * 70)
    req_file = Path('requirements.txt')
    requirements = parse_requirements(req_file)
    print(f"   Dependencias en requirements.txt: {len(requirements)}")
    
    # 2. Obtener paquetes instalados
    print("\n2. VERIFICANDO PAQUETES INSTALADOS")
    print("-" * 70)
    installed = get_installed_packages()
    print(f"   Paquetes instalados: {len(installed)}")
    
    # 3. Comparar
    print("\n3. ANÁLISIS DE DEPENDENCIAS")
    print("-" * 70)
    
    missing = []
    installed_ok = []
    version_mismatch = []
    
    for pkg, required_ver in requirements.items():
        if pkg in installed:
            if required_ver == 'any' or installed[pkg] == required_ver:
                installed_ok.append(pkg)
            else:
                version_mismatch.append((pkg, required_ver, installed[pkg]))
        else:
            missing.append((pkg, required_ver))
    
    # 4. Escanear imports en código
    print("\n4. ESCANEANDO IMPORTS EN CÓDIGO")
    print("-" * 70)
    
    code_imports = set()
    python_files = list(Path('.').rglob('*.py'))
    print(f"   Archivos Python encontrados: {len(python_files)}")
    
    for pyfile in python_files:
        if 'venv' not in str(pyfile) and '.venv' not in str(pyfile):
            imports = scan_imports_in_file(pyfile)
            code_imports.update(imports)
    
    print(f"   Módulos únicos importados: {len(code_imports)}")
    
    # Mapeo común de imports a paquetes pip
    import_to_package = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'dotenv': 'python-dotenv',
    }
    
    # Buscar imports que no están en requirements
    imports_not_in_req = set()
    for imp in code_imports:
        pkg = import_to_package.get(imp, imp)
        if pkg.lower() not in requirements and imp not in ['os', 'sys', 'pathlib', 'datetime', 'json', 'logging', 'warnings', 'typing', 'collections', 'itertools', 're', 'math', 'time', 'random']:
            imports_not_in_req.add(imp)
    
    # 5. REPORTE FINAL
    print("\n" + "=" * 70)
    print("📊 REPORTE FINAL")
    print("=" * 70)
    
    print(f"\n✅ INSTALADAS CORRECTAMENTE: {len(installed_ok)}")
    if len(installed_ok) <= 20:
        for pkg in sorted(installed_ok)[:10]:
            print(f"   ✓ {pkg}")
        if len(installed_ok) > 10:
            print(f"   ... y {len(installed_ok) - 10} más")
    
    if version_mismatch:
        print(f"\n⚠️  VERSIÓN INCORRECTA: {len(version_mismatch)}")
        for pkg, required, actual in version_mismatch:
            print(f"   ⚠️  {pkg}: requerida {required}, instalada {actual}")
    
    if missing:
        print(f"\n❌ FALTANTES: {len(missing)}")
        for pkg, ver in missing:
            print(f"   ❌ {pkg}=={ver}")
    
    if imports_not_in_req:
        print(f"\n⚠️  IMPORTS NO EN REQUIREMENTS: {len(imports_not_in_req)}")
        for imp in sorted(imports_not_in_req)[:10]:
            print(f"   ⚠️  {imp}")
    
    # 6. RECOMENDACIONES
    print("\n" + "=" * 70)
    print("💡 RECOMENDACIONES")
    print("=" * 70)
    
    if missing:
        print("\n🔧 Para instalar dependencias faltantes:")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
    
    if not missing and not version_mismatch:
        print("\n✅ ¡Todas las dependencias están correctamente instaladas!")
    
    return len(missing) + len(version_mismatch)

if __name__ == '__main__':
    exit_code = main()
    sys.exit(min(exit_code, 1))
