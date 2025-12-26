"""
Script de Configuración Automática del Proyecto
Crea toda la estructura de carpetas y archivos base
"""

import os
import sys
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

FOLDERS = [
    'data/Mayusculas',
    'data/Minusculas',
    'data/Numeros',
    'processed_handwriting',
    'checkpoints',
    'training_progress',
    'models',
    'test_images',
    'results',
    'backups',
    'docs'
]

REQUIREMENTS_CONTENT = """# Dependencias del Proyecto OCR Manuscrito
# Instalar con: pip install -r requirements.txt

numpy>=1.21.0
opencv-python>=4.5.0
pillow>=9.0.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
"""

GITIGNORE_CONTENT = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
env/
*.egg-info/

# Dataset (archivos muy grandes)
data/*.png
data/*.jpg
data/*.jpeg
data/*/*.png
data/*/*.jpg

# Archivos procesados (generados automáticamente)
processed_handwriting/*.npz
processed_handwriting/*.png
processed_handwriting/classes/

# Checkpoints y progreso
checkpoints/
training_progress/

# Modelos entrenados
models/*.pkl

# Logs
*.log
training_auto.log

# Resultados temporales
results/*.png
results/*.txt

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Backups
backups/

# Mantener estructura vacía (Git)
!data/.gitkeep
!processed_handwriting/.gitkeep
!checkpoints/.gitkeep
!training_progress/.gitkeep
!models/.gitkeep
!test_images/.gitkeep
!results/.gitkeep
!docs/.gitkeep
"""

README_CONTENT = """# OCR Manuscrito - Proyecto Final Inteligencia Artificial

Sistema OCR implementado desde cero para reconocimiento de texto manuscrito e impreso.

## 🎯 Características

- ✅ Red neuronal implementada desde cero (sin librerías pre-entrenadas)
- ✅ Reconocimiento de texto manuscrito y tipográfico
- ✅ Entrenamiento autónomo con checkpoints
- ✅ Data augmentation automático
- ✅ Preprocesamiento avanzado de imágenes

## 📋 Requisitos

- Python 3.8 o superior
- Dependencias listadas en `requirements.txt`

## 🚀 Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno
# Windows:
venv\\Scripts\\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## 📊 Uso

### Modo Interactivo (Recomendado)
```bash
python main.py
```

### Línea de Comandos
```bash
# Pipeline completo autónomo
python main.py todo auto

# Solo procesar dataset
python main.py procesar

# Solo entrenar (autónomo)
python main.py entrenar auto

# Probar modelo
python main.py probar test_images/prueba.jpg

# Ver estadísticas
python main.py stats
```

### Monitorear Entrenamiento
```bash
# En otra terminal mientras entrena
python monitor.py watch
```

## 📁 Estructura del Proyecto

```
ocr_manuscrito/
├── ocr_complete.py          # Sistema OCR completo
├── dataset_processor.py     # Procesamiento de dataset
├── train_model.py          # Entrenamiento
├── main.py                 # Script principal
├── monitor.py              # Monitor de entrenamiento
├── data/                   # Dataset original
├── processed_handwriting/  # Dataset procesado
├── models/                 # Modelos entrenados
└── checkpoints/           # Checkpoints de entrenamiento
```

## 🎓 Dataset

- **Origen**: Manuscritos de 60 colaboradores
- **Contenido**: Letras mayúsculas, minúsculas y números (0-9)
- **Total**: ~3,720 caracteres base
- **Aumentado**: ~14,880 caracteres (con data augmentation)

## 📈 Resultados

Todos los resultados se generan automáticamente:
- Modelos entrenados (`.pkl`)
- Estadísticas (`.json`)
- Visualizaciones (`.png`)
- Logs detallados (`.log`)

## 🤖 Modo Autónomo

El entrenamiento autónomo permite dejar el proceso corriendo sin supervisión:

- ✅ Checkpoints cada 10 épocas
- ✅ Recuperación automática si se interrumpe
- ✅ Gráficas generadas cada 25 épocas
- ✅ Logs detallados en tiempo real
- ✅ Early stopping automático

## 👤 Autor

[Tu Nombre]
Proyecto Final - Inteligencia Artificial
Universidad [Nombre]

## 📅 Fecha

Diciembre 2025

## 📝 Licencia

Proyecto académico - Todos los derechos reservados
"""

# ============================================================================
# FUNCIONES
# ============================================================================

def create_folder_structure():
    """Crea la estructura de carpetas"""
    print("📁 Creando estructura de carpetas...")
    print("="*60)
    
    created = 0
    already_exists = 0
    
    for folder in FOLDERS:
        folder_path = Path(folder)
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Creada: {folder}/")
            created += 1
            
            # Crear .gitkeep para mantener carpetas vacías en Git
            gitkeep = folder_path / '.gitkeep'
            gitkeep.touch()
        else:
            print(f"  ⚠️  Ya existe: {folder}/")
            already_exists += 1
    
    print(f"\n📊 Resumen:")
    print(f"  - Carpetas creadas: {created}")
    print(f"  - Ya existían: {already_exists}")
    print(f"  - Total: {len(FOLDERS)}")
    
    return created > 0

def create_base_files():
    """Crea archivos base (requirements.txt, .gitignore, README.md)"""
    print("\n📄 Creando archivos base...")
    print("="*60)
    
    files = {
        'requirements.txt': REQUIREMENTS_CONTENT,
        '.gitignore': GITIGNORE_CONTENT,
        'README.md': README_CONTENT
    }
    
    created = 0
    skipped = 0
    
    for filename, content in files.items():
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            print(f"  ✅ Creado: {filename}")
            created += 1
        else:
            print(f"  ⚠️  Ya existe: {filename} (no se sobrescribió)")
            skipped += 1
    
    print(f"\n📊 Resumen:")
    print(f"  - Archivos creados: {created}")
    print(f"  - Ya existían: {skipped}")
    
    return created > 0

def check_python_scripts():
    """Verifica qué scripts principales están presentes"""
    print("\n🔍 Verificando scripts principales...")
    print("="*60)
    
    required_scripts = [
        ('ocr_complete.py', 'Sistema OCR completo'),
        ('dataset_processor.py', 'Procesador de dataset'),
        ('train_model.py', 'Sistema de entrenamiento'),
        ('main.py', 'Script principal'),
    ]
    
    optional_scripts = [
        ('monitor.py', 'Monitor de entrenamiento'),
        ('ocr_utils.py', 'Utilidades adicionales'),
    ]
    
    print("\n📜 Scripts obligatorios:")
    missing_required = []
    for script, description in required_scripts:
        if os.path.exists(script):
            print(f"  ✅ {script:25s} - {description}")
        else:
            print(f"  ❌ {script:25s} - {description} (FALTA)")
            missing_required.append(script)
    
    print("\n📜 Scripts opcionales:")
    for script, description in optional_scripts:
        if os.path.exists(script):
            print(f"  ✅ {script:25s} - {description}")
        else:
            print(f"  ⚠️  {script:25s} - {description} (opcional)")
    
    return len(missing_required) == 0, missing_required

def check_venv():
    """Verifica si existe entorno virtual"""
    print("\n🐍 Verificando entorno virtual...")
    print("="*60)
    
    venv_paths = ['venv', 'env', 'ENV']
    venv_exists = any(os.path.exists(p) for p in venv_paths)
    
    if venv_exists:
        print("  ✅ Entorno virtual encontrado")
    else:
        print("  ⚠️  No se encontró entorno virtual")
        print("\n  💡 Para crear uno:")
        print("     python -m venv venv")
        print("     venv\\Scripts\\activate  (Windows)")
        print("     source venv/bin/activate  (Linux/Mac)")
    
    return venv_exists

def print_next_steps(scripts_ok, missing_scripts):
    """Imprime próximos pasos según el estado"""
    print("\n" + "="*60)
    print("✨ CONFIGURACIÓN COMPLETADA")
    print("="*60)
    
    print("\n📋 PRÓXIMOS PASOS:")
    print("="*60)
    
    if not scripts_ok:
        print("\n❗ PASO 1: Copiar scripts faltantes")
        print("  Necesitas copiar los siguientes archivos:")
        for script in missing_scripts:
            print(f"    - {script}")
        print("\n  Los tienes como artifacts en la conversación.")
    else:
        print("\n✅ PASO 1: Scripts principales presentes")
    
    print("\n📂 PASO 2: Preparar datos")
    print("  1. Descarga tu carpeta del Google Drive")
    print("  2. Copia la estructura dentro de data/:")
    print("     data/")
    print("     ├── Mayusculas/A/*.png")
    print("     ├── Minusculas/a/*.png")
    print("     └── Numeros/0/*.png")
    
    if not check_venv():
        print("\n🐍 PASO 3: Crear entorno virtual")
        print("  python -m venv venv")
        print("  venv\\Scripts\\activate  (Windows)")
    else:
        print("\n✅ PASO 3: Entorno virtual presente")
    
    print("\n📦 PASO 4: Instalar dependencias")
    print("  pip install -r requirements.txt")
    
    print("\n🚀 PASO 5: Ejecutar proyecto")
    print("  python main.py")
    
    print("\n" + "="*60)
    print("💡 CONSEJOS:")
    print("="*60)
    print("  - Usa 'python main.py' para modo interactivo")
    print("  - Usa 'python main.py todo auto' para pipeline completo")
    print("  - Usa 'python monitor.py watch' para monitorear entrenamiento")
    print("  - Lee README.md para más información")
    print("="*60)

def create_example_test_image_readme():
    """Crea un README en test_images/ explicando qué poner ahí"""
    readme_path = Path('test_images') / 'README.txt'
    
    content = """
CARPETA TEST_IMAGES
===================

Esta carpeta es para colocar imágenes de prueba para el OCR.

Formatos soportados:
  - .jpg / .jpeg
  - .png
  - .bmp
  - .tiff

Ejemplos de lo que puedes poner:
  1. Fotos de texto manuscrito
  2. Imágenes de texto impreso
  3. Documentos escaneados
  4. Capturas de pantalla con texto

Para probar el OCR:
  python main.py probar
  
  O directamente:
  python main.py probar test_images/tu_imagen.jpg

Nota: Las imágenes no se incluyen en Git (ver .gitignore)
    """
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content.strip())
    
    print(f"  ℹ️  Creado: {readme_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Función principal"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     CONFIGURACIÓN AUTOMÁTICA DEL PROYECTO OCR              ║
    ║          Sistema de Reconocimiento Manuscrito              ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    print("\nEste script creará toda la estructura necesaria del proyecto.")
    print("Se crearán carpetas, archivos base y se verificarán dependencias.\n")
    
    respuesta = input("¿Continuar? (s/n): ").strip().lower()
    
    if respuesta != 's':
        print("\n❌ Configuración cancelada")
        return
    
    print("\n" + "="*60)
    print("INICIANDO CONFIGURACIÓN")
    print("="*60)
    
    # 1. Crear estructura de carpetas
    create_folder_structure()
    
    # 2. Crear archivos base
    create_base_files()
    
    # 3. Verificar scripts
    scripts_ok, missing_scripts = check_python_scripts()
    
    # 4. Verificar entorno virtual
    check_venv()
    
    # 5. Crear README en test_images
    create_example_test_image_readme()
    
    # 6. Mostrar próximos pasos
    print_next_steps(scripts_ok, missing_scripts)
    
    print("\n✅ Script de configuración completado\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Configuración interrumpida por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error durante la configuración: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)