"""
Script Principal v2.0 - Con Reconocimiento de Texto Completo
Soporta: Caracteres, Palabras, Líneas, Texto manuscrito e impreso
"""

import os
import sys
from pathlib import Path
import json
import cv2

# Importar módulos corregidos
import dataset_processor
import train_model
from ocr_complete import OCRSystem, process_image_file

class Config:
    DATA_PATH = './data'
    DATASET_OUTPUT = 'processed_handwriting/handwritten_dataset.npz'
    MODEL_PATH = 'models/ocr_model.pkl'
    RESULTS_DIR = 'results'
    
    # Crear directorio de resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)


def ejecutar_pipeline_completo():
    """Pipeline completo: Procesar + Entrenar"""
    print("\n" + "="*60)
    print("PIPELINE COMPLETO - OCR MANUSCRITO")
    print("="*60)
    
    print("\n--- PASO 1/2: Procesando Dataset ---")
    dataset_path = dataset_processor.procesar_dataset_completo(Config.DATA_PATH)
    
    if dataset_path is None:
        print("\n❌ Error al procesar dataset. Abortando.")
        return
    
    print("\n--- PASO 2/2: Entrenando Modelo (Modo Autónomo) ---")
    train_model.ejecutar_entrenamiento_autonomo()
    
    print("\n" + "="*60)
    print("✅ Pipeline completado con éxito")
    print("="*60)


def probar_ocr_basico(ruta_imagen):
    """Prueba OCR básica (compatibilidad con versión anterior)"""
    if not os.path.exists(Config.MODEL_PATH):
        print("❌ Error: No existe el modelo entrenado.")
        print("   Entrena primero con: python main.py entrenar")
        return
    
    if not os.path.exists(ruta_imagen):
        print(f"❌ Error: No se encuentra la imagen: {ruta_imagen}")
        return
    
    print(f"\n🔍 Analizando imagen: {ruta_imagen}")
    
    try:
        ocr = OCRSystem(model_path=Config.MODEL_PATH)
        
        # Cargar y procesar imagen
        img = cv2.imread(ruta_imagen)
        if img is None:
            print("❌ No se pudo cargar la imagen.")
            return
        
        # Procesar con confianza
        resultado, confianza = ocr.process_image(img, return_confidence=True)
        
        print(f"\n{'='*60}")
        print("📄 TEXTO RECONOCIDO:")
        print(f"{'='*60}")
        print(resultado)
        print(f"{'='*60}")
        print(f"✓ Confianza promedio: {confianza:.2%}")
        print(f"{'='*60}\n")
        
        # Guardar resultado
        output_file = os.path.join(Config.RESULTS_DIR, 'ultimo_reconocimiento.txt')
        ocr.save_result_to_file(resultado, output_file)
        
        return resultado
        
    except Exception as e:
        print(f"❌ Error durante el reconocimiento: {str(e)}")
        import traceback
        traceback.print_exc()


def probar_ocr_avanzado():
    """Prueba OCR avanzada con opciones"""
    if not os.path.exists(Config.MODEL_PATH):
        print("❌ Error: No existe el modelo entrenado.")
        print("   Entrena primero con: python main.py entrenar")
        return
    
    print("\n" + "="*60)
    print("🔬 OCR AVANZADO - Reconocimiento de Texto")
    print("="*60)
    
    # Seleccionar imagen
    print("\nOpciones:")
    print("1. Ingresar ruta de imagen manualmente")
    print("2. Usar imagen de prueba (test_images/)")
    print("3. Procesar múltiples imágenes (batch)")
    
    opcion = input("\nSelecciona (1-3): ").strip()
    
    if opcion == '1':
        ruta = input("Ruta de la imagen: ").strip()
        if os.path.exists(ruta):
            probar_ocr_basico(ruta)
        else:
            print(f"❌ No se encuentra: {ruta}")
    
    elif opcion == '2':
        # Listar imágenes en test_images
        test_dir = Path('test_images')
        if test_dir.exists():
            imagenes = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
            
            if not imagenes:
                print("❌ No hay imágenes en test_images/")
                return
            
            print("\nImágenes disponibles:")
            for i, img in enumerate(imagenes, 1):
                print(f"  {i}. {img.name}")
            
            try:
                idx = int(input("\nSelecciona número: ")) - 1
                if 0 <= idx < len(imagenes):
                    probar_ocr_basico(str(imagenes[idx]))
                else:
                    print("❌ Número inválido")
            except ValueError:
                print("❌ Entrada inválida")
        else:
            print("❌ Carpeta test_images/ no existe")
    
    elif opcion == '3':
        procesar_batch()


def procesar_batch():
    """Procesa múltiples imágenes en batch"""
    print("\n" + "="*60)
    print("📦 PROCESAMIENTO BATCH")
    print("="*60)
    
    carpeta = input("\nRuta de la carpeta con imágenes: ").strip()
    
    if not os.path.exists(carpeta):
        print(f"❌ No se encuentra la carpeta: {carpeta}")
        return
    
    # Buscar todas las imágenes
    imagenes = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        imagenes.extend(Path(carpeta).glob(ext))
    
    if not imagenes:
        print(f"❌ No se encontraron imágenes en: {carpeta}")
        return
    
    print(f"\n✓ Encontradas {len(imagenes)} imágenes")
    print("Procesando...\n")
    
    # Cargar modelo una vez
    ocr = OCRSystem(model_path=Config.MODEL_PATH)
    
    # Procesar cada imagen
    resultados = []
    for i, img_path in enumerate(imagenes, 1):
        print(f"[{i}/{len(imagenes)}] {img_path.name}...", end=' ')
        
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print("❌ Error al cargar")
                continue
            
            texto, conf = ocr.process_image(img, return_confidence=True)
            
            # Guardar resultado individual
            output_file = os.path.join(
                Config.RESULTS_DIR,
                f"{img_path.stem}_resultado.txt"
            )
            ocr.save_result_to_file(texto, output_file)
            
            resultados.append({
                'archivo': img_path.name,
                'texto': texto,
                'confianza': conf,
                'lineas': len(texto.split('\n'))
            })
            
            print(f"✓ (conf: {conf:.2%})")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DEL BATCH")
    print("="*60)
    print(f"Total procesadas: {len(resultados)}/{len(imagenes)}")
    
    if resultados:
        conf_promedio = sum(r['confianza'] for r in resultados) / len(resultados)
        print(f"Confianza promedio: {conf_promedio:.2%}")
        print(f"\nResultados guardados en: {Config.RESULTS_DIR}/")
        
        # Guardar resumen
        resumen_path = os.path.join(Config.RESULTS_DIR, 'batch_resumen.json')
        with open(resumen_path, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        print(f"Resumen guardado en: {resumen_path}")


def verificar_sistema():
    """Verifica el estado del sistema"""
    print("\n" + "="*60)
    print("🔍 VERIFICACIÓN DEL SISTEMA")
    print("="*60)
    
    checks = []
    
    # 1. Dataset
    if os.path.exists(Config.DATASET_OUTPUT):
        data = dataset_processor.cargar_dataset(Config.DATASET_OUTPUT)
        if data[0] is not None:
            X, y, char_to_idx, idx_to_char = data
            checks.append(('Dataset procesado', True, f"{len(X)} muestras, {len(char_to_idx)} clases"))
        else:
            checks.append(('Dataset procesado', False, 'Error al cargar'))
    else:
        checks.append(('Dataset procesado', False, 'No encontrado'))
    
    # 2. Modelo
    if os.path.exists(Config.MODEL_PATH):
        checks.append(('Modelo entrenado', True, Config.MODEL_PATH))
    else:
        checks.append(('Modelo entrenado', False, 'No encontrado'))
    
    # 3. Mapeo
    mapping_path = Config.MODEL_PATH.replace('.pkl', '_mapping.json')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        checks.append(('Mapeo de caracteres', True, f"{len(mapping)} clases"))
    else:
        checks.append(('Mapeo de caracteres', False, 'No encontrado'))
    
    # 4. Datos de entrada
    if os.path.exists(Config.DATA_PATH):
        categorias = ['Mayusculas', 'Minusculas', 'Numeros']
        total_imgs = 0
        for cat in categorias:
            cat_path = Path(Config.DATA_PATH) / cat
            if cat_path.exists():
                imgs = sum(1 for _ in cat_path.rglob('*.png'))
                imgs += sum(1 for _ in cat_path.rglob('*.jpg'))
                total_imgs += imgs
        
        if total_imgs > 0:
            checks.append(('Datos de entrada', True, f"~{total_imgs} imágenes"))
        else:
            checks.append(('Datos de entrada', False, 'Sin imágenes'))
    else:
        checks.append(('Datos de entrada', False, 'Carpeta no existe'))
    
    # 5. Imágenes de prueba
    if os.path.exists('test_images'):
        test_imgs = list(Path('test_images').glob('*.jpg')) + \
                   list(Path('test_images').glob('*.png'))
        if test_imgs:
            checks.append(('Imágenes de prueba', True, f"{len(test_imgs)} disponibles"))
        else:
            checks.append(('Imágenes de prueba', False, 'Carpeta vacía'))
    else:
        checks.append(('Imágenes de prueba', False, 'Carpeta no existe'))
    
    # Mostrar resultados
    print("")
    for nombre, estado, detalle in checks:
        simbolo = "✓" if estado else "❌"
        print(f"  {simbolo} {nombre:25s}: {detalle}")
    
    # Resumen
    print("\n" + "="*60)
    total = len(checks)
    ok = sum(1 for _, estado, _ in checks if estado)
    
    if ok == total:
        print("✅ Sistema completamente funcional")
    elif ok >= total * 0.6:
        print("⚠️  Sistema parcialmente funcional")
    else:
        print("❌ Sistema requiere configuración")
    
    print(f"   Componentes OK: {ok}/{total}")
    print("="*60)


def mostrar_estadisticas():
    """Muestra estadísticas del sistema"""
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DEL SISTEMA")
    print("="*60)
    
    # Dataset
    if os.path.exists(Config.DATASET_OUTPUT):
        X, y, char_to_idx, idx_to_char = dataset_processor.cargar_dataset(Config.DATASET_OUTPUT)
        
        if X is not None:
            print("\n📦 DATASET:")
            print(f"  Total muestras: {len(X)}")
            print(f"  Clases: {len(char_to_idx)}")
            print(f"  Forma de imagen: {X[0].shape}")
            
            # Distribución por tipo
            from collections import Counter
            etiquetas = [idx_to_char[str(i)] for i in y]
            counter = Counter(etiquetas)
            
            mayus = sum(v for k, v in counter.items() if k.isupper() and k.isalpha())
            minus = sum(v for k, v in counter.items() if k.islower() and k.isalpha())
            nums = sum(v for k, v in counter.items() if k.isdigit())
            
            print(f"\n  Distribución:")
            print(f"    Mayúsculas: {mayus} muestras")
            print(f"    Minúsculas: {minus} muestras")
            print(f"    Números: {nums} muestras")
    
    # Modelo
    if os.path.exists(Config.MODEL_PATH):
        size_mb = os.path.getsize(Config.MODEL_PATH) / (1024 * 1024)
        print(f"\n🤖 MODELO:")
        print(f"  Archivo: {Config.MODEL_PATH}")
        print(f"  Tamaño: {size_mb:.2f} MB")
    
    # Resultados
    if os.path.exists(Config.RESULTS_DIR):
        resultados = list(Path(Config.RESULTS_DIR).glob('*.txt'))
        if resultados:
            print(f"\n📄 RESULTADOS:")
            print(f"  Total archivos: {len(resultados)}")
            print(f"  Directorio: {Config.RESULTS_DIR}/")
    
    print("\n" + "="*60)


def menu_interactivo():
    """Menú interactivo mejorado"""
    while True:
        print("\n" + "="*60)
        print("🔤 SISTEMA OCR MANUSCRITO v2.0")
        print("   Reconocimiento de Texto Completo")
        print("="*60)
        
        print("\n📋 MENÚ PRINCIPAL:")
        print("  1. 🔄 Pipeline Completo (Procesar + Entrenar)")
        print("  2. ⚙️  Solo Procesar Dataset")
        print("  3. 🎓 Solo Entrenar Modelo")
        print("  4. 🔍 Probar OCR (Simple)")
        print("  5. 🔬 Probar OCR (Avanzado)")
        print("  6. 📦 Procesamiento Batch")
        print("  7. 🔍 Verificar Sistema")
        print("  8. 📊 Estadísticas")
        print("  9. 📝 Monitor de Entrenamiento")
        print("  0. ❌ Salir")
        
        opcion = input("\nSelecciona una opción: ").strip()
        
        if opcion == '1':
            ejecutar_pipeline_completo()
            input("\nPresiona ENTER para continuar...")
        
        elif opcion == '2':
            print("\n--- Procesando Dataset ---")
            dataset_processor.procesar_dataset_completo(Config.DATA_PATH)
            input("\nPresiona ENTER para continuar...")
        
        elif opcion == '3':
            print("\n--- Entrenando Modelo ---")
            train_model.ejecutar_entrenamiento_autonomo()
            input("\nPresiona ENTER para continuar...")
        
        elif opcion == '4':
            img_path = input("\nRuta de la imagen: ").strip()
            if img_path:
                probar_ocr_basico(img_path)
            input("\nPresiona ENTER para continuar...")
        
        elif opcion == '5':
            probar_ocr_avanzado()
            input("\nPresiona ENTER para continuar...")
        
        elif opcion == '6':
            procesar_batch()
            input("\nPresiona ENTER para continuar...")
        
        elif opcion == '7':
            verificar_sistema()
            input("\nPresiona ENTER para continuar...")
        
        elif opcion == '8':
            mostrar_estadisticas()
            input("\nPresiona ENTER para continuar...")
        
        elif opcion == '9':
            print("\n💡 Ejecuta en otra terminal:")
            print("   python monitor.py watch")
            input("\nPresiona ENTER para continuar...")
        
        elif opcion == '0':
            print("\n¡Hasta luego! 👋\n")
            break
        
        else:
            print("\n❌ Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Modo línea de comandos
        cmd = sys.argv[1].lower()
        
        if cmd == "todo":
            ejecutar_pipeline_completo()
        
        elif cmd == "entrenar":
            train_model.ejecutar_entrenamiento_autonomo()
        
        elif cmd == "procesar":
            dataset_processor.procesar_dataset_completo(Config.DATA_PATH)
        
        elif cmd == "probar":
            if len(sys.argv) > 2:
                probar_ocr_basico(sys.argv[2])
            else:
                print("❌ Uso: python main.py probar <ruta_imagen>")
        
        elif cmd == "batch":
            if len(sys.argv) > 2:
                procesar_batch()
            else:
                print("❌ Uso: python main.py batch")
        
        elif cmd == "verificar":
            verificar_sistema()
        
        elif cmd == "stats":
            mostrar_estadisticas()
        
        elif cmd == "help":
            print("""
Uso: python main.py [comando] [argumentos]

Comandos disponibles:
  todo                Pipeline completo (procesar + entrenar)
  entrenar           Solo entrenar modelo
  procesar           Solo procesar dataset
  probar <imagen>    Probar OCR con una imagen
  batch              Procesar múltiples imágenes
  verificar          Verificar estado del sistema
  stats              Mostrar estadísticas
  help               Mostrar esta ayuda

Sin argumentos: Modo interactivo con menú

Ejemplos:
  python main.py todo
  python main.py probar test.jpg
  python main.py batch
            """)
        
        else:
            print(f"❌ Comando desconocido: {cmd}")
            print("Usa 'python main.py help' para ver comandos disponibles")
    
    else:
        # Modo interactivo
        menu_interactivo()