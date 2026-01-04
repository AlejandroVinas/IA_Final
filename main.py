import sys, cv2, os, numpy as np, json, pickle
from ocr_complete import OCRSystem, NeuralNetwork
from dataset_processor import HandwritingDatasetProcessor

def cargar_datos_digitales():
    """Carga los datos generados por generar_digital.py desde data_digital/"""
    print("\n[*] Cargando datos DIGITALES desde data_digital/...")
    
    if not os.path.exists('data_digital'):
        print("⚠️  Carpeta 'data_digital' no encontrada. Ejecuta generar_digital.py primero.")
        return np.array([]), []
    
    X_digital = []
    y_digital = []
    
    for char_folder in os.listdir('data_digital'):
        char_path = os.path.join('data_digital', char_folder)
        if not os.path.isdir(char_path):
            continue
            
        for img_file in os.listdir(char_path):
            if img_file.endswith('.png'):
                img_path = os.path.join(char_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Redimensionar a 28x28 (mismo que manuscritos)
                    img_resized = cv2.resize(img, (28, 28))
                    X_digital.append(img_resized)
                    y_digital.append(char_folder)
    
    print(f"    ✓ {len(X_digital)} imágenes digitales cargadas")
    return np.array(X_digital), y_digital

def cargar_y_entrenar():
    print("\n" + "="*70)
    print("🚀 ENTRENAMIENTO HÍBRIDO: MANUSCRITO + DIGITAL")
    print("="*70)
    
    # 1. CARGAR DATOS MANUSCRITOS
    print("\n[*] Iniciando carga de datos MANUSCRITOS...")
    dp = HandwritingDatasetProcessor()
    X_manuscrito, y_manuscrito = dp.cargar_desde_carpetas('data')
    
    if len(X_manuscrito) == 0:
        print("❌ Error: No hay imágenes en 'data'. Revisa la estructura.")
        return False
    
    print(f"    ✓ {len(X_manuscrito)} imágenes manuscritas cargadas")
    
    # 2. CARGAR DATOS DIGITALES
    X_digital, y_digital = cargar_datos_digitales()
    
    # 3. COMBINAR AMBOS DATASETS
    if len(X_digital) > 0:
        print("\n[*] Combinando datasets...")
        X_combined = np.concatenate([X_manuscrito, X_digital], axis=0)
        # Convertir ambos a listas para concatenar correctamente
        y_combined = list(y_manuscrito) + list(y_digital)
        print(f"    ✓ Total combinado: {len(X_combined)} imágenes")
        print(f"      - Manuscritas: {len(X_manuscrito)}")
        print(f"      - Digitales: {len(X_digital)}")
    elif accion == "debug":
        print("\n" + "="*70)
        print("🔍 MODO DEBUG - Procesamiento Detallado")
        print("="*70)
        
        if not os.path.exists('models/ocr_model.pkl'):
            print("❌ Error: Modelo no encontrado. Ejecuta 'python main.py entrenar'")
            sys.exit(1)
        
        ocr = OCRSystem(model_path='models/ocr_model.pkl')
        
        carpeta_proceso = 'imagenes_a_procesar'
        if not os.path.exists(carpeta_proceso):
            print(f"❌ Carpeta '{carpeta_proceso}' no encontrada")
            sys.exit(1)
        
        imagenes = [f for f in os.listdir(carpeta_proceso) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not imagenes:
            print(f"❌ No hay imágenes en '{carpeta_proceso}'")
            sys.exit(1)
        
        # Procesar solo la primera imagen en modo debug
        img_name = imagenes[0]
        print(f"\n📄 Procesando: {img_name} (modo DEBUG)")
        print("   Presiona cualquier tecla para avanzar en cada paso\n")
        
        img_path = os.path.join(carpeta_proceso, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            resultado, confianza = ocr.process_image(img, return_confidence=True, debug=True)
            print(f"\n✅ Resultado final: '{resultado}'")
            print(f"✅ Confianza promedio: {confianza:.1%}")
        
        print("="*70)
    
    else:
        print("\n⚠️  No se encontraron datos digitales, usando solo manuscritos")
        X_combined = X_manuscrito
        y_combined = y_manuscrito
    
    # 4. PREPARAR DATOS PARA ENTRENAMIENTO
    print("\n[*] Preparando datos para entrenamiento...")
    
    # ALFABETO ESTRICTO: Orden 0-9, A-Z, a-z
    clases_ordenadas = sorted(list(set(y_combined)))
    char_to_idx = {char: i for i, char in enumerate(clases_ordenadas)}
    idx_to_char = {i: char for char, i in char_to_idx.items()}
    
    print(f"    ✓ {len(clases_ordenadas)} clases detectadas: {clases_ordenadas}")
    
    # Normalizar y aplanar
    X_flat = X_combined.reshape(len(X_combined), -1) / 255.0
    
    # One-hot encoding
    y_oh = np.zeros((len(y_combined), len(clases_ordenadas)))
    for i, label in enumerate(y_combined):
        y_oh[i, char_to_idx[label]] = 1
    
    # 5. ENTRENAR RED NEURONAL
    print("\n[*] Entrenando red neuronal...")
    print(f"    Arquitectura: 784 -> 256 -> 128 -> {len(clases_ordenadas)}")
    
    nn = NeuralNetwork([784, 256, 128, len(clases_ordenadas)], learning_rate=0.01)
    
    # Entrenamiento con más épocas para mejor convergencia
    epochs = 200
    batch_size = 64
    
    for epoch in range(epochs + 1):
        # Mezclar datos
        idx = np.random.permutation(len(X_flat))
        X_shuffled, y_shuffled = X_flat[idx], y_oh[idx]
        
        # Entrenamiento por batches
        for i in range(0, len(X_flat), batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            nn.backward(nn.forward(batch_x), batch_y)
        
        # Mostrar progreso cada 10 épocas
        if epoch % 10 == 0:
            pred = nn.forward(X_flat)[-1]
            acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_oh, axis=1)) * 100
            print(f"    Época {epoch}/{epochs} - Accuracy: {acc:.2f}%")
    
    # 6. GUARDAR MODELO Y MAPEO
    print("\n[*] Guardando modelo...")
    os.makedirs('models', exist_ok=True)
    
    with open('models/ocr_model.pkl', 'wb') as f:
        pickle.dump(nn, f)
    
    with open('models/ocr_model_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({'idx_to_char': idx_to_char}, f)
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"✓ Modelo guardado en: models/ocr_model.pkl")
    print(f"✓ Mapeo guardado en: models/ocr_model_mapping.json")
    print(f"✓ Accuracy final: {acc:.2f}%")
    print(f"✓ Total de imágenes entrenadas: {len(X_combined)}")
    print("="*70)
    
    return True

if __name__ == "__main__":
    accion = sys.argv[1].lower() if len(sys.argv) > 1 else "ayuda"
    
    if accion == "entrenar":
        exito = cargar_y_entrenar()
        if exito:
            print("\n💡 Siguiente paso: python test_cumplimiento.py")
        
    elif accion == "procesar":
        print("\n" + "="*70)
        print("🔍 PROCESANDO IMÁGENES")
        print("="*70)
        
        if not os.path.exists('models/ocr_model.pkl'):
            print("❌ Error: Modelo no encontrado. Ejecuta 'python main.py entrenar'")
            sys.exit(1)
        
        ocr = OCRSystem(model_path='models/ocr_model.pkl')
        
        # Procesar imágenes de prueba
        carpeta_proceso = 'imagenes_a_procesar'
        if not os.path.exists(carpeta_proceso):
            print(f"❌ Carpeta '{carpeta_proceso}' no encontrada")
            print(f"💡 Crea la carpeta y coloca ahí tus imágenes")
            sys.exit(1)
        
        imagenes = [f for f in os.listdir(carpeta_proceso) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not imagenes:
            print(f"❌ No hay imágenes en '{carpeta_proceso}'")
            sys.exit(1)
        
        print(f"\n📁 Encontradas {len(imagenes)} imágenes\n")
        
        # Crear carpeta para resultados
        carpeta_resultados = 'resultados_ocr'
        os.makedirs(carpeta_resultados, exist_ok=True)
        
        for img_name in imagenes:
            img_path = os.path.join(carpeta_proceso, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"⚠️  No se pudo leer: {img_name}")
                continue
            
            try:
                # Procesar con visualización
                base_name = os.path.splitext(img_name)[0]
                vis_path = os.path.join(carpeta_resultados, f"{base_name}_resultado.png")
                
                resultado, confianza, img_vis = ocr.process_image_with_visualization(img, vis_path)
                
                print(f"📄 {img_name}")
                print(f"   ✓ Resultado: '{resultado}'")
                print(f"   ✓ Confianza: {confianza:.1%}")
                print(f"   ✓ Visualización: {vis_path}")
                
                # Guardar resultado en TXT
                txt_path = os.path.join(carpeta_resultados, f"{base_name}_texto.txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"Archivo: {img_name}\n")
                    f.write(f"Texto reconocido: {resultado}\n")
                    f.write(f"Confianza: {confianza:.1%}\n")
                
                print(f"   ✓ Texto guardado: {txt_path}\n")
                
            except Exception as e:
                print(f"❌ Error procesando {img_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print("="*70)
        print(f"✅ Resultados guardados en: {carpeta_resultados}/")
        print("="*70)
    
    else:
        print("\n" + "="*70)
        print("📖 USO DEL SISTEMA OCR")
        print("="*70)
        print("\nComandos disponibles:")
        print("  python main.py entrenar  - Entrena el modelo con datos híbridos")
        print("  python main.py procesar  - Procesa imágenes y guarda resultados")
        print("  python main.py debug     - Modo debug con visualización paso a paso")
        print("\nPasos recomendados:")
        print("  1. python generar_digital.py    (genera datos digitales)")
        print("  2. python main.py entrenar       (entrena con ambos tipos)")
        print("  3. python test_cumplimiento.py   (verifica funcionamiento)")
        print("\nPara procesar tus imágenes:")
        print("  1. Crea carpeta 'imagenes_a_procesar'")
        print("  2. Coloca tus imágenes ahí (.png, .jpg)")
        print("  3. python main.py procesar")
        print("     O python main.py debug (para ver el proceso paso a paso)")
        print("="*70)