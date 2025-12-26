import sys
import cv2
import os
import numpy as np
import json
import pickle
from ocr_complete import OCRSystem, NeuralNetwork
from dataset_processor import HandwritingDatasetProcessor

def leer_imagen_unicode(ruta, flags=cv2.IMREAD_COLOR):
    """Lee imágenes en rutas con tildes o caracteres especiales."""
    try:
        with open(ruta, "rb") as f:
            chunk = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(chunk, flags)
    except Exception:
        return None

def cargar_y_entrenar():
    """Carga los datos con aumento y entrena la red neuronal."""
    print("\n" + "="*50)
    print("      INICIANDO PROCESO DE ENTRENAMIENTO")
    print("="*50)

    # 1. Cargar y aumentar datos
    dp = HandwritingDatasetProcessor()
    print("[1] Cargando imágenes y generando variaciones (Data Augmentation)...")
    X, y_labels = dp.cargar_desde_carpetas('data')
    
    if len(X) == 0:
        print("❌ ERROR: No se encontraron imágenes en la carpeta 'data'.")
        return False

    # 2. Crear mapeo de clases (Mayúsculas, Minúsculas y Números)
    clases_unicas = sorted(list(set(y_labels)))
    char_to_idx = {char: i for i, char in enumerate(clases_unicas)}
    idx_to_char = {i: char for char, i in char_to_idx.items()}
    num_clases = len(clases_unicas)
    
    print(f"[2] Clases detectadas: {num_clases}")
    print(f"[+] Total de muestras (originales + aumentadas): {len(X)}")

    # 3. Preparar datos para la red
    X_flat = X.reshape(len(X), -1) / 255.0  # Normalizar 0-1
    y_oh = np.zeros((len(y_labels), num_clases))
    for i, label in enumerate(y_labels):
        y_oh[i, char_to_idx[label]] = 1

    # 4. Configurar Red Neuronal (Arquitectura Reforzada)
    # Capas: Entrada(784) -> Oculta1(256) -> Oculta2(128) -> Salida(num_clases)
    # Learning rate bajado a 0.01 para mayor estabilidad
    nn = NeuralNetwork([784, 256, 128, num_clases], learning_rate=0.01)

    # 5. Bucle de entrenamiento
    print(f"[3] Entrenando durante 150 épocas...")
    for epoch in range(151):
        # Mezclar datos en cada época (shuffling)
        idx = np.random.permutation(len(X_flat))
        X_shuffled = X_flat[idx]
        y_shuffled = y_oh[idx]
        
        # Mini-batch training para mejor convergencia
        batch_size = 32
        for i in range(0, len(X_flat), batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            acts = nn.forward(batch_x)
            nn.backward(acts, batch_y)
            
        if epoch % 10 == 0:
            predicciones = nn.forward(X_flat)[-1]
            error = np.mean(np.square(y_oh - predicciones))
            # Calcular precisión simple
            acc = np.mean(np.argmax(predicciones, axis=1) == np.argmax(y_oh, axis=1)) * 100
            print(f"    Época {epoch:3}/150 | Error: {error:.6f} | Precisión: {acc:.2f}%")

    # 6. Guardar el modelo y el mapeo
    os.makedirs('models', exist_ok=True)
    with open('models/ocr_model.pkl', 'wb') as f:
        pickle.dump(nn, f)
    
    with open('models/ocr_model_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({'idx_to_char': idx_to_char}, f, indent=4)
        
    print("\n✅ Modelo entrenado y guardado en 'models/'.")
    return True

def procesar_imagenes():
    """Usa el modelo guardado para leer imágenes en la carpeta de entrada."""
    if not os.path.exists('models/ocr_model.pkl'):
        print("❌ ERROR: No existe un modelo entrenado. Ejecuta 'python main.py entrenar' primero.")
        return

    ocr = OCRSystem()
    ruta_input = 'imagenes_a_procesar'
    archivo_txt = 'resultados.txt'

    if not os.path.exists(ruta_input):
        os.makedirs(ruta_input)
        print(f"[!] Carpeta '{ruta_input}' creada. Añade imágenes allí.")
        return

    archivos = [f for f in os.listdir(ruta_input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not archivos:
        print("[-] No hay imágenes para procesar en 'imagenes_a_procesar'.")
        return

    print(f"[*] Procesando {len(archivos)} imágenes...")
    with open(archivo_txt, 'w', encoding='utf-8') as f:
        f.write("REPORTE OCR - RESULTADOS\n" + "="*30 + "\n")
        for nombre in archivos:
            img_path = os.path.join(ruta_input, nombre)
            img = leer_imagen_unicode(img_path)
            if img is not None:
                resultado = ocr.process_image(img)
                f.write(f"Archivo: {nombre}\nResultado: {resultado}\n" + "-"*30 + "\n")
                print(f"   [OK] {nombre} -> {resultado}")

    print(f"\n✅ Proceso terminado. Resultados guardados en '{archivo_txt}'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUso: python main.py [entrenar | procesar | todo]")
        sys.exit(1)

    comando = sys.argv[1].lower()

    if comando == "entrenar":
        cargar_y_entrenar()
    elif comando == "procesar":
        procesar_imagenes()
    elif comando == "todo":
        if cargar_y_entrenar():
            procesar_imagenes()
    else:
        print(f"❌ Comando '{comando}' no reconocido.")