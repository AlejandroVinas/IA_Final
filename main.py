import sys
import cv2
import os
import numpy as np
import json
from ocr_complete import OCRSystem, NeuralNetwork

def leer_imagen_unicode(ruta, flags=cv2.IMREAD_COLOR):
    """Lee imágenes en rutas con tildes o caracteres especiales (UTF-8)."""
    try:
        # Usamos numpy para cargar el archivo binario y luego decodificarlo con OpenCV
        with open(ruta, "rb") as f:
            chunk = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(chunk, flags)
    except Exception:
        return None

def preparar_dataset_desde_data(ruta_raiz='data'):
    X, y = [], []
    mapeo_caracteres = [] 
    categorias = ['Mayusculas', 'Minusculas', 'Numeros']

    print(f"1. Analizando estructura de carpetas en '{ruta_raiz}'...")

    for cat in categorias:
        ruta_cat = os.path.join(ruta_raiz, cat)
        if not os.path.exists(ruta_cat): continue
        
        items = sorted(os.listdir(ruta_cat))
        for item in items:
            ruta_item = os.path.join(ruta_cat, item)
            if os.path.isdir(ruta_item):
                if item not in mapeo_caracteres:
                    mapeo_caracteres.append(item)

    char_to_idx = {char: idx for idx, char in enumerate(mapeo_caracteres)}
    idx_to_char = {idx: char for idx, char in enumerate(mapeo_caracteres)}

    print(f"2. Cargando imágenes (corrigiendo rutas con caracteres especiales)...")
    for cat in categorias:
        ruta_cat = os.path.join(ruta_raiz, cat)
        if not os.path.exists(ruta_cat): continue
        
        for caracter in os.listdir(ruta_cat):
            ruta_caracter = os.path.join(ruta_cat, caracter)
            if not os.path.isdir(ruta_caracter): continue
                
            label_idx = char_to_idx[caracter]
            for archivo in os.listdir(ruta_caracter):
                if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(ruta_caracter, archivo)
                    
                    # LLAMADA A LA FUNCIÓN CORREGIDA
                    img = leer_imagen_unicode(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        img_res = cv2.resize(img, (28, 28))
                        X.append(img_res.flatten() / 255.0)
                        label = np.zeros(len(mapeo_caracteres))
                        label[label_idx] = 1
                        y.append(label)

    if not os.path.exists('processed_handwriting'): os.makedirs('processed_handwriting')
    with open('processed_handwriting/handwritten_dataset_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({'idx_to_char': idx_to_char}, f)

    return np.array(X), np.array(y), len(mapeo_caracteres)

def cargar_y_entrenar():
    X, y, num_clases = preparar_dataset_desde_data()
    if len(X) == 0:
        print("Error: No se cargaron imágenes. Revisa la carpeta 'data'.")
        return False

    print(f"3. Entrenando red con {len(X)} imágenes cargadas correctamente...")
    nn = NeuralNetwork(layers=[784, 128, num_clases], learning_rate=0.01)
    
    for epoch in range(51):
        acts = nn.forward(X)
        nn.backward(acts, y)
        if epoch % 10 == 0:
            loss = -np.mean(y * np.log(acts[-1] + 1e-8))
            print(f"   Época {epoch}/50 - Error: {loss:.4f}")

    if not os.path.exists('models'): os.makedirs('models')
    nn.save_model('models/ocr_model.pkl')
    print("4. Modelo entrenado con éxito.")
    return True

def procesar_automatico():
    if not cargar_y_entrenar(): return

    ocr = OCRSystem()
    ruta_input = 'imagenes_a_procesar'
    archivo_txt = 'resultados.txt'

    if not os.path.exists(ruta_input):
        os.makedirs(ruta_input)
        print(f"\n[!] Carpeta '{ruta_input}' creada. Pon tus imágenes allí.")
        return

    print(f"\n5. Procesando imágenes de la carpeta '{ruta_input}'...")
    archivos = [f for f in os.listdir(ruta_input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not archivos:
        print("   -> No hay imágenes nuevas en 'imagenes_a_procesar'.")
        return

    with open(archivo_txt, 'w', encoding='utf-8') as f:
        f.write("REPORTE OCR\n" + "="*30 + "\n")
        for nombre in archivos:
            img_path = os.path.join(ruta_input, nombre)
            img = leer_imagen_unicode(img_path) # Usar versión unicode aquí también
            if img is not None:
                resultado = ocr.process_image(img)
                f.write(f"{nombre}: {resultado}\n")
                print(f"   [OK] {nombre} -> {resultado}")

    print(f"\n[FIN] Resultados guardados en '{archivo_txt}'.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "procesar":
        procesar_automatico()
    else:
        print("Uso: python main.py procesar")