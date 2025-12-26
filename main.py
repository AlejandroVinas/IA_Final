import sys, cv2, os, numpy as np, json, pickle
from ocr_complete import OCRSystem, NeuralNetwork
from dataset_processor import HandwritingDatasetProcessor

def cargar_y_entrenar():
    print("\n[*] Iniciando carga de datos...")
    dp = HandwritingDatasetProcessor()
    # Carga desde la carpeta 'data'
    X, y_labels = dp.cargar_desde_carpetas('data')
    
    if len(X) == 0:
        print("❌ Error: No hay imágenes en 'data'. Revisa la estructura.")
        return False

    # ALFABETO ESTRICTO: Forzamos el orden 0-9, A-Z, a-z
    clases_ordenadas = sorted(list(set(y_labels)))
    char_to_idx = {char: i for i, char in enumerate(clases_ordenadas)}
    idx_to_char = {i: char for char, i in char_to_idx.items()}
    
    # Preparar datos para la red
    X_flat = X.reshape(len(X), -1) / 255.0
    y_oh = np.zeros((len(y_labels), len(clases_ordenadas)))
    for i, label in enumerate(y_labels):
        y_oh[i, char_to_idx[label]] = 1

    print(f"[*] Entrenando con {len(X)} imágenes y {len(clases_ordenadas)} clases...")
    # Capas: 784 (entrada) -> 256 -> 128 -> N clases (salida)
    nn = NeuralNetwork([784, 256, 128, len(clases_ordenadas)], learning_rate=0.01)

    
    for epoch in range(151):
        idx = np.random.permutation(len(X_flat))
        X_s, y_s = X_flat[idx], y_oh[idx]
        
        for i in range(0, len(X_flat), 64):
            batch_x, batch_y = X_s[i:i+64], y_s[i:i+64]
            nn.backward(nn.forward(batch_x), batch_y)
            
        if epoch % 10 == 0:
            pred = nn.forward(X_flat)[-1]
            acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_oh, axis=1)) * 100
            print(f"    Época {epoch}/150 - Precisión Entrenamiento: {acc:.2f}%")

    # Guardar modelo y mapeo
    os.makedirs('models', exist_ok=True)
    with open('models/ocr_model.pkl', 'wb') as f:
        pickle.dump(nn, f)
    with open('models/ocr_model_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({'idx_to_char': idx_to_char}, f)
        
    print("✅ Modelo y Mapeo sincronizados y guardados en /models")
    return True

if __name__ == "__main__":
    accion = sys.argv[1].lower() if len(sys.argv) > 1 else "ayuda"
    
    if accion == "entrenar":
        cargar_y_entrenar()
    elif accion == "procesar":
        ocr = OCRSystem()
        # Procesar imágenes de prueba
        if os.path.exists('imagenes_a_procesar'):
            for img_n in os.listdir('imagenes_a_procesar'):
                img = cv2.imread(os.path.join('imagenes_a_procesar', img_n))
                if img is not None:
                    print(f"Resultado {img_n}: {ocr.process_image(img)}")