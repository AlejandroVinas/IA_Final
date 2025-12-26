import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from ocr_complete import NeuralNetwork
from dataset_processor import cargar_dataset

def augment_image(img_flat):
    """Crea una versión rotada o movida de la imagen"""
    img = img_flat.reshape(28, 28)
    # Rotación aleatoria
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    img = cv2.warpAffine(img, M, (28, 28))
    # Traslación aleatoria (mover un poco)
    tx, ty = np.random.uniform(-3, 3, 2)
    M_t = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_t, (28, 28))
    return img.flatten()

def ejecutar_entrenamiento_autonomo():
    print("🚀 Cargando datos y aplicando Aumentación...")
    X, y, char_to_idx, _ = cargar_dataset()
    if X is None: return

    X_norm = X.reshape(X.shape[0], -1) / 255.0
    
    # AUMENTACIÓN: Multiplicar x6 el dataset
    X_aug, y_aug = [], []
    for i in range(len(X_norm)):
        X_aug.append(X_norm[i]); y_aug.append(y[i])
        for _ in range(5): # 5 copias variadas
            X_aug.append(augment_image(X_norm[i]))
            y_aug.append(y[i])
    
    X_train, X_test, y_train, y_test = train_test_split(np.array(X_aug), np.array(y_aug), test_size=0.15)
    num_clases = len(char_to_idx)
    y_train_oh = np.eye(num_clases)[y_train]
    
    # Red Neuronal (Más neuronas para aprender 62 clases)
    modelo = NeuralNetwork([784, 512, 256, num_clases], learning_rate=0.005)
    
    epochs = 150
    batch_size = 64
    print(f"📊 Entrenando con {len(X_train)} muestras aumentadas...")
    
    for e in range(epochs):
        perm = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            acts = modelo.forward(X_train[idx])
            modelo.backward(acts, y_train_oh[idx])
        
        if e % 10 == 0 or e == epochs-1:
            pred = np.argmax(modelo.forward(X_test)[-1], axis=1)
            acc = np.mean(pred == y_test)
            print(f"Época {e}/{epochs} | Precisión Validacion: {acc:.4f}")

    os.makedirs('models', exist_ok=True)
    modelo.save_model('models/ocr_model.pkl')
    print("✅ Modelo optimizado guardado.")

if __name__ == "__main__":
    ejecutar_entrenamiento_autonomo()