import numpy as np
import os
import json
from ocr_complete import NeuralNetwork
from dataset_processor import cargar_dataset
from sklearn.model_selection import train_test_split

def ejecutar_entrenamiento_autonomo():
    X, y, char_to_idx, idx_to_char = cargar_dataset()
    X_flat = X.reshape(len(X), -1) / 255.0
    
    # One-hot encoding
    y_oh = np.zeros((len(y), 62))
    y_oh[np.arange(len(y)), y] = 1
    
    X_train, X_val, y_train, y_val = train_test_split(X_flat, y_oh, test_size=0.15)
    
    # Red más capaz
    modelo = NeuralNetwork([784, 512, 256, 62], learning_rate=0.001)
    
    print(f"📊 Entrenando con {len(X_train)} muestras...")
    for epoch in range(200):
        # Batch training
        idx = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), 64):
            batch_idx = idx[i:i+64]
            acts = modelo.forward(X_train[batch_idx])
            modelo.backward(acts, y_train[batch_idx])
        
        if epoch % 10 == 0:
            val_preds = np.argmax(modelo.forward(X_val)[-1], axis=1)
            acc = np.mean(val_preds == np.argmax(y_val, axis=1))
            print(f"Época {epoch} | Acc: {acc:.4f}")

    os.makedirs('models', exist_ok=True)
    modelo.save_model('models/ocr_model.pkl')
    print("✅ Modelo optimizado guardado.")

if __name__ == "__main__":
    ejecutar_entrenamiento_autonomo()