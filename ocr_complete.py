import numpy as np
import cv2
import pickle
import json
import os

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i]) for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, s)) for s in layers[1:]]
        self.lr = learning_rate

    def forward(self, x):
        acts = [x]
        for i in range(len(self.weights)-1):
            x = np.maximum(0, np.dot(x, self.weights[i]) + self.biases[i])
            acts.append(x)
        exp = np.exp(np.dot(x, self.weights[-1]) + self.biases[-1] - np.max(x))
        acts.append(exp / np.sum(exp, axis=1, keepdims=True))
        return acts

    def backward(self, acts, y):
        delta = acts[-1] - y
        for i in range(len(self.weights)-1, -1, -1):
            dw = np.dot(acts[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            if i > 0: delta = np.dot(delta, self.weights[i].T) * (acts[i] > 0)
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

    def save_model(self, path):
        with open(path, 'wb') as f: pickle.dump(self, f)

class OCRSystem:
    def __init__(self, model_path='models/ocr_model.pkl'):
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f: self.model = pickle.load(f)
        with open('processed_handwriting/handwritten_dataset_mapping.json', 'r') as f:
            self.idx_to_char = json.load(f)['idx_to_char']

    def process_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
        # Mejor binarización para el test
        bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Segmentar (Dilatar un poco para unir trazos débiles)
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(bin, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]
        boxes.sort(key=lambda b: b[0]) # De izquierda a derecha

        texto = ""
        for (x, y, w, h) in boxes:
            char_crop = bin[y:y+h, x:x+w]
            # Normalización idéntica al entrenamiento
            canvas = np.zeros((28, 28), dtype=np.uint8)
            f = 20 / max(w, h)
            nw, nh = max(1, int(w*f)), max(1, int(h*f))
            res = cv2.resize(char_crop, (nw, nh))
            canvas[(28-nh)//2:(28-nh)//2+nh, (28-nw)//2:(28-nw)//2+nw] = res
            
            # Predecir
            out = self.model.forward(canvas.flatten().reshape(1, -1) / 255.0)[-1]
            texto += self.idx_to_char.get(str(np.argmax(out)), "?")
        return texto

def process_image_file(path, model_path='models/ocr_model.pkl', output_path=None):
    ocr = OCRSystem(model_path)
    img = cv2.imread(path)
    txt = ocr.process_image(img)
    if output_path:
        with open(output_path, 'w') as f: f.write(txt)
    return txt