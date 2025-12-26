import numpy as np
import cv2
import pickle
import json
import os

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.001):
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
            dw = np.dot(acts[i].T, delta) / len(y)
            db = np.sum(delta, axis=0, keepdims=True) / len(y)
            if i > 0: delta = np.dot(delta, self.weights[i].T) * (acts[i] > 0)
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

    def save_model(self, path):
        with open(path, 'wb') as f: pickle.dump(self, f)

class OCRSystem:
    def __init__(self, model_path='models/ocr_model.pkl'):
        self.model = None
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f: self.model = pickle.load(f)
        
        mapping_path = 'processed_handwriting/handwritten_dataset_mapping.json'
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.idx_to_char = json.load(f)['idx_to_char']

    def process_image(self, image, return_confidence=False):
        if self.model is None: return ("Error", 0) if return_confidence else "Error"
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
        _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = sorted([cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 5], key=lambda b: b[0])

        texto = ""
        confianzas = []
        for i, (x, y, w, h) in enumerate(boxes):
            if i > 0 and (x - (boxes[i-1][0] + boxes[i-1][2])) > w * 0.7:
                texto += " " # Espacio detectado

            char_crop = bin[y:y+h, x:x+w]
            canvas = np.zeros((28, 28), dtype=np.uint8)
            scale = 20.0 / max(w, h)
            nw, nh = int(w * scale), int(h * scale)
            res = cv2.resize(char_crop, (nw, nh))
            canvas[(28-nh)//2:(28-nh)//2+nh, (28-nw)//2:(28-nw)//2+nw] = res
            
            probs = self.model.forward(canvas.flatten().reshape(1, -1) / 255.0)[-1][0]
            idx = np.argmax(probs)
            texto += self.idx_to_char.get(str(idx), "?")
            confianzas.append(probs[idx])

        return (texto, np.mean(confianzas)) if return_confidence else texto