import numpy as np
import cv2
import pickle
import json

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        # Heavier initialization for better convergence
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
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * (acts[i] > 0)
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

class OCRSystem:
    def __init__(self, model_path='models/ocr_model.pkl'):
        self.model = None
        self.idx_to_char = {}
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f: self.model = pickle.load(f)
            map_path = model_path.replace('.pkl', '_mapping.json')
            if os.path.exists(map_path):
                with open(map_path, 'r') as f: self.idx_to_char = json.load(f)['idx_to_char']

    def process_image(self, image):
        if self.model is None: return "Error: Sin modelo"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
        _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # FILTRO DE SEGMENTACIÓN MEJORADO (Área > 40 y altura > 12)
        boxes = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if cv2.contourArea(c) > 40 and h > 12:
                boxes.append((x, y, w, h))
        boxes = sorted(boxes, key=lambda b: b[0])

        texto = ""
        for i, (x, y, w, h) in enumerate(boxes):
            # Detectar espacios entre letras
            if i > 0 and (x - (boxes[i-1][0] + boxes[i-1][2])) > w * 0.8:
                texto += " "
            
            char_crop = bin[y:y+h, x:x+w]
            # Centrar en 28x28 (idéntico al entrenamiento)
            canvas = np.zeros((28, 28))
            scale = 20.0 / max(w, h)
            char_res = cv2.resize(char_crop, (int(w*scale), int(h*scale)))
            nh, nw = char_res.shape
            canvas[(28-nh)//2:(28-nh)//2+nh, (28-nw)//2:(28-nw)//2+nw] = char_res
            
            out = self.model.forward(canvas.flatten().reshape(1, -1) / 255.0)[-1]
            texto += self.idx_to_char.get(str(np.argmax(out)), "?")
        return texto