import numpy as np
import cv2
import pickle
import json
import os

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        # Inicialización Xavier para mejor convergencia
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
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            map_path = model_path.replace('.pkl', '_mapping.json')
            if os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Asegurar que los índices sean strings para el dict
                    self.idx_to_char = {str(k): v for k, v in data['idx_to_char'].items()}

    def process_image(self, image, return_confidence=False):
        if self.model is None: return ("Error: No hay modelo", 0) if return_confidence else "Error"
        
        # Preprocesamiento
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # DILATACIÓN: Engrosa los trazos para que las letras no se rompan en pedazos
        kernel = np.ones((2,2), np.uint8)
        bin_dilated = cv2.dilate(bin, kernel, iterations=1)
        
        
        
        # Segmentación
        contours, _ = cv2.findContours(bin_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            # Filtro de ruido: ignorar cosas muy pequeñas o muy delgadas
            if h > 12 and w > 4:
                boxes.append((x, y, w, h))
        
        # Ordenar de izquierda a derecha
        boxes = sorted(boxes, key=lambda b: b[0])

        texto = ""
        confianzas = []
        
        for i, (x, y, w, h) in enumerate(boxes):
            # Lógica de detección de espacios
            if i > 0:
                gap = x - (boxes[i-1][0] + boxes[i-1][2])
                if gap > w * 0.8: texto += " "

            # Extraer carácter de la imagen original binaria (no la dilatada)
            char_crop = bin[y:y+h, x:x+w]
            
            # Normalización: Centrar en un lienzo de 28x28 (igual que el dataset)
            
            canvas = np.zeros((28, 28), dtype=np.uint8)
            scale = 20.0 / max(w, h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            char_res = cv2.resize(char_crop, (nw, nh), interpolation=cv2.INTER_AREA)
            
            y_off, x_off = (28 - nh) // 2, (28 - nw) // 2
            canvas[y_off:y_off+nh, x_off:x_off+nw] = char_res
            
            # Predicción
            input_data = canvas.flatten().reshape(1, -1) / 255.0
            probs = self.model.forward(input_data)[-1]
            idx = np.argmax(probs)
            
            confianzas.append(np.max(probs))
            char_pred = self.idx_to_char.get(str(idx), "?")
            texto += char_pred

        if return_confidence:
            return texto, np.mean(confianzas) if confianzas else 0
        return texto