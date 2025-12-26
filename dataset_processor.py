import numpy as np
import cv2
import os
from pathlib import Path
import json
from datetime import datetime

class HandwritingDatasetProcessor:
    def __init__(self, output_dir='processed_handwriting'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.stats = {'total_characters': 0, 'failed_images': []}

    def _leer_imagen_unicode(self, path):
        try:
            with open(str(path), "rb") as f:
                chunk = np.frombuffer(f.read(), dtype=np.uint8)
                return cv2.imdecode(chunk, cv2.IMREAD_UNCHANGED)
        except: return None

    def process_single_character_image(self, image_path):
        image = self._leer_imagen_unicode(image_path)
        if image is None: return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return self.normalize_character(binary[y:y+h, x:x+w])

    def normalize_character(self, char_img, target_size=(28, 28)):
        h, w = char_img.shape
        scale = 20.0 / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(char_img, (nw, nh), interpolation=cv2.INTER_AREA)
        res = np.zeros(target_size, dtype=np.uint8)
        res[(28-nh)//2:(28-nh)//2+nh, (28-nw)//2:(28-nw)//2+nw] = resized
        return res

    def process_directory_structure(self, base_dir):
        base_path = Path(base_dir)
        all_images, all_labels = [], []
        categories = {'Mayusculas': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'Minusculas': 'abcdefghijklmnopqrstuvwxyz', 'Numeros': '0123456789'}
        
        for cat_name, chars in categories.items():
            cat_path = base_path / cat_name
            for char in chars:
                char_dir = cat_path / char
                if not char_dir.exists(): continue
                files = list(char_dir.glob('*.png')) + list(char_dir.glob('*.jpg'))
                for img_path in files:
                    norm = self.process_single_character_image(img_path)
                    if norm is not None:
                        all_images.append(norm); all_labels.append(char)
        return np.array(all_images), np.array(all_labels)

    def save_dataset(self, X, y):
        # DICCIONARIO FIJO DE 62 CLASES (Evita desplazamientos)
        caracteres_fijos = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        char_to_idx = {char: i for i, char in enumerate(caracteres_fijos)}
        idx_to_char = {i: char for char, i in char_to_idx.items()}
        
        X_filtrado, y_idx = [], []
        for i in range(len(y)):
            if y[i] in char_to_idx:
                X_filtrado.append(X[i])
                y_idx.append(char_to_idx[y[i]])
                
        path = os.path.join(self.output_dir, 'handwritten_dataset.npz')
        np.savez_compressed(path, X=np.array(X_filtrado), y=np.array(y_idx))
        with open(path.replace('.npz', '_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)
        return path

def cargar_dataset(dataset_path='processed_handwriting/handwritten_dataset.npz'):
    data = np.load(dataset_path, allow_pickle=True)
    with open(dataset_path.replace('.npz', '_mapping.json'), 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return data['X'], data['y'], mapping['char_to_idx'], mapping['idx_to_char']