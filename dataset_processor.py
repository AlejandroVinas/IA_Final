import numpy as np
import cv2
import os
from pathlib import Path
import json

class HandwritingDatasetProcessor:
    def __init__(self, output_dir='processed_handwriting'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _leer_imagen_unicode(self, path):
        try:
            with open(str(path), "rb") as f:
                chunk = np.frombuffer(f.read(), dtype=np.uint8)
                return cv2.imdecode(chunk, cv2.IMREAD_UNCHANGED)
        except: return None

    def normalize_character(self, char_img, target_size=(28, 28)):
        """Limpia el ruido y centra la letra perfectamente"""
        if char_img is None or char_img.size == 0:
            return np.zeros(target_size, dtype=np.uint8)
        
        # 1. Eliminar bordes vacíos extra
        coords = cv2.findNonZero(char_img)
        if coords is None: return np.zeros(target_size, dtype=np.uint8)
        x, y, w, h = cv2.boundingRect(coords)
        char_img = char_img[y:y+h, x:x+w]

        # 2. Redimensionar manteniendo aspecto
        f = 20 / max(w, h)
        nw, nh = max(1, int(w*f)), max(1, int(h*f))
        res = cv2.resize(char_img, (nw, nh), interpolation=cv2.INTER_AREA)

        # 3. Centrar en lienzo 28x28
        canvas = np.zeros(target_size, dtype=np.uint8)
        y_off, x_off = (28-nh)//2, (28-nw)//2
        canvas[y_off:y_off+nh, x_off:x_off+nw] = res
        return canvas

    def process_directory_structure(self, base_dir):
        base_path = Path(base_dir)
        X, y = [], []
        cats = {'Mayusculas': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'Minusculas': 'abcdefghijklmnopqrstuvwxyz', 'Numeros': '0123456789'}

        for cat, chars in cats.items():
            for char in chars:
                char_dir = base_path / cat / char
                if not char_dir.exists(): continue
                files = list(char_dir.glob('*.png')) + list(char_dir.glob('*.jpg'))
                for img_path in files:
                    img = self._leer_imagen_unicode(img_path)
                    if img is None: continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
                    _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    X.append(self.normalize_character(bin))
                    y.append(char)
                print(f"✓ Clase {char} procesada")
        return np.array(X), np.array(y)

    def save_dataset(self, X, y):
        unique_labels = sorted(list(set(y)))
        char_to_idx = {char: i for i, char in enumerate(unique_labels)}
        idx_to_char = {i: char for char, i in char_to_idx.items()}
        y_idx = np.array([char_to_idx[lbl] for lbl in y])
        path = os.path.join(self.output_dir, 'handwritten_dataset.npz')
        np.savez_compressed(path, X=X, y=y_idx)
        with open(path.replace('.npz', '_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)
        return path

def cargar_dataset(path='processed_handwriting/handwritten_dataset.npz'):
    if not os.path.exists(path): return None, None, None, None
    data = np.load(path)
    with open(path.replace('.npz', '_mapping.json'), 'r') as f:
        m = json.load(f)
    return data['X'], data['y'], m['char_to_idx'], m['idx_to_char']

def procesar_dataset_completo(path):
    p = HandwritingDatasetProcessor()
    X, y = p.process_directory_structure(path)
    return p.save_dataset(X, y) if len(X) > 0 else None