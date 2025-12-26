import numpy as np
import cv2
import os

class HandwritingDatasetProcessor:
    def __init__(self, output_dir='processed_handwriting'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _leer_imagen_unicode(self, path):
        try:
            with open(str(path), "rb") as f:
                chunk = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(chunk, cv2.IMREAD_GRAYSCALE)
                return img
        except Exception as e:
            return None

    def _aumentar_imagen(self, img):
        """Crea variaciones de la imagen original (Rotación y Ruido)"""
        variaciones = []
        h, w = img.shape
        # Rotaciones leves (-7 y 7 grados)
        for angulo in [-7, 7]:
            M = cv2.getRotationMatrix2D((w//2, h//2), angulo, 1)
            variaciones.append(cv2.warpAffine(img, M, (w, h), borderValue=0))
        # Ruido suave
        ruido = np.random.randint(0, 2, size=img.shape, dtype=np.uint8) * 255
        variaciones.append(cv2.addWeighted(img, 0.9, ruido, 0.1, 0))
        return variaciones

    def process_single_character_image(self, image_path):
        image = self._leer_imagen_unicode(image_path)
        if image is None: return None
        
        # Umbralado
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return None
        
        # Obtener el contorno más grande
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # --- PROTECCIÓN CONTRA DIMENSIONES CERO ---
        if w <= 0 or h <= 0:
            return None
            
        char_crop = binary[y:y+h, x:x+w]
        canvas = np.zeros((28, 28), dtype=np.uint8)
        
        # Calcular escala asegurando que no haya división por cero
        max_dim = max(w, h)
        scale = 20.0 / max_dim
        
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        # Redimensionar con seguridad
        try:
            char_res = cv2.resize(char_crop, (new_w, new_h))
            # Centrar en el canvas de 28x28
            y_offset = (28 - new_h) // 2
            x_offset = (28 - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_res
            return canvas
        except Exception:
            return None

    def cargar_desde_carpetas(self, ruta_raiz='data'):
        all_images, all_labels = [], []
        categorias = ['Mayusculas', 'Minusculas', 'Numeros']
        
        if not os.path.exists(ruta_raiz):
            print(f"❌ La carpeta {ruta_raiz} no existe.")
            return np.array([]), np.array([])

        for cat in categorias:
            ruta_cat = os.path.join(ruta_raiz, cat)
            if not os.path.exists(ruta_cat): continue
            
            for char_folder in os.listdir(ruta_cat):
                ruta_folder = os.path.join(ruta_cat, char_folder)
                if not os.path.isdir(ruta_folder): continue
                
                for img_name in os.listdir(ruta_folder):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(ruta_folder, img_name)
                        norm = self.process_single_character_image(path)
                        
                        if norm is not None:
                            all_images.append(norm)
                            all_labels.append(char_folder)
                            # Generar aumentadas
                            for aux in self._aumentar_imagen(norm):
                                all_images.append(aux)
                                all_labels.append(char_folder)
        
        return np.array(all_images), np.array(all_labels)