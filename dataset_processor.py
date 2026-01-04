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
        if image is None: 
            return None
        
        # Umbralado
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: 
            return None
        
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
        """
        Carga datos de múltiples fuentes:
        1. Carpetas originales (Mayusculas, Minusculas, Numeros)
        2. Manuscritos propios (data_manuscrito_propio)
        """
        all_images, all_labels = [], []
        
        print("\n" + "="*70)
        print("📂 CARGANDO DATASETS")
        print("="*70)
        
        # 1. CARGAR DATOS ORIGINALES
        categorias = ['Mayusculas', 'Minusculas', 'Numeros']
        
        if os.path.exists(ruta_raiz):
            print(f"\n[*] Cargando datos originales de: {ruta_raiz}/")
            
            for cat in categorias:
                ruta_cat = os.path.join(ruta_raiz, cat)
                if not os.path.exists(ruta_cat): 
                    continue
                
                cat_count = 0
                for char_folder in os.listdir(ruta_cat):
                    ruta_folder = os.path.join(ruta_cat, char_folder)
                    if not os.path.isdir(ruta_folder): 
                        continue
                    
                    char_count = 0
                    for img_name in os.listdir(ruta_folder):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            path = os.path.join(ruta_folder, img_name)
                            norm = self.process_single_character_image(path)
                            
                            if norm is not None:
                                all_images.append(norm)
                                all_labels.append(char_folder)
                                char_count += 1
                                
                                # Generar aumentadas
                                for aux in self._aumentar_imagen(norm):
                                    all_images.append(aux)
                                    all_labels.append(char_folder)
                    
                    if char_count > 0:
                        cat_count += char_count
                
                if cat_count > 0:
                    print(f"    ✓ {cat}: {cat_count} caracteres base")
        
        original_count = len(all_images)
        print(f"\n[📊] Total datos originales (con aumentación): {original_count}")
        
        # 2. CARGAR MANUSCRITOS PROPIOS
        manuscrito_dir = 'data_manuscrito_propio'
        if os.path.exists(manuscrito_dir):
            print(f"\n[*] Cargando manuscritos propios de: {manuscrito_dir}/")
            manuscritos_base = 0
            manuscritos_total = 0
            
            for char_folder in sorted(os.listdir(manuscrito_dir)):
                char_path = os.path.join(manuscrito_dir, char_folder)
                if not os.path.isdir(char_path):
                    continue
                
                char_count = 0
                for img_file in os.listdir(char_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(char_path, img_file)
                        # Estos ya están en 28x28, solo cargarlos
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        if img is not None and img.shape == (28, 28):
                            all_images.append(img)
                            all_labels.append(char_folder)
                            char_count += 1
                            manuscritos_base += 1
                            manuscritos_total += 1
                            
                            # También aumentar estos
                            for aux in self._aumentar_imagen(img):
                                all_images.append(aux)
                                all_labels.append(char_folder)
                                manuscritos_total += 1
                
                if char_count > 0:
                    print(f"    ✓ {char_folder}: {char_count} imágenes propias")
            
            print(f"\n[📊] Total manuscritos propios:")
            print(f"    - Base: {manuscritos_base}")
            print(f"    - Con aumentación: {manuscritos_total}")
        else:
            print(f"\n[ℹ️] No hay manuscritos propios en: {manuscrito_dir}/")
            print(f"    💡 Usa 'python agregar_manuscritos.py' para añadir tus datos")
        
        print("\n" + "="*70)
        print(f"[✅] TOTAL FINAL: {len(all_images)} imágenes para entrenamiento")
        print("="*70 + "\n")
        
        if len(all_images) == 0:
            print("❌ Error: No hay imágenes. Revisa las carpetas.")
            return np.array([]), np.array([])
        
        return np.array(all_images), np.array(all_labels)