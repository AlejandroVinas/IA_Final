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
        self.debug = False  # Activar para ver imágenes de debug
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            map_path = model_path.replace('.pkl', '_mapping.json')
            if os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Asegurar que los índices sean strings para el dict
                    self.idx_to_char = {str(k): v for k, v in data['idx_to_char'].items()}
                    
        print(f"[OCR] Modelo cargado con {len(self.idx_to_char)} clases")

    def corregir_inclinacion(self, image):
        """Detecta y corrige la inclinación de la imagen"""
        try:
            # Encontrar contornos para calcular ángulo
            coords = cv2.findNonZero(image)
            if coords is None or len(coords) < 10:
                return image
            
            # Obtener el rectángulo mínimo rotado
            angle = cv2.minAreaRect(coords)[-1]
            
            # Normalizar el ángulo
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            # Solo corregir si hay inclinación significativa
            if abs(angle) > 2:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
                return rotated
            
        except Exception as e:
            pass
        
        return image
    
    def preprocesar_imagen(self, image):
        """Preprocesamiento robusto de la imagen"""
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # MEJORA 1: Ecualización de histograma para mejorar contraste
        gray = cv2.equalizeHist(gray)
        
        # MEJORA 2: Aplicar blur bilateral (preserva bordes mejor)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # MEJORA 3: Probar múltiples métodos de binarización y elegir el mejor
        # Método 1: Otsu
        _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Método 2: Adaptativo con parámetros ajustados
        binary_adaptive = cv2.adaptiveThreshold(
            blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            blockSize=21,  # Aumentado para captar mejor los trazos
            C=10  # Ajustado para fondos grises
        )
        
        # Elegir el que tenga más píxeles negros (más contenido detectado)
        if np.sum(binary_otsu) > np.sum(binary_adaptive):
            binary = binary_otsu
        else:
            binary = binary_adaptive
        
        # MEJORA 4: Corregir inclinación
        binary = self.corregir_inclinacion(binary)
        
        # MEJORA 5: Limpiar ruido pequeño
        kernel_small = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # MEJORA 6: Conectar trazos rotos (común en manuscritos)
        kernel_close = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        return gray, binary

    def segmentar_caracteres(self, binary):
        """Segmenta caracteres de la imagen binaria"""
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        h_img, w_img = binary.shape
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filtros para eliminar ruido
            # 1. Tamaño mínimo (reducido para manuscritos)
            if h < 8 or w < 4:
                continue
            
            # 2. Relación de aspecto razonable
            aspect_ratio = w / float(h)
            if aspect_ratio > 4 or aspect_ratio < 0.08:
                continue
            
            # 3. Área mínima (reducida para manuscritos)
            if area < 30:
                continue
            
            # 4. No puede ser más grande que la mitad de la imagen
            if h > h_img * 0.9 or w > w_img * 0.5:
                continue
            
            boxes.append((x, y, w, h))
        
        # MEJORA: Si detectamos muy pocos caracteres, intentar segmentar por proyección
        if len(boxes) < 3:
            boxes_projection = self.segmentar_por_proyeccion(binary)
            if len(boxes_projection) > len(boxes):
                boxes = boxes_projection
        
        # Ordenar de izquierda a derecha
        boxes = sorted(boxes, key=lambda b: b[0])
        
        return boxes
    
    def segmentar_por_proyeccion(self, binary):
        """
        Método alternativo de segmentación usando proyección vertical
        Útil cuando los caracteres están muy conectados
        """
        h, w = binary.shape
        
        # Proyección vertical (suma de píxeles blancos por columna)
        projection = np.sum(binary, axis=0)
        
        # Suavizar proyección (sin scipy)
        window_size = 5
        projection_smooth = np.convolve(projection, np.ones(window_size)/window_size, mode='same')
        
        # Encontrar mínimos locales (espacios entre caracteres)
        threshold = np.mean(projection_smooth) * 0.3
        
        boxes = []
        in_char = False
        start_x = 0
        
        for i in range(w):
            if projection_smooth[i] > threshold and not in_char:
                # Inicio de un carácter
                in_char = True
                start_x = i
            elif projection_smooth[i] <= threshold and in_char:
                # Fin de un carácter
                in_char = False
                width = i - start_x
                
                # Encontrar límites verticales
                char_region = binary[:, start_x:i]
                rows_with_content = np.where(np.sum(char_region, axis=1) > 0)[0]
                
                if len(rows_with_content) > 0:
                    y_min = rows_with_content[0]
                    y_max = rows_with_content[-1]
                    
                    if width > 5 and (y_max - y_min) > 8:
                        boxes.append((start_x, y_min, width, y_max - y_min))
        
        # Último carácter si estamos dentro de uno
        if in_char:
            width = w - start_x
            char_region = binary[:, start_x:]
            rows_with_content = np.where(np.sum(char_region, axis=1) > 0)[0]
            if len(rows_with_content) > 0:
                y_min = rows_with_content[0]
                y_max = rows_with_content[-1]
                if width > 5 and (y_max - y_min) > 8:
                    boxes.append((start_x, y_min, width, y_max - y_min))
        
        return boxes

    def detectar_espacios(self, boxes):
        """Detecta dónde deben ir los espacios entre palabras"""
        if len(boxes) <= 1:
            return []
        
        espacios = []
        anchos = [b[2] for b in boxes]
        ancho_promedio = np.median(anchos)
        
        for i in range(1, len(boxes)):
            # Gap entre el final del carácter anterior y el inicio del actual
            prev_x, prev_y, prev_w, prev_h = boxes[i-1]
            curr_x, curr_y, curr_w, curr_h = boxes[i]
            
            gap = curr_x - (prev_x + prev_w)
            
            # Si el gap es mayor a 1.5 veces el ancho promedio, es un espacio
            if gap > ancho_promedio * 1.5:
                espacios.append(i)
        
        return espacios

    def normalizar_caracter(self, char_img):
        """Normaliza un carácter a 28x28 (igual que el dataset de entrenamiento)"""
        h, w = char_img.shape
        
        # MEJORA 1: Aplicar un pequeño filtro para limpiar ruido
        char_img = cv2.medianBlur(char_img, 3)
        
        # MEJORA 2: Recortar bordes vacíos (tight crop)
        coords = cv2.findNonZero(char_img)
        if coords is not None:
            x, y, w_crop, h_crop = cv2.boundingRect(coords)
            char_img = char_img[y:y+h_crop, x:x+w_crop]
            h, w = char_img.shape
        
        # Crear canvas de 28x28
        canvas = np.zeros((28, 28), dtype=np.uint8)
        
        # MEJORA 3: Calcular escala para que quepa con buen margen
        # Usamos 22 en lugar de 20 para que los caracteres manuscritos se vean mejor
        scale = 22.0 / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        # Redimensionar el carácter
        char_resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # MEJORA 4: Aplicar un poco de dilatación para hacer los trazos más gruesos
        # (los manuscritos suelen tener trazos más finos que los digitales)
        kernel = np.ones((2, 2), np.uint8)
        char_resized = cv2.dilate(char_resized, kernel, iterations=1)
        
        # Centrar en el canvas
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        
        # Asegurar que no nos salimos del canvas
        y_offset = max(0, min(y_offset, 28 - new_h))
        x_offset = max(0, min(x_offset, 28 - new_w))
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_resized
        
        return canvas

    def predecir_caracter(self, char_img_28x28):
        """Predice un carácter normalizado"""
        # Normalizar a [0, 1]
        input_data = char_img_28x28.flatten().reshape(1, -1) / 255.0
        
        # Predecir
        probs = self.model.forward(input_data)[-1]
        idx = np.argmax(probs)
        confidence = np.max(probs)
        
        # Obtener carácter
        char = self.idx_to_char.get(str(idx), "?")
        
        return char, confidence

    def process_image(self, image, return_confidence=False, debug=False):
        """
        Procesa una imagen completa y extrae el texto
        
        Args:
            image: Imagen BGR o escala de grises
            return_confidence: Si True, devuelve (texto, confianza_promedio)
            debug: Si True, muestra imágenes de debug
        
        Returns:
            texto reconocido (y opcionalmente confianza)
        """
        if self.model is None:
            return ("Error: No hay modelo cargado", 0) if return_confidence else "Error: No hay modelo"
        
        self.debug = debug
        
        # 1. PREPROCESAMIENTO
        gray, binary = self.preprocesar_imagen(image)
        
        if debug:
            cv2.imshow("Original", image)
            cv2.imshow("Binaria", binary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # 2. SEGMENTACIÓN
        boxes = self.segmentar_caracteres(binary)
        
        if len(boxes) == 0:
            return ("", 0.0) if return_confidence else ""
        
        print(f"[OCR] Detectados {len(boxes)} caracteres")
        
        # 3. DETECTAR ESPACIOS
        espacios_idx = self.detectar_espacios(boxes)
        
        # 4. RECONOCER CARACTERES
        texto = ""
        confianzas = []
        
        for i, (x, y, w, h) in enumerate(boxes):
            # Añadir espacio si corresponde
            if i in espacios_idx:
                texto += " "
            
            # Extraer carácter de la imagen binaria
            char_crop = binary[y:y+h, x:x+w]
            
            # Normalizar a 28x28
            char_28x28 = self.normalizar_caracter(char_crop)
            
            # Predecir
            char, conf = self.predecir_caracter(char_28x28)
            
            if debug:
                print(f"Carácter {i}: '{char}' (confianza: {conf:.2%})")
                cv2.imshow(f"Char {i}", char_28x28)
                cv2.waitKey(500)
            
            texto += char
            confianzas.append(conf)
        
        if debug:
            cv2.destroyAllWindows()
        
        # 5. RETORNAR RESULTADO
        confianza_promedio = np.mean(confianzas) if confianzas else 0.0
        
        if return_confidence:
            return texto, confianza_promedio
        
        return texto
    
    def process_image_with_visualization(self, image, output_path=None):
        """
        Procesa la imagen y genera una visualización con los caracteres detectados
        
        Args:
            image: Imagen a procesar
            output_path: Ruta donde guardar la visualización (opcional)
        
        Returns:
            texto, confianza, imagen_visualizada
        """
        # Preprocesar
        gray, binary = self.preprocesar_imagen(image)
        
        # Segmentar
        boxes = self.segmentar_caracteres(binary)
        
        # Crear visualización
        vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        texto = ""
        confianzas = []
        
        for i, (x, y, w, h) in enumerate(boxes):
            # Dibujar rectángulo
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extraer y reconocer carácter
            char_crop = binary[y:y+h, x:x+w]
            char_28x28 = self.normalizar_caracter(char_crop)
            char, conf = self.predecir_caracter(char_28x28)
            
            # Añadir etiqueta
            label = f"{char}:{conf:.0%}"
            cv2.putText(vis_img, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            texto += char
            confianzas.append(conf)
        
        confianza_promedio = np.mean(confianzas) if confianzas else 0.0
        
        # Añadir texto resultado
        cv2.putText(vis_img, f"Resultado: {texto}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(vis_img, f"Confianza: {confianza_promedio:.1%}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_img)
            print(f"[OCR] Visualización guardada en: {output_path}")
        
        return texto, confianza_promedio, vis_img