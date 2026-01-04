"""
Herramienta para agregar datos manuscritos propios al dataset
Segmenta caracteres de una imagen y permite etiquetarlos manualmente
"""

import cv2
import numpy as np
import os
import sys

class ManuscritoCollector:
    def __init__(self):
        self.base_dir = 'data_manuscrito_propio'
        os.makedirs(self.base_dir, exist_ok=True)
        
    def preprocesar_imagen(self, image):
        """Mismo preprocesamiento que ocr_complete.py"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Ecualizar histograma
        gray = cv2.equalizeHist(gray)
        
        # Blur bilateral
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Binarización Otsu
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Limpieza
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        kernel_close = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        return gray, binary
    
    def segmentar_caracteres(self, binary):
        """Segmenta caracteres"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        h_img, w_img = binary.shape
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filtros
            if h < 8 or w < 4:
                continue
            
            aspect_ratio = w / float(h)
            if aspect_ratio > 4 or aspect_ratio < 0.08:
                continue
            
            if area < 30:
                continue
            
            if h > h_img * 0.9 or w > w_img * 0.5:
                continue
            
            boxes.append((x, y, w, h))
        
        # Ordenar de izquierda a derecha
        boxes = sorted(boxes, key=lambda b: b[0])
        return boxes
    
    def normalizar_caracter(self, char_img):
        """Normaliza a 28x28"""
        h, w = char_img.shape
        
        # Limpiar
        char_img = cv2.medianBlur(char_img, 3)
        
        # Recortar
        coords = cv2.findNonZero(char_img)
        if coords is not None:
            x, y, w_crop, h_crop = cv2.boundingRect(coords)
            char_img = char_img[y:y+h_crop, x:x+w_crop]
            h, w = char_img.shape
        
        # Canvas 28x28
        canvas = np.zeros((28, 28), dtype=np.uint8)
        
        # Escalar
        scale = 22.0 / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        char_resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Dilatar
        kernel = np.ones((2, 2), np.uint8)
        char_resized = cv2.dilate(char_resized, kernel, iterations=1)
        
        # Centrar
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        
        y_offset = max(0, min(y_offset, 28 - new_h))
        x_offset = max(0, min(x_offset, 28 - new_w))
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_resized
        
        return canvas
    
    def procesar_imagen_interactiva(self, image_path, texto_esperado):
        """
        Procesa una imagen y permite etiquetar cada carácter
        
        Args:
            image_path: Ruta a la imagen
            texto_esperado: Texto que contiene la imagen (ej: "Hola123")
        """
        print("\n" + "="*70)
        print(f"📝 Procesando: {image_path}")
        print(f"📝 Texto esperado: '{texto_esperado}'")
        print("="*70)
        
        # Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ No se pudo cargar la imagen")
            return False
        
        # Preprocesar
        gray, binary = self.preprocesar_imagen(img)
        
        # Mostrar imagen binaria
        cv2.imshow("Imagen binaria (presiona cualquier tecla)", binary)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        # Segmentar
        boxes = self.segmentar_caracteres(binary)
        
        # Preparar texto esperado (quitar espacios)
        texto_chars = [c for c in texto_esperado.upper() if c.strip()]
        
        print(f"\n✓ Detectados {len(boxes)} segmentos")
        print(f"✓ Esperados {len(texto_chars)} caracteres (sin espacios)")
        
        if len(boxes) == 0:
            print("\n❌ No se detectaron caracteres. Verifica la imagen.")
            return False
        
        if len(boxes) != len(texto_chars):
            print(f"\n⚠️  ADVERTENCIA: Número diferente de segmentos ({len(boxes)}) vs caracteres esperados ({len(texto_chars)})")
            print("    Posibles causas:")
            print("    - Caracteres muy juntos (detectados como uno)")
            print("    - Ruido detectado como carácter")
            print("    - Caracteres muy separados")
            respuesta = input("\n¿Continuar de todos modos? (s/n): ")
            if respuesta.lower() != 's':
                return False
        
        # Crear visualización
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Caracteres detectados", vis)
        cv2.waitKey(1000)
        
        # Etiquetar caracteres
        caracteres_guardados = 0
        
        print("\n" + "="*70)
        print("ETIQUETADO DE CARACTERES")
        print("="*70)
        print("\nInstrucciones:")
        print("  - Se mostrará cada carácter detectado")
        print("  - Ingresa la etiqueta correcta (UNA letra/número)")
        print("  - Presiona Enter para usar la sugerencia")
        print("  - Escribe 'skip' para saltar")
        print("  - Escribe 'quit' para terminar")
        print("="*70 + "\n")
        
        for i, (x, y, w, h) in enumerate(boxes):
            # Extraer carácter
            char_crop = binary[y:y+h, x:x+w]
            char_28x28 = self.normalizar_caracter(char_crop)
            
            # Mostrar ampliado
            char_display = cv2.resize(char_28x28, (200, 200), interpolation=cv2.INTER_NEAREST)
            
            # Resaltar en la imagen completa
            vis_temp = vis.copy()
            cv2.rectangle(vis_temp, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            cv2.imshow("Caracter actual (ampliado)", char_display)
            cv2.imshow("Posicion en imagen", vis_temp)
            cv2.waitKey(100)
            
            # Sugerir etiqueta
            if i < len(texto_chars):
                etiqueta_sugerida = texto_chars[i]
            else:
                etiqueta_sugerida = "?"
            
            print(f"\n📌 Carácter {i+1}/{len(boxes)}:")
            etiqueta = input(f"   Etiqueta [sugerida: '{etiqueta_sugerida}']: ").strip().upper()
            
            # Comandos especiales
            if etiqueta.lower() == 'quit':
                print("\n⏹️  Finalizando...")
                break
            
            if etiqueta.lower() == 'skip' or etiqueta == "":
                if etiqueta == "" and etiqueta_sugerida != "?":
                    etiqueta = etiqueta_sugerida
                else:
                    print("   ⏭️  Saltado")
                    continue
            
            # Validar etiqueta
            if len(etiqueta) != 1 or not etiqueta.strip():
                print("   ⚠️  Etiqueta debe ser un solo carácter válido (no espacios), saltando...")
                continue
            
            # Validar que sea alfanumérico o Ñ
            if not (etiqueta.isalnum() or etiqueta in ['Ñ', 'ñ']):
                print("   ⚠️  Solo se permiten letras y números, saltando...")
                continue
            
            # Guardar carácter
            char_dir = os.path.join(self.base_dir, etiqueta)
            os.makedirs(char_dir, exist_ok=True)
            
            # Contar archivos existentes
            existing = len([f for f in os.listdir(char_dir) if f.endswith('.png')])
            filename = f"manuscrito_{existing:03d}.png"
            filepath = os.path.join(char_dir, filename)
            
            cv2.imwrite(filepath, char_28x28)
            caracteres_guardados += 1
            print(f"   ✅ Guardado como: {etiqueta}/{filename}")
        
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print(f"✅ Proceso completado: {caracteres_guardados} caracteres guardados")
        print(f"📁 Carpeta: {self.base_dir}")
        print("="*70)
        
        return True
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas de los datos recopilados"""
        print("\n" + "="*70)
        print("📊 ESTADÍSTICAS DE DATOS MANUSCRITOS PROPIOS")
        print("="*70)
        
        if not os.path.exists(self.base_dir):
            print("❌ No hay datos recopilados aún")
            return
        
        total = 0
        stats = {}
        
        for char_folder in sorted(os.listdir(self.base_dir)):
            char_path = os.path.join(self.base_dir, char_folder)
            if os.path.isdir(char_path):
                count = len([f for f in os.listdir(char_path) if f.endswith('.png')])
                if count > 0:
                    stats[char_folder] = count
                    total += count
        
        print(f"\n✓ Total caracteres: {total}")
        print(f"✓ Clases diferentes: {len(stats)}")
        print("\nDistribución por carácter:")
        
        for char, count in sorted(stats.items()):
            bar = "█" * min(count, 50)
            print(f"  {char}: {count:3d} {bar}")
        
        print("\n" + "="*70)
        print("💡 Para usar estos datos:")
        print(f"   1. Copia la carpeta '{self.base_dir}' a 'data/Manuscrito_Propio'")
        print("   2. O modifica dataset_processor.py para incluir esta carpeta")
        print("   3. Re-entrena: python main.py entrenar")
        print("="*70)


def main():
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🎨 HERRAMIENTA DE ETIQUETADO DE MANUSCRITOS")
        print("="*70)
        print("\nUso:")
        print("  python agregar_manuscritos.py <imagen> <texto>")
        print("\nEjemplo:")
        print('  python agregar_manuscritos.py mi_imagen.png "Hola123"')
        print("\nPara ver estadísticas:")
        print("  python agregar_manuscritos.py stats")
        print("="*70)
        sys.exit(1)
    
    collector = ManuscritoCollector()
    
    # Comando stats
    if sys.argv[1].lower() == 'stats':
        collector.mostrar_estadisticas()
        sys.exit(0)
    
    # Procesar imagen
    if len(sys.argv) < 3:
        print("❌ Error: Falta el texto esperado")
        print('Uso: python agregar_manuscritos.py <imagen> "texto"')
        sys.exit(1)
    
    image_path = sys.argv[1]
    texto_esperado = sys.argv[2]
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Imagen no encontrada: {image_path}")
        sys.exit(1)
    
    success = collector.procesar_imagen_interactiva(image_path, texto_esperado)
    
    if success:
        print("\n💡 Siguiente paso:")
        print("  - Procesa más imágenes para tener más datos")
        print("  - Cuando tengas suficientes, integra los datos y re-entrena")
        collector.mostrar_estadisticas()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)