import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def buscar_fuente(nombre_fuente):
    """Busca la fuente en las carpetas comunes de Windows."""
    rutas_posibles = [
        nombre_fuente,
        os.path.join("C:\\Windows\\Fonts", nombre_fuente),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", nombre_fuente)
    ]
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            return ruta
    return None

def generar_dataset_digital():
    # 1. Configuración
    base_dir = 'data_digital'
    caracteres = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789Ñ"
    
    # Nombres de archivos de fuentes estándar en Windows
    nombres_fuentes = [
        "arial.ttf", 
        "times.ttf", 
        "cour.ttf",   
        "verdana.ttf",
        "georgia.ttf"
    ]
    
    os.makedirs(base_dir, exist_ok=True)
    print("[*] Generando caracteres digitales...")

    contador_total = 0
    for char in caracteres:
        char_dir = os.path.join(base_dir, char)
        os.makedirs(char_dir, exist_ok=True)
        
        for i, nombre in enumerate(nombres_fuentes):
            try:
                # Crear imagen blanca de 64x64
                img = Image.new('L', (64, 64), color=255)
                draw = ImageDraw.Draw(img)
                
                ruta_fuente = buscar_fuente(nombre)
                if ruta_fuente:
                    font = ImageFont.truetype(ruta_fuente, 45)
                else:
                    font = ImageFont.load_default()

                # NUEVA FORMA DE CENTRAR (Compatible con Pillow 10+)
                # Obtenemos el rectángulo delimitador (left, top, right, bottom)
                bbox = draw.textbbox((0, 0), char, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                
                # Calculamos posición para centrar
                pos_x = (64 - w) / 2 - bbox[0]
                pos_y = (64 - h) / 2 - bbox[1]
                
                draw.text((pos_x, pos_y), char, font=font, fill=0)
                
                # Guardar imagen
                img.save(os.path.join(char_dir, f"digital_{i}.png"))
                contador_total += 1
                
            except Exception as e:
                print(f"    [!] Error con fuente {nombre} en carácter {char}: {e}")
                continue
    
    print(f"\n[+] ¡ÉXITO! Se han generado {contador_total} imágenes.")
    print(f"[+] Revisa la carpeta: {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    generar_dataset_digital()