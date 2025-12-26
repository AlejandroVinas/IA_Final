import os
import time
import matplotlib.pyplot as plt
import json

LOG_FILE = 'training_auto.log'
STATS_FILE = 'results/training_stats.json' # Opcional si decides guardar dicts

def mostrar_estado_basico():
    if not os.path.exists(LOG_FILE):
        print("⌛ Esperando a que comience el entrenamiento (archivo log no encontrado)...")
        return

    with open(LOG_FILE, 'r') as f:
        lineas = f.readlines()
        if not lineas: return
        
        ultima = lineas[-1]
        print(f"\n📈 ESTADO ACTUAL: {ultima.strip()}")
        
        # Mostrar progreso visual simple
        if "Época" in ultima:
            try:
                # Extraer época actual: "Época 10/100" -> 10
                progreso = ultima.split('Época ')[1].split('/')[0]
                total = ultima.split('/')[1].split(' |')[0]
                porcentaje = int((int(progreso)/int(total)) * 100)
                barra = "█" * (porcentaje // 5) + "░" * (20 - (porcentaje // 5))
                print(f"Progreso: |{barra}| {porcentaje}%")
            except: pass

def modo_watch():
    print("📺 Iniciando Monitor en Tiempo Real (Ctrl+C para salir)...")
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- MONITOR OCR - {time.strftime('%H:%M:%S')} ---")
            mostrar_estado_basico()
            time.sleep(5) # Actualizar cada 5 segundos
    except KeyboardInterrupt:
        print("\nMonitor detenido.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        modo_watch()
    else:
        mostrar_estado_basico()