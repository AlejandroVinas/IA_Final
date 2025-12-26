"""
Script de Pruebas Automatizadas - Verificación de Cumplimiento
Demuestra que el sistema cumple con las funcionalidades obligatorias:
1. Reconocimiento de texto impreso (tipografía digital)
2. Reconocimiento de texto manuscrito (tipografía manual)
"""

import os
import sys
import numpy as np
import cv2
from datetime import datetime

# Importar sistema OCR
from ocr_complete import OCRSystem
import dataset_processor

class TestCumplimiento:
    """Pruebas automatizadas para verificar cumplimiento"""
    
    def __init__(self):
        self.model_path = 'models/ocr_model.pkl'
        self.results_dir = 'test_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
    
    def log(self, mensaje, tipo='INFO'):
        """Registra mensaje con timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        simbolo = {
            'INFO': 'ℹ️',
            'SUCCESS': '✅',
            'ERROR': '❌',
            'WARNING': '⚠️'
        }.get(tipo, 'ℹ️')
        
        linea = f"[{timestamp}] {simbolo} {mensaje}"
        print(linea)
        
        # Guardar en log
        with open(os.path.join(self.results_dir, 'test_log.txt'), 'a', encoding='utf-8') as f:
            f.write(linea + '\n')
    
    def generar_imagen_texto_impreso(self, texto, filename='test_impreso.png'):
        """Genera imagen con texto impreso sintético"""
        # Crear imagen en blanco
        img = np.ones((200, 600), dtype=np.uint8) * 255
        
        # Añadir texto con diferentes fuentes
        lineas = texto.split('\n')
        y_offset = 30
        
        for linea in lineas:
            cv2.putText(img, linea, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
            y_offset += 40
        
        # Guardar
        filepath = os.path.join(self.results_dir, filename)
        cv2.imwrite(filepath, img)
        
        return filepath, texto
    
    def generar_imagen_texto_manuscrito(self, texto, filename='test_manuscrito.png'):
        """
        Genera imagen simulando texto manuscrito
        (En producción, usarías imágenes reales de manuscritos)
        """
        # Crear imagen en blanco
        img = np.ones((200, 600), dtype=np.uint8) * 255
        
        # Añadir texto con fuente más "manuscrita"
        lineas = texto.split('\n')
        y_offset = 35
        
        for linea in lineas:
            # Simular manuscrito con FONT_HERSHEY_SCRIPT_SIMPLEX
            cv2.putText(img, linea, (25, y_offset), 
                       cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, 0, 2)
            y_offset += 45
        
        # Añadir ruido para simular papel
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Guardar
        filepath = os.path.join(self.results_dir, filename)
        cv2.imwrite(filepath, img)
        
        return filepath, texto
    
    def calcular_accuracy(self, texto_esperado, texto_obtenido):
        """Calcula accuracy a nivel de caracteres"""
        # Normalizar (quitar espacios extra, etc.)
        esperado = ''.join(texto_esperado.split()).lower()
        obtenido = ''.join(texto_obtenido.split()).lower()
        
        if len(esperado) == 0:
            return 0.0
        
        # Calcular caracteres correctos
        correctos = sum(1 for a, b in zip(esperado, obtenido) if a == b)
        total = max(len(esperado), len(obtenido))
        
        return correctos / total if total > 0 else 0.0
    
    def test_modelo_cargado(self):
        """Test 1: Verificar que el modelo existe y se puede cargar"""
        self.log("Test 1: Cargando modelo...", 'INFO')
        
        if not os.path.exists(self.model_path):
            self.log(f"ERROR: Modelo no encontrado en {self.model_path}", 'ERROR')
            self.tests_failed += 1
            return False
        
        try:
            ocr = OCRSystem(model_path=self.model_path)
            self.log("Modelo cargado correctamente", 'SUCCESS')
            self.tests_passed += 1
            return True
        except Exception as e:
            self.log(f"ERROR al cargar modelo: {str(e)}", 'ERROR')
            self.tests_failed += 1
            return False
    
    def test_reconocimiento_impreso(self):
        """Test 2: Reconocimiento de texto impreso (FUNCIONALIDAD OBLIGATORIA 1)"""
        self.log("\nTest 2: Reconocimiento de Texto Impreso", 'INFO')
        
        # Generar imagen de prueba
        texto_original = "HOLA MUNDO\nEsto es una prueba\n12345"
        img_path, _ = self.generar_imagen_texto_impreso(texto_original)
        
        self.log(f"Imagen generada: {img_path}", 'INFO')
        self.log(f"Texto esperado: {texto_original.replace(chr(10), ' ')}", 'INFO')
        
        # Procesar con OCR
        try:
            ocr = OCRSystem(model_path=self.model_path)
            img = cv2.imread(img_path)
            texto_reconocido, confianza = ocr.process_image(img, return_confidence=True)
            
            self.log(f"Texto reconocido: {texto_reconocido.replace(chr(10), ' ')}", 'INFO')
            self.log(f"Confianza: {confianza:.2%}", 'INFO')
            
            # Calcular accuracy
            accuracy = self.calcular_accuracy(texto_original, texto_reconocido)
            self.log(f"Accuracy: {accuracy:.2%}", 'INFO')
            
            # Guardar resultado
            output_path = os.path.join(self.results_dir, 'resultado_impreso.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"TEXTO ORIGINAL:\n{texto_original}\n\n")
                f.write(f"TEXTO RECONOCIDO:\n{texto_reconocido}\n\n")
                f.write(f"CONFIANZA: {confianza:.2%}\n")
                f.write(f"ACCURACY: {accuracy:.2%}\n")
            
            # Verificar cumplimiento
            if accuracy >= 0.7:  # 70% como mínimo aceptable
                self.log("✓ FUNCIONALIDAD 1 (Texto Impreso): CUMPLIDA", 'SUCCESS')
                self.tests_passed += 1
                self.test_results.append({
                    'test': 'Reconocimiento Texto Impreso',
                    'status': 'PASS',
                    'accuracy': accuracy,
                    'confidence': confianza
                })
                return True
            else:
                self.log(f"✗ FUNCIONALIDAD 1: Accuracy insuficiente ({accuracy:.2%})", 'ERROR')
                self.tests_failed += 1
                return False
                
        except Exception as e:
            self.log(f"ERROR en reconocimiento impreso: {str(e)}", 'ERROR')
            self.tests_failed += 1
            import traceback
            traceback.print_exc()
            return False
    
    def test_reconocimiento_manuscrito(self):
        """Test 3: Reconocimiento de texto manuscrito (FUNCIONALIDAD OBLIGATORIA 2)"""
        self.log("\nTest 3: Reconocimiento de Texto Manuscrito", 'INFO')
        
        # Generar imagen de prueba simulando manuscrito
        texto_original = "Hola Mundo\nPrueba manuscrita\n67890"
        img_path, _ = self.generar_imagen_texto_manuscrito(texto_original)
        
        self.log(f"Imagen generada: {img_path}", 'INFO')
        self.log(f"Texto esperado: {texto_original.replace(chr(10), ' ')}", 'INFO')
        
        # Procesar con OCR
        try:
            ocr = OCRSystem(model_path=self.model_path)
            img = cv2.imread(img_path)
            texto_reconocido, confianza = ocr.process_image(img, return_confidence=True)
            
            self.log(f"Texto reconocido: {texto_reconocido.replace(chr(10), ' ')}", 'INFO')
            self.log(f"Confianza: {confianza:.2%}", 'INFO')
            
            # Calcular accuracy
            accuracy = self.calcular_accuracy(texto_original, texto_reconocido)
            self.log(f"Accuracy: {accuracy:.2%}", 'INFO')
            
            # Guardar resultado
            output_path = os.path.join(self.results_dir, 'resultado_manuscrito.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"TEXTO ORIGINAL:\n{texto_original}\n\n")
                f.write(f"TEXTO RECONOCIDO:\n{texto_reconocido}\n\n")
                f.write(f"CONFIANZA: {confianza:.2%}\n")
                f.write(f"ACCURACY: {accuracy:.2%}\n")
            
            # Verificar cumplimiento (más permisivo para manuscrito)
            if accuracy >= 0.6:  # 60% para manuscrito (más difícil)
                self.log("✓ FUNCIONALIDAD 2 (Texto Manuscrito): CUMPLIDA", 'SUCCESS')
                self.tests_passed += 1
                self.test_results.append({
                    'test': 'Reconocimiento Texto Manuscrito',
                    'status': 'PASS',
                    'accuracy': accuracy,
                    'confidence': confianza
                })
                return True
            else:
                self.log(f"✗ FUNCIONALIDAD 2: Accuracy insuficiente ({accuracy:.2%})", 'ERROR')
                self.tests_failed += 1
                return False
                
        except Exception as e:
            self.log(f"ERROR en reconocimiento manuscrito: {str(e)}", 'ERROR')
            self.tests_failed += 1
            import traceback
            traceback.print_exc()
            return False
    
    def test_salida_archivo_txt(self):
        """Test 4: Verificar que se genera archivo .txt"""
        self.log("\nTest 4: Generación de archivo .txt", 'INFO')
        
        # Verificar que existen los archivos de resultado
        archivos_esperados = [
            'resultado_impreso.txt',
            'resultado_manuscrito.txt'
        ]
        
        todos_existen = True
        for archivo in archivos_esperados:
            path = os.path.join(self.results_dir, archivo)
            if os.path.exists(path):
                self.log(f"✓ Archivo generado: {archivo}", 'SUCCESS')
            else:
                self.log(f"✗ Archivo faltante: {archivo}", 'ERROR')
                todos_existen = False
        
        if todos_existen:
            self.tests_passed += 1
            return True
        else:
            self.tests_failed += 1
            return False
    
    def generar_reporte_final(self):
        """Genera reporte final de cumplimiento"""
        self.log("\n" + "="*60, 'INFO')
        self.log("REPORTE FINAL DE CUMPLIMIENTO", 'INFO')
        self.log("="*60, 'INFO')
        
        # Resumen de tests
        total_tests = self.tests_passed + self.tests_failed
        porcentaje = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        self.log(f"\nTests ejecutados: {total_tests}", 'INFO')
        self.log(f"Tests pasados: {self.tests_passed} ({porcentaje:.1f}%)", 'SUCCESS' if porcentaje >= 75 else 'WARNING')
        self.log(f"Tests fallidos: {self.tests_failed}", 'ERROR' if self.tests_failed > 0 else 'INFO')
        
        # Verificar funcionalidades obligatorias
        self.log("\n" + "="*60, 'INFO')
        self.log("FUNCIONALIDADES OBLIGATORIAS:", 'INFO')
        self.log("="*60, 'INFO')
        
        func1_cumplida = any(t['test'] == 'Reconocimiento Texto Impreso' and t['status'] == 'PASS' 
                            for t in self.test_results)
        func2_cumplida = any(t['test'] == 'Reconocimiento Texto Manuscrito' and t['status'] == 'PASS' 
                            for t in self.test_results)
        
        self.log(f"\n1. Texto Impreso (digital): {'✅ CUMPLIDA' if func1_cumplida else '❌ NO CUMPLIDA'}", 
                'SUCCESS' if func1_cumplida else 'ERROR')
        
        self.log(f"2. Texto Manuscrito (manual): {'✅ CUMPLIDA' if func2_cumplida else '❌ NO CUMPLIDA'}", 
                'SUCCESS' if func2_cumplida else 'ERROR')
        
        # Resumen detallado
        if self.test_results:
            self.log("\n" + "="*60, 'INFO')
            self.log("DETALLES DE RESULTADOS:", 'INFO')
            self.log("="*60, 'INFO')
            
            for result in self.test_results:
                self.log(f"\n{result['test']}:", 'INFO')
                self.log(f"  Estado: {result['status']}", 'SUCCESS' if result['status'] == 'PASS' else 'ERROR')
                self.log(f"  Accuracy: {result['accuracy']:.2%}", 'INFO')
                self.log(f"  Confianza: {result['confidence']:.2%}", 'INFO')
        
        # Conclusión
        self.log("\n" + "="*60, 'INFO')
        self.log("CONCLUSIÓN:", 'INFO')
        self.log("="*60, 'INFO')
        
        if func1_cumplida and func2_cumplida:
            self.log("\n✅ ✅ ✅ TODAS LAS FUNCIONALIDADES OBLIGATORIAS CUMPLIDAS ✅ ✅ ✅", 'SUCCESS')
            self.log("\nEl sistema es APTO para entrega.", 'SUCCESS')
        else:
            self.log("\n❌ FUNCIONALIDADES OBLIGATORIAS INCOMPLETAS", 'ERROR')
            self.log("\nEl sistema requiere ajustes antes de entrega.", 'WARNING')
        
        # Guardar reporte
        reporte_path = os.path.join(self.results_dir, 'REPORTE_CUMPLIMIENTO.txt')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("REPORTE DE CUMPLIMIENTO - FUNCIONALIDADES OBLIGATORIAS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Tests ejecutados: {total_tests}\n")
            f.write(f"Tests pasados: {self.tests_passed}\n")
            f.write(f"Tests fallidos: {self.tests_failed}\n\n")
            f.write("FUNCIONALIDADES OBLIGATORIAS:\n")
            f.write(f"1. Texto Impreso: {'CUMPLIDA' if func1_cumplida else 'NO CUMPLIDA'}\n")
            f.write(f"2. Texto Manuscrito: {'CUMPLIDA' if func2_cumplida else 'NO CUMPLIDA'}\n\n")
            
            for result in self.test_results:
                f.write(f"\n{result['test']}:\n")
                f.write(f"  Estado: {result['status']}\n")
                f.write(f"  Accuracy: {result['accuracy']:.2%}\n")
                f.write(f"  Confianza: {result['confidence']:.2%}\n")
            
            f.write("\n" + "="*60 + "\n")
            if func1_cumplida and func2_cumplida:
                f.write("CONCLUSIÓN: TODAS LAS FUNCIONALIDADES CUMPLIDAS\n")
            else:
                f.write("CONCLUSIÓN: FUNCIONALIDADES INCOMPLETAS\n")
        
        self.log(f"\n✓ Reporte guardado en: {reporte_path}", 'SUCCESS')
    
    def ejecutar_todos_los_tests(self):
        """Ejecuta todos los tests de cumplimiento"""
        print("\n" + "="*60)
        print("🧪 PRUEBAS DE CUMPLIMIENTO - OCR MANUSCRITO")
        print("="*60)
        
        # Limpiar log anterior
        log_path = os.path.join(self.results_dir, 'test_log.txt')
        if os.path.exists(log_path):
            os.remove(log_path)
        
        # Ejecutar tests
        self.test_modelo_cargado()
        self.test_reconocimiento_impreso()
        self.test_reconocimiento_manuscrito()
        self.test_salida_archivo_txt()
        
        # Generar reporte
        self.generar_reporte_final()
        
        return self.tests_failed == 0


def main():
    """Función principal"""
    
    # Verificar que existe el modelo
    if not os.path.exists('models/ocr_model.pkl'):
        print("\n❌ ERROR: No se encuentra el modelo entrenado")
        print("\nPara ejecutar estas pruebas necesitas:")
        print("  1. python main.py entrenar")
        print("  2. python test_cumplimiento.py")
        sys.exit(1)
    
    # Ejecutar tests
    tester = TestCumplimiento()
    success = tester.ejecutar_todos_los_tests()
    
    print("\n" + "="*60)
    print(f"Resultados guardados en: {tester.results_dir}/")
    print("="*60)
    
    # Retornar código de salida
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrumpidos por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)