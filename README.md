# ExtremalOptimization

Implementación del algoritmo **Extremal Optimization (EO)** en Python, aplicado al problema de **knapsack 0/1**.  
Permite cargar instancias desde archivos `.txt` y ejecutar el algoritmo con distintos parámetros para obtener soluciones aproximadas.

---

## 📦 Requisitos

- Python 3.10 o superior  
- pip  
- (Opcional) entorno virtual

---

## 🔧 Instalación
pip install numpy pandas

### Ejecución básica:
python main1.py -f archivos/large2.txt -t 2.2 -i 20000 -s 15


ejecucion General:
python main1.py \
    -f <archivo_knapsack> \
    -t <tau> \
    -i <iter_max> \
    -s <seed>

cada instancia debe seguir el siguiente formato:
nombre_instancia
n
capacidad
valor_optimo
-----
listado_de_pesos
-----
listado_de_valores
