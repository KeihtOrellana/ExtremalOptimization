import numpy as np
import pandas as pd
import sys
import time
import os # Usado para manejo de archivos

"""
Implementación del Problema de la Mochila (Knapsack Problem) 
utilizando Extremal Optimisation (EO), basado en los requisitos
del "Trabajo de Programación 3".
"""

# --- Funciones Requeridas (Generadores) [cite: 243, 244] ---

def generar_real_0_1():
    """Genera un número real randómico entre [0 y 1]. [cite: 243]"""
    return np.random.rand()

def generar_entero_1_N(N):
    """Genera un número entero randómico entre [1 y N]. [cite: 244]"""
    return np.random.randint(1, N + 1)

# --- Funciones Principales de EO (Requeridas) [cite: 245-249] ---

def inicializar_ecosistema(n_items):
    """
    Genera una solución inicial aleatoria (ecosistema). [cite: 101, 245]
    Un vector binario donde 1 = item en la mochila, 0 = item fuera. [cite: 212]
    """
    return np.random.randint(0, 2, size=n_items)

def evaluar_ecosistema(solucion, pesos, valores, capacidad):
    """
    Función de evaluación del ecosistema (solución completa). [cite: 249]
    Calcula el valor total de la mochila. [cite: 209, 210]
    Si se excede la capacidad, el fitness es 0 (solución inválida). [cite: 211]
    """
    peso_total = np.dot(solucion, pesos)
    valor_total = np.dot(solucion, valores)

    if peso_total > capacidad:
        return 0  # Penalización por exceder la capacidad [cite: 211]
    else:
        return valor_total # Maximizar el valor [cite: 210]

def calcular_fitness_especies(pesos, valores):
    """
    Función de evaluación del fitness de cada especie (componente). [cite: 246]
    Usamos la densidad de valor (valor/peso) como el "fitness"
    individual de cada item, como heurística[cite: 91].
    """
    # Evitar división por cero si un item tiene peso 0
    with np.errstate(divide='ignore', invalid='ignore'):
        fitness = np.divide(valores, pesos)
        # Reemplazar infinitos (si peso=0 y valor>0) o NaNs (si peso=0 y valor=0)
        fitness[~np.isfinite(fitness)] = 0 
    return fitness

def generar_probabilidades_tau(n, tau):
    """
    Genera el array de probabilidades P basado en el ranking y Tau. [cite: 103, 253]
    Según la Ecuación 1: P_i = i^-tau [cite: 60]
    """
    # i va de 1 a n (para los rangos) [cite: 60]
    i = np.arange(1, n + 1)
    probabilidades = np.power(i.astype(float), -tau)
    
    # Normalizar las probabilidades para que sumen 1 (para RWS)
    return probabilidades / np.sum(probabilidades)

def seleccionar_especie_ruleta(fitness_especies, probabilidades_ranking):
    """
    Función de selección de una especie usando el método de la ruleta. [cite: 247]
    1. Rankea las especies de peor (menor fitness) a mejor (mayor fitness)[cite: 108].
    2. Asigna las probabilidades P al ranking (P[0] es para el peor).
    3. Selecciona un item (índice) usando RWS basado en esas probabilidades[cite: 109].
    """
    n_items = len(fitness_especies)
    
    # 1. Obtener los índices que ordenarían el fitness de peor a mejor
    # (El item en ranking_indices[0] es el peor)
    ranking_indices = np.argsort(fitness_especies) 
    
    # 2. Seleccionar un *rango* (de 0 a n-1) usando RWS
    # np.random.choice usa las probabilidades_ranking para elegir un índice
    # El índice 0 (peor rango) tiene la mayor probabilidad
    rango_seleccionado = np.random.choice(n_items, p=probabilidades_ranking)
    
    # 3. Devolver el *índice del item* que está en ese rango seleccionado
    indice_item_seleccionado = ranking_indices[rango_seleccionado]
    return indice_item_seleccionado

def reemplazar_especie(solucion, indice_a_reemplazar):
    """
    Función de reemplazo de la especie seleccionada. [cite: 248]
    Genera una nueva solución candidata invirtiendo el bit (0 <-> 1)
    del componente seleccionado[cite: 110, 112].
    """
    nueva_solucion = solucion.copy()
    # Invertir el valor: 1 -> 0 ó 0 -> 1
    nueva_solucion[indice_a_reemplazar] = 1 - nueva_solucion[indice_a_reemplazar]
    return nueva_solucion

# --- Programa Principal ---

def main():
    """
    Programa principal que ejecuta el algoritmo EO para el Knapsack Problem.
    Utiliza las bibliotecas Numpy, Pandas, Sys y Time.
    """
    
    # --- 1. Parámetros de Entrada (sintonizables)  ---
    try:
        # sys.argv permite leer parámetros desde la línea de comandos
        archivo_entrada = sys.argv[1] if len(sys.argv) > 1 else "knapsack_data.csv"
        capacidad = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        semilla = int(sys.argv[3]) if len(sys.argv) > 3 else 42
        iteraciones = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
        tau = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
    except (ValueError, IndexError):
        print("Error en los argumentos. Usando valores por defecto.")
        archivo_entrada = "knapsack_data.csv" # [cite: 250]
        capacidad = 50 # [cite: 196]
        semilla = 42 # [cite: 251]
        iteraciones = 2000 # [cite: 252]
        tau = 3.0 # [cite: 253]
    
    # --- Interfaz (Presentación por pantalla) [cite: 263] ---
    print("="*50)
    print("  Optimización Extrema (EO) para el Problema de la Mochila")
    print("="*50)
    print(f"  Archivo de entrada: {archivo_entrada}")
    print(f"  Capacidad Mochila:  {capacidad}")
    print(f"  Valor Semilla:      {semilla}")
    print(f"  Iteraciones:        {iteraciones}")
    print(f"  Valor Tau (τ):      {tau}")
    print("-"*50)

    # Configurar valor semilla [cite: 251]
    np.random.seed(semilla)
    
    # --- 2. Cargar Datos (usando Pandas)  ---
    try:
        df = pd.read_csv(archivo_entrada)
        if 'peso' not in df.columns or 'valor' not in df.columns:
            raise KeyError("El CSV debe tener columnas 'peso' y 'valor'.")
            
        pesos = df['peso'].values
        valores = df['valor'].values
        n_items = len(df)
        print(f"Se cargaron {n_items} items desde '{archivo_entrada}'.\n")
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{archivo_entrada}'.")
        print("Por favor, cree un CSV con columnas 'peso' y 'valor'.")
        # Generar archivo de ejemplo si no existe
        if not os.path.exists(archivo_entrada):
            print(f"Creando archivo de ejemplo: '{archivo_entrada}'...")
            sample_df = pd.DataFrame({
                'item': [f'item_{i}' for i in range(10)],
                'peso': [10, 20, 30, 5, 15, 25, 8, 12, 18, 22],
                'valor': [60, 100, 120, 20, 80, 110, 45, 70, 90, 105]
            })
            sample_df.to_csv(archivo_entrada, index=False)
            print("Archivo de ejemplo creado. Por favor, ejecute el script de nuevo.")
        return
    except KeyError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error inesperado al leer el archivo: {e}") # Manejo de errores [cite: 263]
        return
        
    start_time = time.time() # Iniciar cronómetro 

    # --- 3. Lógica del Algoritmo EO (basado en PDF [cite: 100-116]) ---

    # Calcular fitness de componentes (especies) [cite: 246]
    fitness_especies = calcular_fitness_especies(pesos, valores)
    
    # Generar probabilidades de ranking [cite: 103]
    probabilidades_ranking = generar_probabilidades_tau(n_items, tau)

    # 1: Generar solución inicial [cite: 101]
    solucion_actual = inicializar_ecosistema(n_items) # [cite: 245]
    
    # 2: Evaluar y guardar como la mejor [cite: 102]
    # (Adaptado para maximización [cite: 209], el algoritmo [cite: 114] era para minimización)
    fitness_actual = evaluar_ecosistema(solucion_actual, pesos, valores, capacidad) # [cite: 249]
    solucion_mejor = solucion_actual.copy()
    fitness_mejor = fitness_actual
    
    historial_fitness = [fitness_mejor]

    # 4: Iniciar iteraciones [cite: 103, 252]
    for i in range(iteraciones):
        
        # 6: Seleccionar componente j usando RWS [cite: 109, 247]
        indice_a_reemplazar = seleccionar_especie_ruleta(fitness_especies, probabilidades_ranking)
        
        # 7: Generar nueva solución (reemplazar/mutar) [cite: 110, 112, 248]
        nueva_solucion = reemplazar_especie(solucion_actual, indice_a_reemplazar)
        
        # 8: Evaluar nueva solución [cite: 113]
        nuevo_fitness = evaluar_ecosistema(nueva_solucion, pesos, valores, capacidad)
        
        # EO siempre acepta la nueva solución [cite: 38, 62]
        solucion_actual = nueva_solucion
        fitness_actual = nuevo_fitness
        
        # 9: Comparar con la mejor encontrada [cite: 114]
        # (Adaptado para maximización: > en lugar de <)
        if fitness_actual > fitness_mejor:
            solucion_mejor = solucion_actual.copy()
            fitness_mejor = fitness_actual
        
        historial_fitness.append(fitness_mejor)
        
        # Opcional: Imprimir progreso
        # if (i + 1) % (iteraciones // 10) == 0:
        #     print(f"Iteración {i+1}/{iteraciones} - Mejor Fitness: {fitness_mejor}")

    # --- 4. Resultados (Interfaz) [cite: 263] ---
    end_time = time.time()
    peso_final = np.dot(solucion_mejor, pesos)
    items_seleccionados = df.iloc[solucion_mejor == 1]

    print("--- Resultados de la Optimización ---")
    print(f"Tiempo de ejecución: {end_time - start_time:.4f} segundos")
    print(f"Mejor Valor (Fitness): {fitness_mejor}")
    print(f"Peso Total: {peso_final} (Capacidad: {capacidad})")
    print("\nItems seleccionados en la mochila:")
    if items_seleccionados.empty:
        print("  Ningún item seleccionado.")
    else:
        print(items_seleccionados.to_string(index=False))
    print("="*50)
    
    # Opcional: Graficar (requiere matplotlib: pip install matplotlib)
    # try:
    #     import matplotlib.pyplot as plt
    #     plt.plot(historial_fitness)
    #     plt.title("Evolución del Mejor Fitness")
    #     plt.xlabel("Iteración")
    #     plt.ylabel("Mejor Valor (Fitness)")
    #     plt.grid(True)
    #     plt.show()
    # except ImportError:
    #     print("\n(Para ver el gráfico de evolución, instala matplotlib: pip install matplotlib)")


if __name__ == "__main__":
    main()