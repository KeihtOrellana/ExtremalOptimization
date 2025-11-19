import numpy as np
import pandas as pd

def load_single_knapsack_problem(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Lee todas las líneas, elimina espacios/saltos de línea y filtra la línea '-----'
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('-----')]

    try:
        #  1. Leer la Cabecera (Líneas 0 a 4)
        problem_name = lines[0]                       # Línea 1: Nombre
        n = int(lines[1])                             # Línea 2: Número de elementos (n)
        C = int(lines[2])                             # Línea 3: Capacidad (C)
        
        # El valor óptimo (z) no tiene decimales con coma, usamos float directamente
        optimal_z = float(lines[3])                   # Línea 4: Fitness óptimo (z)
        # lines[4] es 'time', se ignora
        
        #  2. Leer los Datos de los Ítems (Líneas 5 en adelante)
        item_data = []
        
        # Leemos exactamente n ítems a partir de la línea 5
        for item_line in lines[5:5 + n]: 
            
            # ATENCIÓN: Separamos ÚNICAMENTE por coma (,)
            parts = item_line.split(',') 
            
            if len(parts) >= 4:
                # 0: ID | 1: Precio/Valor | 2: Peso | 3: Solución(0/1)
                
                # Se asume que todos son números enteros o floats con punto decimal
                item_id = int(parts[0])
                value = float(parts[1])  
                weight = float(parts[2]) 
                solution = int(parts[3])
                
                item_data.append({
                    'ID': item_id,
                    'Value': value,
                    'Weight': weight,
                    'In_Solution': solution
                })

        df_items = pd.DataFrame(item_data)
        
        #  3. Devolver el Problema
        return {
            'name': problem_name,
            'n_items': n,
            'capacity': C,
            'optimal_z': optimal_z,
            'items_df': df_items.set_index('ID')
        }
    
    except (ValueError, IndexError) as e:
        print(f"Error de parsing: Verifique que las líneas de cabecera y de ítems estén completas y que los campos estén separados solo por comas. Error: {e}")
        return None

def obtener_vectores(problem_data):
        
        
    # 1. Extraer los parámetros
    n_items = problem_data['n_items']
    capacity = problem_data['capacity']
    items_df = problem_data['items_df']
    # 2. Inicializar los vectores de NumPy
    vector_valor = items_df['Value'].values
    vector_peso = items_df['Weight'].values
    return n_items, capacity, vector_valor, vector_peso


def inicializar_solucion(n_items):

    return np.random.randint(0, 2, size=n_items)


def evaluar_global(sol, valores):
    return np.dot(sol, valores)



def calcular_fitness_local_dual(sol, valores, pesos, capacidad):

    peso_actual = np.dot(sol, pesos)
    fitness = np.zeros(len(sol), dtype=float)

    # Relación valor/peso
    ratio = valores / pesos

    if peso_actual <= capacidad:
        # FITNESS 1 → agregar: rankear solo apagados
        for i in range(len(sol)):
            if sol[i] == 0:
                fitness[i] = ratio[i]
            else:
                fitness[i] = 0.0

    else:
        # FITNESS 2 → quitar: rankear solo activos
        for i in range(len(sol)):
            if sol[i] == 1:
                fitness[i] = ratio[i]
            else:
                fitness[i] = 0.0

    return fitness



def rankear_variables(f_local):

    ranks = np.zeros(len(f_local), dtype=int)

    candidates = np.where(f_local > 0)[0]
    if len(candidates) == 0:
        return ranks

    # ordenar PEOR → MEJOR
    sorted_candidates = candidates[np.argsort(f_local[candidates])]

    for r, idx in enumerate(sorted_candidates, start=1):
        ranks[idx] = r

    return ranks



def generar_probabilidades(ranks, tau):
    P = np.zeros(len(ranks), dtype=float)

    candidates = np.where(ranks > 0)[0]
    if len(candidates) == 0:
        return np.ones(len(ranks)) / len(ranks)

    inv = np.power(ranks[candidates].astype(float), -tau)
    inv /= np.sum(inv)

    P[candidates] = inv
    return P



def seleccionar_variable(P):
    acumulado = np.cumsum(P)
    r = np.random.rand()
    return np.searchsorted(acumulado, r)



def mutar_variable(sol, idx):
    nueva = sol.copy()
    nueva[idx] = 1 - nueva[idx]
    return nueva



def extremal_optimization(valores, pesos, capacidad, tau, max_iter,seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = len(valores)
    sol = inicializar_solucion(n)

    best_sol = sol.copy()
    best_f = -np.inf   # Inicializar con -infinito para permitir mejoras


    for _ in range(max_iter):

        # 1. Fitness local dinámico
        f_local = calcular_fitness_local_dual(sol, valores, pesos, capacidad)

        # 2. Ranking EO
        ranks = rankear_variables(f_local)

        # 3. Probabilidades EO
        P = generar_probabilidades(ranks, tau)

        # 4. Elegir variable a modificar
        idx = seleccionar_variable(P)

        # 5. Mutación EO (flip)
        sol = mutar_variable(sol, idx)

        # 6. Evaluación global
        f = evaluar_global(sol, valores)

            

        peso_actual = np.dot(sol, pesos)
        valor_actual = np.dot(sol, valores)

        # SOLO guardar soluciones válidas
        if peso_actual <= capacidad and valor_actual > best_f:
            best_f = valor_actual
            best_sol = sol.copy()


    return best_sol, best_f
