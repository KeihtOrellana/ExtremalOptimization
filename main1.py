import numpy as np
import argparse
from EO1 import extremal_optimization,load_single_knapsack_problem, obtener_vectores

# ejemplo para ejecutar python main1.py -f archivos\large2.txt -t 2.2 -i 20000 -s 15

parser = argparse.ArgumentParser(description="Ejecutar Extremal Optimization para un archivo knapsack")

parser.add_argument("-f", "--file", type=str, required=True,help="Ruta del archivo knapsack (ej: archivos/large2.txt)")
parser.add_argument("-t", "--tau", type=float, required=True,help="Valor del parámetro tau")
parser.add_argument("-i", "--iter_max", type=int, required=True,help="Número máximo de iteraciones")
parser.add_argument(    "-s", "--seed", type=int, required=True, help="Semilla aleatoria")

args = parser.parse_args()

problema = load_single_knapsack_problem(args.file)
n_items, capacidad, valores, pesos = obtener_vectores(problema)

print("====================================")
print("    EXTREMAL OPTIMIZATION - KNAPSACK")
print("====================================")
print(f"Instancia: {problema['name']}")
print(f"Items:     {n_items}")
print(f"Capacidad: {capacidad}")
print("====================================\n")

print("Ejecutando EO...\n")
best_sol, best_value = extremal_optimization(
    valores=valores,
    pesos=pesos,
    capacidad=capacidad,
    tau=args.tau,
    max_iter=args.iter_max,
    seed=args.seed
)

peso_final = np.dot(best_sol, pesos)


print("=========== RESULTADOS ===========")
print(f"Mejor valor encontrado: {best_value:.2f}")
print(f"Peso de la mochila:     {peso_final:.2f}")
print(f"Capacidad máxima:       {capacidad}")

print("Vector solución (1=incluido, 0=no):")
print(best_sol)

print("\nEO finalizado.")
print("=====================================")
