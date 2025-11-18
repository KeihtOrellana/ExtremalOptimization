import numpy as np
from EO1 import extremal_optimization,load_single_knapsack_problem, obtener_vectores





ruta = "archivos\large.txt"  
problema = load_single_knapsack_problem(ruta)

n_items, capacidad, valores, pesos = obtener_vectores(problema)

print("====================================")
print("    EXTREMAL OPTIMIZATION - KNAPSACK")
print("====================================")
print(f"Instancia: {problema['name']}")
print(f"Items:     {n_items}")
print(f"Capacidad: {capacidad}")
print("====================================\n")

tau = 1.2
iter_max = 30000
print("Ejecutando EO...\n")
best_sol, best_value = extremal_optimization(
    valores=valores,
    pesos=pesos,
    capacidad=capacidad,
    tau=tau,
    max_iter=iter_max
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
