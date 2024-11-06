# Install necessary libraries first (run this only once)
# !pip install pulp streamlit matplotlib

import streamlit as st
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
import matplotlib.pyplot as plt

# Configuración de la página de Streamlit
st.set_page_config(page_title="Método Branch and Bound", layout="wide")
st.title("Resolución de Programación Lineal Entera con Branch and Bound")

# Definición del problema de maximización
problem = LpProblem("Maximizar_P", LpMaximize)

# Variables de decisión
x1 = LpVariable("x1", lowBound=0, cat="Integer")
x2 = LpVariable("x2", lowBound=0, cat="Integer")
x3 = LpVariable("x3", lowBound=0, cat="Integer")

# Función objetivo
problem += 4 * x1 + 3 * x2 + 3 * x3, "Función Objetivo"

# Restricciones
problem += 4 * x1 + 2 * x2 + x3 <= 10, "Restricción 1"
problem += 3 * x1 + 4 * x2 + 2 * x3 <= 14, "Restricción 2"
problem += 2 * x1 + x2 + 3 * x3 <= 7, "Restricción 3"

# Resolver el problema inicial
problem.solve()

# Mostrar el estado inicial
st.subheader("Resultado del problema relajado (sin restricciones enteras)")
if problem.status == 1:
    st.write("Solución Relajada:")
    st.write(f"x1 = {value(x1)}, x2 = {value(x2)}, x3 = {value(x3)}")
    st.write(f"Valor óptimo: {value(problem.objective)}")
else:
    st.write("No se encontró una solución.")

# Implementación de Branch and Bound
def branch_and_bound(problem):
    # Función para realizar el método Branch and Bound y almacenar soluciones en una lista
    solutions = []
    stack = [problem]

    while stack:
        current_problem = stack.pop()
        current_problem.solve()

        # Si el problema tiene una solución entera, almacenar
        if all(var.varValue.is_integer() for var in current_problem.variables()):
            solutions.append((value(current_problem.objective), {var.name: var.varValue for var in current_problem.variables()}))
        else:
            # Si no es una solución entera, aplicar Branch
            for var in current_problem.variables():
                if not var.varValue.is_integer():
                    # Crear dos ramas con restricciones enteras
                    floor_val = int(var.varValue)
                    ceil_val = floor_val + 1
                    
                    # Rama con la restricción var <= floor
                    prob_floor = current_problem.copy()
                    prob_floor += (var <= floor_val, f"{var.name} <= {floor_val}")
                    stack.append(prob_floor)
                    
                    # Rama con la restricción var >= ceil
                    prob_ceil = current_problem.copy()
                    prob_ceil += (var >= ceil_val, f"{var.name} >= {ceil_val}")
                    stack.append(prob_ceil)
                    break
    return solutions

# Ejecutar Branch and Bound y mostrar resultados
solutions = branch_and_bound(problem)
st.subheader("Soluciones Óptimas Encontradas")
if solutions:
    for i, (obj_val, solution) in enumerate(solutions):
        st.write(f"Solución {i+1}:")
        for var, val in solution.items():
            st.write(f"{var} = {val}")
        st.write(f"Valor de la función objetivo: {obj_val}")
else:
    st.write("No se encontraron soluciones óptimas.")

# Visualización gráfica de las restricciones
st.subheader("Visualización de Restricciones y Soluciones")

# Generar una gráfica con las restricciones
fig, ax = plt.subplots()

# Definir los valores de x1 y x2 para la visualización de restricciones en 2D
x_vals = range(11)
y_vals = [((10 - 4 * x) / 2) for x in x_vals]  # Restricción 1
y_vals_2 = [((14 - 3 * x) / 4) for x in x_vals]  # Restricción 2
y_vals_3 = [((7 - 2 * x) / 1) for x in x_vals]  # Restricción 3

# Graficar cada restricción
ax.plot(x_vals, y_vals, label="4x1 + 2x2 + x3 <= 10")
ax.plot(x_vals, y_vals_2, label="3x1 + 4x2 + 2x3 <= 14")
ax.plot(x_vals, y_vals_3, label="2x1 + x2 + 3x3 <= 7")

# Ajuste y detalles de la gráfica
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
ax.grid(True)

# Mostrar la gráfica en Streamlit
st.pyplot(fig)
