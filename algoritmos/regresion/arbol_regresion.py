# =============================================================================
# Algoritmo 5: Árbol de Decisión (Regresión)
# Dataset: Producción agrícola de soja/maíz (sintético argentino)
# Contexto: Predicción de toneladas según hectáreas, lluvia y temperatura
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "Árbol de Decisión (Regresión)"
DATASET = "Producción agrícola de soja/maíz en la pampa húmeda (sintético)"

def generar_dataset():
    """
    Variables:
    - hectareas: superficie sembrada (50 a 2000 ha)
    - lluvia_mm: precipitaciones en período crítico (200 a 800 mm)
    - temp_media: temperatura media (15 a 30 °C)
    - fertilizante: kg/ha aplicados (0 a 300)
    Target: toneladas producidas
    """
    np.random.seed(55)
    n = 300
    hectareas    = np.random.uniform(50, 2000, n)
    lluvia_mm    = np.random.uniform(200, 800, n)
    temp_media   = np.random.uniform(15, 30, n)
    fertilizante = np.random.uniform(0, 300, n)

    # Producción: 3 ton/ha base + efecto lluvia + fertilizante - penalidad calor extremo
    toneladas = (3.0 * hectareas
                 + 0.5 * lluvia_mm
                 + 10 * fertilizante
                 - 50 * np.maximum(temp_media - 25, 0)
                 + np.random.normal(0, 500, n))

    X = np.column_stack([hectareas, lluvia_mm, temp_media, fertilizante])
    return X, toneladas

def ejecutar():
    print("  Generando dataset: producción agrícola pampa húmeda...")
    X, y = generar_dataset()
    nombres = ["Hectáreas", "Lluvia (mm)", "Temp. media (°C)", "Fertilizante (kg/ha)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Comparamos profundidades
    profundidades = [3, 5, 10, None]
    resultados = {}
    for prof in profundidades:
        modelo = DecisionTreeRegressor(max_depth=prof, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        resultados[str(prof)] = {
            'modelo': modelo,
            'y_pred': y_pred,
            'r2':   r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

    mejor_key = max(resultados, key=lambda k: resultados[k]['r2'])
    mejor_modelo = resultados[mejor_key]['modelo']

    print(f"\n  {'─'*40}")
    print(f"  COMPARACIÓN DE PROFUNDIDADES")
    print(f"  {'─'*40}")
    for k, res in resultados.items():
        marca = " ← mejor" if k == mejor_key else ""
        print(f"  max_depth={k:<5}: R²={res['r2']:.4f}  RMSE={res['rmse']:.2f}{marca}")
    print(f"  {'─'*40}")

    # Importancia de variables
    importancias = mejor_modelo.feature_importances_
    print(f"\n  Importancia de variables (max_depth={mejor_key}):")
    for nombre, imp in zip(nombres, importancias):
        print(f"    {nombre:<22}: {imp:.4f}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Árbol de Decisión (Regresión)\nProducción Agrícola - Pampa Húmeda", fontsize=13)

    # Importancia de variables
    ax1 = axes[0]
    colores = ['steelblue', 'darkorange', 'green', 'crimson']
    bars = ax1.barh(nombres, importancias, color=colores)
    ax1.set_xlabel("Importancia")
    ax1.set_title("Importancia de variables")
    ax1.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, importancias):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9)

    # Real vs predicho
    ax2 = axes[1]
    y_pred_mejor = resultados[mejor_key]['y_pred']
    ax2.scatter(y_test, y_pred_mejor, alpha=0.4, color='darkorange')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', linewidth=1.5, label='Predicción perfecta')
    ax2.set_xlabel("Valores reales (ton)")
    ax2.set_ylabel("Valores predichos (ton)")
    ax2.set_title(f"Real vs Predicho (depth={mejor_key})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    guardar_grafico(fig, "05_arbol_regresion_agricola.png")

    # --- Informe ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += "VARIABLES\n" + "-"*40 + "\n"
    for n_ in nombres:
        contenido += f"  - {n_}\n"
    contenido += "\nCOMPARACIÓN DE PROFUNDIDADES\n" + "-"*40 + "\n"
    for k, res in resultados.items():
        marca = " ← mejor" if k == mejor_key else ""
        contenido += f"  max_depth={k:<5}: R²={res['r2']:.4f}  RMSE={res['rmse']:.2f}{marca}\n"
    contenido += "\nIMPORTANCIA DE VARIABLES\n" + "-"*40 + "\n"
    for nombre, imp in zip(nombres, importancias):
        contenido += f"  {nombre:<24}: {imp:.4f}\n"
    guardar_informe("05_arbol_regresion_agricola.txt", contenido)
