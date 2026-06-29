# =============================================================================
# Algoritmo 3: Regresión Polinomial
# Dataset: Inflación mensual vs poder adquisitivo (sintético argentino)
# Contexto: Relación no lineal entre acumulado de inflación y salario real
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "Regresión Polinomial"
DATASET = "Inflación mensual vs poder adquisitivo (Argentina sintético)"

def generar_dataset():
    """
    Simula la relación entre inflación acumulada (%) y poder adquisitivo
    del salario (índice 100 = base). A mayor inflación, el poder adquisitivo
    cae de forma no lineal (curva descendente con ruido).
    """
    np.random.seed(21)
    n = 150
    # Inflación acumulada mensual: de 0% a 300%
    inflacion = np.random.uniform(0, 300, n)
    # Poder adquisitivo: decae polinomialmente
    poder = 100 - 0.5 * inflacion + 0.0008 * inflacion**2 + np.random.normal(0, 3, n)
    return inflacion.reshape(-1, 1), poder

def ejecutar():
    print("  Generando dataset: inflación vs poder adquisitivo...")
    X, y = generar_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Comparamos grados 1, 2 y 3
    resultados = {}
    for grado in [1, 2, 3]:
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=grado, include_bias=False)),
            ('reg',  LinearRegression())
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        resultados[grado] = {
            'modelo': pipeline,
            'y_pred': y_pred,
            'r2':   r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

    mejor_grado = max(resultados, key=lambda g: resultados[g]['r2'])

    print(f"\n  {'─'*40}")
    print(f"  COMPARACIÓN DE GRADOS POLINOMIALES")
    print(f"  {'─'*40}")
    for g, res in resultados.items():
        marca = " ← mejor" if g == mejor_grado else ""
        print(f"  Grado {g}: R²={res['r2']:.4f}  RMSE={res['rmse']:.4f}{marca}")
    print(f"  {'─'*40}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Regresión Polinomial\nInflación vs Poder Adquisitivo (Argentina)", fontsize=13)

    colores = {1: 'green', 2: 'crimson', 3: 'darkorange'}
    x_linea = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    ax1 = axes[0]
    ax1.scatter(X_test, y_test, alpha=0.4, color='steelblue', label='Datos reales', zorder=3)
    for g, res in resultados.items():
        y_linea = res['modelo'].predict(x_linea)
        ax1.plot(x_linea, y_linea, color=colores[g], linewidth=2,
                 label=f'Grado {g} (R²={res["r2"]:.3f})')
    ax1.set_xlabel("Inflación acumulada (%)")
    ax1.set_ylabel("Poder adquisitivo (índice)")
    ax1.set_title("Ajuste por grado polinomial")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    mejor = resultados[mejor_grado]
    ax2.scatter(y_test, mejor['y_pred'], alpha=0.5, color='darkorange')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', linewidth=1.5, label='Predicción perfecta')
    ax2.set_xlabel("Valores reales")
    ax2.set_ylabel("Valores predichos")
    ax2.set_title(f"Real vs Predicho (Grado {mejor_grado})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    guardar_grafico(fig, "03_regresion_polinomial.png")

    # --- Informe ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += "DESCRIPCIÓN DEL DATASET\n" + "-"*40 + "\n"
    contenido += "Relación entre inflación acumulada y poder adquisitivo\n"
    contenido += "del salario en Argentina (datos sintéticos).\n"
    contenido += f"  Muestras: 150 | Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}\n\n"
    contenido += "COMPARACIÓN DE GRADOS\n" + "-"*40 + "\n"
    for g, res in resultados.items():
        marca = " ← mejor" if g == mejor_grado else ""
        contenido += f"  Grado {g}: R²={res['r2']:.4f}  RMSE={res['rmse']:.4f}{marca}\n"
    contenido += f"\nMejor grado seleccionado: {mejor_grado}\n"
    guardar_informe("03_regresion_polinomial.txt", contenido)
