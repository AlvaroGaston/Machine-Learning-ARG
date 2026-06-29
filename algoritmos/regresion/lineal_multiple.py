# =============================================================================
# Algoritmo 2: Regresión Lineal Múltiple
# Dataset: Precio de alquiler en CABA (sintético argentino)
# Contexto: Precio de alquiler según m², habitaciones y distancia al centro
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "Regresión Lineal Múltiple"
DATASET = "Precio de alquiler en CABA (m², habitaciones, distancia al centro)"

def generar_dataset():
    """
    Simula precios de alquiler en CABA según:
    - m²: superficie del departamento (30 a 150 m²)
    - habitaciones: cantidad de ambientes (1 a 5)
    - distancia: km al obelisco (0.5 a 20 km)
    Precio base: ~$2000/m² + bonificaciones por ambientes - penalización por distancia
    """
    np.random.seed(7)
    n = 300
    m2          = np.random.uniform(30, 150, n)
    habitaciones = np.random.randint(1, 6, n).astype(float)
    distancia   = np.random.uniform(0.5, 20, n)

    # Precio en miles de pesos
    precio = (2000 * m2
              + 15000 * habitaciones
              - 3000 * distancia
              + np.random.normal(0, 20000, n))

    X = np.column_stack([m2, habitaciones, distancia])
    return X, precio

def ejecutar():
    print("  Generando dataset: alquileres en CABA...")
    X, y = generar_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    nombres = ["m²", "Habitaciones", "Distancia (km)"]
    print(f"\n  {'─'*40}")
    print(f"  RESULTADOS DEL MODELO")
    print(f"  {'─'*40}")
    for nombre, coef in zip(nombres, modelo.coef_):
        print(f"  Coef. {nombre:<18}: {coef:.2f}")
    print(f"  Intercepto              : {modelo.intercept_:.2f}")
    print(f"  MSE                     : {mse:.2f}")
    print(f"  RMSE                    : {rmse:.2f}")
    print(f"  R²                      : {r2:.4f}")
    print(f"  {'─'*40}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Regresión Lineal Múltiple\nAlquileres en CABA", fontsize=13)

    for i, (ax, nombre) in enumerate(zip(axes, nombres)):
        ax.scatter(X_test[:, i], y_test, alpha=0.4, color='steelblue', label='Real')
        ax.scatter(X_test[:, i], y_pred, alpha=0.4, color='crimson', label='Predicho')
        ax.set_xlabel(nombre)
        ax.set_ylabel("Precio ($)")
        ax.set_title(f"Precio vs {nombre}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    guardar_grafico(fig, "02_regresion_lineal_multiple.png")

    # --- Informe ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += "DESCRIPCIÓN DEL DATASET\n"
    contenido += "-" * 40 + "\n"
    contenido += "Simula precios de alquiler en CABA con 3 variables:\n"
    contenido += "  m² (superficie), habitaciones, distancia al Obelisco.\n"
    contenido += f"  Muestras: 300 | Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}\n\n"
    contenido += "COEFICIENTES DEL MODELO\n"
    contenido += "-" * 40 + "\n"
    for nombre, coef in zip(nombres, modelo.coef_):
        contenido += f"  {nombre:<20}: {coef:.2f}\n"
    contenido += f"  {'Intercepto':<20}: {modelo.intercept_:.2f}\n\n"
    contenido += "MÉTRICAS\n"
    contenido += "-" * 40 + "\n"
    contenido += f"  MSE  : {mse:.2f}\n"
    contenido += f"  RMSE : {rmse:.2f}\n"
    contenido += f"  R²   : {r2:.4f}\n\n"
    contenido += "INTERPRETACIÓN\n"
    contenido += "-" * 40 + "\n"
    contenido += f"  Cada m² adicional suma ${modelo.coef_[0]:.0f} al precio.\n"
    contenido += f"  Cada habitación extra suma ${modelo.coef_[1]:.0f}.\n"
    contenido += f"  Cada km de distancia resta ${abs(modelo.coef_[2]):.0f}.\n"
    guardar_informe("02_regresion_lineal_multiple.txt", contenido)
