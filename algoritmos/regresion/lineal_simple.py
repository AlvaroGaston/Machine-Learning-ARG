# =============================================================================
# Algoritmo 1: Regresión Lineal Simple
# Dataset: Precio de nafta vs distancia recorrida (sintético argentino)
# Contexto: Relación entre litros consumidos y km recorridos en rutas nacionales
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "Regresión Lineal Simple"
DATASET = "Precio de nafta vs distancia recorrida (rutas nacionales argentinas)"

def generar_dataset():
    """
    Genera datos sintéticos que simulan la relación entre litros de nafta
    consumidos y kilómetros recorridos en rutas nacionales de Argentina.
    Precio promedio nafta premium: ~$1200/litro (2024).
    Consumo promedio auto: 10 km/litro → gasto = km / 10 * precio_litro.
    """
    np.random.seed(42)
    n = 200
    # Variable independiente: kilómetros recorridos (entre 50 y 1500 km)
    km = np.random.uniform(50, 1500, n)
    # Precio en pesos: ~120 pesos por km (consumo + variación aleatoria)
    precio = 120 * km + np.random.normal(0, 5000, n)
    return km.reshape(-1, 1), precio

def ejecutar():
    print("  Generando dataset: consumo de nafta en rutas argentinas...")
    X, y = generar_dataset()

    # División entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenamiento del modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)

    # Métricas
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  {'─'*40}")
    print(f"  RESULTADOS DEL MODELO")
    print(f"  {'─'*40}")
    print(f"  Coeficiente (pendiente) : {modelo.coef_[0]:.4f}")
    print(f"  Intercepto              : {modelo.intercept_:.2f}")
    print(f"  MSE                     : {mse:.2f}")
    print(f"  RMSE                    : {rmse:.2f}")
    print(f"  R²                      : {r2:.4f}")
    print(f"  {'─'*40}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Regresión Lineal Simple\nNafta vs Distancia (Rutas Argentinas)", fontsize=13)

    # Gráfico 1: Datos reales vs línea de regresión
    ax1 = axes[0]
    x_linea = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    ax1.scatter(X_test, y_test, alpha=0.5, color='steelblue', label='Datos reales')
    ax1.plot(x_linea, modelo.predict(x_linea), color='crimson', linewidth=2, label='Regresión')
    ax1.set_xlabel("Kilómetros recorridos")
    ax1.set_ylabel("Costo en pesos ($)")
    ax1.set_title("Datos vs Línea de Regresión")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Valores reales vs predichos
    ax2 = axes[1]
    ax2.scatter(y_test, y_pred, alpha=0.5, color='darkorange')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', linewidth=1.5, label='Predicción perfecta')
    ax2.set_xlabel("Valores reales ($)")
    ax2.set_ylabel("Valores predichos ($)")
    ax2.set_title("Real vs Predicho")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    guardar_grafico(fig, "01_regresion_lineal_simple.png")

    # --- Informe de texto ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += "DESCRIPCIÓN DEL DATASET\n"
    contenido += "-" * 40 + "\n"
    contenido += "Datos sintéticos que simulan el costo de viaje en auto\n"
    contenido += "por rutas nacionales argentinas según los km recorridos.\n"
    contenido += f"  - Muestras totales : 200\n"
    contenido += f"  - Entrenamiento    : {len(X_train)} muestras\n"
    contenido += f"  - Prueba           : {len(X_test)} muestras\n\n"
    contenido += "RESULTADOS DEL MODELO\n"
    contenido += "-" * 40 + "\n"
    contenido += f"  Coeficiente (pendiente) : {modelo.coef_[0]:.4f}\n"
    contenido += f"  Intercepto              : {modelo.intercept_:.2f}\n"
    contenido += f"  MSE                     : {mse:.2f}\n"
    contenido += f"  RMSE                    : {rmse:.2f}\n"
    contenido += f"  R²                      : {r2:.4f}\n\n"
    contenido += "INTERPRETACIÓN\n"
    contenido += "-" * 40 + "\n"
    contenido += f"  Por cada km adicional, el costo aumenta ${modelo.coef_[0]:.2f}.\n"
    contenido += f"  El modelo explica el {r2*100:.1f}% de la variabilidad del costo.\n"
    guardar_informe("01_regresion_lineal_simple.txt", contenido)
