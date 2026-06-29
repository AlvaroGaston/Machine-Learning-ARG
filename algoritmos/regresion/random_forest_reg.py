# =============================================================================
# Algoritmo 6: Random Forest (Regresión)
# Dataset: Consumo eléctrico por provincia (sintético argentino)
# Contexto: Predicción de consumo según temperatura, población e industria
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "Random Forest (Regresión)"
DATASET = "Consumo eléctrico por provincia argentina (sintético)"

def generar_dataset():
    """
    Simula el consumo eléctrico mensual (MWh) por provincia según:
    - temperatura: temperatura media mensual (5 a 40 °C)
    - poblacion: población en miles (50 a 4000)
    - industria: índice de actividad industrial (0 a 100)
    - mes: mes del año (1 a 12) — estacionalidad
    - precio_kwh: precio del kWh en pesos (0.5 a 5)
    """
    np.random.seed(88)
    n = 400
    temperatura  = np.random.uniform(5, 40, n)
    poblacion    = np.random.uniform(50, 4000, n)
    industria    = np.random.uniform(0, 100, n)
    mes          = np.random.randint(1, 13, n).astype(float)
    precio_kwh   = np.random.uniform(0.5, 5, n)

    # Consumo (MWh): sube con población e industria, baja con precio alto
    # Efecto estacional: más consumo en verano (meses 12,1,2) e invierno (6,7)
    estacional = np.abs(np.sin(mes * np.pi / 6))
    consumo = (50 * poblacion
               + 200 * industria
               + 100 * temperatura
               + 3000 * estacional
               - 500 * precio_kwh
               + np.random.normal(0, 5000, n))

    X = np.column_stack([temperatura, poblacion, industria, mes, precio_kwh])
    return X, consumo

def ejecutar():
    print("  Generando dataset: consumo eléctrico provincial...")
    X, y = generar_dataset()
    nombres = ["Temperatura (°C)", "Población (miles)", "Ind. Industrial",
               "Mes", "Precio kWh ($)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    importancias = modelo.feature_importances_

    print(f"\n  {'─'*40}")
    print(f"  RESULTADOS DEL MODELO")
    print(f"  {'─'*40}")
    print(f"  Estimadores : 100 árboles")
    print(f"  MSE         : {mse:.2f}")
    print(f"  RMSE        : {rmse:.2f}")
    print(f"  R²          : {r2:.4f}")
    print(f"\n  Importancia de variables:")
    for nombre, imp in zip(nombres, importancias):
        print(f"    {nombre:<22}: {imp:.4f}")
    print(f"  {'─'*40}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Random Forest (Regresión)\nConsumo Eléctrico por Provincia", fontsize=13)

    # Importancia de variables
    ax1 = axes[0]
    idx = np.argsort(importancias)
    ax1.barh([nombres[i] for i in idx], importancias[idx],
             color='steelblue', edgecolor='white')
    ax1.set_xlabel("Importancia")
    ax1.set_title("Importancia de variables")
    ax1.grid(True, alpha=0.3, axis='x')

    # Real vs predicho
    ax2 = axes[1]
    ax2.scatter(y_test, y_pred, alpha=0.4, color='darkorange')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', linewidth=1.5, label='Predicción perfecta')
    ax2.set_xlabel("Consumo real (MWh)")
    ax2.set_ylabel("Consumo predicho (MWh)")
    ax2.set_title("Real vs Predicho")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    guardar_grafico(fig, "06_random_forest_consumo_electrico.png")

    # --- Informe ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += "VARIABLES DE ENTRADA\n" + "-"*40 + "\n"
    for n_ in nombres:
        contenido += f"  - {n_}\n"
    contenido += "\nMÉTRICAS\n" + "-"*40 + "\n"
    contenido += f"  Estimadores : 100 árboles\n"
    contenido += f"  MSE         : {mse:.2f}\n"
    contenido += f"  RMSE        : {rmse:.2f}\n"
    contenido += f"  R²          : {r2:.4f}\n\n"
    contenido += "IMPORTANCIA DE VARIABLES\n" + "-"*40 + "\n"
    for nombre, imp in zip(nombres, importancias):
        contenido += f"  {nombre:<24}: {imp:.4f}\n"
    guardar_informe("06_random_forest_consumo_electrico.txt", contenido)
