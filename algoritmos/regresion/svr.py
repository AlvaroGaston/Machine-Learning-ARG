# =============================================================================
# Algoritmo 4: SVR (Support Vector Regression)
# Dataset: Precio del dólar blue histórico (sintético argentino)
# Contexto: Serie temporal simulada del tipo de cambio informal
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "SVR (Support Vector Regression)"
DATASET = "Precio del dólar blue histórico (Argentina sintético)"

def generar_dataset():
    """
    Simula la evolución del dólar blue a lo largo de 36 meses.
    Combina una tendencia exponencial (devaluación) con ciclos y ruido.
    Variable X: mes (1 a 36)
    Variable y: precio del dólar blue en pesos
    """
    np.random.seed(33)
    n = 200
    mes = np.linspace(1, 36, n)
    # Tendencia exponencial + componente estacional + ruido
    precio = (200 * np.exp(0.08 * mes)
              + 500 * np.sin(mes / 3)
              + np.random.normal(0, 300, n))
    return mes.reshape(-1, 1), precio

def ejecutar():
    print("  Generando dataset: dólar blue histórico argentino...")
    X, y = generar_dataset()

    # Escalado obligatorio para SVR
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Comparamos kernels
    kernels = ['linear', 'rbf', 'poly']
    resultados = {}
    for kernel in kernels:
        modelo = SVR(kernel=kernel, C=100, epsilon=0.1)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        resultados[kernel] = {
            'modelo': modelo,
            'y_pred': y_pred,
            'r2':   r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

    mejor = max(resultados, key=lambda k: resultados[k]['r2'])

    print(f"\n  {'─'*40}")
    print(f"  COMPARACIÓN DE KERNELS SVR")
    print(f"  {'─'*40}")
    for k, res in resultados.items():
        marca = " ← mejor" if k == mejor else ""
        print(f"  Kernel {k:<8}: R²={res['r2']:.4f}  RMSE={res['rmse']:.4f}{marca}")
    print(f"  {'─'*40}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("SVR - Dólar Blue Histórico (Argentina)", fontsize=13)

    colores = {'linear': 'green', 'rbf': 'crimson', 'poly': 'darkorange'}
    x_plot = np.linspace(X_scaled.min(), X_scaled.max(), 200).reshape(-1, 1)

    ax1 = axes[0]
    ax1.scatter(X_test, y_test, alpha=0.4, color='steelblue', label='Datos reales', zorder=3)
    for k, res in resultados.items():
        ax1.plot(x_plot, res['modelo'].predict(x_plot), color=colores[k],
                 linewidth=2, label=f'{k} (R²={res["r2"]:.3f})')
    ax1.set_xlabel("Mes (normalizado)")
    ax1.set_ylabel("Precio (normalizado)")
    ax1.set_title("Comparación de kernels")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    y_pred_mejor = resultados[mejor]['y_pred']
    ax2.scatter(y_test, y_pred_mejor, alpha=0.5, color='darkorange')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', linewidth=1.5, label='Predicción perfecta')
    ax2.set_xlabel("Valores reales (norm.)")
    ax2.set_ylabel("Valores predichos (norm.)")
    ax2.set_title(f"Real vs Predicho (kernel={mejor})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    guardar_grafico(fig, "04_svr_dolar_blue.png")

    # --- Informe ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += "DESCRIPCIÓN DEL DATASET\n" + "-"*40 + "\n"
    contenido += "Serie temporal simulada del dólar blue argentino (36 meses).\n"
    contenido += "Incluye tendencia exponencial + estacionalidad + ruido.\n\n"
    contenido += "COMPARACIÓN DE KERNELS\n" + "-"*40 + "\n"
    for k, res in resultados.items():
        marca = " ← mejor" if k == mejor else ""
        contenido += f"  {k:<10}: R²={res['r2']:.4f}  RMSE={res['rmse']:.4f}{marca}\n"
    contenido += f"\nMejor kernel: {mejor}\n"
    guardar_informe("04_svr_dolar_blue.txt", contenido)
