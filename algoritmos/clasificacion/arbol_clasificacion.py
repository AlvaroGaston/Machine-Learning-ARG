# =============================================================================
# Algoritmo 11: Árbol de Decisión (Clasificación)
# Dataset: Riesgo crediticio de pymes argentinas (sintético)
# Contexto: Clasificar nivel de riesgo en Bajo / Medio / Alto
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "Árbol de Decisión (Clasificación)"
DATASET = "Riesgo crediticio de pymes argentinas (sintético)"

CLASES = ["Riesgo Bajo", "Riesgo Medio", "Riesgo Alto"]

def generar_dataset():
    """
    Variables de una pyme:
    - facturacion_anual: millones de pesos (1 a 500)
    - empleados: cantidad de empleados (1 a 200)
    - antiguedad_años: años de vida de la empresa (0 a 30)
    - deuda_ratio: deuda/activos (0 a 1.5)
    - sector: 0=agro, 1=comercio, 2=industria, 3=servicios
    - morosidad: días promedio de mora (0 a 180)
    Target: 0=Bajo, 1=Medio, 2=Alto
    """
    np.random.seed(77)
    n = 450

    facturacion = np.random.exponential(50, n)
    facturacion = np.clip(facturacion, 1, 500)
    empleados   = np.random.randint(1, 201, n).astype(float)
    antiguedad  = np.random.uniform(0, 30, n)
    deuda_ratio = np.random.uniform(0, 1.5, n)
    sector      = np.random.randint(0, 4, n).astype(float)
    morosidad   = np.random.exponential(20, n)
    morosidad   = np.clip(morosidad, 0, 180)

    # Score de riesgo: mayor = más riesgo
    score = (-0.002 * facturacion
             - 0.01 * empleados
             - 0.05 * antiguedad
             + 1.5 * deuda_ratio
             + 0.01 * morosidad
             + np.random.normal(0, 0.3, n))

    # Dividimos en 3 niveles
    q33, q66 = np.percentile(score, [33, 66])
    y = np.where(score < q33, 0, np.where(score < q66, 1, 2))

    X = np.column_stack([facturacion, empleados, antiguedad,
                         deuda_ratio, sector, morosidad])
    return X, y

def ejecutar():
    print("  Generando dataset: riesgo crediticio de pymes...")
    X, y = generar_dataset()
    nombres = ["Facturación (M$)", "Empleados", "Antigüedad (años)",
               "Ratio deuda", "Sector", "Morosidad (días)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Probamos distintas profundidades
    profundidades = [3, 5, 8, None]
    resultados = {}
    for prof in profundidades:
        m = DecisionTreeClassifier(max_depth=prof, random_state=42)
        m.fit(X_train, y_train)
        y_pred_p = m.predict(X_test)
        resultados[str(prof)] = {
            'modelo': m,
            'y_pred': y_pred_p,
            'acc': accuracy_score(y_test, y_pred_p)
        }

    mejor_key   = max(resultados, key=lambda k: resultados[k]['acc'])
    mejor_modelo = resultados[mejor_key]['modelo']
    y_pred = resultados[mejor_key]['y_pred']
    rep = classification_report(y_test, y_pred, target_names=CLASES)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n  {'─'*40}")
    print(f"  COMPARACIÓN DE PROFUNDIDADES")
    print(f"  {'─'*40}")
    for k, res in resultados.items():
        marca = " ← mejor" if k == mejor_key else ""
        print(f"  max_depth={k:<5}: Accuracy={res['acc']:.4f}{marca}")
    print(f"\n{rep}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Árbol de Decisión - Riesgo Crediticio Pymes Argentinas", fontsize=13)

    # Importancia de variables
    ax1 = axes[0]
    importancias = mejor_modelo.feature_importances_
    idx = np.argsort(importancias)
    colores = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(nombres)))
    ax1.barh([nombres[i] for i in idx], importancias[idx],
             color=[colores[i] for i in idx])
    ax1.set_xlabel("Importancia")
    ax1.set_title(f"Importancia de variables (depth={mejor_key})")
    ax1.grid(True, alpha=0.3, axis='x')

    # Matriz de confusión
    ax2 = axes[1]
    im = ax2.imshow(cm, interpolation='nearest', cmap='YlOrRd')
    ax2.set_title(f"Matriz de Confusión\nAccuracy={resultados[mejor_key]['acc']:.3f}")
    plt.colorbar(im, ax=ax2)
    tick_marks = np.arange(len(CLASES))
    ax2.set_xticks(tick_marks); ax2.set_xticklabels(CLASES, rotation=30, ha='right')
    ax2.set_yticks(tick_marks); ax2.set_yticklabels(CLASES)
    ax2.set_ylabel("Real"); ax2.set_xlabel("Predicho")
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax2.text(j, i, str(cm[i,j]), ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black',
                     fontsize=13, fontweight='bold')

    plt.tight_layout()
    guardar_grafico(fig, "11_arbol_clasificacion_pymes.png")

    # --- Informe ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += "COMPARACIÓN DE PROFUNDIDADES\n" + "-"*40 + "\n"
    for k, res in resultados.items():
        marca = " ← mejor" if k == mejor_key else ""
        contenido += f"  max_depth={k:<5}: Accuracy={res['acc']:.4f}{marca}\n"
    contenido += "\nREPORTE DE CLASIFICACIÓN\n" + "-"*40 + "\n"
    contenido += rep
    contenido += "\nIMPORTANCIA DE VARIABLES\n" + "-"*40 + "\n"
    for nombre, imp in zip(nombres, mejor_modelo.feature_importances_):
        contenido += f"  {nombre:<24}: {imp:.4f}\n"
    guardar_informe("11_arbol_clasificacion_pymes.txt", contenido)
