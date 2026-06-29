# =============================================================================
# Algoritmo 8: K Vecinos Más Cercanos (KNN)
# Dataset: Clasificación de vinos mendocinos (sintético argentino)
# Contexto: Clasificar variedad de uva según propiedades fisicoquímicas
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "KNN (K Vecinos Más Cercanos)"
DATASET = "Clasificación de vinos mendocinos por varietal (sintético)"

CLASES = ["Malbec", "Cabernet Sauvignon", "Torrontés"]

def generar_dataset():
    """
    Propiedades de vinos de Mendoza:
    - alcohol: % vol (11 a 16)
    - acidez: pH (3.0 a 4.0)
    - taninos: índice (0 a 10)
    - color: intensidad de color (1 a 10)
    - azucar: g/L de azúcar residual (0 a 20)
    Clases: 0=Malbec, 1=Cabernet Sauvignon, 2=Torrontés
    """
    np.random.seed(64)
    n_por_clase = 150
    datos = []

    # Malbec: alto color, taninos medios-altos, alcohol medio-alto
    malbec = np.column_stack([
        np.random.normal(13.5, 0.8, n_por_clase),  # alcohol
        np.random.normal(3.5,  0.2, n_por_clase),  # acidez
        np.random.normal(7.0,  1.0, n_por_clase),  # taninos
        np.random.normal(8.0,  0.8, n_por_clase),  # color
        np.random.normal(2.5,  1.0, n_por_clase),  # azucar
    ])

    # Cabernet Sauvignon: muy tánico, color intenso, más ácido
    cabernet = np.column_stack([
        np.random.normal(13.0, 0.7, n_por_clase),
        np.random.normal(3.3,  0.2, n_por_clase),
        np.random.normal(8.5,  1.0, n_por_clase),
        np.random.normal(7.5,  0.8, n_por_clase),
        np.random.normal(2.0,  0.8, n_por_clase),
    ])

    # Torrontés: blanco, bajo tanino, más dulce, más ácido
    torrontes = np.column_stack([
        np.random.normal(12.5, 0.6, n_por_clase),
        np.random.normal(3.2,  0.2, n_por_clase),
        np.random.normal(1.5,  0.5, n_por_clase),
        np.random.normal(3.0,  0.8, n_por_clase),
        np.random.normal(8.0,  2.0, n_por_clase),
    ])

    X = np.vstack([malbec, cabernet, torrontes])
    y = np.array([0]*n_por_clase + [1]*n_por_clase + [2]*n_por_clase)
    return X, y

def ejecutar():
    print("  Generando dataset: vinos mendocinos...")
    X, y = generar_dataset()
    nombres = ["Alcohol (%)", "Acidez (pH)", "Taninos", "Color", "Azúcar (g/L)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Evaluamos distintos valores de K
    ks = range(1, 21)
    acc_por_k = []
    for k in ks:
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train_s, y_train)
        acc_por_k.append(accuracy_score(y_test, m.predict(X_test_s)))

    mejor_k   = ks[np.argmax(acc_por_k)]
    mejor_acc = max(acc_por_k)

    modelo = KNeighborsClassifier(n_neighbors=mejor_k)
    modelo.fit(X_train_s, y_train)
    y_pred = modelo.predict(X_test_s)
    rep = classification_report(y_test, y_pred, target_names=CLASES)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n  {'─'*40}")
    print(f"  RESULTADOS DEL MODELO")
    print(f"  {'─'*40}")
    print(f"  Mejor K  : {mejor_k}")
    print(f"  Accuracy : {mejor_acc:.4f}")
    print(f"\n{rep}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("KNN - Clasificación de Vinos Mendocinos", fontsize=13)

    # Accuracy vs K
    ax1 = axes[0]
    ax1.plot(list(ks), acc_por_k, marker='o', color='steelblue', linewidth=2)
    ax1.axvline(mejor_k, color='crimson', linestyle='--', label=f'K óptimo={mejor_k}')
    ax1.set_xlabel("Valor de K")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs K")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Matriz de confusión
    ax2 = axes[1]
    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
    ax2.set_title(f"Matriz de Confusión (K={mejor_k})")
    plt.colorbar(im, ax=ax2)
    tick_marks = np.arange(len(CLASES))
    ax2.set_xticks(tick_marks); ax2.set_xticklabels(CLASES, rotation=30, ha='right')
    ax2.set_yticks(tick_marks); ax2.set_yticklabels(CLASES)
    ax2.set_ylabel("Real"); ax2.set_xlabel("Predicho")
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black',
                     fontsize=13, fontweight='bold')

    plt.tight_layout()
    guardar_grafico(fig, "08_knn_vinos_mendoza.png")

    # --- Informe ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += f"  Mejor K  : {mejor_k}\n"
    contenido += f"  Accuracy : {mejor_acc:.4f}\n\n"
    contenido += "REPORTE DE CLASIFICACIÓN\n" + "-"*40 + "\n"
    contenido += rep
    guardar_informe("08_knn_vinos_mendoza.txt", contenido)
