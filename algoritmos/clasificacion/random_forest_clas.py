# =============================================================================
# Algoritmo 12: Random Forest (Clasificación)
# Dataset: Diagnóstico de cultivos pampeanos (sintético argentino)
# Contexto: Clasificar estado del cultivo en Óptimo / Regular / Deficiente
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "Random Forest (Clasificación)"
DATASET = "Diagnóstico de cultivos pampeanos argentinos (sintético)"

CLASES = ["Óptimo", "Regular", "Deficiente"]

def generar_dataset():
    """
    Variables agronómicas de parcelas en la pampa húmeda:
    - humedad_suelo: % de humedad del suelo (10 a 80)
    - ph_suelo: pH del suelo (4.5 a 8.5)
    - temp_min: temperatura mínima nocturna en °C (0 a 20)
    - lluvia_7dias: lluvia últimos 7 días en mm (0 a 150)
    - nitrogeno: nitrógeno disponible en kg/ha (0 a 200)
    - fosforo: fósforo en ppm (0 a 60)
    - dias_sin_lluvia: días desde la última lluvia (0 a 60)
    Target: 0=Óptimo, 1=Regular, 2=Deficiente
    """
    np.random.seed(123)
    n = 500

    humedad      = np.random.uniform(10, 80, n)
    ph           = np.random.uniform(4.5, 8.5, n)
    temp_min     = np.random.uniform(0, 20, n)
    lluvia_7dias = np.random.uniform(0, 150, n)
    nitrogeno    = np.random.uniform(0, 200, n)
    fosforo      = np.random.uniform(0, 60, n)
    dias_seco    = np.random.uniform(0, 60, n)

    # Score de salud: mayor = mejor estado del cultivo
    score = (0.4 * humedad
             - 5 * np.abs(ph - 6.5)          # pH ideal ~6.5
             + 0.2 * lluvia_7dias
             + 0.3 * nitrogeno
             + 0.5 * fosforo
             - 0.8 * dias_seco
             + 0.3 * temp_min
             + np.random.normal(0, 5, n))

    q33, q66 = np.percentile(score, [33, 66])
    y = np.where(score >= q66, 0, np.where(score >= q33, 1, 2))

    X = np.column_stack([humedad, ph, temp_min, lluvia_7dias,
                         nitrogeno, fosforo, dias_seco])
    return X, y

def ejecutar():
    print("  Generando dataset: diagnóstico de cultivos pampeanos...")
    X, y = generar_dataset()
    nombres = ["Humedad suelo (%)", "pH suelo", "Temp. mín. (°C)",
               "Lluvia 7 días (mm)", "Nitrógeno (kg/ha)",
               "Fósforo (ppm)", "Días sin lluvia"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(n_estimators=150, max_depth=10,
                                    random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    rep  = classification_report(y_test, y_pred, target_names=CLASES)
    cm   = confusion_matrix(y_test, y_pred)
    importancias = modelo.feature_importances_

    print(f"\n  {'─'*40}")
    print(f"  RESULTADOS DEL MODELO")
    print(f"  {'─'*40}")
    print(f"  Estimadores : 150 árboles")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"\n{rep}")
    print(f"  Importancia de variables:")
    for nombre, imp in zip(nombres, importancias):
        print(f"    {nombre:<24}: {imp:.4f}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Random Forest (Clasificación)\nDiagnóstico de Cultivos Pampeanos", fontsize=13)

    # Importancia de variables
    ax1 = axes[0]
    idx = np.argsort(importancias)
    colores = plt.cm.Greens(np.linspace(0.3, 0.9, len(nombres)))
    ax1.barh([nombres[i] for i in idx], importancias[idx],
             color=[colores[i] for i in idx])
    ax1.set_xlabel("Importancia")
    ax1.set_title("Importancia de variables")
    ax1.grid(True, alpha=0.3, axis='x')

    # Matriz de confusión
    ax2 = axes[1]
    im = ax2.imshow(cm, interpolation='nearest', cmap='Greens')
    ax2.set_title(f"Matriz de Confusión\nAccuracy={acc:.3f}")
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
    guardar_grafico(fig, "12_random_forest_cultivos.png")

    # --- Informe ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += "VARIABLES AGRONÓMICAS\n" + "-"*40 + "\n"
    for n_ in nombres:
        contenido += f"  - {n_}\n"
    contenido += f"\n  Estimadores : 150 árboles\n"
    contenido += f"  Accuracy    : {acc:.4f}\n\n"
    contenido += "REPORTE DE CLASIFICACIÓN\n" + "-"*40 + "\n"
    contenido += rep
    contenido += "\nIMPORTANCIA DE VARIABLES\n" + "-"*40 + "\n"
    for nombre, imp in zip(nombres, importancias):
        contenido += f"  {nombre:<26}: {imp:.4f}\n"
    guardar_informe("12_random_forest_cultivos.txt", contenido)
