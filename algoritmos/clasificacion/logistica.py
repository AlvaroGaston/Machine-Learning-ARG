# =============================================================================
# Algoritmo 7: Regresión Logística
# Dataset: Aprobación de crédito bancario (sintético argentino)
# Contexto: Clasificación binaria — crédito aprobado o rechazado
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "Regresión Logística"
DATASET = "Aprobación de crédito bancario argentino (sintético)"

def generar_dataset():
    """
    Variables del solicitante:
    - ingresos_miles: ingresos mensuales en miles de pesos (20 a 500)
    - deuda_pct: porcentaje de deuda sobre ingresos (0 a 120%)
    - antiguedad: años de antigüedad laboral (0 a 30)
    - historial: score crediticio (0 a 100)
    - monto_solicitado: miles de pesos (50 a 2000)
    Target: 1 = aprobado, 0 = rechazado
    """
    np.random.seed(17)
    n = 500
    ingresos         = np.random.uniform(20, 500, n)
    deuda_pct        = np.random.uniform(0, 120, n)
    antiguedad       = np.random.uniform(0, 30, n)
    historial        = np.random.uniform(0, 100, n)
    monto_solicitado = np.random.uniform(50, 2000, n)

    # Probabilidad de aprobación basada en lógica de negocio
    score = (0.003 * ingresos
             - 0.02 * deuda_pct
             + 0.05 * antiguedad
             + 0.04 * historial
             - 0.001 * monto_solicitado
             + np.random.normal(0, 0.5, n))

    y = (score > 0.5).astype(int)
    X = np.column_stack([ingresos, deuda_pct, antiguedad, historial, monto_solicitado])
    return X, y

def ejecutar():
    print("  Generando dataset: solicitudes de crédito bancario...")
    X, y = generar_dataset()
    nombres = ["Ingresos (miles $)", "Deuda (%)", "Antigüedad (años)",
               "Score crediticio", "Monto solicitado (miles $)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_train_s, y_train)
    y_pred      = modelo.predict(X_test_s)
    y_prob      = modelo.predict_proba(X_test_s)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    rep  = classification_report(y_test, y_pred,
                                  target_names=["Rechazado", "Aprobado"])

    print(f"\n  {'─'*40}")
    print(f"  RESULTADOS DEL MODELO")
    print(f"  {'─'*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"\n{rep}")
    print(f"  {'─'*40}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Regresión Logística\nAprobación de Crédito Bancario", fontsize=13)

    # Matriz de confusión
    ax1 = axes[0]
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title(f"Matriz de Confusión\nAccuracy={acc:.3f}")
    plt.colorbar(im, ax=ax1)
    clases = ["Rechazado", "Aprobado"]
    tick_marks = np.arange(len(clases))
    ax1.set_xticks(tick_marks); ax1.set_xticklabels(clases, rotation=30)
    ax1.set_yticks(tick_marks); ax1.set_yticklabels(clases)
    ax1.set_ylabel("Real"); ax1.set_xlabel("Predicho")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black',
                     fontsize=14, fontweight='bold')

    # Curva ROC
    ax2 = axes[1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax2.plot(fpr, tpr, color='crimson', lw=2, label=f'ROC (AUC={auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Aleatorio')
    ax2.set_xlabel("Tasa de Falsos Positivos")
    ax2.set_ylabel("Tasa de Verdaderos Positivos")
    ax2.set_title("Curva ROC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    guardar_grafico(fig, "07_logistica_credito_bancario.png")

    # --- Informe ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += "VARIABLES\n" + "-"*40 + "\n"
    for n_ in nombres:
        contenido += f"  - {n_}\n"
    contenido += "\nMÉTRICAS\n" + "-"*40 + "\n"
    contenido += f"  Accuracy : {acc:.4f}\n"
    contenido += f"  AUC-ROC  : {auc:.4f}\n\n"
    contenido += "REPORTE DE CLASIFICACIÓN\n" + "-"*40 + "\n"
    contenido += rep
    contenido += "\nMATRIZ DE CONFUSIÓN\n" + "-"*40 + "\n"
    contenido += f"  VN={cm[0,0]}  FP={cm[0,1]}\n"
    contenido += f"  FN={cm[1,0]}  VP={cm[1,1]}\n"
    guardar_informe("07_logistica_credito_bancario.txt", contenido)
