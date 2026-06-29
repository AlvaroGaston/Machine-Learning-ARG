# =============================================================================
# Algoritmo 9: SVM (Support Vector Machine - Clasificación)
# Dataset: Detección de spam en correos con temática AFIP (sintético)
# Contexto: Clasificar emails como spam/legítimo según frecuencia de palabras
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "SVM (Support Vector Machine - Clasificación)"
DATASET = "Detección de spam en correos con temática AFIP (sintético)"

PALABRAS = ["AFIP", "urgente", "dinero", "premio", "gratis",
            "RUT", "declaración", "juicio", "banco", "clave"]

def generar_dataset():
    """
    Simula frecuencias de palabras clave en emails.
    Spam: alta frecuencia de 'urgente', 'dinero', 'premio', 'gratis', 'banco'
    Legítimo: alta frecuencia de 'AFIP', 'RUT', 'declaración'
    Target: 0 = legítimo, 1 = spam
    """
    np.random.seed(99)
    n = 600

    # Emails legítimos (300)
    legitimo = np.column_stack([
        np.random.poisson(5, n//2),    # AFIP (alta)
        np.random.poisson(1, n//2),    # urgente (baja)
        np.random.poisson(0.5, n//2),  # dinero
        np.random.poisson(0.2, n//2),  # premio
        np.random.poisson(0.1, n//2),  # gratis
        np.random.poisson(3, n//2),    # RUT (alta)
        np.random.poisson(4, n//2),    # declaración (alta)
        np.random.poisson(0.3, n//2),  # juicio
        np.random.poisson(1, n//2),    # banco
        np.random.poisson(2, n//2),    # clave
    ])

    # Emails spam (300)
    spam = np.column_stack([
        np.random.poisson(1, n//2),    # AFIP (baja, suplanta)
        np.random.poisson(5, n//2),    # urgente (alta)
        np.random.poisson(6, n//2),    # dinero (alta)
        np.random.poisson(4, n//2),    # premio (alta)
        np.random.poisson(5, n//2),    # gratis (alta)
        np.random.poisson(0.5, n//2),  # RUT
        np.random.poisson(0.5, n//2),  # declaración
        np.random.poisson(3, n//2),    # juicio (alta)
        np.random.poisson(4, n//2),    # banco (alta)
        np.random.poisson(0.5, n//2),  # clave
    ])

    X = np.vstack([legitimo, spam])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y

def ejecutar():
    print("  Generando dataset: detección de spam AFIP...")
    X, y = generar_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Comparamos kernels
    kernels = ['linear', 'rbf', 'poly']
    resultados = {}
    for kernel in kernels:
        m = SVC(kernel=kernel, C=1.0, random_state=42, probability=True)
        m.fit(X_train_s, y_train)
        y_pred_k = m.predict(X_test_s)
        resultados[kernel] = {
            'modelo': m,
            'y_pred': y_pred_k,
            'acc': accuracy_score(y_test, y_pred_k)
        }

    mejor = max(resultados, key=lambda k: resultados[k]['acc'])
    y_pred = resultados[mejor]['y_pred']
    rep = classification_report(y_test, y_pred, target_names=["Legítimo", "Spam"])
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n  {'─'*40}")
    print(f"  COMPARACIÓN DE KERNELS SVM")
    print(f"  {'─'*40}")
    for k, res in resultados.items():
        marca = " ← mejor" if k == mejor else ""
        print(f"  Kernel {k:<8}: Accuracy={res['acc']:.4f}{marca}")
    print(f"\n{rep}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("SVM - Detección de Spam (Correos AFIP)", fontsize=13)

    # Accuracy por kernel
    ax1 = axes[0]
    colores = ['steelblue', 'crimson', 'darkorange']
    accs = [resultados[k]['acc'] for k in kernels]
    bars = ax1.bar(kernels, accs, color=colores, edgecolor='white', width=0.5)
    ax1.set_ylim(0.5, 1.05)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy por Kernel")
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', fontsize=11)

    # Matriz de confusión
    ax2 = axes[1]
    im = ax2.imshow(cm, interpolation='nearest', cmap='Oranges')
    ax2.set_title(f"Matriz de Confusión (kernel={mejor})")
    plt.colorbar(im, ax=ax2)
    clases = ["Legítimo", "Spam"]
    tick_marks = np.arange(len(clases))
    ax2.set_xticks(tick_marks); ax2.set_xticklabels(clases)
    ax2.set_yticks(tick_marks); ax2.set_yticklabels(clases)
    ax2.set_ylabel("Real"); ax2.set_xlabel("Predicho")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i,j]), ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black',
                     fontsize=14, fontweight='bold')

    plt.tight_layout()
    guardar_grafico(fig, "09_svm_spam_afip.png")

    # --- Informe ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += "PALABRAS CLAVE ANALIZADAS\n" + "-"*40 + "\n"
    for p in PALABRAS:
        contenido += f"  - {p}\n"
    contenido += "\nCOMPARACIÓN DE KERNELS\n" + "-"*40 + "\n"
    for k, res in resultados.items():
        marca = " ← mejor" if k == mejor else ""
        contenido += f"  {k:<10}: Accuracy={res['acc']:.4f}{marca}\n"
    contenido += "\nREPORTE DE CLASIFICACIÓN\n" + "-"*40 + "\n"
    contenido += rep
    guardar_informe("09_svm_spam_afip.txt", contenido)
