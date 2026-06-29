# =============================================================================
# Algoritmo 10: Naive Bayes
# Dataset: Clasificación de noticias argentinas (Política/Economía/Deportes)
# Contexto: Clasificación de texto con TF-IDF y Multinomial Naive Bayes
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import guardar_grafico, guardar_informe, encabezado_informe

TITULO  = "Naive Bayes (Multinomial)"
DATASET = "Noticias argentinas: Política / Economía / Deportes (sintético)"

CLASES = ["Política", "Economía", "Deportes"]

# Titulares sintéticos representativos de medios argentinos
TITULARES = {
    "Política": [
        "El presidente anunció nuevas medidas de gobierno en Casa Rosada",
        "El Congreso debatió el proyecto de ley de presupuesto nacional",
        "El Senado aprobó por mayoría la reforma constitucional",
        "El gobernador bonaerense presentó su plan de gestión provincial",
        "La oposición rechazó el decreto de necesidad y urgencia presidencial",
        "El canciller viajó a Brasil para reunión del Mercosur",
        "El oficialismo logró quórum para tratar la reforma del Estado",
        "Los ministros debatieron la política exterior argentina en el gabinete",
        "El jefe de gabinete presentó el informe al Congreso de la Nación",
        "La Corte Suprema dictó sentencia en el caso de derechos civiles",
        "El intendente inauguró obras de infraestructura en el municipio",
        "El partido opositor lanzó su candidato para las elecciones primarias",
        "El vice presidente presidió la sesión del Senado de la Nación",
        "La legislatura provincial sancionó la ley de ordenamiento territorial",
        "El gobierno nacional firmó convenios con las provincias patagónicas",
    ],
    "Economía": [
        "El dólar blue subió diez pesos y cerró en nuevo récord histórico",
        "La inflación de noviembre fue del 12 por ciento según el INDEC",
        "El BCRA vendió reservas para sostener el tipo de cambio oficial",
        "La bolsa de comercio de Buenos Aires cerró con suba del dos por ciento",
        "El FMI aprobó el desembolso de fondos para la Argentina",
        "La tasa de desempleo bajó al ocho por ciento según el último censo",
        "Las exportaciones de soja alcanzaron un récord en dólares este año",
        "El gobierno anunció baja de retenciones para el campo argentino",
        "Los supermercados reportaron caída en ventas por pérdida del salario",
        "El índice de pobreza llegó al cuarenta por ciento según INDEC",
        "El peso se depreció frente al dólar en la jornada de hoy",
        "La deuda externa argentina alcanzó los trescientos mil millones",
        "El precio del petróleo impacta en el consumo de nafta local",
        "Las pymes sufrieron caída de ventas por la suba de tasas bancarias",
        "El Banco Nación redujo la tasa de interés para créditos hipotecarios",
    ],
    "Deportes": [
        "La selección argentina goleó a Brasil en el Monumental de Núñez",
        "Boca Juniors venció a River Plate en el superclásico del fútbol argentino",
        "Lionel Messi marcó hat trick en la Copa América celebrada en Buenos Aires",
        "San Lorenzo clasificó a la semifinal de la Copa Libertadores de América",
        "Racing Club se consagró campeón del torneo Apertura de primera división",
        "El Millonario presentó a su nuevo delantero en el Estadio Monumental",
        "Independiente de Avellaneda cayó en el clásico de la avenida Mitre",
        "Estudiantes de La Plata logró el ascenso a la primera división",
        "La Bombonera se llenó para el debut del nuevo técnico xeneize",
        "Huracán ganó el clásico porteño ante San Lorenzo con gol agónico",
        "El equipo nacional de básquet clasificó al mundial de baloncesto",
        "Los Pumas vencieron a los All Blacks en el Rugby Championship",
        "Del Potro volvió a competir en el tenis argentino tras lesión",
        "Vélez Sarsfield cortó racha adversa con triunfo ante Lanús",
        "Talleres de Córdoba sorprendió a River con empate en el Monumental",
    ],
}

def generar_dataset():
    """Combina los titulares sintéticos y crea etiquetas."""
    textos = []
    etiquetas = []
    for i, clase in enumerate(CLASES):
        for texto in TITULARES[clase]:
            # Augmentamos con variaciones leves
            textos.append(texto)
            etiquetas.append(i)
            textos.append(texto + " según fuentes oficiales")
            etiquetas.append(i)
            textos.append("Hoy: " + texto)
            etiquetas.append(i)
    return np.array(textos), np.array(etiquetas)

def ejecutar():
    print("  Generando dataset: noticias argentinas (Política/Economía/Deportes)...")
    X_texto, y = generar_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X_texto, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v  = vectorizer.transform(X_test)

    modelo = MultinomialNB(alpha=0.5)
    modelo.fit(X_train_v, y_train)
    y_pred = modelo.predict(X_test_v)

    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=CLASES)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n  {'─'*40}")
    print(f"  RESULTADOS DEL MODELO")
    print(f"  {'─'*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"\n{rep}")

    # --- Gráfico ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Naive Bayes - Clasificación de Noticias Argentinas", fontsize=13)

    # Matriz de confusión
    ax1 = axes[0]
    im = ax1.imshow(cm, interpolation='nearest', cmap='Purples')
    ax1.set_title(f"Matriz de Confusión\nAccuracy={acc:.3f}")
    plt.colorbar(im, ax=ax1)
    tick_marks = np.arange(len(CLASES))
    ax1.set_xticks(tick_marks); ax1.set_xticklabels(CLASES, rotation=30, ha='right')
    ax1.set_yticks(tick_marks); ax1.set_yticklabels(CLASES)
    ax1.set_ylabel("Real"); ax1.set_xlabel("Predicho")
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax1.text(j, i, str(cm[i,j]), ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black',
                     fontsize=14, fontweight='bold')

    # Top palabras por clase
    ax2 = axes[1]
    feature_names = vectorizer.get_feature_names_out()
    top_n = 5
    colores_clase = ['steelblue', 'darkorange', 'green']
    y_pos = 0
    yticks_labels = []
    yticks_pos = []
    for i, (clase, color) in enumerate(zip(CLASES, colores_clase)):
        log_prob = modelo.feature_log_prob_[i]
        top_idx  = log_prob.argsort()[-top_n:][::-1]
        for idx in top_idx:
            ax2.barh(y_pos, log_prob[idx], color=color, alpha=0.8)
            yticks_labels.append(feature_names[idx])
            yticks_pos.append(y_pos)
            y_pos += 1
        y_pos += 0.5  # espacio entre clases

    ax2.set_yticks(yticks_pos)
    ax2.set_yticklabels(yticks_labels, fontsize=8)
    ax2.set_xlabel("Log-probabilidad")
    ax2.set_title("Top palabras por categoría")

    from matplotlib.patches import Patch
    leyenda = [Patch(color=c, label=cl) for c, cl in zip(colores_clase, CLASES)]
    ax2.legend(handles=leyenda, loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    guardar_grafico(fig, "10_naive_bayes_noticias.png")

    # --- Informe ---
    contenido = encabezado_informe(TITULO, DATASET)
    contenido += f"  Muestras totales  : {len(X_texto)}\n"
    contenido += f"  Entrenamiento     : {len(X_train)}\n"
    contenido += f"  Prueba            : {len(X_test)}\n"
    contenido += f"  Accuracy          : {acc:.4f}\n\n"
    contenido += "REPORTE DE CLASIFICACIÓN\n" + "-"*40 + "\n"
    contenido += rep
    guardar_informe("10_naive_bayes_noticias.txt", contenido)
