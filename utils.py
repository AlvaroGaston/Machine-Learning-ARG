# =============================================================================
# utils.py - Funciones utilitarias compartidas
# =============================================================================

import os
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica (para guardar PNG)
import matplotlib.pyplot as plt
from datetime import datetime

CARPETA_INFORMES = "informes"

def asegurar_carpeta_informes():
    """Crea la carpeta informes/ si no existe."""
    if not os.path.exists(CARPETA_INFORMES):
        os.makedirs(CARPETA_INFORMES)

def guardar_grafico(fig, nombre_archivo):
    """
    Guarda un gráfico matplotlib como PNG en la carpeta informes/.
    Retorna la ruta del archivo guardado.
    """
    asegurar_carpeta_informes()
    ruta = os.path.join(CARPETA_INFORMES, nombre_archivo)
    fig.savefig(ruta, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f"  [✓] Gráfico guardado: {ruta}")
    return ruta

def guardar_informe(nombre_archivo, contenido):
    """
    Guarda un informe de texto (.txt) en la carpeta informes/.
    Retorna la ruta del archivo guardado.
    """
    asegurar_carpeta_informes()
    ruta = os.path.join(CARPETA_INFORMES, nombre_archivo)
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(contenido)
    print(f"  [✓] Informe guardado: {ruta}")
    return ruta

def encabezado_informe(titulo, dataset):
    """Genera un encabezado estándar para los informes de texto."""
    linea = "=" * 60
    ahora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return (
        f"{linea}\n"
        f"TP Parcial 02 - Machine Learning\n"
        f"Universidad Abierta Interamericana\n"
        f"{linea}\n"
        f"Algoritmo : {titulo}\n"
        f"Dataset   : {dataset}\n"
        f"Fecha/Hora: {ahora}\n"
        f"{linea}\n\n"
    )

def separador():
    """Imprime una línea separadora en consola."""
    print("\n" + "=" * 60 + "\n")
