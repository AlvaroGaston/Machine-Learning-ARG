# =============================================================================
# menu.py - Menú interactivo de consola
# =============================================================================

from utils import separador

# Importaciones de algoritmos de regresión
from algoritmos.regresion.lineal_simple       import ejecutar as reg_lineal_simple
from algoritmos.regresion.lineal_multiple     import ejecutar as reg_lineal_multiple
from algoritmos.regresion.polinomial          import ejecutar as reg_polinomial
from algoritmos.regresion.svr                 import ejecutar as reg_svr
from algoritmos.regresion.arbol_regresion     import ejecutar as reg_arbol
from algoritmos.regresion.random_forest_reg   import ejecutar as reg_random_forest

# Importaciones de algoritmos de clasificación
from algoritmos.clasificacion.logistica             import ejecutar as clas_logistica
from algoritmos.clasificacion.knn                   import ejecutar as clas_knn
from algoritmos.clasificacion.svm                   import ejecutar as clas_svm
from algoritmos.clasificacion.naive_bayes           import ejecutar as clas_naive_bayes
from algoritmos.clasificacion.arbol_clasificacion   import ejecutar as clas_arbol
from algoritmos.clasificacion.random_forest_clas    import ejecutar as clas_random_forest

OPCIONES = {
    # --- REGRESIÓN ---
    "1":  ("Regresión Lineal Simple       | Precio de nafta vs distancia recorrida",  reg_lineal_simple),
    "2":  ("Regresión Lineal Múltiple     | Precio de alquiler en CABA",              reg_lineal_multiple),
    "3":  ("Regresión Polinomial          | Inflación mensual vs poder adquisitivo",  reg_polinomial),
    "4":  ("SVR                           | Precio del dólar blue histórico",         reg_svr),
    "5":  ("Árbol de Decisión (Regresión) | Producción agrícola (soja/maíz)",         reg_arbol),
    "6":  ("Random Forest (Regresión)     | Consumo eléctrico por provincia",         reg_random_forest),
    # --- CLASIFICACIÓN ---
    "7":  ("Regresión Logística           | Aprobación de crédito bancario",          clas_logistica),
    "8":  ("KNN                           | Clasificación de vinos mendocinos",       clas_knn),
    "9":  ("SVM                           | Detección de spam (correos AFIP)",        clas_svm),
    "10": ("Naive Bayes                   | Noticias (Política/Economía/Deportes)",   clas_naive_bayes),
    "11": ("Árbol de Decisión (Clasif.)   | Riesgo crediticio de pymes",              clas_arbol),
    "12": ("Random Forest (Clasificación) | Diagnóstico de cultivos pampeanos",       clas_random_forest),
}

def mostrar_menu():
    """Bucle principal del menú de consola."""
    while True:
        separador()
        print("  TP Parcial 02 - Machine Learning")
        print("  Universidad Abierta Interamericana\n")
        print("  ── REGRESIÓN ──────────────────────────────────────")
        for k in ["1","2","3","4","5","6"]:
            print(f"  {k:>2}. {OPCIONES[k][0]}")
        print("\n  ── CLASIFICACIÓN ──────────────────────────────────")
        for k in ["7","8","9","10","11","12"]:
            print(f"  {k:>2}. {OPCIONES[k][0]}")
        print("\n   0. Salir")
        separador()

        opcion = input("  Seleccioná una opción: ").strip()

        if opcion == "0":
            print("\n  ¡Hasta luego!\n")
            break
        elif opcion in OPCIONES:
            separador()
            print(f"  Ejecutando: {OPCIONES[opcion][0]}\n")
            try:
                OPCIONES[opcion][1]()  # Llama a la función ejecutar() del módulo
            except Exception as e:
                print(f"\n  [ERROR] {e}")
            input("\n  Presioná Enter para volver al menú...")
        else:
            print("\n  [!] Opción inválida. Ingresá un número del 0 al 12.")
