# Machine Learning - ARG

Proyecto educativo con implementaciones en Python de algoritmos de aprendizaje automático (regresión y clasificación), acompañado de ejemplos prácticos y visualizaciones.

## Descripción
Este repositorio contiene implementaciones didácticas de varios algoritmos de regresión y clasificación, un menú interactivo para ejecutar los ejemplos, y una carpeta `informes/` con gráficos y resúmenes explicativos. Es ideal para aprender los pasos básicos de cada algoritmo y ver cómo se generan las visualizaciones asociadas.

## Características
- Implementaciones separadas por tipo: `algoritmos/regresion` y `algoritmos/clasificacion`.
- Menú interactivo (`main.py` / `menu.py`) para seleccionar y ejecutar ejemplos.
- Informes con gráficos (PNG) y archivos de texto con explicación por cada ejemplo (`informes/`).
- Código en Python pensado para entender el flujo completo: carga de datos, entrenamiento, evaluación y visualización.

## Estructura del repositorio
```
.
├── algoritmos/
│   ├── regresion/
│   │   ├── lineal_simple.py
│   │   ├── lineal_multiple.py
│   │   ├── polinomial.py
│   │   ├── arbol_regresion.py
│   │   ├── random_forest_reg.py
│   │   └── svr.py
│   └── clasificacion/
│       ├── logistica.py
│       ├── knn.py
│       ├── svm.py
│       ├── naive_bayes.py
│       ├── random_forest_clas.py
│       └── arbol_clasificacion.py
├── informes/                  # PNGs y .txt explicativos por ejemplo
├── main.py                    # Entrada: ejecuta el menú
├── menu.py                    # Menú interactivo
├── utils.py                   # Utilidades (gráficos, guardado, helpers)
└── README.md
```

## Requisitos
- Python 3.8 o superior
- Recomendado instalar:
```bash
pip install numpy pandas scikit-learn matplotlib
```
(Crea un entorno virtual antes si lo deseas.)

## Cómo usar
1. Clona el repositorio:
```bash
git clone https://github.com/AlvaroGaston/Machine-Learning-ARG.git
cd Machine-Learning-ARG
```
2. Instala dependencias (opcional):
```bash
pip install -r requirements.txt  # si añades este archivo
```
3. Ejecuta el menú:
```bash
python main.py
```
Sigue las instrucciones en pantalla para seleccionar el algoritmo o ejemplo que quieras correr. Los gráficos y resultados se guardarán en `informes/`.

## Buenas prácticas sugeridas
- Añadir un `requirements.txt` con versiones de paquetes.
- Documentar para cada script qué dependencias concretas usa.
- Añadir ejemplos de uso y datos de muestra (o scripts que los descarguen/los generen).
- Considerar convertir las funciones principales en un paquete instalable (`setup.py`/`pyproject.toml`) si quieres reutilizar módulos.

## Contribuir
- Si quieres contribuir, abre un issue con la idea o crea un pull request con mejoras (añadir README detallado para cada algoritmo, tests, o automatización del menú).
- Mantén el estilo de nombre y estructura de carpetas para que los informes sigan generándose en `informes/`.

## Créditos
Autor original: AlvaroGaston

---
Si quieres, puedo:
- Generar un `requirements.txt` con las dependencias mínimas.
- Añadir ejemplos de ejecución automáticos (scripts en `examples/`).
- Refactorizar el menú para aceptar argumentos CLI y ejecutar ejemplos desde la terminal sin interacción.
