# DEVELOPMENT_LOG.md

## Registro de Desarrollo del Proyecto: Sistema de Predicción de Calidad de Pinturas

---

### 📅 **Fecha de Inicio del Registro:** 16 de junio de 2025

Este documento sirve como un registro manual de los hitos y decisiones clave en el desarrollo del proyecto "Sistema de Predicción de Calidad de Pinturas", especialmente después de un reinicio del historial de Git.

---

### 🎯 **Propósito del Proyecto**

El objetivo principal de este proyecto es desarrollar una aplicación de Machine Learning capaz de predecir la calidad y propiedades clave de formulaciones de pintura, como la resistencia al fregado, viscosidad, poder cubriente, brillo, estabilidad y características de color (L, a, b), además de clasificar el estado final de la pintura (Éxito/Falla). El objetivo es proporcionar una herramienta interactiva para que los usuarios puedan experimentar con diferentes proporciones de ingredientes y obtener predicciones en tiempo real.

---

### 🚀 **Hitos y Desarrollo Inicial**

* **Generación de Datos Simulados (`generate_paint_data.py`):**
    * Se creó un script para generar un dataset sintético (`formulaciones_pintura_simuladas.csv`) con diversas características de formulación y propiedades de rendimiento de pintura, incluyendo una columna de 'Estado Final' para la clasificación.
* **Entrenamiento de Modelos de Machine Learning (`ml_paint_models.py`):**
    * Se implementó el entrenamiento de múltiples modelos de regresión (RandomForestRegressor) para cada propiedad de la pintura (Resistencia al Fregado, Viscosidad, etc.) y un clasificador (RandomForestClassifier) para el 'Estado Final'.
    * Se utilizó `OneHotEncoder` y `StandardScaler` para el preprocesamiento de datos a través de un `ColumnTransformer`.
    * Se aplicó **SMOTE** para abordar el desbalance de clases en la variable objetivo 'Estado Final' del clasificador.
    * Los modelos entrenados y el preprocesador se guardaron como archivos `.pkl` en la carpeta `trained_models/`.
* **Desarrollo de la Aplicación Interactiva (`app.py`):**
    * Se construyó una interfaz de usuario interactiva utilizando Streamlit, permitiendo a los usuarios ingresar los componentes de la formulación y visualizar las predicciones.
    * La aplicación se diseñó para cargar los modelos y preprocesadores guardados.

---

### 🚧 **Desafíos Principales y Soluciones Implementadas**

El mayor desafío encontrado fue la gestión de los modelos `.pkl` grandes, lo que impidió un despliegue directo en GitHub y, por ende, en Streamlit Cloud.

* **Problema:** Los archivos de modelo (`.pkl`) superaban el límite de tamaño de archivo de GitHub (100 MB), generando errores `GH001: Large files detected` y bloqueando los `git push`. A pesar de intentar soluciones estándar de Git como `git rm --cached` y configurar `.gitignore`, el problema persistió debido a que los archivos ya estaban incrustados en el historial de commits.
* **Solución Adoptada: Hosting de Modelos en Hugging Face Hub:**
    * **Decisión:** Se optó por utilizar Hugging Face Hub como una plataforma externa para almacenar los modelos grandes.
    * **Implementación en `app.py`:** Se modificó `app.py` para que, en lugar de cargar los modelos directamente desde una carpeta local en el repositorio de código, los **descargue desde Hugging Face Hub en tiempo de ejecución** utilizando la librería `huggingface_hub`. Esto permite que el repositorio de código en GitHub sea ligero.
    * **Impacto:** Esta estrategia liberó el repositorio de GitHub de los archivos grandes, permitiendo un despliegue exitoso en Streamlit Cloud.

* **Problema Persistente: Historial de Git Contaminado:**
    * A pesar de la estrategia de Hugging Face, el historial de Git local seguía intentando subir las versiones antiguas de los archivos grandes que estaban incrustadas en los commits pasados, lo que resultaba en continuos errores de `git push`.
    * **Solución Adoptada: Reinicio Completo del Repositorio Git Local:**
        * **Decisión:** Ante la persistencia del problema y para evitar la complejidad de herramientas avanzadas de reescritura de historial (como BFG Repo-Cleaner, que requería Java), se tomó la decisión de **borrar el historial de Git local** (`.git` folder) y **reiniciar el repositorio desde cero**.
        * **Proceso:** Se hizo una copia de seguridad completa del proyecto, se eliminó la carpeta `.git`, se re-inicializó el repositorio (`git init`), se aseguraron las configuraciones de `.gitignore` para ignorar `trained_models/` desde el primer commit, y se realizó un nuevo `git add .` seguido de un `git commit` inicial y un `git push` a un nuevo repositorio vacío en GitHub.
        * **Impacto:** Esta acción garantizó un repositorio de GitHub completamente limpio y funcional, aunque implicó la pérdida del historial de commits anterior. Para mitigar esta pérdida, se decidió crear este `DEVELOPMENT_LOG.md`.

---

### 💡 **Lecciones Aprendidas Clave**

* **Gestión de Archivos Grandes en Git:** Es crucial planificar desde el inicio cómo manejar archivos grandes en Git (modelos ML, datasets, etc.). Git LFS es una opción, pero para despliegues en plataformas específicas, el alojamiento externo (como Hugging Face Hub) es a menudo más robusto y compatible.
* **Importancia del `.gitignore`:** Configurar `.gitignore` correctamente desde el principio es vital, pero no resuelve problemas con archivos ya rastreados en el historial.
* **Caché de Streamlit (`@st.cache_resource`):** Fundamental para cargar modelos eficientemente y evitar descargas o recargas innecesarias en cada ejecución de la aplicación.
* **Persistencia en la Resolución de Problemas:** Los problemas de Git pueden ser complejos, pero con un enfoque sistemático y la voluntad de probar diferentes soluciones, se pueden superar.

---

### 📝 **Resumen de Mensajes de Commit Sugeridos (Historial Perdido)**

Aunque el historial de Git se ha reiniciado, los siguientes mensajes de commit representan las intenciones y los cambios significativos que se realizaron:

* `"Initial clean commit of paint prediction app (models from Hugging Face)"` (Este será el primer commit en el nuevo historial)
* `"Integrate Hugging Face model download in app.py"` (Representa el cambio clave para descargar modelos de HF)
* `"Update .gitignore to ignore trained_models folder"` (Representa el intento inicial de ignorar la carpeta)
* `"Remove trained_models from Git tracking"` (Representa el intento de untrackear la carpeta)
* `"Fix: Adjust model loading paths for deployment and add .gitignore"` (Posible commit inicial de despliegue)
* `"Add Streamlit app logic for paint prediction"` (Desarrollo de la UI y lógica de predicción)
* `"Refactor ML model training script and save assets"` (Refactorización y guardado de modelos)
* `"Implement data generation script"` (Creación del script de datos)
* `"Initial project setup"` (Primeros pasos del proyecto)

---

### ⏭️ **Próximos Pasos y Mejoras Futuras**

* Explorar la re-entrenamiento de modelos en la nube.
* Añadir funcionalidades de optimización de formulaciones.
* Mejorar la visualización de los resultados de predicción.
* Considerar la expansión del dataset con datos más reales.

---