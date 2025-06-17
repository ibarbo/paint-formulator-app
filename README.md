# Sistema de Análisis y Predicción de Formulaciones de Pintura

## Descripción General

Este proyecto implementa un sistema integral para el análisis, modelado predictivo y validación de formulaciones de pintura. Utilizando **ciencia de datos y aprendizaje automático (Machine Learning)**, nuestro objetivo principal es desentrañar las complejas relaciones entre los ingredientes de una pintura y sus propiedades de desempeño finales. Buscamos predecir el resultado de nuevas formulaciones, identificar patrones que conducen a éxitos o fallas, y proporcionar una herramienta que optimice el proceso de I+D en la industria de recubrimientos.

El sistema permite a los formuladores entender el impacto de sus decisiones en las características clave de la pintura, facilitando la experimentación, reduciendo los tiempos de desarrollo y mejorando la calidad del producto final.

## Características Principales

* **Generación de Datos Simulados:** Creación de un dataset sintético pero representativo de formulaciones de pintura, cubriendo un amplio rango de composiciones y resultados.
* **Análisis Exploratorio de Datos (EDA) Profundo:** Identificación de tendencias, correlaciones clave entre ingredientes y propiedades, y el impacto de variables categóricas.
* **Modelado Predictivo Robusto:** Implementación de modelos de Machine Learning (Random Forest) para predecir múltiples propiedades continuas y clasificar el estado final de las formulaciones.
* **Validación de Modelos:** Evaluación de la precisión de las predicciones y la capacidad de clasificación a través de métricas estándar de la industria.
* **Análisis de Importancia de Características:** Identificación de los ingredientes más influyentes en el rendimiento de la pintura, crucial para la optimización de formulaciones.
* **Aplicación Web Interactiva (MVP):** Una interfaz amigable basada en Streamlit que permite a los usuarios introducir nuevas formulaciones y obtener predicciones en tiempo real.
* **Actualización y Reentrenamiento Continúo:** Los modelos están diseñados para ser fácilmente reentrenados y actualizados con nuevos datos o mejoras.

## Componentes Clave del Proyecto

### 1. **Generación de Datos Simulados `generate_paint_data.py`**

Este script es la base de nuestro análisis, ya que crea un conjunto de datos sintéticos (`formulaciones_pintura_simuladas.csv`) que simula diversas formulaciones de pintura. Incluye proporciones de ingredientes (Pigmento Blanco (TiO2), Extender (CaCO3), Ligante, Coalescente, Dispersante, Aditivo Antimicrobiano, Otros Aditivos, Agua), tipo de pintura, proveedor, y propiedades finales clave como Viscosidad, Poder Cubriente, Brillo, Resistencia al Fregado, Estabilidad y valores de color L, a, b. Además, asigna un "Estado Final" (Éxito o diferentes tipos de Falla, como Baja Adhesión, Viscosidad Alta, etc.) basado en umbrales predefinidos.

### 2. **Análisis Exploratorio de Datos (EDA) `analyze_paint_data.py`**

A través de este script, se realiza un exhaustivo Análisis Exploratorio de Datos. Esta fase es fundamental para comprender la "historia" que los datos nos cuentan, incluyendo:

* **Distribución del `Estado Final`**: Se examina la frecuencia de formulaciones exitosas versus los diversos tipos de fallas, lo cual es vital para priorizar áreas de mejora.
* **Matriz de Correlación**: Se visualizan las interrelaciones numéricas entre todos los componentes y las propiedades finales, revelando cómo un cambio en un ingrediente puede impactar múltiples características del desempeño.
* **Relación Ingredientes-Propiedades Clave**: Se analiza el efecto específico de componentes críticos (ej., Ligante, Agua) en propiedades como la Viscosidad Final, identificando rangos óptimos y potenciales riesgos.
* **Impacto de Variables Categóricas**: Se evalúa cómo factores como el `Tipo de Pintura` influyen en el `Brillo` y la `Resistencia al Fregado`, y cómo el `Proveedor de Pigmento Blanco` afecta el `Poder Cubriente`, ofreciendo *insights* para la selección de materias primas.

El EDA proporciona una comprensión profunda del dataset, informando la ingeniería de características y la selección de modelos.

### 3. **Modelado Predictivo con Machine Learning `train_models.py`**

Este script se encarga de implementar y entrenar múltiples modelos de Machine Learning (en este caso, Random Forests) para abordar las tareas de predicción:

* **Regresión:** Se entrenan modelos individuales para predecir cada una de las propiedades continuas de la pintura: `Resistencia al Fregado (Ciclos)`, `Viscosidad Final (KU)`, `Poder Cubriente (m²/L)`, `Brillo (60°)`, `Estabilidad (meses)`, `L`, `a`, y `b`.
* **Clasificación:** Se entrena un modelo para predecir el `Estado Final` de una formulación (`Éxito` o los diversos tipos de `Falla`).

El proceso incluye el preprocesamiento de datos (escalado numérico y codificación One-Hot para categóricas) utilizando un `ColumnTransformer`, que se guarda junto con los modelos para asegurar la consistencia en la inferencia. Los modelos se reentrenan sin errores, garantizando que estén actualizados y listos para su uso.

#### **Evaluación y Salida de los Modelos:**

La efectividad de los modelos se evalúa a través de métricas clave:
* **Para Modelos de Regresión**: Se utilizan **Error Cuadrático Medio (MSE)**, **Error Absoluto Medio (MAE)** y el **Coeficiente de Determinación (R²)** para cuantificar la precisión de las predicciones numéricas.
* **Para Modelos de Clasificación**: Se emplean **Precisión (Accuracy)**, **Recall**, **F1-Score** y el **Área bajo la Curva ROC (AUC)** para evaluar la capacidad del modelo para identificar correctamente los diferentes estados de la formulación.

Un análisis de la **Importancia de las Características (Feature Importance)** acompaña cada modelo, lo que permite determinar qué ingredientes o proporciones son los más influyentes en cada propiedad o en la probabilidad de falla. Esta información es invaluable para los formuladores, ya que guía los esfuerzos de optimización.

### 4. **Aplicación Interactiva Streamlit `app.py`**

La aplicación Streamlit es la interfaz de usuario de este proyecto. Permite a cualquier usuario interactuar con los modelos entrenados de forma intuitiva:
* **Entrada de Parámetros:** Campos amigables para ingresar los componentes de una nueva formulación, con validación básica de entrada y redondeo a dos decimales.
* **Predicciones en Tiempo Real:** Al hacer clic en un botón, la aplicación predice instantáneamente todas las propiedades de regresión (Resistencia al Fregado, Viscosidad, etc., incluyendo los valores de color L, a, b) y el `Estado Final` de la formulación.
* **Feedback Visual:** Proporciona retroalimentación visual clara para el usuario, incluyendo un efecto de globos para las predicciones de "Éxito" y mensajes de advertencia para las "Fallas".
* **Copiado Rápido:** Incluye botones para copiar fácilmente los parámetros de entrada, facilitando la experimentación.
* **Información de Uso:** Un expansor "Lo que puedo/no puedo hacer aquí" guía al usuario sobre las capacidades de la aplicación.

## Estructura del Proyecto
```Markdown
.
├── data/
│   ├── formulaciones_pintura_simuladas.csv # Dataset simulado de formulaciones
│   └── (otros archivos de datos, si los hubiera)
├── env/                                #Entorno virtual (env)
├── trained_models/                     # Modelos de ML entrenados (archivos .pkl/.joblib del         preprocesador, clasificador y regresores)
├── .gitignore                          # Archivo para especificar elementos a ignorar por Git
├── generate_paint_data.py              # Script para generar el dataset simulado
├── analyze_paint_data.py               # Script para el ETL y el Análisis Exploratorio de Datos (EDA)
├── train_models.py                     # Script para entrenar y guardar los modelos de Machine Learning
├── streamlit_app.py                       # Aplicación Streamlit para interactuar con los modelos
├── requirements.txt                    # Dependencias del proyecto
├── LICENSE                             # Este proyecto está bajo la Licencia MIT
└── README.md                           # Este archivo
```
## Configuración y Ejecución Local

Para replicar y ejecutar este proyecto en tu entorno local:

### Prerrequisitos
* `Python 3.8` o superior.
* `git` instalado.

### Pasos para la Configuración:

#### 1. Clonar el Repositorio

Abre tu terminal y ejecuta:

```bash
git clone https://github.com/ibarbo/paint-formulation-optimizer
cd paint-formulation-optimizer
```
#### 2. Configurar el Entorno Virtual
Es altamente recomendable usar un entorno virtual para gestionar las dependencias del proyecto de forma aislada:
```Bash
python -m venv env
# En Windows:
.\env\Scripts\activate
# En macOS/Linux:
source env/bin/activate
```
#### 3. Instalar Dependencias
Con tu entorno virtual activado, instala todas las librerías necesarias:
```Bash
pip install -r requirements.txt
```
#### 4. Generar Datos Simulados
Si aún no tienes el dataset, o si deseas generar uno nuevo, ejecuta:
```Bash
python generate_paint_data.py
```
#### 5. Realizar el Análisis Exploratorio de Datos (EDA)
Para ejecutar el análisis exploratorio y ver los insights del dataset:
```Bash
python analyze_paint_data.py
```
#### 6. Entrenar y Guardar Modelos de Machine Learning
Este paso entrenará los modelos de regresión y clasificación, y los guardará en la carpeta `trained_models/`:
```Bash
python train_models.py
```
#### 7. Ejecutar la Aplicación Streamlit
Finalmente, para interactuar con los modelos entrenados a través de la interfaz web:
```Bash
streamlit run app.py
```
Esto abrirá la aplicación en tu navegador web por defecto.
### Licencia
Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo LICENSE. 📝