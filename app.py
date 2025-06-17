import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from huggingface_hub import hf_hub_download # Importa la función para descargar archivos de Hugging Face Hub
import requests # Mantenida por si acaso se requiriera para futuras funcionalidades de descarga robusta, aunque hf_hub_download es la principal aquí.

# --- Configuración del Repositorio de Hugging Face Hub ---
HF_REPO_ID = "ibarbo/paint-prediction-models" 

# Ajusta el path del sistema para asegurar que los módulos personalizados
# (como 'ml_paint_models.py' si estuvieran en una carpeta superior) sean accesibles.
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# --- Configuración General de la Página de Streamlit ---
st.set_page_config(
    page_title="Sistema de Predicción de Calidad de Pinturas", # Título visible en la pestaña del navegador
    page_icon="🎨", # Icono de la pestaña del navegador
    layout="wide", # Usa un diseño ancho para la aplicación
    initial_sidebar_state="expanded" # Expande la barra lateral por defecto
)

# --- Carga de Modelos y Preprocesadores de Machine Learning ---
@st.cache_resource # Decorador de Streamlit para cachear objetos pesados como modelos ML y preprocesadores.
def load_ml_assets():
    """
    Carga todos los modelos de Machine Learning y preprocesadores necesarios para la aplicación.
    
    Los archivos son descargados desde un repositorio de Hugging Face Hub a una carpeta local
    ('trained_models') si no existen previamente, y luego cargados en memoria.
    Esta función se ejecuta una sola vez gracias a `@st.cache_resource`, optimizando el rendimiento.

    Returns:
        tuple: Contiene todos los objetos ML cargados y las configuraciones de los inputs.
            - preprocessor_full (ColumnTransformer): Objeto para preprocesar las características de entrada.
            - model_features (list): Lista de los nombres de las características transformadas.
            - classifier_estado (RandomForestClassifier): Modelo para predecir el estado final (Éxito/Falla).
            - input_ranges (dict): Diccionario con los rangos (min, max, default, step, format) para los inputs numéricos.
            - paint_types_options (tuple): Opciones de tipo de pintura para el selectbox.
            - suppliers_options (tuple): Opciones de proveedor para el selectbox.
            - regressor_fregado (RandomForestRegressor): Modelo para Resistencia al Fregado.
            - regressor_viscosidad (RandomForestRegressor): Modelo para Viscosidad Final.
            - regressor_cubriente (RandomForestRegressor): Modelo para Poder Cubriente.
            - regressor_brillo (RandomForestRegressor): Modelo para Brillo.
            - regressor_estabilidad (RandomForestRegressor): Modelo para Estabilidad.
            - regressor_l (RandomForestRegressor): Modelo para el componente de color L.
            - regressor_a (RandomForestRegressor): Modelo para el componente de color a.
            - regressor_b (RandomForestRegressor): Modelo para el componente de color b.
    
    Raises:
        st.stop: Detiene la ejecución de la aplicación si los modelos no pueden ser
                 descargados o cargados, indicando un error crítico.
    """
    models_dir = 'trained_models' # Define la carpeta local para almacenar temporalmente los modelos
    os.makedirs(models_dir, exist_ok=True) # Crea la carpeta si no existe

    # Lista de los nombres de archivo PKL a descargar desde Hugging Face Hub.
    # Es crucial que estos nombres coincidan EXACTAMENTE con los archivos subidos al repositorio HF.
    model_files = [
        'preprocessor_full.pkl',
        'model_features.pkl',
        'random_forest_classifier_estado.pkl',
        'random_forest_regressor_Resistencia_al_Fregado_Ciclos.pkl',
        'random_forest_regressor_Viscosidad_Final_KU.pkl',
        'random_forest_regressor_Poder_Cubriente_m2_L.pkl',
        'random_forest_regressor_Brillo_60.pkl',
        'random_forest_regressor_Estabilidad_meses.pkl',
        'random_forest_regressor_L.pkl',
        'random_forest_regressor_a.pkl',
        'random_forest_regressor_b.pkl',
        # Si se utiliza un 'label_encoder_estado_final.pkl' para decodificar etiquetas,
        # descomentar o añadir aquí su nombre:
        # 'label_encoder_estado_final.pkl',
    ]

    try:
        # Itera sobre la lista de archivos para descargarlos si no están ya presentes localmente
        for filename in model_files:
            local_filepath = os.path.join(models_dir, filename)
            if not os.path.exists(local_filepath):
                with st.spinner(f"Descargando {filename} desde Hugging Face Hub..."):
                    try:
                        hf_hub_download(
                            repo_id=HF_REPO_ID,
                            filename=filename,
                            local_dir=models_dir,
                            local_dir_use_symlinks=False # Asegura que se descargue el archivo real, no un enlace simbólico
                        )
                        print(f"✔️ Archivo '{filename}' descargado exitosamente.")
                    except Exception as e:
                        st.error(f"❌ Error crítico: No se pudo descargar '{filename}' desde Hugging Face Hub. "
                                 f"Por favor, verifica que el 'HF_REPO_ID' sea correcto ({HF_REPO_ID}), "
                                 f"que el nombre del archivo en HF sea '{filename}', "
                                 f"y que el repositorio sea público. Detalle: {e}")
                        st.stop() # Detiene la aplicación si la descarga falla
            # Si el archivo ya existe, no se hace nada, se cargará desde el disco.

        # Una vez que todos los archivos están en 'models_dir' (ya sea por descarga o porque ya existían), se procede a cargarlos en memoria.
        preprocessor_full = joblib.load(os.path.join(models_dir, 'preprocessor_full.pkl'))
        
        # Cargar los nombres de las características generadas por el ColumnTransformer (preprocesador).
        model_features = joblib.load(os.path.join(models_dir, 'model_features.pkl'))
        
        # Cargar el clasificador entrenado para predecir el estado de la pintura.
        classifier_estado = joblib.load(os.path.join(models_dir, 'random_forest_classifier_estado.pkl'))
        
        # Cargar todos los modelos de regresión para las diferentes propiedades de la pintura.
        regressor_fregado = joblib.load(os.path.join(models_dir, 'random_forest_regressor_Resistencia_al_Fregado_Ciclos.pkl'))
        regressor_viscosidad = joblib.load(os.path.join(models_dir, 'random_forest_regressor_Viscosidad_Final_KU.pkl'))
        regressor_cubriente = joblib.load(os.path.join(models_dir, 'random_forest_regressor_Poder_Cubriente_m2_L.pkl'))
        regressor_brillo = joblib.load(os.path.join(models_dir, 'random_forest_regressor_Brillo_60.pkl'))
        regressor_estabilidad = joblib.load(os.path.join(models_dir, 'random_forest_regressor_Estabilidad_meses.pkl'))
        regressor_l = joblib.load(os.path.join(models_dir, 'random_forest_regressor_L.pkl'))
        regressor_a = joblib.load(os.path.join(models_dir, 'random_forest_regressor_a.pkl'))
        regressor_b = joblib.load(os.path.join(models_dir, 'random_forest_regressor_b.pkl'))

        # Define los rangos mínimos, máximos, valores por defecto, pasos y formato para cada input numérico
        # Esto asegura que los controles deslizantes o campos de entrada en la UI estén correctamente configurados.
        input_ranges = {
            'Pigmento Blanco (TiO2)': {'min': 200.00, 'max': 600.00, 'default': 400.00, 'step': 0.01, 'format': "%.2f"},
            'Extender (CaCO3)': {'min': 50.00, 'max': 400.00, 'default': 200.00, 'step': 0.01, 'format': "%.2f"},
            'Ligante (Resina Acrílica)': {'min': 200.00, 'max': 700.00, 'default': 450.00, 'step': 0.01, 'format': "%.2f"},
            'Coalescente': {'min': 0.00, 'max': 30.00, 'default': 10.00, 'step': 0.01, 'format': "%.2f"},
            'Dispersante': {'min': 0.00, 'max': 15.00, 'default': 5.00, 'step': 0.01, 'format': "%.2f"},
            'Aditivo Antimicrobiano': {'min': 0.00, 'max': 5.00, 'default': 1.50, 'step': 0.01, 'format': "%.2f"},
            'Otros Aditivos': {'min': 0.00, 'max': 25.00, 'default': 7.50, 'step': 0.01, 'format': "%.2f"},
            'Agua': {'min': 20.00, 'max': 350.00, 'default': 150.00, 'step': 0.01, 'format': "%.2f"}
        }

        # Define las opciones para los campos selectbox, asegurando consistencia con los datos de entrenamiento.
        paint_types_options = ('Mate', 'Satinado', 'Brillante')
        suppliers_options = ('Proveedor A', 'Proveedor B', 'Proveedor C', 'Proveedor D') 

        return (preprocessor_full, model_features, classifier_estado, input_ranges, 
                paint_types_options, suppliers_options, regressor_fregado, 
                regressor_viscosidad, regressor_cubriente, regressor_brillo, 
                regressor_estabilidad, regressor_l, regressor_a, regressor_b)
    
    except FileNotFoundError as e:
        # Manejo de errores si un archivo modelo no se encuentra localmente después de intentar la descarga.
        st.error(f"Error al cargar los modelos o preprocesadores localmente. Archivo no encontrado: {e.filename}")
        st.error("Asegúrate de que los archivos se hayan descargado correctamente o que los nombres en la lista 'model_files' sean correctos.")
        st.stop() # Detiene la ejecución de la aplicación si los modelos críticos no están disponibles
    except Exception as e:
        # Manejo de cualquier otro error inesperado durante la carga de los activos.
        st.error(f"Ocurrió un error inesperado al cargar los activos del modelo: {e}")
        st.stop()

# Llama a la función para cargar todos los activos de Machine Learning al inicio de la aplicación.
# Los resultados se cachean automáticamente.
(preprocessor_full, model_features, classifier_estado, input_ranges, 
 paint_types_options, suppliers_options, regressor_fregado, 
 regressor_viscosidad, regressor_cubriente, regressor_brillo, 
 regressor_estabilidad, regressor_l, regressor_a, regressor_b) = load_ml_assets()

# --- Título y Descripción Principal de la Aplicación ---
st.title("🎨 Sistema de Predicción de Calidad de Pinturas") 
st.markdown("Bienvenido. Aquí puedes ingresar los componentes de una formulación de pintura y predecir sus propiedades clave y su estado final.")

# --- Expander de Información "Qué puedo/no puedo hacer" ---
# Proporciona una guía clara al usuario sobre las capacidades de la herramienta.
with st.expander("🤔 ¿Qué puedo/no puedo hacer aquí?"):
    st.markdown("""
    Esta herramienta te permite:
    * **Predecir** la Resistencia al Fregado, Viscosidad Final, Poder Cubriente, Brillo, Estabilidad y el Estado Final (Éxito/Falla) de una nueva formulación de pintura.
    * **Estimar** los valores de color L, a, y b de la formulación.
    * **Ajustar los componentes** de la formulación usando los campos de entrada numérica.
    * Ver los **resultados en tiempo real** al hacer clic en el botón de predicción.

    **Lo que NO puedo hacer:**
    * Optimizar automáticamente la formulación (por ahora, es una herramienta de predicción, no de optimización).
    * Generar nuevas formulaciones aleatorias (debes ingresar los valores manualmente).
    * Proporcionar datos de formulaciones no simuladas (la base de datos es de ejemplos simulados).
    """)

st.markdown("---") # Separador visual en la UI

# --- Sección de Interfaz de Usuario para Ingreso de Datos ---
st.header("🧪 Componentes de la Formulación")
st.markdown("Ajusta los valores de cada componente en **unidades de 'partes' o gramos por lote/litro**, tal como se usaron para entrenar el modelo.")
st.markdown("Los valores de entrada se **redondean automáticamente a dos decimales** para su procesamiento.")

# Divide los inputs en dos columnas para una mejor organización visual de la interfaz.
col1, col2 = st.columns(2)

with col1:
    tipo_pintura_input = st.selectbox(
        "Tipo de Pintura",
        paint_types_options,
        index=0 # Establece "Mate" como la opción predeterminada
    )
    pigmento_blanco_tio2_input = st.number_input(
        "Pigmento Blanco (TiO2)", 
        min_value=input_ranges['Pigmento Blanco (TiO2)']['min'], 
        max_value=input_ranges['Pigmento Blanco (TiO2)']['max'], 
        value=input_ranges['Pigmento Blanco (TiO2)']['default'], 
        step=input_ranges['Pigmento Blanco (TiO2)']['step'], 
        format=input_ranges['Pigmento Blanco (TiO2)']['format']
    )
    extender_caco3_input = st.number_input(
        "Extender (CaCO3)", 
        min_value=input_ranges['Extender (CaCO3)']['min'], 
        max_value=input_ranges['Extender (CaCO3)']['max'], 
        value=input_ranges['Extender (CaCO3)']['default'], 
        step=input_ranges['Extender (CaCO3)']['step'], 
        format=input_ranges['Extender (CaCO3)']['format']
    )
    ligante_resina_acrilica_input = st.number_input(
        "Ligante (Resina Acrílica)", 
        min_value=input_ranges['Ligante (Resina Acrílica)']['min'], 
        max_value=input_ranges['Ligante (Resina Acrílica)']['max'], 
        value=input_ranges['Ligante (Resina Acrílica)']['default'], 
        step=input_ranges['Ligante (Resina Acrílica)']['step'], 
        format=input_ranges['Ligante (Resina Acrílica)']['format']
    )
    coalescente_input = st.number_input(
        "Coalescente", 
        min_value=input_ranges['Coalescente']['min'], 
        max_value=input_ranges['Coalescente']['max'], 
        value=input_ranges['Coalescente']['default'], 
        step=input_ranges['Coalescente']['step'], 
        format=input_ranges['Coalescente']['format']
    )

with col2:
    dispersante_input = st.number_input(
        "Dispersante", 
        min_value=input_ranges['Dispersante']['min'], 
        max_value=input_ranges['Dispersante']['max'], 
        value=input_ranges['Dispersante']['default'], 
        step=input_ranges['Dispersante']['step'], 
        format=input_ranges['Dispersante']['format']
    )
    aditivo_antimicrobiano_input = st.number_input(
        "Aditivo Antimicrobiano", 
        min_value=input_ranges['Aditivo Antimicrobiano']['min'], 
        max_value=input_ranges['Aditivo Antimicrobiano']['max'], 
        value=input_ranges['Aditivo Antimicrobiano']['default'], 
        step=input_ranges['Aditivo Antimicrobiano']['step'], 
        format=input_ranges['Aditivo Antimicrobiano']['format']
    )
    otros_aditivos_input = st.number_input(
        "Otros Aditivos", 
        min_value=input_ranges['Otros Aditivos']['min'], 
        max_value=input_ranges['Otros Aditivos']['max'], 
        value=input_ranges['Otros Aditivos']['default'], 
        step=input_ranges['Otros Aditivos']['step'], 
        format=input_ranges['Otros Aditivos']['format']
    )
    agua_input = st.number_input(
        "Agua", 
        min_value=input_ranges['Agua']['min'], 
        max_value=input_ranges['Agua']['max'], 
        value=input_ranges['Agua']['default'], 
        step=input_ranges['Agua']['step'], 
        format=input_ranges['Agua']['format']
    )
    proveedor_pigmento_blanco_input = st.selectbox(
        "Proveedor Pigmento Blanco",
        suppliers_options,
        index=0 # Establece "Proveedor A" como la opción predeterminada
    )

st.markdown("---") # Separador visual

# --- Botón de Predicción ---
# Este bloque de código se ejecuta solo cuando el usuario hace clic en el botón.
if st.button("🚀 Predecir Propiedades y Estado", use_container_width=True, type="primary"):
    # Crea un DataFrame de Pandas con los valores ingresados por el usuario.
    # ES CRÍTICO que los nombres de las columnas y su ORDEN coincidan EXACTAMENTE
    # con los nombres y el orden de las características de entrada utilizadas
    # durante el entrenamiento del modelo y la configuración del ColumnTransformer.
    input_data = pd.DataFrame({
        'Pigmento Blanco (TiO2)': [pigmento_blanco_tio2_input],
        'Extender (CaCO3)': [extender_caco3_input],
        'Ligante (Resina Acrílica)': [ligante_resina_acrilica_input],
        'Coalescente': [coalescente_input],
        'Dispersante': [dispersante_input],
        'Aditivo Antimicrobiano': [aditivo_antimicrobiano_input],
        'Otros Aditivos': [otros_aditivos_input],
        'Agua': [agua_input],
        'Tipo de Pintura': [tipo_pintura_input],
        'Proveedor Pigmento Blanco': [proveedor_pigmento_blanco_input]
    })

    # --- Ingeniería de Características Adicionales ---
    # Crea características derivadas (features engineering) que fueron importantes para el modelo.
    # Estas operaciones DEBEN COINCIDIR EXACTAMENTE con las realizadas en el script de entrenamiento
    # (ej. 'ml_paint_models.py').
    input_data['Relacion_TiO2_Ligante'] = input_data['Pigmento Blanco (TiO2)'] / input_data['Ligante (Resina Acrílica)']
    input_data['Porcentaje_Solidos_Totales_Formula'] = (
        input_data['Pigmento Blanco (TiO2)'] + input_data['Extender (CaCO3)'] +
        input_data['Ligante (Resina Acrílica)'] + input_data['Aditivo Antimicrobiano'] +
        input_data['Dispersante'] + input_data['Otros Aditivos']
    )
    
    # --- Preprocesamiento de los Datos de Entrada ---
    with st.spinner("Preprocesando datos..."):
        # Aplica el preprocesador completo (que incluye escalado, codificación, etc.)
        # a los datos de entrada del usuario.
        input_processed = preprocessor_full.transform(input_data)
        # Convierte el array NumPy resultante del preprocesamiento de nuevo a un DataFrame
        # usando los nombres de características finales que el modelo espera.
        input_for_prediction = pd.DataFrame(input_processed, columns=model_features)

    # --- Realizar Predicciones con los Modelos Entrenados ---
    with st.spinner("Calculando predicciones..."):
        # Realiza predicciones utilizando cada modelo de regresión cargado.
        # [0] se usa para extraer el valor escalar de la predicción (ya que .predict() retorna un array).
        pred_fregado = regressor_fregado.predict(input_for_prediction)[0]
        pred_viscosidad = regressor_viscosidad.predict(input_for_prediction)[0]
        pred_cubriente = regressor_cubriente.predict(input_for_prediction)[0]
        pred_brillo = regressor_brillo.predict(input_for_prediction)[0]
        pred_estabilidad = regressor_estabilidad.predict(input_for_prediction)[0]
        pred_l = regressor_l.predict(input_for_prediction)[0]
        pred_a = regressor_a.predict(input_for_prediction)[0]
        pred_b = regressor_b.predict(input_for_prediction)[0]

        # Realiza la predicción del estado final utilizando el modelo clasificador.
        # Se asume que el clasificador predice directamente las etiquetas 'Éxito' o 'Falla'
        # Si el clasificador predijera valores numéricos (ej. 0 o 1), se necesitaría
        # un LabelEncoder para decodificarlos de nuevo a texto.
        pred_estado_encoded = classifier_estado.predict(input_for_prediction)[0]
        predicted_estado_final = pred_estado_encoded 

    # --- Sección de Visualización de Resultados ---
    st.subheader("📊 Resultados de la Predicción")
    st.markdown("---")

    # Muestra los resultados de las predicciones en un formato de tres columnas para mayor claridad.
    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        st.metric(label="Resistencia al Fregado", value=f"{pred_fregado:.0f} ciclos", help="Número de ciclos de fregado antes de que la película de pintura se rompa.")
        st.metric(label="Viscosidad Final", value=f"{pred_viscosidad:.1f} KU", help="Viscosidad de la pintura en unidades Krebs (KU), que indica su fluidez.")
    
    with col_res2:
        st.metric(label="Poder Cubriente", value=f"{pred_cubriente:.2f} m²/L", help="Área que puede cubrir un litro de pintura con una sola capa.")
        st.metric(label="Brillo (60°)", value=f"{pred_brillo:.0f}", help="Medida del brillo de la superficie de la pintura a un ángulo de 60 grados.")
        
    with col_res3:
        st.metric(label="Estabilidad", value=f"{pred_estabilidad:.0f} meses", help="Tiempo estimado en meses que la pintura mantiene sus propiedades óptimas antes de degradarse.")
        st.markdown("##### Color (Lab):") # Encabezado para los valores de color Lab
        st.markdown(f"**L:** `{pred_l:.2f}` (Luminosidad)") # Componente L: de negro (0) a blanco (100)
        st.markdown(f"**a:** `{pred_a:.2f}` (Eje rojo-verde)") # Componente a: de verde (-) a rojo (+)
        st.markdown(f"**b:** `{pred_b:.2f}` (Eje amarillo-azul)") # Componente b: de azul (-) a amarillo (+)
    
    st.markdown("---") # Separador visual

    # Muestra el resultado de la clasificación del estado final con retroalimentación visual.
    if predicted_estado_final == 'Éxito':
        st.success(f"**Estado Final de la Pintura:** `{predicted_estado_final}` 🎉")
        st.balloons() # Pequeña animación de celebración para un resultado exitoso
    else:
        st.error(f"**Estado Final de la Pintura:** `{predicted_estado_final}` ❗")
        st.warning("Esta formulación se clasifica como 'Falla'. Se recomienda revisar los componentes de la formulación.")

    # --- Sección de Acciones Rápidas (Copiar a Portapapeles) ---
    st.markdown("---") # Separador visual
    st.markdown("### Acciones Rápidas")
    # Formatea los inputs actuales del usuario en un bloque de código para facilitar su copia y reutilización.
    copy_input_code = f"""
Tipo de Pintura: '{tipo_pintura_input}'
Pigmento Blanco (TiO2): {pigmento_blanco_tio2_input:.2f}
Extender (CaCO3): {extender_caco3_input:.2f}
Ligante (Resina Acrílica): {ligante_resina_acrilica_input:.2f}
Coalescente: {coalescente_input:.2f}
Dispersante: {dispersante_input:.2f}
Aditivo Antimicrobiano: {aditivo_antimicrobiano_input:.2f}
Otros Aditivos: {otros_aditivos_input:.2f}
Agua: {agua_input:.2f}
Proveedor Pigmento Blanco: '{proveedor_pigmento_blanco_input}'
    """
    st.code(copy_input_code, language='text')
    st.info("Puedes copiar la formulación actual desde el cuadro de arriba para reutilizarla fácilmente.")

# Pie de página o información adicional del desarrollador.
st.markdown("---")
st.markdown("Desarrollado por Víctor 2025")