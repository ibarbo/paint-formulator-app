import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from huggingface_hub import hf_hub_download # Importa esta línea
import requests # Necesario para la función de descarga robusta si la requieres, aunque hf_hub_download es la principal aquí

# --- Configuración de Hugging Face Hub ---
HF_REPO_ID = "ibarbo/paint-prediction-models" 

# Ajustar el path para cargar los módulos personalizados
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Sistema de Predicción de Calidad de Pinturas", # Nombre de página genérico
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Carga de Modelos y Preprocesadores ---
@st.cache_resource # Usar st.cache_resource para objetos de ML (modelos, encoders)
def load_ml_assets():
    models_dir = 'trained_models' # Carpeta donde se guardarán temporalmente tus modelos
    
    # Asegúrate de que la carpeta 'trained_models' exista
    os.makedirs(models_dir, exist_ok=True)
    
    # Lista de los nombres de archivo PKL que necesitas descargar de Hugging Face Hub
    # Estos nombres deben coincidir EXACTAMENTE con los nombres que subiste a HF
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
        # Si tienes 'label_encoder_estado_final.pkl' y lo usas, descomenta o añade aquí:
        # 'label_encoder_estado_final.pkl',
    ]

    try:
        # Descargar cada archivo si no existe localmente
        for filename in model_files:
            local_filepath = os.path.join(models_dir, filename)
            if not os.path.exists(local_filepath):
                with st.spinner(f"Descargando {filename} desde Hugging Face Hub..."):
                    try:
                        hf_hub_download(
                            repo_id=HF_REPO_ID,
                            filename=filename,
                            local_dir=models_dir,
                            local_dir_use_symlinks=False # Importante para asegurar que se descargue el archivo real
                        )
                        st.success(f"✔️ {filename} descargado.")
                    except Exception as e:
                        st.error(f"❌ Error crítico: No se pudo descargar '{filename}' desde Hugging Face Hub. "
                                 f"Por favor, verifica que 'HF_REPO_ID' sea correcto ({HF_REPO_ID}), "
                                 f"que el nombre del archivo en HF sea '{filename}', "
                                 f"y que el repositorio sea público. Detalle: {e}")
                        st.stop() # Detiene la app si no se puede descargar

        # Una vez que todos los archivos están en 'models_dir' (ya sea descargados o ya existían), procede a cargarlos
        preprocessor_full = joblib.load(os.path.join(models_dir, 'preprocessor_full.pkl'))
        
        # Cargar los nombres de las características finales (ColumnTransformer output features)
        model_features = joblib.load(os.path.join(models_dir, 'model_features.pkl'))
        
        # Cargar el clasificador
        classifier_estado = joblib.load(os.path.join(models_dir, 'random_forest_classifier_estado.pkl'))
        
        # Cargar todos los regresores
        regressor_fregado = joblib.load(os.path.join(models_dir, 'random_forest_regressor_Resistencia_al_Fregado_Ciclos.pkl'))
        regressor_viscosidad = joblib.load(os.path.join(models_dir, 'random_forest_regressor_Viscosidad_Final_KU.pkl'))
        regressor_cubriente = joblib.load(os.path.join(models_dir, 'random_forest_regressor_Poder_Cubriente_m2_L.pkl'))
        regressor_brillo = joblib.load(os.path.join(models_dir, 'random_forest_regressor_Brillo_60.pkl'))
        regressor_estabilidad = joblib.load(os.path.join(models_dir, 'random_forest_regressor_Estabilidad_meses.pkl'))
        regressor_l = joblib.load(os.path.join(models_dir, 'random_forest_regressor_L.pkl'))
        regressor_a = joblib.load(os.path.join(models_dir, 'random_forest_regressor_a.pkl'))
        regressor_b = joblib.load(os.path.join(models_dir, 'random_forest_regressor_b.pkl'))

        # Diccionario con los rangos mínimos y máximos para cada input numérico
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

        # Opciones para selectbox (deben coincidir con las usadas en el entrenamiento)
        paint_types_options = ('Mate', 'Satinado', 'Brillante')
        suppliers_options = ('Proveedor A', 'Proveedor B', 'Proveedor C', 'Proveedor D') 

        return (preprocessor_full, model_features, classifier_estado, input_ranges, 
                paint_types_options, suppliers_options, regressor_fregado, 
                regressor_viscosidad, regressor_cubriente, regressor_brillo, 
                regressor_estabilidad, regressor_l, regressor_a, regressor_b)
    
    except FileNotFoundError as e:
        # Este error ahora debería ser raro si la descarga funciona
        st.error(f"Error al cargar los modelos o preprocesadores localmente. Archivo no encontrado: {e.filename}")
        st.error("Asegúrate de que los archivos se hayan descargado correctamente o que los nombres en la lista 'model_files' sean correctos.")
        st.stop() # Detiene la ejecución de la app si los modelos no están disponibles
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al cargar los activos del modelo: {e}")
        st.stop()

# Cargar todos los activos ML
(preprocessor_full, model_features, classifier_estado, input_ranges, 
 paint_types_options, suppliers_options, regressor_fregado, 
 regressor_viscosidad, regressor_cubriente, regressor_brillo, 
 regressor_estabilidad, regressor_l, regressor_a, regressor_b) = load_ml_assets()

# --- Título y Descripción de la Aplicación ---
st.title("🎨 Sistema de Predicción de Calidad de Pinturas") # Título genérico
st.markdown("Bienvenido. Aquí puedes ingresar los componentes de una formulación de pintura y predecir sus propiedades clave y su estado final.") # Descripción genérica

# --- Expander de "Lo que puedo/no puedo hacer" ---
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

st.markdown("---")

# --- Interfaz de Usuario para Ingreso de Datos ---
st.header("🧪 Componentes de la Formulación")
st.markdown("Ajusta los valores de cada componente en **unidades de 'partes' o gramos por lote/litro**, tal como se usaron para entrenar el modelo.")
st.markdown("Los valores de entrada se **redondean automáticamente a dos decimales** para su procesamiento.")

col1, col2 = st.columns(2)

with col1:
    tipo_pintura_input = st.selectbox(
        "Tipo de Pintura",
        paint_types_options,
        index=0 # Default a Mate
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
        index=0 # Default a Proveedor A
    )

st.markdown("---")

# Botón de Predicción
if st.button("🚀 Predecir Propiedades y Estado", use_container_width=True, type="primary"):
    # Crear DataFrame con los inputs del usuario (DEBE COINCIDIR CON EL ORDEN Y NOMBRES DEL ENTRENAMIENTO)
    # Las columnas deben estar en el orden en que el ColumnTransformer espera verlas
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

    # --- Ingeniería de Características (DEBE COINCIDIR CON ML_PAINT_MODELS.PY) ---
    input_data['Relacion_TiO2_Ligante'] = input_data['Pigmento Blanco (TiO2)'] / input_data['Ligante (Resina Acrílica)']
    input_data['Porcentaje_Solidos_Totales_Formula'] = (
        input_data['Pigmento Blanco (TiO2)'] + input_data['Extender (CaCO3)'] +
        input_data['Ligante (Resina Acrílica)'] + input_data['Aditivo Antimicrobiano'] +
        input_data['Dispersante'] + input_data['Otros Aditivos']
    )
    
    # --- Preprocesamiento (Aplicar el preprocesador completo) ---
    with st.spinner("Preprocesando datos..."):
        input_processed = preprocessor_full.transform(input_data)
        input_for_prediction = pd.DataFrame(input_processed, columns=model_features)

    # --- Realizar Predicciones ---
    with st.spinner("Calculando predicciones..."):
        # Predicción de propiedades de regresión
        pred_fregado = regressor_fregado.predict(input_for_prediction)[0]
        pred_viscosidad = regressor_viscosidad.predict(input_for_prediction)[0]
        pred_cubriente = regressor_cubriente.predict(input_for_prediction)[0]
        pred_brillo = regressor_brillo.predict(input_for_prediction)[0]
        pred_estabilidad = regressor_estabilidad.predict(input_for_prediction)[0]
        pred_l = regressor_l.predict(input_for_prediction)[0]
        pred_a = regressor_a.predict(input_for_prediction)[0]
        pred_b = regressor_b.predict(input_for_prediction)[0]

        # Predicción del estado final
        # Si el clasificador predice etiquetas codificadas y necesitas decodificarlas, necesitarías cargar 'label_encoder_estado_final.pkl'
        # Por ahora, asumo que 'Éxito' o 'Falla' ya son las salidas del clasificador si lo entrenaste así.
        pred_estado_encoded = classifier_estado.predict(input_for_prediction)[0]
        predicted_estado_final = pred_estado_encoded 

    st.subheader("📊 Resultados de la Predicción")
    st.markdown("---")

    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        st.metric(label="Resistencia al Fregado", value=f"{pred_fregado:.0f} ciclos", help="Número de ciclos de fregado antes de que la película de pintura se rompa.")
        st.metric(label="Viscosidad Final", value=f"{pred_viscosidad:.1f} KU", help="Viscosidad de la pintura en unidades Krebs (KU).")
    
    with col_res2:
        st.metric(label="Poder Cubriente", value=f"{pred_cubriente:.2f} m²/L", help="Área que puede cubrir un litro de pintura.")
        st.metric(label="Brillo (60°)", value=f"{pred_brillo:.0f}", help="Medida del brillo de la superficie de la pintura a 60 grados.")
        
    with col_res3:
        st.metric(label="Estabilidad", value=f"{pred_estabilidad:.0f} meses", help="Tiempo estimado en meses que la pintura mantiene sus propiedades.")
        st.markdown("##### Color (Lab):")
        st.markdown(f"**L:** `{pred_l:.2f}` (Luminosidad)")
        st.markdown(f"**a:** `{pred_a:.2f}` (Eje rojo-verde)")
        st.markdown(f"**b:** `{pred_b:.2f}` (Eje amarillo-azul)")
    
    st.markdown("---")

    # Mostrar resultado de Clasificación
    if predicted_estado_final == 'Éxito':
        st.success(f"**Estado Final de la Pintura:** `{predicted_estado_final}` 🎉")
        st.balloons() # Pequeña celebración para el éxito
    else:
        st.error(f"**Estado Final de la Pintura:** `{predicted_estado_final}` ❗")
        st.warning("Esta formulación se clasifica como 'Falla'. Se recomienda revisar los componentes.")

    # Botones de Copy-to-Clipboard
    st.markdown("---")
    st.markdown("### Acciones Rápidas")
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
    st.info("Puedes copiar la formulación actual desde el cuadro de arriba para reutilizarla.")

# Pie de página o información adicional
st.markdown("---")
st.markdown("Desarrollado por Víctor 2025")