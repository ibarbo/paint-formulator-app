"""
Script principal para el entrenamiento y guardado de modelos de Machine Learning
para la predicción de propiedades y estado final de formulaciones de pintura.

Este script realiza los siguientes pasos:
1. Carga el dataset de formulaciones de pintura.
2. Realiza ingeniería de características para crear nuevas variables predictoras.
3. Prepara los datos utilizando un ColumnTransformer para escalado y codificación one-hot.
4. Entrena múltiples modelos de Random Forest Regressor para predecir propiedades numéricas (ej., Viscosidad, Brillo).
5. Entrena un modelo de Random Forest Classifier para predecir el estado final (Éxito/Falla),
   aplicando SMOTE para manejar el desequilibrio de clases si es necesario.
6. Evalúa el rendimiento de los modelos entrenados.
7. Guarda todos los modelos y preprocesadores entrenados en la carpeta 'trained_models/'
   para su posterior uso en la aplicación Streamlit.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix, r2_score
import os
import sys
from imblearn.over_sampling import SMOTE # Asegúrate de que imblearn esté instalado: pip install imbalanced-learn
from collections import Counter

# --- Funciones de Carga de Datos ---
def load_data(file_name='data/formulaciones_pintura_simuladas.csv'):
    """
    Carga el dataset de formulaciones de pintura desde un archivo CSV.

    Realiza verificaciones de existencia del archivo y maneja errores de carga.
    Además, asegura que las columnas categóricas clave sean del tipo correcto.

    Args:
        file_name (str): Ruta al archivo CSV que contiene los datos.

    Returns:
        pd.DataFrame or None: El DataFrame cargado si la operación fue exitosa,
                              None en caso de error.
    """
    print(f"\nIntentando cargar el archivo desde: {os.path.abspath(file_name)}\n")
    
    # Verifica si el archivo existe antes de intentar cargarlo.
    if not os.path.exists(file_name):
        print(f"ERROR: El archivo '{file_name}' NO se encontró en la ruta esperada.")
        print("Por favor, verifica que 'generate_paint_data.py' lo haya guardado correctamente y que esté en la carpeta 'data/'.")
        return None
    
    # Intenta leer el archivo CSV.
    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        print(f"ERROR: Error al leer el archivo CSV '{file_name}': {e}")
        return None

    print("\n--- Columnas en el DataFrame cargado (LISTA COMPLETA) ---")
    loaded_columns_list = df.columns.tolist()
    print(loaded_columns_list) 
    print("----------------------------------------------------------\n")

    # Nombres esperados para las columnas categóricas que deben ser convertidas.
    expected_col_tipo_pintura = 'Tipo de Pintura'
    expected_col_proveedor = 'Proveedor Pigmento Blanco'

    # Verificación y conversión de 'Tipo de Pintura'.
    print(f"DEBUG: Buscando columna '{expected_col_tipo_pintura}'")
    if expected_col_tipo_pintura in df.columns:
        df[expected_col_tipo_pintura] = df[expected_col_tipo_pintura].astype('category')
        print(f"DEBUG: Columna '{expected_col_tipo_pintura}' encontrada y convertida.")
    else:
        print(f"ADVERTENCIA: La columna '{expected_col_tipo_pintura}' no se encontró en el CSV.")

    # Verificación y conversión de 'Proveedor Pigmento Blanco'.
    # Se incluye un debug detallado para ayudar a identificar problemas de nombres de columna.
    print(f"\nDEBUG: Buscando columna '{expected_col_proveedor}'")
    found_proveedor = False
    for col_name in df.columns:
        if expected_col_proveedor == col_name:
            found_proveedor = True
            print(f"DEBUG: ¡ÉXITO! La columna '{col_name}' fue encontrada y es IDÉNTICA a la esperada.")
            print(f"DEBUG: Representación de la columna: {repr(col_name)}")
            print(f"DEBUG: Representación de la esperada: {repr(expected_col_proveedor)}")
            break
        elif expected_col_proveedor in col_name:
            print(f"DEBUG: ¡ALERTA! La columna '{col_name}' CONTIENE la subcadena esperada, pero NO es idéntica.")
            print(f"DEBUG: Representación de la columna encontrada: {repr(col_name)}")
            print(f"DEBUG: Representación de la cadena esperada: {repr(expected_col_proveedor)}")
            found_proveedor = True
            break # Si la encontramos (exacta o por subcadena), podemos salir

    if found_proveedor:
        df[expected_col_proveedor] = df[expected_col_proveedor].astype('category')
        print(f"DEBUG: Columna '{expected_col_proveedor}' convertida a categoría.")
    else:
        print(f"ERROR CRÍTICO: La columna '{expected_col_proveedor}' NO se encontró en el archivo de datos.")
        print("Por favor, asegúrate de que 'generate_paint_data.py' se haya ejecutado correctamente y haya creado esta columna.")
        print(f"DEBUG: La columna '{expected_col_proveedor}' NO fue encontrada en el DataFrame.")
        print(f"DEBUG: Representación de la cadena que se buscaba: {repr(expected_col_proveedor)}")
        return None
    
    return df

# Bloque principal de ejecución del script.
# Se ejecuta solo cuando el script es corrido directamente (no cuando es importado).
if __name__ == "__main__":
    print(f"Directorio de trabajo actual: {os.getcwd()}\n")
    
    print("🚀 Entrenamiento de Modelos de Machine Learning para Pinturas")
    print("Este script entrena los modelos de regresión y clasificación y los guarda para su uso en la aplicación Streamlit.")

    # Carga los datos; el script se detiene si la carga falla.
    df = load_data()
    if df is None:
        print("\n¡ATENCIÓN! El DataFrame no pudo ser cargado. Revisar mensajes de error anteriores.")
        sys.exit(1) # Sale del script con un código de error

    print(f"\nDatos cargados. Número de filas: {len(df)}")
    print("\nBalance inicial de clases en 'Estado Final':")
    print(df['Estado Final'].value_counts())

    # --- Ingeniería de Características ---
    # Creación de nuevas características a partir de las existentes.
    # Es VITAL que estas características se repliquen EXACTAMENTE en la aplicación Streamlit (app.py)
    # para asegurar la consistencia entre el entrenamiento y la inferencia.
    df['Relacion_TiO2_Ligante'] = df['Pigmento Blanco (TiO2)'] / df['Ligante (Resina Acrílica)']
    df['Porcentaje_Solidos_Totales_Formula'] = (
        df['Pigmento Blanco (TiO2)'] + df['Extender (CaCO3)'] +
        df['Ligante (Resina Acrílica)'] + df['Aditivo Antimicrobiano'] +
        df['Dispersante'] + df['Otros Aditivos']
    )

    # --- Preparación de Datos para Modelos ---
    # Define las características de entrada (X) que se usarán para entrenar los modelos.
    input_features = [
        'Pigmento Blanco (TiO2)', 'Extender (CaCO3)', 'Ligante (Resina Acrílica)',
        'Coalescente', 'Dispersante', 'Aditivo Antimicrobiano',
        'Otros Aditivos', 'Agua',
        'Relacion_TiO2_Ligante', 'Porcentaje_Solidos_Totales_Formula', # Características ingenierizadas
        'Tipo de Pintura', 'Proveedor Pigmento Blanco' # Características categóricas
    ]

    # Define las variables objetivo para los modelos de regresión (salidas continuas).
    target_regressors = [
        'Resistencia al Fregado (Ciclos)',
        'Viscosidad Final (KU)',
        'Poder Cubriente (m²/L)',
        'Brillo (60°)',
        'Estabilidad (meses)',
        'L', 'a', 'b'
    ]
    # Define la variable objetivo para el modelo clasificador (salida categórica: 'Éxito'/'Falla').
    target_classifier = 'Estado Final'

    # Separa las características (X) de las variables objetivo (y_reg para regresión, y_class para clasificación).
    X = df[input_features]
    y_reg = df[target_regressors] # DataFrame con múltiples columnas objetivo para regresores
    y_class = df[target_classifier] # Serie con la única columna objetivo para el clasificador

    # Define las características numéricas y categóricas para el preprocesamiento.
    # Estas listas se usan para indicar al ColumnTransformer qué transformaciones aplicar a qué columnas.
    numeric_features_for_scaling = [
        'Pigmento Blanco (TiO2)', 'Extender (CaCO3)', 'Ligante (Resina Acrílica)',
        'Coalescente', 'Dispersante', 'Aditivo Antimicrobiano',
        'Otros Aditivos', 'Agua',
        'Relacion_TiO2_Ligante', 'Porcentaje_Solidos_Totales_Formula'
    ]
    categorical_features_for_ohe = ['Tipo de Pintura', 'Proveedor Pigmento Blanco']

    # --- Configuración del ColumnTransformer ---
    # Crea un ColumnTransformer para aplicar diferentes transformaciones a diferentes tipos de columnas.
    # StandardScaler: Escala características numéricas a media 0 y varianza 1.
    # OneHotEncoder: Convierte variables categóricas en un formato numérico binario (columnas dummy).
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features_for_scaling), # Aplica escalado a columnas numéricas
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_for_ohe) # Aplica One-Hot Encoding a categóricas
        ],
        remainder='drop' # Descarta cualquier columna en X que no esté explícitamente en 'numeric_features' o 'categorical_features'
    )

    print("\nAjustando preprocesador y transformando datos...")
    X_processed = preprocessor.fit_transform(X) # Ajusta (aprende los parámetros) y transforma los datos de entrenamiento
    print("Preprocesamiento completado.")

    # Obtiene los nombres de las características resultantes después de la transformación (incluyendo las de One-Hot Encoder).
    # Esto es FUNDAMENTAL para que la aplicación Streamlit pueda crear el DataFrame de entrada con las columnas correctas
    # al pasar nuevos datos al modelo.
    final_feature_names = preprocessor.get_feature_names_out()

    # Convierte los datos preprocesados (que son un array NumPy) de nuevo a un DataFrame de Pandas,
    # asignando los nombres de columna correctos.
    X_processed_df = pd.DataFrame(X_processed, columns=final_feature_names, index=X.index)

    # --- Directorio para Guardar Modelos ---
    # Crea el directorio donde se almacenarán todos los modelos entrenados y preprocesadores.
    models_dir = 'trained_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # --- Guardado de Activos de ML ---
    # Guarda el preprocesador completo. Es esencial guardarlo para que la aplicación Streamlit
    # pueda aplicar las mismas transformaciones a los nuevos datos de entrada.
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor_full.pkl'))
    print(f"ColumnTransformer completo guardado en {os.path.join(models_dir, 'preprocessor_full.pkl')}")

    # Guarda los nombres de las características finales. Necesario para reconstruir el DataFrame
    # con las columnas correctas en la aplicación Streamlit después del preprocesamiento.
    joblib.dump(final_feature_names, os.path.join(models_dir, 'model_features.pkl'))
    print(f"Nombres de las características finales guardados en {os.path.join(models_dir, 'model_features.pkl')}")

    # Guarda las clases únicas de la variable 'Estado Final'. Esto puede ser útil si el clasificador
    # produce etiquetas codificadas numéricamente y necesitas mapearlas de nuevo a sus nombres originales
    # en la aplicación Streamlit. Si el clasificador predice directamente texto, esto es más bien informativo.
    joblib.dump(y_class.unique().tolist(), os.path.join(models_dir, 'label_encoder_estado_final.pkl'))
    print(f"Clases únicas de 'Estado Final' guardadas en {os.path.join(models_dir, 'label_encoder_estado_final.pkl')}")

    # --- Entrenamiento y Evaluación de Modelos Regresores (Múltiples Salidas) ---
    regressor_models = {} # Diccionario para almacenar los modelos regresores entrenados
    for target_col in target_regressors:
        print(f"\n--- Entrenando Modelo Regresor para '{target_col}' ---")
        # Divide los datos en conjuntos de entrenamiento y prueba para cada objetivo de regresión.
        # Se usa la misma división de X_processed_df para consistencia entre modelos.
        X_train_reg, X_test_reg, y_train_reg_col, y_test_reg_col = train_test_split(
            X_processed_df, y_reg[target_col], test_size=0.2, random_state=42
        )
        
        # Inicializa y entrena el modelo RandomForestRegressor.
        # n_estimators: número de árboles en el bosque.
        # n_jobs=-1: usa todos los núcleos de la CPU disponibles para acelerar el entrenamiento.
        regressor = RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1)
        regressor.fit(X_train_reg, y_train_reg_col)
        
        # Guarda cada regresor entrenado con un nombre de archivo descriptivo basado en el nombre de la columna objetivo.
        # Los caracteres especiales en el nombre de la columna se limpian para formar un nombre de archivo válido.
        safe_name = target_col.replace(' (', '_').replace(')', '').replace(' ', '_').replace('/', '_').replace('__', '_')
        joblib.dump(regressor, os.path.join(models_dir, f'random_forest_regressor_{safe_name}.pkl'))
        print(f"Regresor para '{target_col}' guardado en {os.path.join(models_dir, f'random_forest_regressor_{safe_name}.pkl')}")

        # Evalúa el rendimiento del regresor en el conjunto de prueba.
        y_pred_reg_col = regressor.predict(X_test_reg)
        mae_reg = mean_absolute_error(y_test_reg_col, y_pred_reg_col) # Error Absoluto Medio
        r2_reg = r2_score(y_test_reg_col, y_pred_reg_col) # Coeficiente de Determinación R2
        print(f"Modelo Regresor entrenado para '{target_col}'. MAE: {mae_reg:.2f}, R2 Score: {r2_reg:.2f}")
        regressor_models[target_col] = regressor # Almacena el modelo en un diccionario (opcional, para uso posterior en el mismo script)

    # --- Entrenamiento del Modelo Clasificador (Estado Final) ---
    print("\n--- Entrenando Modelo Clasificador (Estado Final) con SMOTE... ---")
    # Divide los datos para el clasificador, utilizando 'stratify' para mantener la proporción
    # de las clases en los conjuntos de entrenamiento y prueba.
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_processed_df, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    print("\nBalance de clases ANTES de SMOTE (conjunto de entrenamiento):")
    print(Counter(y_train_class))

    # Aplica SMOTE (Synthetic Minority Over-sampling Technique) para balancear las clases.
    # SMOTE crea nuevas muestras sintéticas para la clase minoritaria, ayudando al modelo
    # a aprender mejor de ella. Se aplica solo si hay suficientes muestras por clase.
    min_samples_per_class = min(Counter(y_train_class).values())
    if len(y_train_class.unique()) > 1 and min_samples_per_class > 1:
        # k_neighbors debe ser menor o igual al número de muestras de la clase minoritaria - 1
        smote = SMOTE(random_state=42, k_neighbors=min_samples_per_class - 1)
        X_train_class_resampled, y_train_class_resampled = smote.fit_resample(X_train_class, y_train_class)
        print(f"SMOTE aplicado con k_neighbors={min_samples_per_class - 1}")
    else:
        print("ADVERTENCIA: SMOTE no se aplicó debido a que no hay suficientes clases o ejemplos por clase para el remuestreo (se requiere al menos 2 ejemplos por clase para k_neighbors > 0).")
        X_train_class_resampled, y_train_class_resampled = X_train_class, y_train_class # Usa los datos sin remuestrear si SMOTE no es aplicable

    print("\nBalance de clases DESPUÉS de SMOTE (conjunto de entrenamiento remuestreado):")
    print(Counter(y_train_class_resampled))

    # Inicializa y entrena el modelo RandomForestClassifier.
    # class_weight='balanced': Asigna automáticamente pesos a las clases inversamente proporcionales
    # a sus frecuencias de clase. Esto es beneficioso para datasets desequilibrados, incluso con SMOTE.
    classifier = RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1, class_weight='balanced')
    classifier.fit(X_train_class_resampled, y_train_class_resampled)
    # Guarda el clasificador entrenado.
    joblib.dump(classifier, os.path.join(models_dir, 'random_forest_classifier_estado.pkl'))
    print(f"Clasificador para 'Estado Final' guardado en {os.path.join(models_dir, 'random_forest_classifier_estado.pkl')}")

    print("\nModelo Clasificador entrenado.")

    # --- Evaluación del Clasificador ---
    print("\n--- Evaluación Detallada del Clasificador: ---")
    y_pred_class = classifier.predict(X_test_class) # Realiza predicciones en el conjunto de prueba
    # Muestra un informe de clasificación detallado (precisión, recall, f1-score para cada clase).
    print(classification_report(y_test_class, y_pred_class))
    print("\nMatriz de Confusión:")
    # Muestra la matriz de confusión para entender el rendimiento por clase.
    print(pd.DataFrame(confusion_matrix(y_test_class, y_pred_class, labels=classifier.classes_), 
                        index=[f'Real {c}' for c in classifier.classes_], 
                        columns=[f'Predicho {c}' for c in classifier.classes_]))

    print("\n¡Entrenamiento de modelos completado y guardado con ÉXITO!")