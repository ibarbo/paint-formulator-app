import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler # Importar StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix, r2_score # Importar r2_score
import os
import sys
from imblearn.over_sampling import SMOTE # Asegúrate de que imblearn esté instalado: pip install imbalanced-learn
from collections import Counter

# --- Cargar Datos ---
def load_data(file_name='data/formulaciones_pintura_simuladas.csv'):
    print(f"\nIntentando cargar el archivo desde: {os.path.abspath(file_name)}\n")
    
    if not os.path.exists(file_name):
        print(f"ERROR: El archivo '{file_name}' NO se encontró en la ruta esperada.")
        print("Por favor, verifica que 'generate_paint_data.py' lo haya guardado correctamente y que esté en la carpeta 'data/'.")
        return None
    
    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        print(f"ERROR: Error al leer el archivo CSV '{file_name}': {e}")
        return None

    print("\n--- Columnas en el DataFrame cargado (LISTA COMPLETA) ---")
    loaded_columns_list = df.columns.tolist()
    print(loaded_columns_list) 
    print("----------------------------------------------------------\n")

    expected_col_tipo_pintura = 'Tipo de Pintura'
    expected_col_proveedor = 'Proveedor Pigmento Blanco'

    print(f"DEBUG: Buscando columna '{expected_col_tipo_pintura}'")
    if expected_col_tipo_pintura in df.columns:
        df[expected_col_tipo_pintura] = df[expected_col_tipo_pintura].astype('category')
        print(f"DEBUG: Columna '{expected_col_tipo_pintura}' encontrada y convertida.")
    else:
        print(f"ADVERTENCIA: La columna '{expected_col_tipo_pintura}' no se encontró en el CSV.")

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

if __name__ == "__main__":
    print(f"Directorio de trabajo actual: {os.getcwd()}\n")
    
    print("🚀 Entrenamiento de Modelos de Machine Learning para Pinturas")
    print("Este script entrena los modelos de regresión y clasificación y los guarda para su uso en la aplicación Streamlit.")

    df = load_data()
    
    if df is None:
        print("\n¡ATENCIÓN! El DataFrame no pudo ser cargado. Revisar mensajes de error anteriores.")
        sys.exit(1)

    print(f"\nDatos cargados. Número de filas: {len(df)}")
    print("\nBalance inicial de clases en 'Estado Final':")
    print(df['Estado Final'].value_counts())

    # --- Ingeniería de Características (DEBE COINCIDIR CON APP.PY) ---
    # Calcular las características ingenierizadas
    df['Relacion_TiO2_Ligante'] = df['Pigmento Blanco (TiO2)'] / df['Ligante (Resina Acrílica)']
    df['Porcentaje_Solidos_Totales_Formula'] = (
        df['Pigmento Blanco (TiO2)'] + df['Extender (CaCO3)'] +
        df['Ligante (Resina Acrílica)'] + df['Aditivo Antimicrobiano'] +
        df['Dispersante'] + df['Otros Aditivos']
    )

    # --- Preparación de Datos para Modelos ---
    # Características de entrada para los modelos
    input_features = [
        'Pigmento Blanco (TiO2)', 'Extender (CaCO3)', 'Ligante (Resina Acrílica)',
        'Coalescente', 'Dispersante', 'Aditivo Antimicrobiano',
        'Otros Aditivos', 'Agua',
        'Relacion_TiO2_Ligante', 'Porcentaje_Solidos_Totales_Formula', # Características ingenierizadas
        'Tipo de Pintura', 'Proveedor Pigmento Blanco' # Características categóricas
    ]

    # Variables objetivo para regresores
    target_regressors = [
        'Resistencia al Fregado (Ciclos)',
        'Viscosidad Final (KU)',
        'Poder Cubriente (m²/L)',
        'Brillo (60°)',
        'Estabilidad (meses)',
        'L', 'a', 'b'
    ]
    # Variable objetivo para el clasificador
    target_classifier = 'Estado Final'

    X = df[input_features] # X ahora contiene todas las características de entrada
    y_reg = df[target_regressors] # DataFrame de múltiples objetivos para regresión
    y_class = df[target_classifier] # Serie para el clasificador

    # Definir las características numéricas y categóricas para el ColumnTransformer
    # Asegúrate de que esto incluye todas las características de 'input_features'
    numeric_features_for_scaling = [
        'Pigmento Blanco (TiO2)', 'Extender (CaCO3)', 'Ligante (Resina Acrílica)',
        'Coalescente', 'Dispersante', 'Aditivo Antimicrobiano',
        'Otros Aditivos', 'Agua',
        'Relacion_TiO2_Ligante', 'Porcentaje_Solidos_Totales_Formula'
    ]
    categorical_features_for_ohe = ['Tipo de Pintura', 'Proveedor Pigmento Blanco']

    # --- Creación del ColumnTransformer ---
    # Este preprocesador se aplicará a todas las características de entrada (X)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features_for_scaling), # Escalado de características numéricas
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_for_ohe) # Codificación One-Hot para categóricas
        ],
        remainder='drop' # Descarta cualquier columna no especificada, asegura que solo las features deseadas pasen
    )

    print("\nAjustando preprocesador y transformando datos...")
    X_processed = preprocessor.fit_transform(X)
    print("Preprocesamiento completado.")

    # Obtener los nombres de las características finales después del preprocesamiento
    # Esto es crucial para recrear el DataFrame de entrada en la app de Streamlit
    final_feature_names = preprocessor.get_feature_names_out()

    # Convertir X_processed a un DataFrame para mantener los nombres de las columnas
    X_processed_df = pd.DataFrame(X_processed, columns=final_feature_names, index=X.index)

    # --- Directorio para guardar modelos ---
    models_dir = 'trained_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # --- Guardar el ColumnTransformer completo ---
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor_full.pkl'))
    print(f"ColumnTransformer completo guardado en {os.path.join(models_dir, 'preprocessor_full.pkl')}")

    # Guardar los nombres de las características finales para la app de Streamlit
    joblib.dump(final_feature_names, os.path.join(models_dir, 'model_features.pkl'))
    print(f"Nombres de las características finales guardados en {os.path.join(models_dir, 'model_features.pkl')}")

    # Guardar las clases únicas del Estado Final (si el clasificador predice etiquetas codificadas)
    # Si el RandomForestClassifier predice directamente las etiquetas de texto, esto es solo informativo
    joblib.dump(y_class.unique().tolist(), os.path.join(models_dir, 'label_encoder_estado_final.pkl'))
    print(f"Clases únicas de 'Estado Final' guardadas en {os.path.join(models_dir, 'label_encoder_estado_final.pkl')}")

    # --- Entrenamiento y Evaluación de Regresores (Múltiples) ---
    regressor_models = {}
    for target_col in target_regressors:
        print(f"\n--- Entrenando Modelo Regresor para '{target_col}' ---")
        # División de datos específica para cada regresor (si y_reg es un DataFrame de múltiples columnas)
        # Usamos la misma división de X_processed_df para consistencia
        X_train_reg, X_test_reg, y_train_reg_col, y_test_reg_col = train_test_split(
            X_processed_df, y_reg[target_col], test_size=0.2, random_state=42
        )
        
        regressor = RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1)
        regressor.fit(X_train_reg, y_train_reg_col)
        
        # Guardar cada regresor con un nombre descriptivo
        # Reemplazar caracteres especiales en el nombre de la columna para el nombre del archivo
        safe_name = target_col.replace(' (', '_').replace(')', '').replace(' ', '_').replace('/', '_').replace('__', '_') # Eliminar doble underscore
        joblib.dump(regressor, os.path.join(models_dir, f'random_forest_regressor_{safe_name}.pkl'))
        print(f"Regresor '{target_col}' guardado en {os.path.join(models_dir, f'random_forest_regressor_{safe_name}.pkl')}")

        # Evaluación del regresor
        y_pred_reg_col = regressor.predict(X_test_reg)
        mae_reg = mean_absolute_error(y_test_reg_col, y_pred_reg_col)
        r2_reg = r2_score(y_test_reg_col, y_pred_reg_col)
        print(f"Modelo Regresor entrenado para '{target_col}'. MAE: {mae_reg:.2f}, R2 Score: {r2_reg:.2f}")
        regressor_models[target_col] = regressor # Guardar referencia en un diccionario si se necesitara

    # --- Entrenamiento del Modelo Clasificador (Estado Final) ---
    print("\n--- Entrenando Modelo Clasificador (Estado Final) con SMOTE... ---")
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_processed_df, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    print("\nBalance de clases ANTES de SMOTE:")
    print(Counter(y_train_class))

    # Aplicar SMOTE solo si hay más de una clase y al menos 2 muestras por clase (para k_neighbors)
    # k_neighbors debe ser <= min(n_samples_per_class) - 1
    min_samples_per_class = min(Counter(y_train_class).values())
    if len(y_train_class.unique()) > 1 and min_samples_per_class > 1:
        smote = SMOTE(random_state=42, k_neighbors=min_samples_per_class - 1)
        X_train_class_resampled, y_train_class_resampled = smote.fit_resample(X_train_class, y_train_class)
        print(f"SMOTE aplicado con k_neighbors={min_samples_per_class - 1}")
    else:
        print("ADVERTENCIA: SMOTE no se aplicó debido a que no hay suficientes clases o ejemplos por clase para el remuestreo (se requiere al menos 2 ejemplos por clase para k_neighbors > 0).")
        X_train_class_resampled, y_train_class_resampled = X_train_class, y_train_class # Usar los datos sin remuestrear

    print("\nBalance de clases DESPUÉS de SMOTE:")
    print(Counter(y_train_class_resampled))

    # Usar class_weight='balanced' para dar más peso a las clases minoritarias,
    # lo cual es buena práctica junto con SMOTE o por sí solo en desequilibrios.
    classifier = RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1, class_weight='balanced')
    classifier.fit(X_train_class_resampled, y_train_class_resampled)
    joblib.dump(classifier, os.path.join(models_dir, 'random_forest_classifier_estado.pkl'))
    print(f"Clasificador 'Estado Final' guardado en {os.path.join(models_dir, 'random_forest_classifier_estado.pkl')}")

    print("\nModelo Clasificador entrenado.")

    # --- Evaluación del Clasificador ---
    print("\n--- Evaluación Detallada del Clasificador: ---")
    y_pred_class = classifier.predict(X_test_class)
    print(classification_report(y_test_class, y_pred_class))
    print("\nMatriz de Confusión:")
    print(pd.DataFrame(confusion_matrix(y_test_class, y_pred_class, labels=classifier.classes_), 
                         index=[f'Real {c}' for c in classifier.classes_], 
                         columns=[f'Predicho {c}' for c in classifier.classes_]))

    print("\n¡Entrenamiento de modelos completado y guardado con ÉXITO!")