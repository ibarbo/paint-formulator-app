import pandas as pd
import numpy as np
import os

def generate_paint_data(num_samples=10000):
    # Definición de rangos para ingredientes de formulaciones "buenas"
    ingredient_ranges = {
        'Pigmento Blanco (TiO2)': (300, 500),
        'Extender (CaCO3)': (100, 250),
        'Ligante (Resina Acrílica)': (200, 600),
        'Coalescente': (5, 20),
        'Dispersante': (0.5, 10),
        'Aditivo Antimicrobiano': (0.1, 3),
        'Otros Aditivos': (1, 15),
        'Agua': (50, 300)
    }

    # Tipos de pintura
    paint_types = ['Brillante', 'Mate', 'Satinado']
    # Proveedores de Pigmento Blanco
    suppliers = ['Proveedor A', 'Proveedor B', 'Proveedor C']

    data = {}
    for ingredient, (low, high) in ingredient_ranges.items():
        data[ingredient] = np.random.uniform(low, high, num_samples)

    data['Tipo de Pintura'] = np.random.choice(paint_types, num_samples)
    data['Proveedor Pigmento Blanco'] = np.random.choice(suppliers, num_samples)
    data['Estado Final'] = 'Éxito' # Inicialmente todos son éxito

    df = pd.DataFrame(data)

    def calculate_performance_properties(row):
        # Base values derived from ingredients (simplified for simulation)
        # Viscosidad: Aumenta con Pigmento Blanco, Ligante, Extender, disminuye con Agua, Dispersante
        viscosity = 90 + (row['Pigmento Blanco (TiO2)'] * 0.02 + row['Ligante (Resina Acrílica)'] * 0.05 + row['Extender (CaCO3)'] * 0.03) / (row['Agua'] * 0.1 + row['Dispersante'] * 0.5)
        # Aseguramos un rango razonable para la viscosidad
        viscosity = max(80, min(viscosity, 120))
        
        # --- CAMBIO CRÍTICO AQUÍ: Resistencia al Fregado ---
        # Aumenta con Ligante, disminuye con Extender y exceso de Agua.
        # Para muestras ÉXITO, garantizamos que el valor base esté por encima del umbral de 1000.
        # La influencia de los ingredientes es fuerte para reflejar el impacto.
        scrub_resistance_base = (row['Ligante (Resina Acrílica)'] * 1.0 - row['Extender (CaCO3)'] * 0.5) # Influencia fuerte
        scrub_resistance = 1000 + scrub_resistance_base + np.random.randint(50, 400) # Base + derivado + ruido aleatorio positivo
        # Aseguramos que no baje de 1000 para las muestras "buenas" y un cap para no excederse
        scrub_resistance = max(1000, min(scrub_resistance, 2500)) 

        # Poder Cubriente: Aumenta con Pigmento Blanco
        hiding_power = 8 + row['Pigmento Blanco (TiO2)'] * 0.02 + np.random.uniform(0, 5)
        hiding_power = max(5, min(hiding_power, 30)) # Rango razonable

        # Brillo: Depende del tipo de pintura y la calidad de los aditivos
        gloss = np.random.randint(10, 80) if row['Tipo de Pintura'] == 'Brillante' else \
                (np.random.randint(5, 30) if row['Tipo de Pintura'] == 'Mate' else np.random.randint(15, 50))
        gloss = max(0, min(gloss, 90)) # Rango razonable

        # Estabilidad: Aumenta con Aditivo Antimicrobiano y Dispersante, disminuye con Otros Aditivos
        stability = 12 + (row['Aditivo Antimicrobiano'] * 2 + row['Dispersante'] * 0.5 - row['Otros Aditivos'] * 0.1 + np.random.uniform(0, 10))
        stability = max(6, min(stability, 36)) # Rango razonable

        # Color LAB (random para esta simulación, asumiendo color base neutro)
        L = np.random.uniform(85, 98)
        a = np.random.uniform(-5, 5)
        b = np.random.uniform(-5, 5)

        return pd.Series({
            'Resistencia al Fregado (Ciclos)': scrub_resistance,
            'Viscosidad Final (KU)': viscosity,
            'Poder Cubriente (m²/L)': hiding_power,
            'Brillo (60°)': gloss,
            'Estabilidad (meses)': stability,
            'L': L, 'a': a, 'b': b
        })

    # Aplicar las propiedades de rendimiento a todo el DataFrame
    df[['Resistencia al Fregado (Ciclos)', 'Viscosidad Final (KU)',
        'Poder Cubriente (m²/L)', 'Brillo (60°)', 'Estabilidad (meses)',
        'L', 'a', 'b']] = df.apply(calculate_performance_properties, axis=1)

    # --- INICIO DEL CÓDIGO AGREGADO PARA EL REDONDEO ---
    # Define la cantidad de decimales para cada variable numérica de ENTRADA
    # Esta configuración debe coincidir con el 'format' y 'step' en st.number_input() en Streamlit.
    decimal_places_for_inputs = {
        'Pigmento Blanco (TiO2)': 2,
        'Extender (CaCO3)': 2,
        'Ligante (Resina Acrílica)': 2,
        'Coalescente': 2,
        'Dispersante': 2,
        'Aditivo Antimicrobiano': 2,
        'Otros Aditivos': 2,
        'Agua': 2,
        # Si tienes otras variables de entrada que no son ingredientes, inclúyelas aquí.
        # Las variables de resultado (Viscosidad, Resistencia, etc.) se redondean más adelante si es necesario,
        # pero para las entradas del modelo, nos centramos en las partes de la formulación.
    }

    print("\nAplicando redondeo a las variables de entrada de la formulación...")
    for col, dp in decimal_places_for_inputs.items():
        if col in df.columns:
            df[col] = df[col].round(dp)
    print("Redondeo aplicado.")
    # --- FIN DEL CÓDIGO AGREGADO PARA EL REDONDEO ---


    # --- Inyección de Fallas Específicas (aproximadamente 20% del dataset total) ---
    num_failures_to_inject = int(num_samples * 0.20)
    # Distribuir las fallas entre los tipos definidos
    failure_types = [
        'Falla - Baja Adhesión',
        'Falla - Viscosidad Alta',
        'Falla - Baja Estabilidad',
        'Falla - Bajo Poder Cubriente'
    ]
    num_each_failure_type = num_failures_to_inject // len(failure_types)

    all_indices = df.index.tolist()
    np.random.shuffle(all_indices) # Mezclar índices para seleccionar aleatoriamente

    # Inyectar cada tipo de falla
    current_idx = 0
    for failure_type in failure_types:
        indices_for_this_failure = all_indices[current_idx : current_idx + num_each_failure_type]
        current_idx += num_each_failure_type

        if failure_type == 'Falla - Baja Adhesión':
            # Asegúrate de redondear también los valores inyectados para la coherencia
            df.loc[indices_for_this_failure, 'Ligante (Resina Acrílica)'] = np.random.uniform(50, 200, num_each_failure_type).round(decimal_places_for_inputs['Ligante (Resina Acrílica)'])
            df.loc[indices_for_this_failure, 'Extender (CaCO3)'] = np.random.uniform(250, 400, num_each_failure_type).round(decimal_places_for_inputs['Extender (CaCO3)'])
            df.loc[indices_for_this_failure, 'Estado Final'] = 'Falla - Baja Adhesión'
            df.loc[indices_for_this_failure, 'Resistencia al Fregado (Ciclos)'] = np.random.uniform(100, 999, num_each_failure_type)

        elif failure_type == 'Falla - Viscosidad Alta':
            df.loc[indices_for_this_failure, 'Agua'] = np.random.uniform(20, 80, num_each_failure_type).round(decimal_places_for_inputs['Agua'])
            df.loc[indices_for_this_failure, 'Pigmento Blanco (TiO2)'] = np.random.uniform(450, 600, num_each_failure_type).round(decimal_places_for_inputs['Pigmento Blanco (TiO2)'])
            df.loc[indices_for_this_failure, 'Estado Final'] = 'Falla - Viscosidad Alta'
            # Redondea también las propiedades de rendimiento para estas fallas si se usan como características
            df.loc[indices_for_this_failure, 'Viscosidad Final (KU)'] = np.random.uniform(105.1, 120, num_each_failure_type).round(1) # Asumo 1 decimal para Viscosidad Final

        elif failure_type == 'Falla - Baja Estabilidad':
            df.loc[indices_for_this_failure, 'Aditivo Antimicrobiano'] = np.random.uniform(0.01, 0.5, num_each_failure_type).round(decimal_places_for_inputs['Aditivo Antimicrobiano'])
            df.loc[indices_for_this_failure, 'Otros Aditivos'] = np.random.uniform(10, 25, num_each_failure_type).round(decimal_places_for_inputs['Otros Aditivos'])
            df.loc[indices_for_this_failure, 'Estado Final'] = 'Falla - Baja Estabilidad'
            df.loc[indices_for_this_failure, 'Estabilidad (meses)'] = np.random.uniform(1, 11.9, num_each_failure_type).round(1) # Asumo 1 decimal para Estabilidad

        elif failure_type == 'Falla - Bajo Poder Cubriente':
            df.loc[indices_for_this_failure, 'Pigmento Blanco (TiO2)'] = np.random.uniform(100, 250, num_each_failure_type).round(decimal_places_for_inputs['Pigmento Blanco (TiO2)'])
            df.loc[indices_for_this_failure, 'Extender (CaCO3)'] = np.random.uniform(200, 400, num_each_failure_type).round(decimal_places_for_inputs['Extender (CaCO3)'])
            df.loc[indices_for_this_failure, 'Estado Final'] = 'Falla - Bajo Poder Cubriente'
            df.loc[indices_for_this_failure, 'Poder Cubriente (m²/L)'] = np.random.uniform(1, 7.9, num_each_failure_type).round(2) # Asumo 2 decimales para Poder Cubriente


    # --- Aplicar reglas de umbral finales para asegurar consistencia ---
    # Estas reglas pueden reclasificar algunas muestras "Éxito" si sus propiedades
    # generadas aleatoriamente caen fuera de los umbrales de calidad, o reforzar fallas inyectadas.
    # Aseguramos que la columna sea numérica para las comparaciones
    # Aunque ya deberian ser numericas, una conversion explicita es buena practica.
    df['Viscosidad Final (KU)'] = pd.to_numeric(df['Viscosidad Final (KU)'], errors='coerce')
    df['Resistencia al Fregado (Ciclos)'] = pd.to_numeric(df['Resistencia al Fregado (Ciclos)'], errors='coerce')
    df['Estabilidad (meses)'] = pd.to_numeric(df['Estabilidad (meses)'], errors='coerce')
    df['Poder Cubriente (m²/L)'] = pd.to_numeric(df['Poder Cubriente (m²/L)'], errors='coerce')

    # Aplicar umbrales para sobrescribir 'Estado Final' si es necesario
    df.loc[df['Viscosidad Final (KU)'] > 105, 'Estado Final'] = 'Falla - Viscosidad Alta'
    df.loc[df['Resistencia al Fregado (Ciclos)'] < 1000, 'Estado Final'] = 'Falla - Baja Adhesión'
    df.loc[df['Estabilidad (meses)'] < 12, 'Estado Final'] = 'Falla - Baja Estabilidad'
    df.loc[df['Poder Cubriente (m²/L)'] < 8, 'Estado Final'] = 'Falla - Bajo Poder Cubriente'

    # Opcional: Redondear las columnas de RESULTADO final, si también se usan como features o para visualización
    # Esto es independiente del redondeo de las ENTRADAS
    decimal_places_for_results = {
        'Resistencia al Fregado (Ciclos)': 0,
        'Viscosidad Final (KU)': 1,
        'Poder Cubriente (m²/L)': 2,
        'Brillo (60°)': 0,
        'Estabilidad (meses)': 1,
        'L': 2, 'a': 2, 'b': 2
    }
    for col, dp in decimal_places_for_results.items():
        if col in df.columns:
            df[col] = df[col].round(dp)


    # Exportar a CSV
    file_path = 'data/formulaciones_pintura_simuladas.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Dataset de {num_samples} formulaciones generado y guardado en '{file_path}'")
    
    # Imprimir la distribución para verificar el balance
    print("\nDistribución de 'Estado Final' en el dataset generado:")
    print(df['Estado Final'].value_counts())

    return df

if __name__ == '__main__':
    # Generar un dataset de 10,000 muestras para el entrenamiento inicial
    generate_paint_data(num_samples=10000)