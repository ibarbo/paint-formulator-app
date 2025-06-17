import pandas as pd
import numpy as np
import os

def generate_paint_data(num_samples=10000):
    """
    Genera un dataset sintético de formulaciones de pintura con propiedades de rendimiento
    simuladas y un porcentaje de 'fallas' inyectadas.

    Este dataset se utiliza para el entrenamiento de modelos de Machine Learning
    que predicen las propiedades de la pintura y su estado final (Éxito/Falla).

    Args:
        num_samples (int): Número total de formulaciones de pintura a generar.
                           Por defecto, 10,000 muestras.

    Returns:
        pandas.DataFrame: Un DataFrame de Pandas que contiene las formulaciones simuladas,
                          sus propiedades de rendimiento y su estado final.

    El dataset se guarda también en un archivo CSV en 'data/formulaciones_pintura_simuladas.csv'.
    """
    # Definición de rangos de valores para los ingredientes base de formulaciones consideradas "buenas".
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

    # Definición de categorías discretas para las formulaciones.
    paint_types = ['Brillante', 'Mate', 'Satinado']
    suppliers = ['Proveedor A', 'Proveedor B', 'Proveedor C']

    # Genera datos aleatorios para cada ingrediente y categoría.
    data = {}
    for ingredient, (low, high) in ingredient_ranges.items():
        data[ingredient] = np.random.uniform(low, high, num_samples)

    data['Tipo de Pintura'] = np.random.choice(paint_types, num_samples)
    data['Proveedor Pigmento Blanco'] = np.random.choice(suppliers, num_samples)
    data['Estado Final'] = 'Éxito' # Inicialmente, todas las muestras se etiquetan como 'Éxito'.

    df = pd.DataFrame(data)

    def calculate_performance_properties(row):
        """
        Calcula las propiedades de rendimiento de la pintura basándose en la formulación de una fila.
        Estas relaciones son simplificadas y simuladas para generar datos variados.
        """
        # Viscosidad (KU): Aumenta con Pigmento Blanco, Ligante, Extender; disminuye con Agua, Dispersante.
        # Se añaden límites para mantener la viscosidad en un rango realista.
        viscosity = 90 + (row['Pigmento Blanco (TiO2)'] * 0.02 + row['Ligante (Resina Acrílica)'] * 0.05 + row['Extender (CaCO3)'] * 0.03) / (row['Agua'] * 0.1 + row['Dispersante'] * 0.5)
        viscosity = max(80, min(viscosity, 120))
        
        # Resistencia al Fregado (Ciclos): Aumenta con Ligante, disminuye con Extender y exceso de Agua.
        # Para muestras inicialmente 'Éxito', se garantiza que el valor base esté por encima del umbral de 1000.
        # Se simula una fuerte influencia de ingredientes clave.
        scrub_resistance_base = (row['Ligante (Resina Acrílica)'] * 1.0 - row['Extender (CaCO3)'] * 0.5) 
        scrub_resistance = 1000 + scrub_resistance_base + np.random.randint(50, 400) # Se añade ruido positivo para 'Éxito'.
        scrub_resistance = max(1000, min(scrub_resistance, 2500)) # Se asegura que no baje de 1000 y se establece un tope.

        # Poder Cubriente (m²/L): Aumenta principalmente con el Pigmento Blanco.
        hiding_power = 8 + row['Pigmento Blanco (TiO2)'] * 0.02 + np.random.uniform(0, 5)
        hiding_power = max(5, min(hiding_power, 30)) # Se ajusta a un rango razonable.

        # Brillo (60°): Depende del tipo de pintura y se simula aleatoriamente dentro de rangos para cada tipo.
        gloss = np.random.randint(10, 80) if row['Tipo de Pintura'] == 'Brillante' else \
                (np.random.randint(5, 30) if row['Tipo de Pintura'] == 'Mate' else np.random.randint(15, 50))
        gloss = max(0, min(gloss, 90)) # Se asegura que el brillo esté en un rango lógico.

        # Estabilidad (meses): Aumenta con Aditivo Antimicrobiano y Dispersante; disminuye con Otros Aditivos.
        stability = 12 + (row['Aditivo Antimicrobiano'] * 2 + row['Dispersante'] * 0.5 - row['Otros Aditivos'] * 0.1 + np.random.uniform(0, 10))
        stability = max(6, min(stability, 36)) # Se establece un rango de estabilidad razonable.

        # Color LAB (L, a, b): Valores de color simulados como neutros o en un rango limitado.
        # Estos valores son generados de forma aleatoria para la simulación.
        L = np.random.uniform(85, 98)
        a = np.random.uniform(-5, 5)
        b = np.random.uniform(-5, 5)

        # Retorna las propiedades calculadas como una Serie de Pandas.
        return pd.Series({
            'Resistencia al Fregado (Ciclos)': scrub_resistance,
            'Viscosidad Final (KU)': viscosity,
            'Poder Cubriente (m²/L)': hiding_power,
            'Brillo (60°)': gloss,
            'Estabilidad (meses)': stability,
            'L': L, 'a': a, 'b': b
        })

    # Aplica la función `calculate_performance_properties` a cada fila del DataFrame
    # para generar las propiedades de rendimiento iniciales.
    df[['Resistencia al Fregado (Ciclos)', 'Viscosidad Final (KU)',
        'Poder Cubriente (m²/L)', 'Brillo (60°)', 'Estabilidad (meses)',
        'L', 'a', 'b']] = df.apply(calculate_performance_properties, axis=1)

    # --- Redondeo de Variables de Entrada Numéricas ---
    # Define la cantidad de decimales para cada variable de ENTRADA numérica de la formulación.
    # Es crucial que estas configuraciones de redondeo coincidan con los 'format' y 'step'
    # definidos para los widgets `st.number_input()` en la aplicación Streamlit (`app.py`).
    decimal_places_for_inputs = {
        'Pigmento Blanco (TiO2)': 2,
        'Extender (CaCO3)': 2,
        'Ligante (Resina Acrílica)': 2,
        'Coalescente': 2,
        'Dispersante': 2,
        'Aditivo Antimicrobiano': 2,
        'Otros Aditivos': 2,
        'Agua': 2,
    }

    print("\nAplicando redondeo a las variables de entrada de la formulación...")
    for col, dp in decimal_places_for_inputs.items():
        if col in df.columns:
            df[col] = df[col].round(dp)
    print("Redondeo aplicado a las variables de entrada.")

    # --- Inyección de Fallas Específicas en el Dataset ---
    # Se inyecta un porcentaje de muestras con características y propiedades que simulan 'fallas'.
    num_failures_to_inject = int(num_samples * 0.20) # Aproximadamente el 20% del dataset total.
    
    # Tipos de fallas predefinidas para inyectar en el dataset.
    failure_types = [
        'Falla - Baja Adhesión',
        'Falla - Viscosidad Alta',
        'Falla - Baja Estabilidad',
        'Falla - Bajo Poder Cubriente'
    ]
    num_each_failure_type = num_failures_to_inject // len(failure_types)

    all_indices = df.index.tolist()
    np.random.shuffle(all_indices) # Mezcla los índices para seleccionar muestras aleatorias para inyectar fallas.

    # Itera sobre cada tipo de falla e inyecta las condiciones de fallo en un subconjunto de muestras.
    current_idx = 0
    for failure_type in failure_types:
        indices_for_this_failure = all_indices[current_idx : current_idx + num_each_failure_type]
        current_idx += num_each_failure_type

        if failure_type == 'Falla - Baja Adhesión':
            # Se ajustan los ingredientes para simular baja adhesión y se etiqueta la falla.
            # Los valores inyectados también se redondean para mantener la coherencia.
            df.loc[indices_for_this_failure, 'Ligante (Resina Acrílica)'] = np.random.uniform(50, 200, num_each_failure_type).round(decimal_places_for_inputs['Ligante (Resina Acrílica)'])
            df.loc[indices_for_this_failure, 'Extender (CaCO3)'] = np.random.uniform(250, 400, num_each_failure_type).round(decimal_places_for_inputs['Extender (CaCO3)'])
            df.loc[indices_for_this_failure, 'Estado Final'] = 'Falla - Baja Adhesión'
            # Se simula una Resistencia al Fregado baja.
            df.loc[indices_for_this_failure, 'Resistencia al Fregado (Ciclos)'] = np.random.uniform(100, 999, num_each_failure_type)

        elif failure_type == 'Falla - Viscosidad Alta':
            # Se ajustan los ingredientes y propiedades para simular alta viscosidad.
            df.loc[indices_for_this_failure, 'Agua'] = np.random.uniform(20, 80, num_each_failure_type).round(decimal_places_for_inputs['Agua'])
            df.loc[indices_for_this_failure, 'Pigmento Blanco (TiO2)'] = np.random.uniform(450, 600, num_each_failure_type).round(decimal_places_for_inputs['Pigmento Blanco (TiO2)'])
            df.loc[indices_for_this_failure, 'Estado Final'] = 'Falla - Viscosidad Alta'
            # Se inyecta una Viscosidad Final alta, redondeada a 1 decimal.
            df.loc[indices_for_this_failure, 'Viscosidad Final (KU)'] = np.random.uniform(105.1, 120, num_each_failure_type).round(1) 

        elif failure_type == 'Falla - Baja Estabilidad':
            # Se ajustan los ingredientes y propiedades para simular baja estabilidad.
            df.loc[indices_for_this_failure, 'Aditivo Antimicrobiano'] = np.random.uniform(0.01, 0.5, num_each_failure_type).round(decimal_places_for_inputs['Aditivo Antimicrobiano'])
            df.loc[indices_for_this_failure, 'Otros Aditivos'] = np.random.uniform(10, 25, num_each_failure_type).round(decimal_places_for_inputs['Otros Aditivos'])
            df.loc[indices_for_this_failure, 'Estado Final'] = 'Falla - Baja Estabilidad'
            # Se inyecta una Estabilidad baja, redondeada a 1 decimal.
            df.loc[indices_for_this_failure, 'Estabilidad (meses)'] = np.random.uniform(1, 11.9, num_each_failure_type).round(1)

        elif failure_type == 'Falla - Bajo Poder Cubriente':
            # Se ajustan los ingredientes y propiedades para simular bajo poder cubriente.
            df.loc[indices_for_this_failure, 'Pigmento Blanco (TiO2)'] = np.random.uniform(100, 250, num_each_failure_type).round(decimal_places_for_inputs['Pigmento Blanco (TiO2)'])
            df.loc[indices_for_this_failure, 'Extender (CaCO3)'] = np.random.uniform(200, 400, num_each_failure_type).round(decimal_places_for_inputs['Extender (CaCO3)'])
            df.loc[indices_for_this_failure, 'Estado Final'] = 'Falla - Bajo Poder Cubriente'
            # Se inyecta un Poder Cubriente bajo, redondeado a 2 decimales.
            df.loc[indices_for_this_failure, 'Poder Cubriente (m²/L)'] = np.random.uniform(1, 7.9, num_each_failure_type).round(2)


    # --- Aplicación de Reglas de Umbral Finales para Consistencia de 'Estado Final' ---
    # Estas reglas re-evalúan el 'Estado Final' de las muestras (incluyendo las inicialmente 'Éxito')
    # basándose en umbrales de calidad definidos para las propiedades de rendimiento.
    # Esto asegura que el campo 'Estado Final' sea consistente con las propiedades generadas.
    # Se asegura que las columnas sean numéricas antes de la comparación.
    df['Viscosidad Final (KU)'] = pd.to_numeric(df['Viscosidad Final (KU)'], errors='coerce')
    df['Resistencia al Fregado (Ciclos)'] = pd.to_numeric(df['Resistencia al Fregado (Ciclos)'], errors='coerce')
    df['Estabilidad (meses)'] = pd.to_numeric(df['Estabilidad (meses)'], errors='coerce')
    df['Poder Cubriente (m²/L)'] = pd.to_numeric(df['Poder Cubriente (m²/L)'], errors='coerce')

    # Reclasificación de 'Estado Final' basándose en umbrales críticos de calidad.
    df.loc[df['Viscosidad Final (KU)'] > 105, 'Estado Final'] = 'Falla - Viscosidad Alta'
    df.loc[df['Resistencia al Fregado (Ciclos)'] < 1000, 'Estado Final'] = 'Falla - Baja Adhesión'
    df.loc[df['Estabilidad (meses)'] < 12, 'Estado Final'] = 'Falla - Baja Estabilidad'
    df.loc[df['Poder Cubriente (m²/L)'] < 8, 'Estado Final'] = 'Falla - Bajo Poder Cubriente'

    # --- Redondeo de Variables de RESULTADO Finales ---
    # Opcionalmente, se redondean las columnas de RESULTADO final para su uso en visualización
    # o si se van a usar como características en otros modelos.
    # Este redondeo es independiente del aplicado a las variables de entrada.
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

    # --- Exportar el Dataset Generado a CSV ---
    file_path = 'data/formulaciones_pintura_simuladas.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True) # Crea el directorio 'data' si no existe.
    df.to_csv(file_path, index=False) # Guarda el DataFrame en CSV, sin incluir el índice.
    print(f"Dataset de {num_samples} formulaciones generado y guardado en '{file_path}'")
    
    # Imprime la distribución de los estados finales para verificar el balance de clases generado.
    print("\nDistribución de 'Estado Final' en el dataset generado:")
    print(df['Estado Final'].value_counts())

    return df

if __name__ == '__main__':
    # Bloque principal para ejecutar la generación de datos cuando el script es ejecutado directamente.
    # Genera un dataset de 10,000 muestras para el entrenamiento inicial de los modelos.
    generate_paint_data(num_samples=10000)