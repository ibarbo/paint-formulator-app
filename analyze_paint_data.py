import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización para que los gráficos sean más agradables
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6) # Tamaño por defecto de las figuras

# --- 1. Extracción (Carga de Datos) ---
file_name = 'data/formulaciones_pintura_simuladas.csv' # RUTA RELATIVA CORRECTA
try:
    df = pd.read_csv(file_name)
    print(f"Datos cargados exitosamente desde '{file_name}'.")
    print("\nNombres de las columnas en el DataFrame:")
    print(df.columns.tolist())
except FileNotFoundError:
    print(f"Error: El archivo '{file_name}' no se encontró. Asegúrate de haber ejecutado 'generate_paint_data.py' primero.")
    exit()

# --- 2. Transformación y Carga (Preprocesamiento Básico y Carga a DataFrame) ---

# Convertir 'Estado Final' a tipo 'category' para optimizar memoria y facilitar análisis
df['Estado Final'] = df['Estado Final'].astype('category')
df['Tipo de Pintura'] = df['Tipo de Pintura'].astype('category')
df['Proveedor Pigmento Blanco'] = df['Proveedor Pigmento Blanco'].astype('category')

# Definición de ingredientes_cols - ¡Asegúrate de que esta lista esté siempre antes de donde se usa!
ingredientes_cols = [
    'Pigmento Blanco (TiO2)', 'Extender (CaCO3)', 'Ligante (Resina Acrílica)',
    'Coalescente', 'Dispersante', 'Agua', 'Aditivo Antimicrobiano', 'Otros Aditivos'
]

# Ingeniería de Características: Ejemplos útiles para este dataset
df['Relacion_TiO2_Ligante'] = df['Pigmento Blanco (TiO2)'] / df['Ligante (Resina Acrílica)']
df['Porcentaje_Solidos_Totales_Formula'] = (
    df['Pigmento Blanco (TiO2)'] + df['Extender (CaCO3)'] +
    df['Ligante (Resina Acrílica)'] + df['Aditivo Antimicrobiano'] +
    df['Dispersante'] + df['Otros Aditivos']
)


print("\n--- Vista General de los Datos ---")
print(df.head())
print("\n--- Información del DataFrame ---")
df.info()
print("\n--- Estadísticas Descriptivas ---")
print(df.describe().T)
print("\n--- Conteo de 'Estado Final' ---")
print(df['Estado Final'].value_counts())


# --- 3. Análisis Exploratorio de Datos (EDA) ---

print("\n--- EDA: Distribución de 'Estado Final' ---")
plt.figure(figsize=(10, 6))
sns.countplot(y='Estado Final', data=df, palette='viridis', order=df['Estado Final'].value_counts().index)
plt.title('Distribución de Tipos de Resultado de Formulación')
plt.xlabel('Número de Fórmulas')
plt.ylabel('Estado Final')
plt.tight_layout()
plt.show()

print("\n--- EDA: Correlación entre Variables Numéricas ---")
numeric_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(14, 10))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación de Variables Numéricas')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\n--- EDA: Relación entre Ingredientes Clave y Viscosidad Final ---")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Agua', y='Viscosidad Final (KU)', data=df, hue='Estado Final', palette='coolwarm', alpha=0.7)
plt.title('Viscosidad Final vs. Porcentaje de Agua')
plt.xlabel('Agua (%)')
plt.ylabel('Viscosidad Final (KU)')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Ligante (Resina Acrílica)', y='Viscosidad Final (KU)', data=df, hue='Estado Final', palette='coolwarm', alpha=0.7)
plt.title('Viscosidad Final vs. Porcentaje de Ligante')
plt.xlabel('Ligante (Resina Acrílica) (%)')
plt.ylabel('Viscosidad Final (KU)')
plt.tight_layout()
plt.show()

print("\n--- EDA: Impacto del Tipo de Pintura en la Resistencia al Fregado y Brillo ---")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Tipo de Pintura', y='Resistencia al Fregado (Ciclos)', data=df, palette='pastel')
plt.title('Resistencia al Fregado por Tipo de Pintura')
plt.xlabel('Tipo de Pintura')
plt.ylabel('Resistencia al Fregado (Ciclos)')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
sns.boxplot(x='Tipo de Pintura', y='Brillo (60°)', data=df, palette='pastel')
plt.title('Brillo por Tipo de Pintura')
plt.xlabel('Tipo de Pintura')
plt.ylabel('Brillo (60°)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- EDA: Impacto del Proveedor de Pigmento Blanco en el Poder Cubriente ---")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Proveedor Pigmento Blanco', y='Poder Cubriente (m²/L)', data=df, palette='muted')
plt.title('Poder Cubriente por Proveedor de Pigmento Blanco')
plt.xlabel('Proveedor de Pigmento Blanco')
plt.ylabel('Poder Cubriente (m²/L)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- EDA Completo: ¡Análisis Preliminar Finalizado! ---")
print("Revisa los gráficos y la salida de la consola para obtener insights iniciales.")


# --- INICIO DEL ANÁLISIS DETALLADO DE FÓRMULAS CON 'FALLA' (BLOQUE AGREGADO) ---
print("\n" + "="*50)
print("--- ANÁLISIS DETALLADO DE FÓRMULAS CON 'FALLA' ---")
print("="*50)

df_fallas = df[df['Estado Final'].str.contains('Falla', na=False)] # na=False para evitar errores si hay NaNs

if not df_fallas.empty:
    print(f"\nSe encontraron {len(df_fallas)} fórmulas con estado 'Falla'.")
    
    print("\nEstadísticas descriptivas de ingredientes para fórmulas con 'Falla':")
    columnas_interes = ingredientes_cols + ['Relacion_TiO2_Ligante', 'Porcentaje_Solidos_Totales_Formula', 'Viscosidad Final (KU)', 'Resistencia al Fregado (Ciclos)']
    
    # Filtrar solo las columnas que realmente existen en df_fallas para evitar KeyError
    columnas_existentes = [col for col in columnas_interes if col in df_fallas.columns]
    
    print(df_fallas[columnas_existentes].describe().T)

    print("\nEjemplos aleatorios de Fórmulas con 'Falla' (5 muestras):")
    columnas_ejemplo = ingredientes_cols + ['Tipo de Pintura', 'Proveedor Pigmento Blanco', 'Estado Final', 'Viscosidad Final (KU)', 'Resistencia al Fregado (Ciclos)']
    
    # Filtrar solo las columnas que realmente existen en df_fallas
    columnas_ejemplo_existentes = [col for col in columnas_ejemplo if col in df_fallas.columns]

    print(df_fallas[columnas_ejemplo_existentes].sample(min(5, len(df_fallas)), random_state=42))
    
    print("\nConteo de 'Tipo de Pintura' en las fallas:")
    print(df_fallas['Tipo de Pintura'].value_counts())
    
    print("\nConteo de 'Proveedor Pigmento Blanco' en las fallas:")
    print(df_fallas['Proveedor Pigmento Blanco'].value_counts())

else:
    print("¡Advertencia: No se encontraron fórmulas con estado 'Falla' en el dataset!")
    print("Esto podría indicar que tus datos simulados solo contienen éxitos, o que la etiqueta 'Falla' no se está generando.")
    print("Si este es el caso, las predicciones de 'Falla' serán difíciles de conseguir.")

print("\n" + "="*50)
print("--- FIN DEL ANÁLISIS DE FALLAS ---")
print("="*50)
# --- FIN DEL ANÁLISIS DETALLADO DE FÓRMULAS CON 'FALLA' (BLOQUE AGREGADO) ---