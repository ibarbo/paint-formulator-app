"""
Este script realiza un Análisis Exploratorio de Datos (EDA) exhaustivo
sobre el dataset simulado de formulaciones de pintura.

Los objetivos principales de este EDA son:
1.  Cargar los datos desde un archivo CSV.
2.  Realizar un preprocesamiento básico, incluyendo la conversión de tipos de datos.
3.  Generar nuevas características (Feature Engineering) que pueden ser útiles para el modelado.
4.  Proporcionar una visión general de la estructura y estadísticas de los datos.
5.  Visualizar distribuciones, correlaciones y relaciones entre variables clave
    mediante varios tipos de gráficos (histogramas, boxplots, scatterplots, mapas de calor).
6.  Realizar un análisis detallado de las formulaciones clasificadas como 'Falla'
    para identificar posibles patrones o causas de los resultados negativos.

Este análisis ayuda a comprender mejor el dataset antes de proceder al modelado
de Machine Learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización para mejorar la estética de los gráficos.
sns.set_style("whitegrid") # Establece el estilo de cuadrícula de Seaborn.
plt.rcParams['figure.figsize'] = (12, 6) # Define el tamaño por defecto de las figuras de Matplotlib.

# --- 1. Extracción (Carga de Datos) ---
# Define la ruta relativa al archivo CSV del dataset.
file_name = 'data/formulaciones_pintura_simuladas.csv' 
try:
    df = pd.read_csv(file_name) # Intenta cargar el DataFrame desde el archivo CSV.
    print(f"Datos cargados exitosamente desde '{file_name}'.")
    print("\nNombres de las columnas en el DataFrame:")
    print(df.columns.tolist()) # Imprime la lista de nombres de columnas.
except FileNotFoundError:
    # Manejo de error si el archivo no se encuentra.
    print(f"Error: El archivo '{file_name}' no se encontró. Asegúrate de haber ejecutado 'generate_paint_data.py' primero.")
    exit() # Sale del script si el archivo no está presente.

# --- 2. Transformación y Carga (Preprocesamiento Básico y Carga a DataFrame) ---

# Convierte columnas categóricas a tipo 'category' de Pandas.
# Esto optimiza el uso de memoria y puede acelerar ciertas operaciones de análisis
# y visualización al tratar estas columnas como tipos discretos.
df['Estado Final'] = df['Estado Final'].astype('category')
df['Tipo de Pintura'] = df['Tipo de Pintura'].astype('category')
df['Proveedor Pigmento Blanco'] = df['Proveedor Pigmento Blanco'].astype('category')

# Definición de las columnas que representan los ingredientes principales de la formulación.
# Esta lista se usa para filtrado y análisis específicos de estos componentes.
ingredientes_cols = [
    'Pigmento Blanco (TiO2)', 'Extender (CaCO3)', 'Ligante (Resina Acrílica)',
    'Coalescente', 'Dispersante', 'Agua', 'Aditivo Antimicrobiano', 'Otros Aditivos'
]

# Ingeniería de Características: Creación de nuevas características a partir de las existentes.
# Estas características derivadas pueden capturar relaciones importantes en los datos
# y mejorar el rendimiento de los modelos de Machine Learning.
df['Relacion_TiO2_Ligante'] = df['Pigmento Blanco (TiO2)'] / df['Ligante (Resina Acrílica)']
df['Porcentaje_Solidos_Totales_Formula'] = (
    df['Pigmento Blanco (TiO2)'] + df['Extender (CaCO3)'] +
    df['Ligante (Resina Acrílica)'] + df['Aditivo Antimicrobiano'] +
    df['Dispersante'] + df['Otros Aditivos']
)

# --- Vista Preliminar de los Datos ---
print("\n--- Vista General de los Datos (primeras 5 filas) ---")
print(df.head()) # Muestra las primeras filas del DataFrame.
print("\n--- Información del DataFrame ---")
df.info() # Proporciona un resumen conciso del DataFrame, incluyendo tipos de datos y valores no nulos.
print("\n--- Estadísticas Descriptivas de Columnas Numéricas ---")
print(df.describe().T) # Muestra estadísticas descriptivas para columnas numéricas (transpuesto para mejor legibilidad).
print("\n--- Conteo de Ocurrencias para 'Estado Final' ---")
print(df['Estado Final'].value_counts()) # Muestra la frecuencia de cada categoría en 'Estado Final'.


# --- 3. Análisis Exploratorio de Datos (EDA) ---

# Gráfico de barras para visualizar la distribución de la variable objetivo 'Estado Final'.
# Ayuda a identificar el balance de clases (ej., 'Éxito' vs. 'Falla').
print("\n--- EDA: Distribución de 'Estado Final' ---")
plt.figure(figsize=(10, 6))
sns.countplot(y='Estado Final', data=df, palette='viridis', order=df['Estado Final'].value_counts().index)
plt.title('Distribución de Tipos de Resultado de Formulación')
plt.xlabel('Número de Fórmulas')
plt.ylabel('Estado Final')
plt.tight_layout() # Ajusta automáticamente los parámetros de la figura para un diseño ajustado.
plt.show()

# Mapa de calor para visualizar la matriz de correlación entre variables numéricas.
# Los valores cercanos a 1 o -1 indican correlaciones fuertes (positiva o negativa).
print("\n--- EDA: Correlación entre Variables Numéricas ---")
numeric_cols = df.select_dtypes(include=np.number).columns # Selecciona solo las columnas numéricas.
plt.figure(figsize=(14, 10))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación de Variables Numéricas')
plt.xticks(rotation=45, ha='right') # Rota las etiquetas del eje X para mejor legibilidad.
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Gráficos de dispersión para explorar la relación entre ingredientes clave y la 'Viscosidad Final'.
# Se usa 'hue' para diferenciar por 'Estado Final' y observar patrones.
print("\n--- EDA: Relación entre Ingredientes Clave y Viscosidad Final ---")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1) # Crea un subplot 1x2, selecciona el primero.
sns.scatterplot(x='Agua', y='Viscosidad Final (KU)', data=df, hue='Estado Final', palette='coolwarm', alpha=0.7)
plt.title('Viscosidad Final vs. Porcentaje de Agua')
plt.xlabel('Agua (%)')
plt.ylabel('Viscosidad Final (KU)')

plt.subplot(1, 2, 2) # Selecciona el segundo subplot.
sns.scatterplot(x='Ligante (Resina Acrílica)', y='Viscosidad Final (KU)', data=df, hue='Estado Final', palette='coolwarm', alpha=0.7)
plt.title('Viscosidad Final vs. Porcentaje de Ligante')
plt.xlabel('Ligante (Resina Acrílica) (%)')
plt.ylabel('Viscosidad Final (KU)')
plt.tight_layout()
plt.show()

# Gráficos de caja (boxplots) para analizar el impacto del 'Tipo de Pintura'
# en propiedades clave como 'Resistencia al Fregado' y 'Brillo'.
print("\n--- EDA: Impacto del Tipo de Pintura en la Resistencia al Fregado y Brillo ---")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Tipo de Pintura', y='Resistencia al Fregado (Ciclos)', data=df, palette='pastel')
plt.title('Resistencia al Fregado por Tipo de Pintura')
plt.xlabel('Tipo de Pintura')
plt.ylabel('Resistencia al Fregado (Ciclos)')
plt.xticks(rotation=45, ha='right') # Rota las etiquetas del eje X.

plt.subplot(1, 2, 2)
sns.boxplot(x='Tipo de Pintura', y='Brillo (60°)', data=df, palette='pastel')
plt.title('Brillo por Tipo de Pintura')
plt.xlabel('Tipo de Pintura')
plt.ylabel('Brillo (60°)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Gráfico de caja para visualizar el impacto del 'Proveedor Pigmento Blanco'
# en el 'Poder Cubriente'.
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

# Filtra el DataFrame para incluir solo las formulaciones cuyo 'Estado Final' contiene 'Falla'.
# 'na=False' asegura que las entradas NaN en 'Estado Final' no causen errores.
df_fallas = df[df['Estado Final'].str.contains('Falla', na=False)] 

if not df_fallas.empty:
    print(f"\nSe encontraron {len(df_fallas)} fórmulas con estado 'Falla'.")
    
    print("\nEstadísticas descriptivas de ingredientes y propiedades clave para fórmulas con 'Falla':")
    # Define las columnas de interés para un análisis descriptivo específico de las fallas.
    columnas_interes = ingredientes_cols + ['Relacion_TiO2_Ligante', 'Porcentaje_Solidos_Totales_Formula', 'Viscosidad Final (KU)', 'Resistencia al Fregado (Ciclos)']
    
    # Filtra las columnas para asegurar que solo se incluyan las que realmente existen en el DataFrame.
    columnas_existentes = [col for col in columnas_interes if col in df_fallas.columns]
    
    print(df_fallas[columnas_existentes].describe().T) # Muestra estadísticas descriptivas de estas columnas.

    print("\nEjemplos aleatorios de Fórmulas con 'Falla' (5 muestras):")
    # Define las columnas a mostrar en los ejemplos de fallas para una inspección rápida.
    columnas_ejemplo = ingredientes_cols + ['Tipo de Pintura', 'Proveedor Pigmento Blanco', 'Estado Final', 'Viscosidad Final (KU)', 'Resistencia al Fregado (Ciclos)']
    
    # Filtra las columnas para asegurar que solo se incluyan las que realmente existen en el DataFrame.
    columnas_ejemplo_existentes = [col for col in columnas_ejemplo if col in df_fallas.columns]

    # Muestra una muestra aleatoria de fórmulas con 'Falla', limitado a 5 o menos si hay menos fallas.
    print(df_fallas[columnas_ejemplo_existentes].sample(min(5, len(df_fallas)), random_state=42))
    
    print("\nConteo de 'Tipo de Pintura' en las fallas:")
    print(df_fallas['Tipo de Pintura'].value_counts()) # Frecuencia de tipos de pintura en formulaciones fallidas.
    
    print("\nConteo de 'Proveedor Pigmento Blanco' en las fallas:")
    print(df_fallas['Proveedor Pigmento Blanco'].value_counts()) # Frecuencia de proveedores en formulaciones fallidas.

else:
    print("¡Advertencia: No se encontraron fórmulas con estado 'Falla' en el dataset!")
    print("Esto podría indicar que tus datos simulados solo contienen éxitos, o que la etiqueta 'Falla' no se está generando.")
    print("Si este es el caso, las predicciones de 'Falla' serán difíciles de conseguir con los modelos.")

print("\n" + "="*50)
print("--- FIN DEL ANÁLISIS DE FALLAS ---")
print("="*50)
# --- FIN DEL ANÁLISIS DETALLADO DE FÓRMULAS CON 'FALLA' (BLOQUE AGREGADO) ---