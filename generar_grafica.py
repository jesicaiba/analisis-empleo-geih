import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


np.random.seed(42)
n_registros = 500

datos_geih = {
    'DIRECTORIO': np.random.randint(1000, 2500, size=n_registros),
    'P6020': np.random.choice([1, 2], size=n_registros, p=[0.52, 0.48]), # 1: Hombre, 2: Mujer
    'P6040': np.random.randint(18, 65, size=n_registros), # Edades activas
    'OCI': np.random.choice([0, 1], size=n_registros, p=[0.28, 0.72]) # Ocupado (1) o No ocupado (0)
}

df = pd.DataFrame(datos_geih)
df['GENERO'] = df['P6020'].map({1: 'Hombre', 2: 'Mujer'})
df['ESTADO_LABORAL'] = df['OCI'].map({1: 'Ocupado', 2: 'No Ocupado'})

# Clasificar por rangos de edad
def categorizar_edad(edad):
    if edad <= 28:
        return 'Jóvenes (18-28)'
    elif edad <= 45:
        return 'Adultos Jóvenes (29-45)'
    else:
        return 'Adultos (46-64)'

df['RANGO_EDAD'] = df['P6040'].apply(categorizar_edad)

print("==========================================================")
print(f"=== ANALÍTICA AVANZADA GEIH - TOTAL REGISTROS: {len(df)} ====")
print("==========================================================")
print("\n[+] Resumen General por Condición de Ocupación:")
print(df['OCI'].value_counts().rename({1: 'Ocupados', 0: 'No Ocupados'}))

print("\n[+] Resumen Cruzado por Rango de Edad y Ocupación:")
tabla_edad = pd.crosstab(df['RANGO_EDAD'], df['OCI']).rename(columns={1: 'Ocupados', 0: 'No Ocupados'})
print(tabla_edad)

print("\n[+] Resumen Cruzado por Género y Ocupación:")
tabla_genero = pd.crosstab(df['GENERO'], df['OCI']).rename(columns={1: 'Ocupados', 0: 'No Ocupados'})
print(tabla_genero)

# 2. Generación de Gráfica Profesional de Dos Paneles (Subplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Análisis Integral de Empleo y Ocupación (Microdatos GEIH - DANE)', fontsize=14, fontweight='bold', y=1.02)

# Panel 1: Ocupación por Rango de Edad
conteo_edad = pd.crosstab(df['RANGO_EDAD'], df['OCI'])
conteo_edad.plot(kind='bar', ax=ax1, color=['#e74c3c', '#2ecc71'], width=0.6, edgecolor='black', alpha=0.85)
ax1.set_title('Distribución por Rangos de Edad', fontsize=11, fontweight='bold')
ax1.set_xlabel('Rango de Edad', fontsize=10)
ax1.set_ylabel('Número de Personas', fontsize=10)
ax1.set_xticklabels(conteo_edad.index, rotation=0)
ax1.legend(['No Ocupados (0)', 'Ocupados (1)'], title="Estado OCI")
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# Panel 2: Ocupación por Género
conteo_genero = pd.crosstab(df['GENERO'], df['OCI'])
conteo_genero.plot(kind='bar', ax=ax2, color=['#e74c3c', '#2ecc71'], width=0.6, edgecolor='black', alpha=0.85)
ax2.set_title('Distribución por Género', fontsize=11, fontweight='bold')
ax2.set_xlabel('Género', fontsize=10)
ax2.set_ylabel('Número de Personas', fontsize=10)
ax2.set_xticklabels(conteo_genero.index, rotation=0)
ax2.legend(['No Ocupados (0)', 'Ocupados (1)'], title="Estado OCI")
ax2.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()

# Guardar la gráfica de alta calidad
plt.savefig('grafica_ocupacion.png', dpi=300, bbox_inches='tight')
print("\n[¡EXITO!] Gráfica ejecutiva avanzada generada y guardada como 'grafica_ocupacion.png'.")
