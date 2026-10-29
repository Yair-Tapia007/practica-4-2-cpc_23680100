import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# 1. Configuración de la página (Icono y layout ancho)
st.set_page_config(layout="wide", page_title="Práctica 4.2 - Regresión", page_icon="💧")

# 2. Encabezado principal con mejor diseño
st.title("💧 Panel Interactivo: Predicción del Consumo de Agua (CPC)")
st.markdown("Dashboard analítico de **Regresión Lineal Múltiple** procesado mediante **DuckDB**.")
st.markdown("TAPIA VAZQUEZ YAIR GUSTAVO 23680100.")
st.divider()

# Generación de datos
np.random.seed(42)
n = 100
temperatura = np.random.normal(25, 5, n)
humedad = np.random.normal(60, 10, n)
personas = np.random.normal(4, 1, n)
bidon = np.random.normal(2, 0.5, n)
cpc = 54.49 + (2.40 * temperatura) - (0.79 * humedad) - (1.18 * personas) + (14.88 * bidon) + np.random.normal(0, 2, n)

df_simulado = pd.DataFrame({'temperatura_max': temperatura, 'humedad_max': humedad, 
                            'personas_familia': personas, 'consumo_bidon_pc': bidon, 'cpc_consumo': cpc})
con = duckdb.connect(database=':memory:')
con.register('datos_temp', df_simulado)
con.execute("CREATE TABLE datos_agua AS SELECT * FROM datos_temp")
df_agua = con.execute("SELECT * FROM datos_agua").df()

# --- DISEÑO DEL DASHBOARD ---

# PASO 1: Tabla moderna
st.subheader("🗄️ Paso 1: Base de Datos Generada (DuckDB)")
# En Streamlit Cloud sí podemos usar el dataframe interactivo
st.dataframe(df_agua.head(10), use_container_width=True)
st.divider()

# PASO 2: Matrices de correlación con color
st.subheader("🔗 Paso 2: Análisis de Correlación")
col1, col2 = st.columns(2)
matriz_correlacion = df_agua.corr()

with col1:
    st.markdown("**Matriz de Correlación de Pearson (r)**")
    # Tabla con gradiente de color automático
    st.dataframe(matriz_correlacion.style.background_gradient(cmap='coolwarm'), use_container_width=True)

with col2:
    st.markdown("**Impacto en el Consumo de Agua (cpc_consumo)**")
    corr_objetivo = matriz_correlacion['cpc_consumo'].drop('cpc_consumo').to_frame()
    # Gráfica de barras dentro de la tabla
    st.dataframe(corr_objetivo.style.bar(color=['#d65f5f', '#5fba7d']), use_container_width=True)
    
st.divider()

# ENTRENAMIENTO DEL MODELO
X = sm.add_constant(df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']])
Y = df_agua['cpc_consumo']
modelo = sm.OLS(Y, X).fit()
y_pred = modelo.predict(X)
residuos = modelo.resid

# PASOS 4 Y 5: MÉTRICAS EN BLOQUES ESTILO KPI
st.subheader("🎯 Rendimiento y Diagnóstico del Modelo")
stat, p_value = stats.shapiro(residuos)
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuos)
r2 = r2_score(Y, y_pred)
rmse = np.sqrt(mean_squared_error(Y, y_pred))

# Mostramos los resultados en 4 columnas con indicadores de estado
m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Coef. de Determinación (R²)", value=f"{r2*100:.2f}%", delta="Excelente precisión")
m2.metric(label="Error Cuadrático (RMSE)", value=f"{rmse:.2f}", delta="- Margen de error", delta_color="inverse")
m3.metric(label="Shapiro-Wilk (p-valor)", value=f"{p_value:.4f}", delta="Normalidad OK" if p_value > 0.05 else "Revisar")
m4.metric(label="Durbin-Watson", value=f"{dw:.2f}", delta="Independencia OK" if 1.5 < dw < 2.5 else "Revisar")
st.divider()

# PASO 3: RESUMEN OLS LIMPIO
st.subheader("📈 Paso 3: Resumen Estadístico Oficial (OLS)")
# Encapsulamos el texto flotante en una caja de código delimitada
st.code(modelo.summary().as_text(), language='plaintext')
st.divider()

# PANEL VISUAL
st.subheader("📊 Panel de Diagnóstico Visual")
sns.set_theme(style="darkgrid") # Adaptado al tema oscuro de Streamlit
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.heatmap(df_agua.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación')

sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='b', ax=axes[0, 1], line_kws={'color': 'red'})
axes[0, 1].set_title('2. Tendencia')

axes[0, 2].scatter(y_pred, Y, alpha=0.7, color='teal')
axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)
axes[0, 2].set_title('3. Reales vs Predichos')

axes[1, 0].scatter(y_pred, residuos, color='purple', alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('4. Homocedasticidad')

sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Q-Q Normal')

axes[1, 2].axis('off')
plt.tight_layout()

st.pyplot(fig)
