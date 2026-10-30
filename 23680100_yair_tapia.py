import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de página
st.set_page_config(layout="wide", page_title="Práctica 4.2 - Regresión Múltiple", page_icon="⚡")

# --- INYECCIÓN DE CSS PARA DISEÑO LLAMATIVO (TEMA NEÓN/TECH) ---
st.markdown("""
<style>
    /* Efecto de tarjetas para las métricas (KPIs) */
    div[data-testid="metric-container"] {
        background-color: #1A1A24;
        border: 1px solid #00FFAA;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 0px 15px rgba(0, 255, 170, 0.15);
    }
    /* Color de los valores en las métricas */
    div[data-testid="stMetricValue"] {
        color: #00FFAA !important;
        font-weight: 800;
        font-size: 2.2rem;
    }
    /* Color de los subtítulos de métricas */
    div[data-testid="stMetricLabel"] {
        color: #FF00FF !important;
        font-weight: 700;
        font-size: 1.1rem;
    }
    /* Títulos de los pasos */
    h2 {
        color: #FF9E64 !important;
        border-bottom: 2px solid #3B4252;
        padding-bottom: 10px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado principal personalizado
st.markdown("<h1 style='text-align: center; color: #00D2FF; text-shadow: 0px 0px 10px #00D2FF;'>⚡ Práctica 4.2: Panel Predictivo de Consumo de Agua (CPC)</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #B0C4DE;'>Modelo de Regresión Lineal Múltiple Interactivo</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; background-color: #1A1A24; padding: 10px; border-radius: 8px; display: inline-block; margin: 0 auto;'><b>👨‍💻 Autor:</b> Yair Gustavo Tapia Vazquez &nbsp;&nbsp;|&nbsp;&nbsp; <b>🎓 No. Control:</b> 23680100</p>", unsafe_allow_html=True)
st.write("") # Espaciador

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

# Entrenamiento del modelo
X = sm.add_constant(df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']])
Y = df_agua['cpc_consumo']
modelo = sm.OLS(Y, X).fit()
y_pred = modelo.predict(X)
residuos = modelo.resid

# --- PASO 1 ---
st.header("🗄️ PASO 1: Generación de Datos con DuckDB")
st.info("Vista previa de los primeros 10 registros procesados en memoria.")
st.dataframe(df_agua.head(10), use_container_width=True)

# --- PASO 2 ---
st.header("🔥 PASO 2: Matriz de Correlación de Pearson (r)")
col1, col2 = st.columns([2, 1])
matriz_correlacion = df_agua.corr()

with col1:
    st.markdown("**Mapa de Calor de Variables (Plasma)**")
    # Cambiamos a 'plasma' para colores neon muy llamativos
    st.dataframe(matriz_correlacion.style.background_gradient(cmap='plasma'), use_container_width=True)
with col2:
    st.markdown("**Impacto en el Consumo (CPC)**")
    corr_objetivo = matriz_correlacion['cpc_consumo'].drop('cpc_consumo').to_frame()
    # Barra de color magenta neon
    st.dataframe(corr_objetivo.style.bar(color='#FF00FF'), use_container_width=True)

# --- PASO 3 ---
st.header("📊 PASO 3: Resumen del Modelo OLS (Statsmodels)")
with st.expander("Haz clic aquí para desplegar el reporte estadístico oficial completo", expanded=True):
    st.code(modelo.summary().as_text(), language='plaintext')

# --- PASO 4 ---
st.header("🎯 PASO 4: Diagnóstico de Residuos y Supuestos")
stat, p_value = stats.shapiro(residuos)
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuos)

col_diag1, col_diag2 = st.columns(2)
with col_diag1:
    if p_value > 0.05:
        st.success(f"✅ **NORMALIDAD CUMPLIDA:** \n\nEstadístico de Shapiro-Wilk = {stat:.4f} \np-valor = {p_value:.4f}")
    else:
        st.error(f"❌ **NO CUMPLE:** \n\nEstadístico = {stat:.4f}, p-valor = {p_value:.4f}")

with col_diag2:
    if 1.5 < dw < 2.5:
        st.success(f"✅ **INDEPENDENCIA CUMPLIDA:** \n\nEstadístico Durbin-Watson = {dw:.4f}")
    else:
        st.warning(f"⚠️ **REVISAR INDEPENDENCIA:** \n\nEstadístico = {dw:.4f}")

# --- PASO 5 ---
st.header("🚀 PASO 5: Métricas de Rendimiento (Scikit-learn)")
r2 = r2_score(Y, y_pred)
rmse = np.sqrt(mean_squared_error(Y, y_pred))

m1, m2 = st.columns(2)
m1.metric(label="Coeficiente de Determinación (R²)", value=f"{r2*100:.2f}%", delta="Excelente Ajuste")
m2.metric(label="Error Cuadrático Medio Raíz (RMSE)", value=f"{rmse:.4f}", delta="Litros/hab-día", delta_color="inverse")

# --- PANEL VISUAL ---
st.header("🖼️ Panel de Diagnóstico Visual")
# Configurar Matplotlib para tema oscuro y colores neón
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#0E1117') # Igual al fondo oscuro de Streamlit
for ax in axes.flat:
    ax.set_facecolor('#0E1117')
    ax.grid(color='#3B4252', linestyle='--', linewidth=0.5)

sns.heatmap(df_agua.corr(), annot=True, fmt=".2f", cmap="plasma", ax=axes[0, 0], cbar_kws={'label': 'Pearson r'})
axes[0, 0].set_title('1. Matriz de Correlación', color='#00D2FF', fontweight='bold')

sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='#00FFAA', ax=axes[0, 1], scatter_kws={'alpha':0.6}, line_kws={'color': '#FF00FF', 'linewidth': 3})
axes[0, 1].set_title('2. Tendencia (Humedad vs CPC)', color='#00D2FF', fontweight='bold')

axes[0, 2].scatter(y_pred, Y, alpha=0.7, color='#FF9E64')
axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='#FF00FF', linestyle='--', lw=3)
axes[0, 2].set_title('3. Reales vs Predichos', color='#00D2FF', fontweight='bold')

axes[1, 0].scatter(y_pred, residuos, color='#00D2FF', alpha=0.6)
axes[1, 0].axhline(y=0, color='#FF00FF', linestyle='--', lw=3)
axes[1, 0].set_title('4. Homocedasticidad', color='#00D2FF', fontweight='bold')

sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1], markerfacecolor='#00FFAA', markeredgecolor='#00FFAA', alpha=0.6)
axes[1, 1].get_lines()[1].set_color('#FF00FF')
axes[1, 1].get_lines()[1].set_linewidth(3)
axes[1, 1].set_title('5. Q-Q Normal', color='#00D2FF', fontweight='bold')

axes[1, 2].axis('off')
plt.tight_layout()

st.pyplot(fig)
