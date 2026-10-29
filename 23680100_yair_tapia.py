import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de la página web
st.set_page_config(layout="wide", page_title="Práctica 4.2 - Regresión")
st.title("📊 Panel Interactivo: Predicción del Consumo de Agua (CPC)")

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

st.subheader("--- PASO 1: PRIMEROS 5 REGISTROS OBTENIDOS DE DUCKDB ---")
st.table(df_agua.head())

# Paso 2
st.subheader("--- PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON (r) ---")
matriz_correlacion = df_agua.corr()
st.text(matriz_correlacion.to_string())
st.text(f"\nCorrelación lineal de cada variable con la respuesta 'cpc_consumo':\n{matriz_correlacion['cpc_consumo'].drop('cpc_consumo').to_string()}")

# Entrenamiento del modelo
X = sm.add_constant(df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']])
Y = df_agua['cpc_consumo']
modelo = sm.OLS(Y, X).fit()
y_pred = modelo.predict(X)
residuos = modelo.resid

# Paso 3
st.subheader("--- PASO 3: RESUMEN DEL MODELO OLS (statsmodels) ---")
st.text(modelo.summary().as_text())

# Paso 4
st.subheader("--- PASO 4: DIAGNÓSTICO DE RESIDUOS Y SUPUESTOS ---")
stat, p_value = stats.shapiro(residuos)
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuos)

diag_text = f"1. Prueba de Normalidad (Shapiro-Wilk): Estadístico = {stat:.4f}, valor p = {p_value:.4f}\n"
if p_value > 0.05:
    diag_text += "   -> CONCLUSIÓN: Los residuos cumplen con el supuesto de NORMALIDAD (p > 0.05).\n"
diag_text += f"2. Estadístico Durbin-Watson (Independencia): {dw:.4f}"
st.text(diag_text)

# Paso 5
st.subheader("--- PASO 5: MÉTRICAS DE RENDIMIENTO DEL MODELO (scikit-learn) ---")
r2 = r2_score(Y, y_pred)
rmse = np.sqrt(mean_squared_error(Y, y_pred))
st.text(f"Coeficiente de Determinación (R^2):\t{r2*100:.3f}%\nError Cuadrático Medio Raíz (RMSE):\t{rmse:.4f} lt/hab-día")

# Panel Visual
st.subheader("--- PANEL DE DIAGNÓSTICO VISUAL ---")
sns.set_theme(style="whitegrid")
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
