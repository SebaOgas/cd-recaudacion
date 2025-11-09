import joblib
import sys
from sklearn.base import BaseEstimator, TransformerMixin
import streamlit as st
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
import numpy as np

st.html("""
    <style>
        .stMainBlockContainer {
            max-width:80rem;
        }
    </style>
    """
)

# Carga del modelo
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)

sys.modules['__main__'].DropColumns = DropColumns

loaded_grid = joblib.load('data/model.pkl')

# Carga de datos originales
df_montos = pd.read_csv('data/montos.csv')
df_montos['anio'] = pd.to_numeric(df_montos['anio'], errors='coerce').astype('Int64')
df_montos['mes']  = pd.to_numeric(df_montos['mes'],  errors='coerce').astype('Int64')
df_montos = df_montos.set_index(['anio', 'mes'])

def preparar_ipcs(df):
    df['ipc_santafe'] = df['ipc_santafe'] / 100
    df['ipc_gba'] = df['ipc_gba'].pct_change(fill_method=None)
    df['ipc_mendoza'] = df['ipc_mendoza'].pct_change(fill_method=None)
    return df

def mapear_provincia_ipc(poblacion_df):
    def match_provincias_ipc(provincia):
        match provincia:
            case "24 partidos del Gran Buenos Aires":
                return "ipc_gba"
            case "Buenos Aires":
                return "ipc_gba"
            case "Córdoba":
                return "ipc_cordoba"
            case "Tucumán":
                return "ipc_tucuman"
            case "Santa Fe":
                return "ipc_santafe"
            case "Mendoza":
                return "ipc_mendoza"
            case _:
                return pd.NA
    poblacion_df["mapeo_ipc"] = poblacion_df["provincia"].apply(lambda p : p.strip()).apply(match_provincias_ipc)
    return poblacion_df

def filtrar_provincias(poblacion_df):
    return poblacion_df[poblacion_df['mapeo_ipc'].notna()]

def agrupar_por_mapeo_ipc(poblacion_df):
    return poblacion_df.groupby('mapeo_ipc').sum()

def dejar_poblacion_df_bonito(poblacion_df):
    poblacion_df = poblacion_df.drop(columns=['provincia'])
    poblacion_df['valor'] = poblacion_df['valor'].astype('int64')
    return poblacion_df

poblacion_path = 'data/poblacion_2010.csv'
poblacion_df = pd.read_csv(poblacion_path)

poblacion_df = (
    poblacion_df
    .pipe(mapear_provincia_ipc)
    .pipe(filtrar_provincias)
    .pipe(agrupar_por_mapeo_ipc)
    .pipe(dejar_poblacion_df_bonito)
)

def juntar_ipc_provincias(df):
    ipc_cols = poblacion_df.index.to_list()

    def weighted_avg(row):
        valid_ipcs = row[ipc_cols].dropna()
        if valid_ipcs.empty:
            return pd.NA
        relevant_poblacion = poblacion_df.loc[valid_ipcs.index]
        weights = relevant_poblacion['valor']
        if weights.sum() == 0:
            return pd.NA
        return (valid_ipcs * weights).sum() / weights.sum()

    df['ipc_provincias'] = df.apply(weighted_avg, axis=1)
    return df

def limpiar_ipc(df):
    df['ipc'] = df['ipc_argentina'].fillna(df['ipc_provincias'])
    cols = ['ipc_argentina', 'ipc_provincias']
    cols.extend(poblacion_df.index.to_list())
    df = df.drop(columns=cols)
    return df

def ajustar_recaudacion_miles_a_millones(df):
    df['recaudacion'] = df['recaudacion'].where(
        df.index.get_level_values('anio') >= 2021,
        df['recaudacion'] / 1000
    )
    return df

def limpiar_nas_inicio(df):
    mask = (df['recaudacion'].notna()) | (df.index.get_level_values('anio') > 2008)
    df = df[mask]
    return df

df_montos = (
    df_montos
    .pipe(preparar_ipcs)
    .pipe(juntar_ipc_provincias)
    .pipe(limpiar_ipc)
    .pipe(ajustar_recaudacion_miles_a_millones)
    .pipe(limpiar_nas_inicio)
)

def predict_missing_values(df, target_year, target_month, alpha=0.5):
    """Predict values using exponential moving average of same month in previous years"""
    predictions = {}
    same_month_data = df[df.index.get_level_values('mes') == target_month]
    same_month_data = same_month_data[same_month_data.index.get_level_values('anio') < target_year]
    same_month_data = same_month_data.sort_index()
    
    if len(same_month_data) < 1:
        return {}
    
    for col in df.columns:
        values = same_month_data[col].dropna()
        if len(values) == 0:
            predictions[col] = None
            continue
        ema = values.iloc[0]
        for val in values.iloc[1:]:
            ema = alpha * val + (1 - alpha) * ema
        predictions[col] = ema
    
    return predictions

def get_value_or_predict(df, year, month, column, predictions):
    """Get value from dataframe if exists, otherwise use prediction. Returns (value, is_predicted)"""
    if (year, month) in df.index:
        value = df.loc[(year, month), column]
        if pd.notna(value):
            return value, False
    if predictions and column in predictions and predictions[column] is not None:
        return predictions[column], True
    return None, False

# UI
st.title("Predictor de Recaudación Argentina")

# Month picker
st.markdown("### Seleccionar Periodo")

# Get available periods from data + next 12 months
periodos_disponibles = set()
for anio, mes in df_montos.index:
    if pd.notna(anio) and pd.notna(mes):
        periodos_disponibles.add((int(anio), int(mes)))

# Add next 12 months
current_date = datetime.now()
for i in range(12):
    future_date = current_date + relativedelta(months=i)
    periodos_disponibles.add((future_date.year, future_date.month))

# Sort periods
periodos_disponibles = sorted(list(periodos_disponibles))

# Get unique years that have data
years_disponibles = sorted(list(set([y for y, m in periodos_disponibles])))

meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", 
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

col1, col2 = st.columns(2)

# Year selector
with col2:
    anio_sel = st.selectbox("Año", years_disponibles,
                           index=years_disponibles.index(current_date.year) if current_date.year in years_disponibles else len(years_disponibles)-1)

# Month selector - only show months available for selected year
meses_disponibles = sorted([m for y, m in periodos_disponibles if y == anio_sel])
with col1:
    default_mes = current_date.month if current_date.month in meses_disponibles else meses_disponibles[0]
    mes_sel = st.selectbox("Mes", meses_disponibles,
                           format_func=lambda x: meses[x-1],
                           index=meses_disponibles.index(default_mes))

periodo_anterior = (anio_sel - 1, mes_sel)

# Predict values for selected period if not available
predicted_values_yearago = predict_missing_values(df_montos, anio_sel - 1, mes_sel)
predicted_values_current = predict_missing_values(df_montos, anio_sel, mes_sel)

# Variables
variables = [
    ('dolar_oficial', 'Dólar Oficial'),
    ('ipc', 'IPC'),
    ('exportaciones', 'Exportaciones'),
    ('emae', 'EMAE'),
    ('importaciones', 'Importaciones'),
    ('monedas_publico', 'Billetes y monedas en poder del público'),
    ('dolar_blue', 'Dólar Blue')
]

unidades = {
    'dolar_oficial': 'ARS',
    'ipc': '%',
    'exportaciones': 'millones USD',
    'emae': 'índice',
    'importaciones': 'millones USD',
    'monedas_publico': 'millones ARS',
    'dolar_blue': 'ARS'
}

# Create input table
st.markdown("### Variables Económicas")
col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 2, 1], vertical_alignment='center')

with col1:
    st.markdown("**Variable**")
with col2:
    st.markdown(f"**{meses[mes_sel-1]} {anio_sel - 1}**")
with col3:
    st.markdown("")
with col4:
    st.markdown(f"**{meses[mes_sel-1]} {anio_sel}**")
with col5:
    st.markdown("")

# Store input values
valores_hace_anio = {}
valores_actuales = {}

for var_key, var_label in variables:
    col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 2, 1], vertical_alignment='center')
    
    with col1:
        st.write(var_label)
    
    valor_anterior, is_pred_anterior = get_value_or_predict(df_montos, anio_sel - 1, mes_sel, var_key, predicted_values_yearago)
    valor_actual, is_pred_actual = get_value_or_predict(df_montos, anio_sel, mes_sel, var_key, predicted_values_current)
    
    with col2:
        val_hace_anio = st.text_input(
            "Valor", 
            value=f"{valor_anterior:.2f}{'*' if is_pred_anterior else ''}" if valor_anterior is not None else "",
            key=f"{var_key}_hace_anio_{anio_sel}_{mes_sel}",
            label_visibility="collapsed"
        )
        valores_hace_anio[var_key] = float(val_hace_anio.replace('*', '')) if val_hace_anio else 0
    
    with col3:
        st.write(unidades.get(var_key, 'unidad'))
    
    with col4:
        val_actual = st.text_input(
            "Valor",
            value=f"{valor_actual:.2f}{'*' if is_pred_actual else ''}" if valor_actual is not None else "",
            key=f"{var_key}_actual_{anio_sel}_{mes_sel}",
            label_visibility="collapsed"
        )
        valores_actuales[var_key] = float(val_actual.replace('*', '')) if val_actual else 0
    
    with col5:
        st.write(unidades.get(var_key, 'unidad'))

# Recaudación hace un año
recaudacion_anterior, is_pred_recaud = get_value_or_predict(df_montos, anio_sel - 1, mes_sel, 'recaudacion', predicted_values_yearago)

st.markdown("---")
col1, col2, col3 = st.columns([2, 5, 1], vertical_alignment='center')
with col1:
    st.write(f"**Recaudación {meses[mes_sel-1]} {anio_sel - 1}:**")
with col2:
    recaudacion_hace_anio = st.text_input(
        "Recaudación",
        value=f"{recaudacion_anterior:.2f}{'*' if is_pred_recaud else ''}" if recaudacion_anterior is not None else "",
        key=f"recaudacion_hace_anio_{anio_sel}_{mes_sel}",
        label_visibility="collapsed"
    )
with col3:
    st.write("millones ARS")

# Predict button
st.info("Los valores marcados con asterisco (*) son estimaciones automáticas")

if st.button("Predecir Recaudación", type="primary", use_container_width=True):
    try:
        diferencias = {}
        for var_key, _ in variables:
            hace_anio = valores_hace_anio[var_key]
            actual = valores_actuales[var_key]
            if hace_anio != 0:
                diferencias[var_key] = ((actual - hace_anio) / hace_anio)
            else:
                diferencias[var_key] = 0
        
        input_data = pd.DataFrame([diferencias])
        input_data['ingresos_familiares'] = 0
        input_data['anio'] = anio_sel
        input_data['mes'] = mes_sel
        
        cambio_predicho = loaded_grid.predict(input_data)[0]
        recaudacion_base = float(recaudacion_hace_anio.replace('*', '')) if recaudacion_hace_anio else 0
        recaudacion_predicha = recaudacion_base * (1 + cambio_predicho)

        recaudacion_actual, is_pred_recaud_actual = get_value_or_predict(df_montos, anio_sel, mes_sel, 'recaudacion', predicted_values_current)
        
        st.markdown("---")
        st.markdown("### Resultados")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Variación Interanual (modelo)", f"{cambio_predicho:.2%}")
        with col2:
            st.metric("Recaudación Predicha", f"{recaudacion_predicha:,.2f} millones ARS")
        
        if not is_pred_recaud_actual and recaudacion_actual is not None:
            st.markdown("#### Comparación con valores reales")
            variacion_real = (recaudacion_actual - recaudacion_base) / recaudacion_base if recaudacion_base != 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Variación Real", f"{variacion_real:.2%}")
            with col2:
                st.metric("Recaudación Real", f"{recaudacion_actual:,.2f} millones ARS")
            
            error_abs = abs(recaudacion_predicha - recaudacion_actual)
            error_pct = error_abs / recaudacion_actual if recaudacion_actual != 0 else 0
            st.metric("Error", f"{error_pct:.2%}")
        
    except Exception as e:
        st.error(f"Error al predecir: {str(e)}")