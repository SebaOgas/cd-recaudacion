import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import sys
from sklearn.base import BaseEstimator, TransformerMixin

st.html("""
    <style>
        .stMainBlockContainer {
            max-width:80rem;
        }
    </style>
    """
)

# Cargar el modelo
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)

sys.modules['__main__'].DropColumns = DropColumns

loaded_grid = joblib.load('data/model.pkl')

df = pd.read_csv('data/df_result_interanual.csv')
test_predictions = pd.read_csv('data/test_predictions.csv')

# Realizar predicciones completas
y_pred_full = loaded_grid.predict(df.drop(columns=["recaudacion"]))

plot_df_full = pd.DataFrame({
    "Real": df["recaudacion"],
    "Predicho": y_pred_full
})

plot_df_full.index = df.index
plot_df_full = plot_df_full.sort_index()

# Preparar datos para Altair
plot_df_full = plot_df_full.reset_index()
plot_df_full['anio'] = df['anio'].values
plot_df_full['mes'] = df['mes'].values
plot_df_full['index_num'] = range(len(plot_df_full))

# Crear datasets separados para línea Real (sin puntos) y puntos de test
real_line = plot_df_full[['index_num', 'anio', 'mes', 'Real']].copy()
real_line['Tipo'] = 'Real'
real_line = real_line.rename(columns={'Real': 'Recaudacion'})

# Preparar puntos de predicción solo para datos de test
test_predictions['anio'] = pd.to_numeric(test_predictions['anio'], errors='coerce').astype('Int64')
test_predictions['mes'] = pd.to_numeric(test_predictions['mes'], errors='coerce').astype('Int64')

predicho_points = plot_df_full.merge(
    test_predictions[['anio', 'mes']], 
    on=['anio', 'mes'], 
    how='inner'
)[['index_num', 'anio', 'mes', 'Predicho']].copy()
predicho_points['Tipo'] = 'Predicho'
predicho_points = predicho_points.rename(columns={'Predicho': 'Recaudacion'})

# Combinar ambos datasets
plot_df_combined = pd.concat([real_line, predicho_points], ignore_index=True)

# Crear datos para las marcas de año en el eje X
year_positions = df.groupby('anio').apply(lambda x: x.index[0]).reset_index()
year_positions.columns = ['anio', 'position']
year_positions['index_num'] = [plot_df_full[plot_df_full.index == pos]['index_num'].values[0] 
                                 for pos in year_positions['position']]

# Línea Real sin puntos
line_real = alt.Chart(real_line).mark_line(point=False).encode(
    x=alt.X('index_num:Q',
            axis=alt.Axis(
                title='Año',
                values=year_positions['index_num'].tolist(),
                labelExpr=f"datum.value == {year_positions['index_num'].iloc[0]} ? '{year_positions['anio'].iloc[0]}' : " + 
                         ' : '.join([f"datum.value == {row['index_num']} ? '{row['anio']}'" 
                                   for _, row in year_positions.iloc[1:].iterrows()]) + " : ''",
                labelAngle=-90,
                grid=True
            )),
    y=alt.Y('Recaudacion:Q', 
            title='Variación Interanual de Recaudación',
            scale=alt.Scale(zero=False),
            axis=alt.Axis(grid=True, format='.1%')),
    color=alt.Color('Tipo:N', legend=alt.Legend(title=''), scale=alt.Scale(domain=['Real', 'Predicho'], range=['#1f77b4', '#ff7f0e'])),
    tooltip=[
        alt.Tooltip('anio:O', title='Año'),
        alt.Tooltip('mes:O', title='Mes'),
        alt.Tooltip('Recaudacion:Q', title='Recaudación', format='.2%')
    ]
)

# Puntos de predicción
points_predicho = alt.Chart(predicho_points).mark_point(size=60, filled=True).encode(
    x='index_num:Q',
    y='Recaudacion:Q',
    color=alt.Color('Tipo:N', legend=None, scale=alt.Scale(domain=['Real', 'Predicho'], range=['#1f77b4', '#ff7f0e'])),
    tooltip=[
        alt.Tooltip('anio:O', title='Año'),
        alt.Tooltip('mes:O', title='Mes'),
        alt.Tooltip('Recaudacion:Q', title='Predicción', format='.2%')
    ]
)

# Línea horizontal de referencia en 0
reference_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[5, 5], color='gray').encode(
    y='y:Q'
)

chart = (line_real + points_predicho + reference_line).properties(
    width=800,
    height=400,
    title='Variación Interanual de Recaudación Real vs. Predicha'
).interactive()

# Gráfico de estacionalidad
slider_anio = alt.binding_range(
    min=int(df['anio'].min()),
    max=int(df['anio'].max()),
    step=1,
    name='Seleccionar año: '
)
param_anio = alt.param(
    name='year_selector',  
    value=int(df['anio'].max()), 
    bind=slider_anio
)

# Calcular escala Y global
y_min = df['recaudacion'].min()
y_max = df['recaudacion'].max()
y_padding = (y_max - y_min) * 0.1

chart2 = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x=alt.X('mes:O', 
                title='Mes', 
                axis=alt.Axis(values=list(range(1, 13)), grid=True),
                scale=alt.Scale(domain=list(range(1, 13)))),
        y=alt.Y('recaudacion:Q', 
                title='Variación Interanual de Recaudación', 
                axis=alt.Axis(grid=True),
                scale=alt.Scale(domain=[y_min - y_padding, y_max + y_padding])),
        color=alt.value('#1f77b4'),
        tooltip=[
            alt.Tooltip('anio:O', title='Año'),
            alt.Tooltip('mes:O', title='Mes'),
            alt.Tooltip('recaudacion:Q', title='Recaudación', format=',')
        ]
    )
    .transform_filter(alt.datum.anio == param_anio)
    .add_params(param_anio)
    .properties(
        width=700,
        height=400,
        title='Estacionalidad de la Recaudación por Año'
    )
)

# Gráficos de IPC
df_montos = pd.read_csv('data/montos.csv')
df_montos['anio'] = pd.to_numeric(df_montos['anio'], errors='coerce').astype('Int64')
df_montos['mes'] = pd.to_numeric(df_montos['mes'], errors='coerce').astype('Int64')
df_montos = df_montos.set_index(['anio', 'mes'])

# Preparar IPCs como en el código original
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
    poblacion_df["mapeo_ipc"] = poblacion_df["provincia"].apply(lambda p: p.strip()).apply(match_provincias_ipc)
    return poblacion_df

def filtrar_provincias(poblacion_df):
    return poblacion_df[poblacion_df['mapeo_ipc'].notna()]

def agrupar_por_mapeo_ipc(poblacion_df):
    return poblacion_df.groupby('mapeo_ipc').sum()

def dejar_poblacion_df_bonito(poblacion_df):
    poblacion_df = poblacion_df.drop(columns=['provincia'])
    poblacion_df['valor'] = poblacion_df['valor'].astype('int64')
    return poblacion_df

poblacion_df = pd.read_csv('data/poblacion_2010.csv')
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

df_montos = (
    df_montos
    .pipe(preparar_ipcs)
    .pipe(juntar_ipc_provincias)
)

# Filtrar desde 2008
df_montos = df_montos[df_montos.index.get_level_values('anio') >= 2008]

# Gráfico 1: IPC de cada provincia vs promedio provincial
ipc_cols = poblacion_df.index.to_list()
df_ipc_prov = df_montos[ipc_cols + ['ipc_provincias']].reset_index()
df_ipc_prov['periodo'] = pd.to_datetime(df_ipc_prov['anio'].astype(str) + '-' + df_ipc_prov['mes'].astype(str).str.zfill(2))

df_ipc_prov_long = df_ipc_prov.melt(
    id_vars=['anio', 'mes', 'periodo'],
    value_vars=ipc_cols + ['ipc_provincias'],
    var_name='variable',
    value_name='valor'
)

rename_map = {
    'ipc_gba': 'GBA',
    'ipc_cordoba': 'Córdoba',
    'ipc_tucuman': 'Tucumán',
    'ipc_santafe': 'Santa Fe',
    'ipc_mendoza': 'Mendoza',
    'ipc_provincias': 'IPC Provincial Promedio'
}
df_ipc_prov_long['variable'] = df_ipc_prov_long['variable'].map(rename_map)

chart_ipc_prov = alt.Chart(df_ipc_prov_long).mark_line(point=False).encode(
    x=alt.X('periodo:T', title='Año', axis=alt.Axis(format='%Y-%m')),
    y=alt.Y('valor:Q', title='Valor IPC'),
    color=alt.Color('variable:N', legend=alt.Legend(title='Serie')),
    strokeWidth=alt.condition(
        alt.datum.variable == 'IPC Provincial Promedio',
        alt.value(3),
        alt.value(1)
    ),
    tooltip=['anio:O', 'mes:O', 'variable:N', alt.Tooltip('valor:Q', format='.4f')]
).properties(
    width=800,
    height=400,
    title='IPC de cada provincia VS IPC provincial promedio'
).interactive()

# Gráfico 2: IPC Argentina vs promedio provincial
if 'ipc_argentina' in df_montos.columns:
    df_ipc_arg = df_montos[['ipc_argentina', 'ipc_provincias']].reset_index()
    df_ipc_arg['periodo'] = pd.to_datetime(df_ipc_arg['anio'].astype(str) + '-' + df_ipc_arg['mes'].astype(str).str.zfill(2))
    
    # Calcular error porcentual
    df_ipc_comparison = df_ipc_arg[['ipc_argentina', 'ipc_provincias']].dropna()
    df_ipc_comparison['dif'] = df_ipc_comparison['ipc_provincias'] - df_ipc_comparison['ipc_argentina']
    df_ipc_comparison['dif_pct'] = df_ipc_comparison['dif'] / df_ipc_comparison['ipc_argentina'] * 100
    error_promedio = df_ipc_comparison['dif_pct'].mean()
    
    df_ipc_arg_long = df_ipc_arg.melt(
        id_vars=['anio', 'mes', 'periodo'],
        value_vars=['ipc_argentina', 'ipc_provincias'],
        var_name='variable',
        value_name='valor'
    )
    
    df_ipc_arg_long['variable'] = df_ipc_arg_long['variable'].map({
        'ipc_argentina': 'IPC Argentina',
        'ipc_provincias': 'IPC Provincial Promedio'
    })
    
    chart_ipc_arg = alt.Chart(df_ipc_arg_long).mark_line(point=False).encode(
        x=alt.X('periodo:T', title='Año', axis=alt.Axis(format='%Y-%m')),
        y=alt.Y('valor:Q', title='Valor IPC'),
        color=alt.Color('variable:N', legend=alt.Legend(title='Serie')),
        tooltip=['anio:O', 'mes:O', 'variable:N', alt.Tooltip('valor:Q', format='.4f')]
    ).properties(
        width=800,
        height=400,
        title=f'IPC de Argentina VS IPC provincial promedio (Error promedio: {error_promedio:.4f}%)'
    ).interactive()

# Mostrar gráficos
st.title("Análisis de Recaudación e Inflación")

st.subheader("Variación Interanual")
st.altair_chart(chart, use_container_width=True)

# Nuevo gráfico: Recaudación Nominal
# Cargar datos de recaudación nominal del CSV original
df_recaudacion_nominal = pd.read_csv('data/montos.csv')
df_recaudacion_nominal['anio'] = pd.to_numeric(df_recaudacion_nominal['anio'], errors='coerce').astype('Int64')
df_recaudacion_nominal['mes'] = pd.to_numeric(df_recaudacion_nominal['mes'], errors='coerce').astype('Int64')

# Filtrar desde 2008
df_recaudacion_nominal = df_recaudacion_nominal[df_recaudacion_nominal['anio'] >= 2008]

# Ajustar recaudación de miles a millones para años anteriores a 2021
df_recaudacion_nominal['recaudacion'] = df_recaudacion_nominal.apply(
    lambda row: row['recaudacion'] / 1000 if row['anio'] < 2021 else row['recaudacion'],
    axis=1
)

# Reindexar después del filtrado
df_recaudacion_nominal = df_recaudacion_nominal.reset_index(drop=True)
df_recaudacion_nominal['index_num'] = range(len(df_recaudacion_nominal))

# Preparar datos reales
real_nominal = df_recaudacion_nominal[['anio', 'mes', 'index_num', 'recaudacion']].copy()
real_nominal['Tipo'] = 'Real'
real_nominal = real_nominal.rename(columns={'recaudacion': 'Recaudacion'})

# Calcular recaudación predicha nominal a partir de las variaciones
# Primero, crear un mapeo de index_num para los datos filtrados
df_full_filtered = df.merge(
    df_recaudacion_nominal[['anio', 'mes', 'index_num']],
    on=['anio', 'mes'],
    how='inner'
)

predicho_nominal = df_full_filtered.merge(
    test_predictions[['anio', 'mes']], 
    on=['anio', 'mes'], 
    how='inner'
)

# Calcular recaudación predicha: recaudacion_año_anterior * (1 + variacion_predicha)
predicho_nominal_list = []
for _, row in predicho_nominal.iterrows():
    anio_anterior = row['anio'] - 1
    mes_actual = row['mes']
    
    recaud_anterior = df_recaudacion_nominal[
        (df_recaudacion_nominal['anio'] == anio_anterior) & 
        (df_recaudacion_nominal['mes'] == mes_actual)
    ]
    
    if not recaud_anterior.empty:
        recaud_base = recaud_anterior['recaudacion'].values[0]
        # Obtener la variación predicha del modelo
        variacion_predicha = y_pred_full[df[(df['anio'] == row['anio']) & (df['mes'] == mes_actual)].index[0]]
        recaud_predicha = recaud_base * (1 + variacion_predicha)
        
        predicho_nominal_list.append({
            'anio': row['anio'],
            'mes': row['mes'],
            'index_num': row['index_num'],
            'Recaudacion': recaud_predicha,
            'Tipo': 'Predicho'
        })

predicho_nominal_df = pd.DataFrame(predicho_nominal_list)

# Crear gráfico de recaudación nominal
year_positions_nom = df_recaudacion_nominal.groupby('anio')['index_num'].first().reset_index()

line_real_nominal = alt.Chart(real_nominal).mark_line(point=False).encode(
    x=alt.X('index_num:Q',
            axis=alt.Axis(
                title='Año',
                values=year_positions_nom['index_num'].tolist(),
                labelExpr=f"datum.value == {year_positions_nom['index_num'].iloc[0]} ? '{year_positions_nom['anio'].iloc[0]}' : " + 
                         ' : '.join([f"datum.value == {row['index_num']} ? '{row['anio']}'" 
                                   for _, row in year_positions_nom.iloc[1:].iterrows()]) + " : ''",
                labelAngle=-90,
                grid=True
            )),
    y=alt.Y('Recaudacion:Q', 
            title='Recaudación (millones ARS)',
            axis=alt.Axis(grid=True, format=',.0f')).scale(type="log"),
    color=alt.Color('Tipo:N', legend=alt.Legend(title=''), scale=alt.Scale(domain=['Real', 'Predicho'], range=['#1f77b4', '#ff7f0e'])),
    tooltip=[
        alt.Tooltip('anio:O', title='Año'),
        alt.Tooltip('mes:O', title='Mes'),
        alt.Tooltip('Recaudacion:Q', title='Recaudación', format=',.2f')
    ]
)

points_predicho_nominal = alt.Chart(predicho_nominal_df).mark_point(size=60, filled=True).encode(
    x='index_num:Q',
    y='Recaudacion:Q',
    color=alt.Color('Tipo:N', legend=None, scale=alt.Scale(domain=['Real', 'Predicho'], range=['#1f77b4', '#ff7f0e'])),
    tooltip=[
        alt.Tooltip('anio:O', title='Año'),
        alt.Tooltip('mes:O', title='Mes'),
        alt.Tooltip('Recaudacion:Q', title='Predicción', format=',.2f')
    ]
)

chart_nominal = (line_real_nominal + points_predicho_nominal).properties(
    width=800,
    height=400,
    title='Recaudación Nominal Real vs. Predicha (escala logarítmica)'
).interactive()

st.subheader("Recaudación Nominal")
st.altair_chart(chart_nominal, use_container_width=True)

st.subheader("Estacionalidad por Año")
st.altair_chart(chart2, use_container_width=True)

st.subheader("IPC por Provincia")
st.altair_chart(chart_ipc_prov, use_container_width=True)

if 'ipc_argentina' in df_montos.columns:
    st.subheader("IPC Argentina vs. Provincias")
    st.altair_chart(chart_ipc_arg, use_container_width=True)