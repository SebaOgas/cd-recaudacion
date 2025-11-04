import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import sys
from sklearn.base import BaseEstimator, TransformerMixin

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

# Realizar predicciones
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
plot_df_full['index_num'] = range(len(plot_df_full))

# Transformar a formato largo
plot_df_long = plot_df_full.melt(
    id_vars=['index', 'anio', 'index_num'],
    value_vars=['Real', 'Predicho'],
    var_name='Tipo',
    value_name='Recaudacion'
)

# Crear datos para las marcas de año en el eje X
year_positions = df.groupby('anio').apply(lambda x: x.index[0]).reset_index()
year_positions.columns = ['anio', 'position']
year_positions['index_num'] = [plot_df_full[plot_df_full.index == pos]['index_num'].values[0] 
                                 for pos in year_positions['position']]

chart = alt.Chart(plot_df_long).mark_line(point=True).encode(
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
            axis=alt.Axis(grid=True)),
    color=alt.Color('Tipo:N', 
                    legend=alt.Legend(title=''),
                    scale=alt.Scale(domain=['Real', 'Predicho'], range=['#1f77b4', '#ff7f0e'])),
    tooltip=[
        alt.Tooltip('anio:O', title='Año'),
        alt.Tooltip('Tipo:N', title='Tipo'),
        alt.Tooltip('Recaudacion:Q', title='Recaudación', format='.2f')
    ]
).properties(
    width=800,
    height=400,
    title='Variación Interanual de Recaudación Real vs. Predicha'
).interactive()

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

chart2 = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x=alt.X('mes:O', title='Mes', axis=alt.Axis(values=list(range(1, 13)), grid=True)),
        y=alt.Y('recaudacion:Q', title='Recaudación', axis=alt.Axis(grid=True)),
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

st.subheader("Variación Interanual")
st.altair_chart(chart, use_container_width=True)

st.subheader("Estacionalidad por Año")
st.altair_chart(chart2, use_container_width=True)