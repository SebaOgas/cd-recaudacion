import streamlit as st

file_path = 'README.md'

with open(file_path, 'r', encoding="utf-8") as file:
    contenido = file.read()

st.markdown(contenido)