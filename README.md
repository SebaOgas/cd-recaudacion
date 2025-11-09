# Predicción de Recaudación de ARCA

Este proyecto desarrollado con streamlit demuestra los hallazgos realizados por los alumnos y el modelo entrenado para la predicción de la recaudación de ARCA.

Alumnos:
* Duran, Tatiana
* Espeche, Marcos
* Ogás, Sebastián

Cátedra: Ciencia de Datos

Año: 2025

El proyecto puede verse en ejecución en [recaudacion.streamlit.app](https://recaudacion.streamlit.app).

Repositorio de [GitHub](https://github.com/SebaOgas/cd-recaudacion).

## Funcionamiento

A partir de variables macroeconómicas desde 2008, se entrenó un modelo de Regresión de Bosque Aleatorio para predecir la recaudación.

Dados los efectos de la inflación y la estacionalidad en la economía, las entradas usadas por el modelo son las variaciones interanuales de las variables económicas, así como el monto de la recaudación 12 meses antes del periodo a predecir.

## Comandos útiles

Para ejecución local, clonar repositorio y:

```
pip install -r requirements.txt
streamlit run ./Inicio.py
```