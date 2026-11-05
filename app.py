import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.title("🎵 ¿Llegará esta canción al #1 de Billboard?")
st.caption("Desarrollado por Andrea Fernanda Tovez Corrales · UNAH Campus Comayagua")

carpeta = Path(__file__).resolve().parent

with (carpeta / "modelo.pkl").open("rb") as archivo:
    arbol = pickle.load(archivo)

with (carpeta / "config_modelo.json").open(encoding="utf-8") as archivo:
    config = json.load(archivo)

opciones = config["opciones"]

# Los escenarios del portal PHP pueden precargar el formulario.
def indice_inicial(campo):
    valores = opciones[campo]
    valor = st.query_params.get(campo)
    return valores.index(valor) if valor in valores else 0

try:
    duracion_inicial = int(st.query_params.get("duration", "210"))
except ValueError:
    duracion_inicial = 210

if not 1 <= duracion_inicial <= 2**53 - 1:
    duracion_inicial = 210

with st.container(border=True):
    st.subheader("🎛️ Configura tu canción")
    st.caption("Cambia cualquier opción para actualizar la predicción.")
    izquierda, derecha = st.columns(2)

    with izquierda:
        artista = st.selectbox(
            "Tipo de artista",
            opciones["artist_type"],
            index=indice_inicial("artist_type"),
        )
        genero = st.selectbox(
            "Género musical",
            opciones["genre"],
            index=indice_inicial("genre"),
        )

    with derecha:
        tempo = st.selectbox(
            "Tempo",
            opciones["tempo"],
            index=indice_inicial("tempo"),
        )
        duracion = st.number_input(
            "Duración en segundos",
            min_value=1,
            value=duracion_inicial,
            step=1,
        )

cancion = pd.DataFrame([{
    "artist_type": artista,
    "tempo": tempo,
    "genre": genero,
    "durationSeg": duracion,
}])

entrada = pd.get_dummies(cancion).reindex(
    columns=config["columnas_modelo"],
    fill_value=0,
)

prediccion = arbol.predict(entrada)[0]
probabilidades = dict(zip(arbol.classes_, arbol.predict_proba(entrada)[0]))
puntaje_top1 = float(probabilidades.get(1, 0.0))

with st.container(border=True):
    st.subheader("📊 Predicción en vivo")

    if prediccion == 1:
        st.success("El modelo predice que la canción llegaría al #1.")
    else:
        st.info("El modelo predice que la canción no llegaría al #1.")

    st.metric("Probabilidad estimada de Top 1", f"{puntaje_top1:.1%}")
    st.progress(puntaje_top1)
    st.caption(
        "La estimación se basa en la proporción de casos Top 1 "
        "en la hoja del árbol a la que llega esta canción."
    )
