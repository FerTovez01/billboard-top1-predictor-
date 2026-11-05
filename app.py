import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Billboard Lab", page_icon="🎵")

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(ellipse at top, #24113e 0%, #10091d 45%, #08060f 80%);
        color: #f8f3ff;
    }

    [data-testid="stHeader"] {
        background: #10091d;
    }

    [data-testid="stHeader"] button,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] label p,
    [data-testid="stMetricValue"] {
        color: #f8f3ff;
    }

    [data-testid="stCaptionContainer"] p {
        color: #d8c7f2;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #171025;
        border-color: #513278;
        border-radius: 18px;
    }

    [data-testid="stForm"] {
        background: linear-gradient(135deg, #3e1679, #210d46);
        border: 1px solid #bc80ff;
        border-radius: 18px;
        box-shadow: 0 0 24px #913bff55, inset 0 0 20px #913bff22;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] [data-baseweb="input"],
    [data-testid="stNumberInput"] input,
    [data-testid="stNumberInput"] button {
        background: #1b0c32;
        color: #f8f3ff;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border-color: #a16aff;
    }

    [data-testid="stSelectbox"] svg {
        color: #dabaff;
        fill: currentColor;
    }

    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #7024d9, #8c30e6);
        border: 1px solid #bc80ff;
        color: white;
        font-weight: 700;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background: #6124bb;
        border-color: #e4c8ff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎵 ¿Llegará esta canción al #1 de Billboard?")
st.caption(
    "Desarrollado por Andrea Fernanda Tovez Corrales · UNAH Campus Comayagua"
)


@st.cache_resource
def cargar_archivos():
    carpeta = Path(__file__).resolve().parent

    with (carpeta / "modelo.pkl").open("rb") as archivo:
        modelo = pickle.load(archivo)

    with (carpeta / "config_modelo.json").open(encoding="utf-8") as archivo:
        configuracion = json.load(archivo)

    return modelo, configuracion


try:
    arbol, config = cargar_archivos()
except FileNotFoundError as error:
    st.error(f"Falta {Path(error.filename).name} junto a app.py.")
    st.stop()

opciones = config["opciones"]


def indice_inicial(campo):
    """Selecciona el valor enviado por el portal PHP, si es válido."""
    valores = opciones[campo]
    valor = st.query_params.get(campo)
    return valores.index(valor) if valor in valores else 0


duracion_maxima = 2**53 - 1

try:
    duracion_inicial = int(st.query_params.get("duration", "210"))
except ValueError:
    duracion_inicial = 210

if not 1 <= duracion_inicial <= duracion_maxima:
    duracion_inicial = 210

with st.form("cancion"):
    st.subheader("🎛️ Configura tu canción")
    st.caption("Elige las características y pulsa «Predecir canción».")

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
            max_value=duracion_maxima,
            value=duracion_inicial,
            step=1,
        )

    predecir = st.form_submit_button(
        "🎵 Predecir canción",
        type="primary",
        width="stretch",
    )

if not predecir:
    st.info("Pulsa «Predecir canción» para ver el resultado.")
    st.stop()

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
hoja = int(arbol.apply(entrada)[0])

with st.container(border=True):
    st.subheader("📊 Resultado de la predicción")
    st.caption(
        f"Escenario evaluado · {artista} · {genero} · {tempo} · {duracion} s"
    )

    if prediccion == 1:
        st.success("El modelo predice que la canción llegaría al #1.")
    else:
        st.info("El modelo predice que la canción no llegaría al #1.")

    st.metric("Probabilidad estimada de Top 1", f"{puntaje_top1:.1%}")
    st.progress(puntaje_top1)
    st.caption(
        f"Hoja del árbol: {hoja}. El porcentaje representa la proporción "
        "de casos Top 1 en esa hoja."
    )
  
