import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Para poder importar el código de la carpeta src/
sys.path.insert(0, str(Path(__file__).parent / "src"))
from predecir import cargar_modelo, predecir

st.set_page_config(page_title="¿De qué país es?", page_icon="🌎")


@st.cache_resource  # el modelo se carga una sola vez
def obtener_modelo():
    return cargar_modelo()


modelo, paises = obtener_modelo()

st.title("🌎 ¿De qué país es la pregunta?")
st.write(
    "Escribe una pregunta de cultura general latinoamericana (geografía, fauna, "
    "flora, tradiciones, platos, personajes...) y la red neuronal GRU "
    "intentará adivinar a qué país se refiere."
)

# Ejemplos de prueba
ejemplos = [
    "¿Qué es la cabecera cantonal del Cantón Marcabelí?",
    "¿Cuál es el plato típico de Oaxaca llamado mole negro?",
    "¿Quién fue el cantante Maluma?",
    "¿En qué comuna se encuentra la bahía de Cucao?",
]
ejemplo = st.selectbox("Prueba con un ejemplo (opcional):", ["—"] + ejemplos)

pregunta = st.text_area(
    "Tu pregunta:",
    value="" if ejemplo == "—" else ejemplo,
    placeholder="Ej: ¿Cuál es la capital del departamento de Chuquisaca?",
)

if st.button("Predecir país", type="primary"):
    if not pregunta.strip():
        st.warning("Escribe una pregunta primero.")
    else:
        resultado = predecir(modelo, paises, pregunta)
        mejor_pais, mejor_prob = resultado[0]

        st.success(f"País más probable: **{mejor_pais.title()}** ({mejor_prob:.1%})")

        # Gráfica con los 5 países más probables
        top5 = pd.DataFrame(resultado[:5], columns=["País", "Probabilidad"])
        st.bar_chart(top5.set_index("País"))

        if mejor_prob < 0.5:
            st.info(
                "El modelo no está muy seguro: muchas preguntas "
                "(por ejemplo sobre especies) no dan pistas del país."
            )
