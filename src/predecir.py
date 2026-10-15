"""Carga el modelo entrenado y predice el país de una pregunta."""
import json
import sys

import numpy as np
from tensorflow import keras

from datos import RAIZ

CARPETA_MODELO = RAIZ / "modelo"


def cargar_modelo():
    """Devuelve el modelo entrenado y la lista de países."""
    modelo = keras.models.load_model(CARPETA_MODELO / "modelo.keras")
    paises = json.loads((CARPETA_MODELO / "paises.json").read_text())
    return modelo, paises


def predecir(modelo, paises, pregunta):
    """Devuelve una lista de mayor a menor."""
    probabilidades = modelo.predict(np.array([pregunta], dtype=object), verbose=0)[0]
    resultado = list(zip(paises, probabilidades))
    return sorted(resultado, key=lambda par: par[1], reverse=True)


if __name__ == "__main__":
    pregunta = sys.argv[1] if len(sys.argv) > 1 else "¿Qué es la cueca?"
    modelo, paises = cargar_modelo()
    for pais, prob in predecir(modelo, paises, pregunta)[:5]:
        print(f"{pais:<22} {prob:.1%}")
