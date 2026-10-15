"""Funciones para cargar, limpiar y dividir el dataset."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Carpeta raíz
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "dataset.xlsx"
SEMILLA = 42


def cargar_datos(ruta=RUTA_DATOS):
    """Lee el Excel y deja solo filas importantes para el modelo."""
    df = pd.read_excel(ruta)

    # Quitar filas sin pregunta o sin país
    df = df.dropna(subset=["Question", "Country"])

    # Quitar espacios sobrantes
    df["Question"] = df["Question"].str.strip()

    # Quitar preguntas repetidas del mismo país
    df = df.drop_duplicates(subset=["Question", "Country"])

    return df.reset_index(drop=True)


def dividir_datos(df):
    divisor = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEMILLA)
    idx_train, idx_resto = next(divisor.split(df, groups=df["Entity"]))
    train = df.iloc[idx_train]
    resto = df.iloc[idx_resto]

    divisor = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEMILLA)
    idx_val, idx_test = next(divisor.split(resto, groups=resto["Entity"]))
    val = resto.iloc[idx_val]
    test = resto.iloc[idx_test]

    return train, val, test
