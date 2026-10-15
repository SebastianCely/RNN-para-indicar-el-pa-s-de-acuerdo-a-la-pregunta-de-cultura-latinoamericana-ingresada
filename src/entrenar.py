"""Entrena el modelo y lo guarda en la carpeta modelo/."""
import json
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from datos import RAIZ, SEMILLA, cargar_datos, dividir_datos
from modelo import crear_modelo, crear_vectorizador

CARPETA_MODELO = RAIZ / "modelo"
EPOCAS = 15


def main(tipo="gru"):
    keras.utils.set_random_seed(SEMILLA)

    # Datos
    df = cargar_datos()
    train, val, test = dividir_datos(df)
    print(f"Entrenamiento: {len(train)} | Validación: {len(val)} | Prueba: {len(test)}")

    # Cada país pasa a ser un número
    paises = sorted(df["Country"].unique())
    pais_a_numero = {pais: i for i, pais in enumerate(paises)}
    y_train = train["Country"].map(pais_a_numero).to_numpy()
    y_val = val["Country"].map(pais_a_numero).to_numpy()
    y_test = test["Country"].map(pais_a_numero).to_numpy()

    # Textos
    x_train = train["Question"].to_numpy(dtype=object)
    x_val = val["Question"].to_numpy(dtype=object)
    x_test = test["Question"].to_numpy(dtype=object)

    # Los países con pocos datos pesan más
    pesos = compute_class_weight("balanced", classes=np.arange(len(paises)), y=y_train)
    pesos = dict(enumerate(pesos))

    # Modelo
    vectorizador = crear_vectorizador(x_train)
    modelo = crear_modelo(vectorizador, len(paises), tipo)
    modelo.summary()

    # Entrenamiento 
    parar_temprano = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    historial = modelo.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCAS,
        batch_size=128,
        class_weight=pesos,
        callbacks=[parar_temprano],
    )

    # Evaluación con datos que el modelo nunca vio
    predicciones = modelo.predict(x_test, batch_size=256).argmax(axis=1)
    accuracy = accuracy_score(y_test, predicciones)
    f1_macro = f1_score(y_test, predicciones, average="macro")
    print(f"\nAccuracy en prueba: {accuracy:.3f}")
    print(f"F1 macro en prueba: {f1_macro:.3f}")

    # 8. Guardar todo
    CARPETA_MODELO.mkdir(exist_ok=True)
    modelo.save(CARPETA_MODELO / "modelo.keras")
    (CARPETA_MODELO / "paises.json").write_text(json.dumps(paises, ensure_ascii=False))
    (CARPETA_MODELO / "metricas.json").write_text(json.dumps(
        {"tipo": tipo, "accuracy": accuracy, "f1_macro": f1_macro}, indent=2))
    print("Modelo guardado en", CARPETA_MODELO)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gru")
