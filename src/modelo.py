"""Construcción de la red neuronal (GRU o LSTM)."""
from tensorflow import keras
from tensorflow.keras import layers

MAX_PALABRAS = 20000   # tamaño del vocabulario
LARGO_TEXTO = 30       # cada pregunta se ajusta a 30 palabras


def crear_vectorizador(textos_entrenamiento):
    #Convertir texto en números
    vectorizador = layers.TextVectorization(
        max_tokens=MAX_PALABRAS,
        output_sequence_length=LARGO_TEXTO,
    )
    vectorizador.adapt(textos_entrenamiento)  # aprende el vocabulario
    return vectorizador


def crear_modelo(vectorizador, n_paises, tipo="gru"):
    """Crea el modelo. tipo puede ser "gru" o "lstm"."""
    capa_recurrente = {"gru": layers.GRU, "lstm": layers.LSTM}[tipo]

    modelo = keras.Sequential([
        keras.Input(shape=(), dtype="string"),        
        vectorizador,                                  
        layers.Embedding(MAX_PALABRAS, 32, mask_zero=True),  
        layers.Dropout(0.5),                           
        layers.Bidirectional(capa_recurrente(64)),     
        layers.Dropout(0.5),                           
        layers.Dense(n_paises, activation="softmax"),  
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo
