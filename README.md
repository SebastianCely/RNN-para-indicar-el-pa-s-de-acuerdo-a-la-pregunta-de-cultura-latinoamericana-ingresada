# 🌎 Clasificador de países con GRU

Proyecto de Deep Learning con una **red recurrente (GRU)** para clasificar texto:
dada una pregunta de cultura general latinoamericana, el modelo predice **a qué país se refiere**
(18 países). Incluye una aplicación en **Streamlit** para probarlo.

## Estructura del repositorio

```
clasificador_paises/
├── data/dataset.xlsx                  # Dataset (105.000 preguntas y respuestas)
├── notebook_experimentacion.ipynb     # Notebook: EDA, entrenamiento, métricas y predicciones
├── src/
│   ├── datos.py                       # Cargar, limpiar y dividir los datos
│   ├── modelo.py                      # Construcción de la red (GRU / LSTM)
│   ├── entrenar.py                    # Entrena y guarda el modelo
│   └── predecir.py                    # Carga el modelo y predice
├── app.py                             # Aplicación Streamlit
├── modelo/                            # Modelo entrenado (modelo.keras, paises.json, metricas.json)
├── imagenes/                          # Gráficas generadas por el notebook
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

Requiere Python 3.10 o superior.

```bash
# 1. (Opcional) crear un entorno virtual
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Abrir la aplicación (el modelo ya viene entrenado en modelo/)
streamlit run app.py
```

Otras opciones:

```bash
# Volver a entrenar el modelo (tarda unos 2-3 minutos en CPU)
python src/entrenar.py            # con GRU
python src/entrenar.py lstm       # con LSTM

# Predecir desde la terminal
python src/predecir.py "¿En qué comuna se encuentra la bahía de Cucao?"

# Abrir el notebook
jupyter notebook notebook_experimentacion.ipynb
```

## El dataset

Preguntas y respuestas en español sobre cultura de 18 países latinoamericanos.

| Columna | Descripción |
|---|---|
| `Entity` | Tema de la pregunta (una especie, un lugar, un plato...) |
| `Country` | País (**variable a predecir**) |
| `Category` | geografía, fauna, flora, tradición, plato, personaje u objeto |
| `Difficulty` | FÁCIL, INTERMEDIA o DIFÍCIL |
| `Question` | Texto de la pregunta (**entrada del modelo**) |
| `Answer` | Respuesta |

Puntos importantes encontrados en el análisis exploratorio:
- Clases **desbalanceadas**: Chile tiene ~19.000 preguntas y Nicaragua ~560.
- Preguntas cortas (unas 10 palabras en promedio).
- Cada entidad aparece en 3 filas (una por dificultad).

## Metodología

1. **Limpieza:** se eliminan filas sin pregunta o país, espacios sobrantes y preguntas repetidas.
2. **División 70 / 15 / 15** (entrenamiento / validación / prueba) **por entidad**, para que la misma
   entidad no aparezca en entrenamiento y en prueba (evita fuga de datos).
3. **Preprocesamiento de texto:** una capa `TextVectorization` (minúsculas, sin signos, vocabulario de 20.000
   palabras, largo fijo de 30) incluida dentro del modelo.
4. **Red neuronal:**
   `Embedding(32) → Dropout → Bidirectional(GRU 64) → Dropout → Dense(18, softmax)`
5. **Entrenamiento:** Adam, pesos por clase para compensar el desbalance y *early stopping* sobre la pérdida
   de validación.

## Resultados (conjunto de prueba)

| Modelo | Accuracy | F1 macro |
|---|---|---|
| Referencia (siempre "Chile") | 0.179 | — |
| **GRU bidireccional** | **0.429** | **0.362** |
| LSTM bidireccional | 0.422 | 0.348 |

El modelo supera con claridad la línea base, pero **no llega a valores altos** y es normal:
muchas preguntas (por ejemplo «¿Qué tipo de especie es *Calomys sorellus*?») no contienen ninguna pista del país.
Acierta más cuando aparecen nombres propios o términos administrativos ("comuna", "cantón", "departamento").

Las gráficas del entrenamiento, la matriz de confusión y las 15 predicciones de prueba están en el notebook
y en la carpeta `imagenes/`.

## Aplicación Streamlit

`app.py` carga el modelo guardado (`modelo/modelo.keras`). El usuario escribe una pregunta y la app muestra el
país más probable junto con una gráfica de los 5 países con mayor probabilidad.

## Posibles mejoras

- Usar también la respuesta (`Answer`) como entrada.
- Probar embeddings preentrenados en español.
- Probar un modelo tipo transformer (por ejemplo BETO).
