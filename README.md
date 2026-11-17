# Detección de noticias falsas

Clasificador de noticias falsas y verdaderas con procesamiento de lenguaje natural y regresión logística, hecho con scikit-learn.

Es un ejercicio de un curso de Machine Learning.

## Resultados

| Notebook | Entrenamiento | Test | Accuracy |
|---|---|---|---|
| [logi_regresion.ipynb](logi_regresion.ipynb) | 1.000 noticias | 500 noticias | 95,0 % |
| [df_grande.ipynb](df_grande.ipynb) | 40.000 noticias | 2.000 noticias | 98,7 % |

## Cómo funciona

1. **Carga de datos.** Se unen `True.csv` y `Fake.csv` y se añade la etiqueta `REAL` o `FAKE`.
2. **Limpieza del texto**, con la función `preprocesar_texto()`:
   - quitar el prefijo de agencia (`WASHINGTON (Reuters) -`), para que el modelo no aprenda a reconocer la fuente en lugar del contenido
   - quitar etiquetas HTML (BeautifulSoup) y URLs
   - pasar a minúsculas y quitar signos de puntuación
   - tokenizar, eliminar *stopwords* y aplicar *stemming* (NLTK)
3. **Vectorización** con `CountVectorizer` (bolsa de palabras).
4. **Entrenamiento** de un modelo `LogisticRegression`.
5. **Evaluación** con *accuracy* sobre noticias que el modelo no ha visto.

En `logi_regresion.ipynb` se explica cada paso con ejemplos. `df_grande.ipynb` aplica el mismo proceso a casi todo el dataset y prueba el modelo con noticias nuevas, mostrando la probabilidad de cada clase.

## Dataset

Los datos no se incluyen en el repositorio por su tamaño (unos 110 MB). Se pueden descargar de Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

Hay que guardar los archivos así:

```
datasets/
├── Fake.csv
└── True.csv
```

## Cómo ejecutarlo

Requiere Python 3.

```bash
pip install -r requirements.txt
jupyter notebook
```

Los recursos de NLTK (`punkt`, `punkt_tab` y `stopwords`) se descargan automáticamente al ejecutar los notebooks.
