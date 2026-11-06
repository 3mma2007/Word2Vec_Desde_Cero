# Word2Vec desde cero (Skip-gram con Negative Sampling)

Implementación de Word2Vec en **NumPy**, escrita desde cero. El modelo aprende embeddings de palabras sobre el corpus **text8** (extracto de Wikipedia en inglés). TensorFlow se usa únicamente para generar los pares skip-gram y muestrear las palabras negativas; el forward, la loss, el backpropagation y la actualización de pesos están escritos paso a paso.

## Contenido

1. [Resumen del proyecto](#resumen-del-proyecto)
2. [Arquitectura](#arquitectura)
3. [Cantidad de pesos y cómo se calcula](#cantidad-de-pesos-y-cómo-se-calcula)
4. [Las matemáticas del modelo](#las-matemáticas-del-modelo)
5. [Pipeline de datos](#pipeline-de-datos)
6. [Hiperparámetros](#hiperparámetros)
7. [Resultados](#resultados)
8. [Cómo usarlo](#cómo-usarlo)
9. [Limitaciones y mejoras posibles](#limitaciones-y-mejoras-posibles)

---

## Resumen del proyecto

| Elemento | Valor |
|---|---|
| Modelo | Skip-gram con negative sampling |
| Framework del modelo | NumPy |
| Corpus | text8, primeras 10 000 000 de palabras |
| Vocabulario | 27 777 (27 776 palabras con al menos 15 apariciones + `<UNK>`) |
| Dimensión del embedding | 128 |
| Ventana de contexto | 2 |
| Negativas por par positivo | 4 |
| Pares de entrenamiento | 36 855 552 |
| Pesos del modelo | 7 110 912 (`2 × 27 777 × 128`) |

---

## Arquitectura

Skip-gram recibe una palabra central y aprende a distinguir su contexto real de palabras al azar.

```
palabra central ──► W1[centro] = h ──┬─► h · W2[contexto]    ─► σ(·) → 1   (positivo)
   (índice)           (N = 128)      ├─► h · W2[negativa 1]  ─► σ(·) → 0
                                     ├─► h · W2[negativa 2]  ─► σ(·) → 0
                                     ├─► h · W2[negativa 3]  ─► σ(·) → 0
                                     └─► h · W2[negativa 4]  ─► σ(·) → 0
```

- `W1` (V × N): matriz de **embeddings**. La fila `i` es el vector de la palabra `i` como palabra central. Es el resultado final del proyecto.
- `W2` (V × N): matriz de **contexto**. La fila `j` es el vector de la palabra `j` cuando actúa como contexto o como negativa.
- No hay bias ni función de activación en la capa oculta.
- El contexto **no entra a la red**: solo se usa para leer una fila de `W2` y calcular el error.

---

## Cantidad de pesos y cómo se calcula

El modelo tiene exactamente **dos matrices de pesos** y ningún bias.

```
W1 : V × N
W2 : V × N
```

donde `V` es el tamaño del vocabulario (`vocab_size`) y `N = 128` es la dimensión del embedding.

### Fórmula

```
pesos(W1) = V × N
pesos(W2) = V × N
─────────────────
total     = 2 × V × N  =  2 × V × 128  =  256 × V
```

### Cómo se obtiene cada término

Cada matriz tiene una fila por palabra del vocabulario y una columna por dimensión del embedding. El número de elementos de una matriz es `filas × columnas`, y como las dos matrices tienen la misma forma, el total es el doble de una de ellas.

### Valor concreto

Con `min_count = 15` sobre las 10 M de palabras, el vocabulario quedó en:

```
vocab_size = len(vocab) = 27 777      (27 776 palabras + el token <UNK>)
```

El cálculo paso a paso:

```
W1 = 27 777 × 128 =  3 555 456 pesos
W2 = 27 777 × 128 =  3 555 456 pesos
                     ───────────────
Total              =  7 110 912 pesos
```


| Matriz | Forma | Pesos |
|---|---|---|
| `W1` (embeddings) | (27 777, 128) | 3 555 456 |
| `W2` (contexto) | (27 777, 128) | 3 555 456 |
| **Total** | | **7 110 912** (≈ 7.1 M) |



### Pesos que se tocan en cada paso

Aunque el modelo tiene millones de pesos, cada par de entrenamiento solo modifica unos pocos:

```
filas de W1 que se actualizan : 1       (la palabra central)
filas de W2 que se actualizan : 1 + 4   (contexto real + 4 negativas)

pesos por par  = (1 + 5) × N = 6 × 128 = 768
pesos por lote = batch_size × 6 × N = 1024 × 6 × 128 = 786 432   (como máximo)
```

El máximo se alcanza cuando no hay palabras repetidas dentro del lote. Con palabras repetidas el número de filas distintas es menor, pero sus gradientes se acumulan con `np.add.at`.

Esta es la ventaja de negative sampling frente al softmax completo: con softmax cada paso tocaría **toda** `W2` (V × N pesos), mientras que aquí solo se tocan 5 filas por ejemplo. La cantidad total de pesos es la misma en ambos casos (`2 × V × N`); lo que cambia es el costo por paso.

### Inicialización

| Matriz | Inicialización |
|---|---|
| `W1` | Uniforme en `[-límite, +límite]`, con `límite = sqrt(6 / (V + N))` (Xavier/Glorot) |
| `W2` | Ceros |

Con `W2 = 0`, todos los scores iniciales valen 0, `σ(0) = 0.5` y la loss inicial es `(1 + k ) × ln 2 (k = número de palabras negativas por cada par positivo) = 5 × 0.693 ≈ 3.47`.

---

## Las matemáticas del modelo

Para un lote de `m` pares `(centro, contexto, 4 negativas)`:

### Forward

```
h         = W1[centro]                         (m, N)
pos_vec   = W2[contexto]                       (m, N)
neg_vec   = W2[negativas]                      (m, k, N)

score_pos = suma_N( h * pos_vec )              (m,)     producto punto con el contexto real
score_neg = einsum('mn,mkn->mk', h, neg_vec)   (m, k)   producto punto con cada negativa
```

### Loss (entropía cruzada binaria)

```
L = -log σ(score_pos) - Σ_k log σ(-score_neg)
```

Se calcula de forma estable con `logaddexp`:

```
-log σ(x)  = log(1 + e^(-x))
-log σ(-x) = log(1 + e^( x))
```

### Backpropagation

El gradiente tiene la misma forma que en un MLP: `predicción - objetivo`.

```
d_pos = σ(score_pos) - 1                       (m,)       objetivo 1
d_neg = σ(score_neg)                           (m, k)     objetivo 0

dh      = d_pos * pos_vec + Σ_k d_neg * neg_vec    / m    (m, N)
dW2_pos = d_pos * h                                / m    (m, N)
dW2_neg = d_neg * h                                / m    (m, k, N)
```

### Actualización

```
W1[centro]    -= ln * dh
W2[contexto]  -= ln * dW2_pos
W2[negativas] -= ln * dW2_neg
```

Se usa `np.add.at` en lugar de la asignación directa, porque una misma palabra puede aparecer varias veces en el lote (como centro, contexto o negativa) y sus gradientes deben **sumarse**, no pisarse.

---

## Pipeline de datos

1. **Descarga** de text8 (≈ 17 M de palabras) y recorte a las primeras 10 M.
2. **Vocabulario**: se cuentan las frecuencias y se conservan las palabras con `min_count ≥ 15`. El índice 0 se reserva para `<UNK>`; las palabras raras se mapean a él (≈ 4.19 % del texto).
3. **Pares positivos** `(centro, contexto)` con `tf.keras.preprocessing.sequence.skipgrams`, ventana 2 y `negative_samples=0`. Keras omite el índice 0, así que ningún par contiene `<UNK>`. Resultado: `(36 855 552, 2)`.
4. **Negativas**: 4 por cada par positivo, muestreadas con probabilidad proporcional a `frecuencia^0.75` (distribución unigrama suavizada de Word2Vec). Se excluye `<UNK>` y se vuelve a muestrear si alguna negativa coincide con el contexto real. Resultado: `(36 855 552, 4)`.
5. **Entrenamiento por mini-batches** mezclados al azar en cada época.

Para el muestreo, `tf.random.categorical` recibe los logaritmos de las probabilidades (`tf.math.log(p)`). Como `freq[0] = 0`, `log(0) = -inf`, lo que da probabilidad cero a `<UNK>`.

---

## Hiperparámetros

| Parámetro | Valor |
|---|---|
| `embedd_dim` (N) | 128 |
| `window_size` | 2 |
| `num_ns` (k) | 4 |
| `min_count` | 15 |
| Palabras del corpus | 10 000 000 |
| `epocas` | 8 |
| `ln` (learning rate) | 10 |
| `batch_size` | 1024 |

El learning rate es alto (10) porque los gradientes se dividen entre el tamaño del lote `m`, por lo que el paso efectivo por ejemplo es `ln / m`.

Pasos por época: `36 855 552 / 1024 ≈ 35 992` (35 991 lotes completos y uno de 768 pares).

---

## Resultados

### Loss

La loss empieza cerca de 3.47 y baja a ≈ 1.85 después de 8 épocas:

| Época | Loss (último lote) |
|---|---|
| 0 | 2.0253 |
| 1 | 2.0122 |
| 2 | 1.9505 |
| 3 | 1.8796 |
| 4 | 1.9242 |
| 5 | 1.8917 |
| 6 | 1.9356 |
| 7 | 1.8537 |

El valor impreso es el del último lote de cada época, por eso fluctúa. En negative sampling la loss nunca se acerca a 0, y valores entre 1.5 y 2.5 son normales con `k = 4`.

### Vecinos más cercanos (similitud coseno)

| Palabra | Vecinos |
|---|---|
| `king` | tsar (0.638), emperor (0.635), cleopatra (0.633), prince (0.630), kazimierz (0.627) |
| `france` | spain (0.741), belgium (0.726), hungary (0.716), germany (0.708), galicia (0.682) |
| `water` | sediment (0.662), brackish (0.633), oxygen (0.622), moisture (0.618), salt (0.618) |
| `january` | november (0.897), june (0.890), september (0.882), july (0.872), october (0.868) |
| `one` | gtb (0.772), nine (0.752), poz (0.747), cyg (0.742), cet (0.735) |

El modelo agrupa gobernantes, países, sustancias del campo del agua y meses sin supervisión. Los vecinos de `one` incluyen ruido (`gtb`, `poz`, `cyg`, `cet`): son abreviaturas de catálogos y tablas de Wikipedia que aparecen junto a números. Con la búsqueda filtrada por frecuencia mínima, la lista se limpia y aparecen más números (`nine`, `eight`, `six`, `seven`, `two`).

### Similitud coseno

```python
E = E / np.linalg.norm(E, axis=1, keepdims=True)   # vectores de longitud 1
sim = E @ E[vocab[palabra]]                         # producto punto = coseno del ángulo
```

Un valor de 1 indica la misma dirección, 0 perpendiculares y -1 direcciones opuestas. Es una medida de orden (qué palabra está más cerca), no una probabilidad.

---

## Cómo usarlo

El proyecto está en el notebook `Word2Vec_desde_Cero.ipynb` (pensado para Google Colab). Ejecuta las celdas en orden:

1. Importar librerías.
2. Definir `sigmoid` y la clase `Word2Vec`.
3. Descargar text8 y construir el vocabulario.
4. Generar pares positivos y negativos.
5. Entrenar:

```python
model = Word2Vec(vocab_size=vocab_size, embedd_dim=128)
historial = model.fit(centro, contexto, negative_skip_grams,
                      epocas=8, ln=10, batch_size=1024)
```

6. Consultar vecinos:

```python
print(vecinos("king", model))
```

7. Exportar para el [Embedding Projector](https://projector.tensorflow.org): se generan `vectors.tsv` y `metadata.tsv` (filas alineadas, incluyendo `<UNK>` en la fila 0).

> El vector de `<UNK>` no se entrena: se excluyó de los pares y de las negativas, así que conserva su inicialización aleatoria. En el Projector aparece como un punto sin significado.

**Requisitos**: `numpy`, `tensorflow`, `matplotlib`. La generación de los 36.8 M de pares con Keras ocupa varios GB de RAM, por lo que conviene un entorno con memoria suficiente (por ejemplo, Colab).

---

## Limitaciones y mejoras posibles

- **Subsampling** de palabras muy frecuentes (`the`, `of`, `one`), que reduce pares repetitivos y suele mejorar los vecinos.
- **Ruido de text8**: abreviaturas y restos de citas (`pmid`, `pg`, `gtb`) contaminan algunos vecinos.
- **Más épocas o más corpus**: la loss seguía bajando al terminar la época 8, y text8 completo tiene 17 M de palabras.
- **Negativas fijas**: se generaron una sola vez. Muestrearlas de nuevo en cada lote suele dar embeddings algo mejores a cambio de más cómputo.
- **Rendimiento**: `np.add.at` es lento; para escalar más conviene una implementación vectorizada o en GPU.
- **Evaluación**: aún no se probaron analogías (`man : king :: woman : ?`) ni se compara contra una referencia como `gensim`.

---

## Referencias

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013).
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (2013).
- Corpus text8: <http://mattmahoney.net/dc/textdata.html>
