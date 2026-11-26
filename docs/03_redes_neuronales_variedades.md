# 3. Redes neuronales sobre variedades

En los apartados anteriores se ha visto que cada ensayo de EEG se resume en una matriz de
covarianza espacial $C \in \mathcal{S}^{++}_{22}$ (ver también [`eeg_2_spd.md`](eeg_2_spd.md)), y
que el conjunto $\mathcal{S}^{++}_n$ de matrices simétricas definidas positivas (SPD) es una
variedad riemanniana con su propia noción de distancia, de geodésica y de media. Este apartado
explica cómo construir una **red neuronal** cuyas capas reciben y devuelven matrices SPD, sin
salir nunca de la variedad.

Nos centramos en **SPDNet** (Huang & Van Gool, 2017), la arquitectura de referencia:

- §3.1 motiva por qué hace falta una red "geométrica";
- §3.2 describe cada una de sus capas, cómo se calculan sus gradientes y cómo se optimizan
  sus pesos;
- §3.3 repasa las extensiones del estado del arte, en particular para imaginación motora;
- §3.4 describe las implementaciones existentes y la nuestra.

El código está en [`src/riemannian_eeg/spdnet.py`](../src/riemannian_eeg/spdnet.py) y sus
propiedades geométricas se verifican en [`tests/test_spdnet.py`](../tests/test_spdnet.py).

**Notación.** $\mathcal{S}_n$ son las matrices simétricas $n \times n$ y $\mathcal{S}^{++}_n$ las
SPD. Para $X \in \mathcal{S}_n$ escribimos su descomposición espectral $X = U \Sigma U^\top$, con
$U$ ortogonal (autovectores en columnas) y $\Sigma = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$.
$\operatorname{sym}(A) = \tfrac12 (A + A^\top)$, y $\circ$ es el producto elemento a elemento.
$\delta(A, B) = \lVert \log(A^{-1/2} B A^{-1/2}) \rVert_F$ es la distancia afín-invariante (AIRM).

---

## 3.1. Motivación

### 3.1.1. Los métodos riemannianos clásicos usan representaciones fijas

Los métodos riemannianos clásicos (apartado 4) explotan la geometría de $\mathcal{S}^{++}_n$ de dos
formas:

- **MDM** (*Minimum Distance to Mean*) calcula la media riemanniana de cada clase y asigna cada
  ensayo a la clase cuya media está más cerca según $\delta$.
- **Espacio tangente** (Barachant et al., 2012): aplana la variedad alrededor de la media
  riemanniana $M$ de los datos con $\log(M^{-1/2} C M^{-1/2})$, vectoriza el resultado y aplica un
  clasificador lineal.

Los dos trabajan con la covarianza de los 22 electrodos **tal cual**: no aprenden qué
combinaciones de electrodos son relevantes. CSP sí aprende filtros espaciales, pero con un
criterio indirecto (el cociente de potencias entre dos clases) y promediando las covarianzas con la
media aritmética, es decir, ignorando la geometría.

### 3.1.2. Una red euclídea no respeta la estructura SPD

La alternativa obvia sería vectorizar $C$ (253 componentes distintas para $n = 22$) y entrenar
una red neuronal convencional. Esto tiene tres problemas:

1. **Ignora la geometría.** La red mediría diferencias en $\mathbb{R}^{253}$, donde multiplicar
   por 100 la potencia de una dirección débil parece un cambio pequeño y variar un 1 % una
   dirección potente parece uno grande. Las potencias se comparan de forma natural por
   **cocientes**, que es lo que mide $\delta$. La red tendría que aprender esa estructura a partir
   de los datos, y con 288 ensayos por sujeto y sesión no hay suficientes.
2. **Sus capas no conservan la positividad.** Una capa $x \mapsto Ax + b$ puede producir
   "matrices" con autovalores negativos, que ya no son covarianzas. Las representaciones
   intermedias pierden su significado físico.
3. **Pierde las invariancias.** La AIRM cumple $\delta(A C_1 A^\top, A C_2 A^\top) = \delta(C_1, C_2)$
   para toda $A$ invertible. Si entre sesiones cambia la mezcla de fuentes que capta cada electrodo
   ($x \mapsto Ax$, por ejemplo por la colocación del gorro o las impedancias), las distancias entre
   ensayos no cambian. Una red euclídea no tiene esta propiedad.

### 3.1.3. Requisitos de una red sobre $\mathcal{S}^{++}_n$

La idea de las redes sobre variedades es combinar las tres cosas: **aprender** transformaciones
como CSP, **respetar la geometría** como los métodos riemannianos y **apilar capas** entrenadas de
extremo a extremo como una red neuronal. Para ello, cada capa debe ser

$$
f_k : \mathcal{S}^{++}_{n_{k-1}} \to \mathcal{S}^{++}_{n_k}, \qquad f_k \text{ derivable},
$$

y solo la última etapa debe salir de la variedad hacia un espacio vectorial, donde ya se puede
usar un clasificador ordinario.

---

## 3.2. SPDNet

### 3.2.1. Arquitectura

SPDNet define tres tipos de capa que hacen el papel de las capas lineales, las activaciones y el
aplanado de una red convencional:

$$
C \xrightarrow{\ \text{BiMap}\ } W_1^\top C\, W_1 \xrightarrow{\ \text{ReEig}\ } \cdots
\xrightarrow{\ \text{BiMap}\ }\xrightarrow{\ \text{ReEig}\ } X_L
\xrightarrow{\ \text{LogEig}\ } \log X_L \xrightarrow{\ \mathrm{vec}\ } \mathbb{R}^{d}
\xrightarrow{\ \text{lineal + softmax}\ } p(y \mid C).
$$

| Red euclídea | SPDNet | Propiedad que conserva |
|---|---|---|
| Capa lineal $Wx$ | BiMap $W^\top X W$ | La salida es SPD |
| ReLU $\max(0, x)$ | ReEig: $\max(\varepsilon, \lambda)$ sobre los autovalores | La salida es SPD |
| Aplanado (*flatten*) | LogEig $\log X$ + vectorización | Distancias del espacio tangente |
| Capa densa + softmax | Igual | — |
| Descenso de gradiente | Descenso de gradiente en la variedad de Stiefel | $W^\top W = I$ |

Con la configuración usada en el proyecto, `SPDNet([22, 16, 8], n_classes=4)`, un ensayo recorre
la red así: $\mathcal{S}^{++}_{22} \to \mathcal{S}^{++}_{16} \to \mathcal{S}^{++}_{8} \to \mathcal{S}_8 \to \mathbb{R}^{36} \to \mathbb{R}^4$.

### 3.2.2. BiMap: transformación bilineal

$$
f_{\text{BiMap}}(X) = W^\top X\, W, \qquad W \in \mathbb{R}^{n_{\text{in}} \times n_{\text{out}}},\;
n_{\text{out}} \le n_{\text{in}}.
$$

**Conserva la positividad.** Para todo $v \neq 0$,
$v^\top (W^\top X W) v = (Wv)^\top X (Wv) > 0$ siempre que $Wv \neq 0$, es decir, siempre que $W$
tenga rango completo por columnas. Con $n_{\text{out}} < n_{\text{in}}$, la capa además reduce la
dimensión.

**Restricción de Stiefel.** Para garantizar el rango completo durante todo el entrenamiento y
eliminar la redundancia de escala (multiplicar $W$ por $a$ multiplica la salida por $a^2$ sin
aportar información), se exige que $W$ tenga columnas ortonormales:

$$
W \in \operatorname{St}(n_{\text{in}}, n_{\text{out}}) = \{ W \in \mathbb{R}^{n_{\text{in}} \times n_{\text{out}}} : W^\top W = I \}.
$$

Este conjunto es la **variedad de Stiefel**: en SPDNet no solo los datos, sino también los pesos,
viven en una variedad (§3.2.6).

**Interpretación: generaliza CSP.** Cada columna $w_j$ de $W$ es un filtro espacial que combina
los electrodos en un "electrodo virtual" $s_j(t) = w_j^\top x(t)$. Como la potencia de cualquier
combinación de electrodos es $w^\top C w$ ([`eeg_2_spd.md`](eeg_2_spd.md), §5), la salida
$W^\top C\, W$ es la **covarianza de los electrodos virtuales**: su diagonal contiene sus potencias
y el resto sus covarianzas mutuas. CSP + LDA corresponde a:

- un BiMap con $W$ **fijado** por el problema de autovalores generalizado $C_1 w = \lambda C_2 w$;
- quedarse solo con la **diagonal** de $W^\top C W$ y tomar su logaritmo;
- un clasificador lineal.

SPDNet aprende $W$ minimizando directamente el error de clasificación de las cuatro clases a la
vez, conserva la matriz completa y permite apilar varias transformaciones.

### 3.2.3. ReEig: la no linealidad

**Necesidad.** Dos BiMap consecutivos equivalen a uno:
$W_2^\top (W_1^\top X W_1) W_2 = (W_1 W_2)^\top X (W_1 W_2)$. Sin una no linealidad, la profundidad
no añade capacidad.

**Por qué no sirve una función elemento a elemento.** Aplicar una ReLU a cada elemento de la
matriz no conserva la positividad en general. Por ejemplo, existen matrices SPD de $5 \times 5$
(con autovalores $0.1, 0.1, 0.1, 12.1, 20.1$) cuya ReLU elemento a elemento tiene un autovalor
de $-1.11$. Además, los elementos dependen de la base elegida (qué electrodos usamos como ejes),
mientras que la información física de una covarianza está en sus direcciones principales
(autovectores) y en la potencia a lo largo de ellas (autovalores).

**Definición.** ReEig rectifica los **autovalores**:

$$
f_{\text{ReEig}}(X) = U \max(\varepsilon I, \Sigma)\, U^\top, \qquad \varepsilon > 0.
$$

Los autovalores mayores que $\varepsilon$ no cambian, los menores se elevan a $\varepsilon$ y los
autovectores se conservan. Es el análogo matricial de la ReLU, con umbral $\varepsilon > 0$ en
lugar de $0$, de modo que la salida es estrictamente definida positiva. Tiene tres efectos:

- introduce la no linealidad (un "codo" en $\lambda = \varepsilon$);
- impone un **suelo de potencia**: las direcciones con potencia despreciable, dominadas por ruido,
  quedan igualadas a $\varepsilon$;
- impide que los autovalores colapsen hacia $0$ tras varias reducciones de dimensión. Eso
  acercaría las matrices al borde de la variedad, que está a distancia riemanniana infinita, y
  haría divergir el logaritmo de la capa LogEig.

El valor de $\varepsilon$ solo tiene sentido en relación con la escala de los datos. Por eso las
covarianzas deben normalizarse, por ejemplo dividiéndolas por su traza media, para que sus
autovalores sean del orden de $1$.

### 3.2.4. LogEig: proyección al espacio tangente

$$
f_{\text{LogEig}}(X) = \log X = U \log(\Sigma)\, U^\top.
$$

El logaritmo matricial se calcula aplicando el logaritmo escalar a los autovalores y no a los
elementos. Lleva $\mathcal{S}^{++}_n$ de forma biyectiva sobre el espacio vectorial $\mathcal{S}_n$:
las matrices con autovalores en $(0, 1)$ pasan a tener autovalores negativos, y el borde de la
variedad ($\lambda \to 0$) se envía a $-\infty$. En $\mathcal{S}_n$ se pueden sumar, restar y
escalar matrices sin restricciones, y su inversa $\exp$ devuelve siempre una matriz SPD.

**Interpretación geométrica.** El espacio tangente a $\mathcal{S}^{++}_n$ en cualquier punto es
$\mathcal{S}_n$. Con la AIRM, las geodésicas que parten de la identidad son
$\gamma(t) = \exp(tS)$, $S \in \mathcal{S}_n$. Son el análogo matricial de cambiar la potencia de
cada dirección principal a un ritmo **relativo** constante: en el caso escalar, la curva con
velocidad relativa $a'(t)/a(t)$ constante es la exponencial. Para llegar a $X$ en $t = 1$ hace
falta $S = \log X$, así que $\log X$ es el vector tangente en $I$ que apunta hacia $X$, y su
longitud es exactamente la distancia geodésica:

$$
\lVert \log X \rVert_F = \delta(I, X).
$$

LogEig es así la aplicación logarítmica $\operatorname{Log}_I$: "desdobla" la variedad sobre su
plano tangente en la identidad, como una proyección cartográfica azimutal equidistante. Las
distancias desde el centro se conservan exactamente. Entre otros dos puntos, la distancia
$\lVert \log A - \log B \rVert_F$ es la distancia **Log-Euclídea** (Arsigny et al., 2007), que
aproxima la AIRM y es más precisa cuanto más cerca de $I$ están los datos.

**Relación con el clasificador en espacio tangente.** El método clásico aplana alrededor de la
media $M$ de los datos, $\log(M^{-1/2} C M^{-1/2})$. Por la invariancia afín, la congruencia por
$M^{-1/2}$ (un blanqueado) lleva $M$ a $I$ sin alterar ninguna distancia, así que aplanar en $M$
equivale a recentrar y aplanar en $I$. SPDNet aplana siempre en $I$, pero las capas BiMap pueden
aprender a llevar los datos a una región adecuada. La batch normalization riemanniana (§3.3.1)
hace este recentrado de forma explícita.

**Vectorización.** La matriz $S = \log X_L \in \mathcal{S}_n$ se convierte en un vector con su
triángulo superior, multiplicando los elementos fuera de la diagonal por $\sqrt{2}$:

$$
\operatorname{vec}(S) = \big(S_{11}, \sqrt2\, S_{12}, \dots, \sqrt2\, S_{1n}, S_{22}, \sqrt2\, S_{23}, \dots, S_{nn}\big) \in \mathbb{R}^{n(n+1)/2}.
$$

Cada elemento fuera de la diagonal aparece dos veces en la matriz, así que el factor $\sqrt2$ hace
que $\lVert \operatorname{vec}(S) \rVert_2 = \lVert S \rVert_F$. La vectorización es una isometría
y conserva las distancias del espacio tangente. Es la misma convención que usa el clasificador en
espacio tangente de pyRiemann. Sobre este vector se aplica una capa lineal con softmax, entrenada
con entropía cruzada.

### 3.2.5. Retropropagación a través de autovalores

ReEig y LogEig, así como la raíz cuadrada, la inversa de la raíz y la exponencial que aparecen en
§3.3.1, son **funciones espectrales**:

$$
F = f(X) = U \operatorname{diag}\big(f(\lambda_1), \dots, f(\lambda_n)\big)\, U^\top.
$$

Para entrenar por retropropagación, cada capa recibe $G = \partial L / \partial F$ (el gradiente de
la pérdida respecto a su salida) y debe devolver $\partial L / \partial X$.

**La dificultad.** Una perturbación $X \mapsto X + H$ cambia tanto los autovalores como los
autovectores. Escribiendo $\tilde H = U^\top H U$, la teoría de perturbaciones da
$d\lambda_i = \tilde H_{ii}$ y

$$
du_i = \sum_{j \neq i} \frac{\tilde H_{ji}}{\lambda_i - \lambda_j}\, u_j .
$$

El término $1/(\lambda_i - \lambda_j)$ diverge cuando dos autovalores coinciden. En ese caso los
autovectores no están bien definidos: cualquier combinación de dos direcciones con la misma
potencia es igualmente principal. Derivar automáticamente a través de la descomposición produce
gradientes infinitos o NaN. Es una situación frecuente: ReEig iguala a $\varepsilon$ todos los
autovalores que rectifica.

**La fórmula de Daleckii–Krein.** Aunque los autovectores sean inestables, $F$ no lo es: si
$\lambda_i = \lambda_j$, entonces $f(\lambda_i) = f(\lambda_j)$ y el giro de $u_i, u_j$ no afecta a
$F$. Al sustituir las expresiones anteriores en $dF$, el efecto del giro en la posición $(i, j)$
aparece multiplicado por $f(\lambda_i) - f(\lambda_j)$. El resultado (Bhatia, 1997, cap. V) es

$$
dF = U \big( K \circ (U^\top H U) \big) U^\top, \qquad
K_{ij} = \begin{cases} \dfrac{f(\lambda_i) - f(\lambda_j)}{\lambda_i - \lambda_j} & \lambda_i \neq \lambda_j, \\[2mm]
f'(\lambda_i) & \lambda_i = \lambda_j. \end{cases}
$$

$K$ es la matriz de **cocientes de diferencias** de $f$: la pendiente de $f$ entre $\lambda_j$ y
$\lambda_i$. Cuando $\lambda_i \to \lambda_j$ tiende a $f'(\lambda_i)$, así que es siempre finita.

- Los elementos diagonales de $\tilde H$ (cambios de potencia) se multiplican por $f'(\lambda_i)$,
  la regla de la cadena escalar.
- Los no diagonales (giros entre direcciones) se multiplican por la pendiente entre los dos
  autovalores.

La fórmula se comprueba fácilmente en dos casos:

- $f(\lambda) = \lambda$ da $K \equiv 1$ y $dF = H$;
- $f(\lambda) = \lambda^2$ da $K_{ij} = \lambda_i + \lambda_j$ y $dF = XH + HX$, la regla del
  producto.

**Gradiente.** Como $K$ es simétrica, la aplicación $H \mapsto dF$ es autoadjunta para el producto
escalar de Frobenius, y el gradiente tiene la misma forma:

$$
\frac{\partial L}{\partial X} = U \big( K \circ (U^\top \operatorname{sym}(G)\, U) \big) U^\top .
$$

Solo requiere la descomposición ya calculada en la pasada hacia delante y unos productos de
matrices.

Para las dos capas de SPDNet:

- **LogEig:** $K_{ij} = \frac{\log\lambda_i - \log\lambda_j}{\lambda_i - \lambda_j}$ y
  $K_{ii} = 1/\lambda_i$.
- **ReEig:** $K_{ij} = 1$ si ambos autovalores superan $\varepsilon$ (el gradiente pasa), $0$ si
  ninguno lo supera (se bloquea) y un valor intermedio si solo uno lo supera. Es el equivalente
  matricial de la máscara de la ReLU.

Ionescu, Vantzos & Sminchisescu (2015) obtuvieron estos gradientes derivando $U$ y $\Sigma$ por
separado (*matrix backpropagation*), que es lo que usó el SPDNet original. La forma de
Daleckii–Krein es equivalente y evita explícitamente la división por $\lambda_i - \lambda_j$.

### 3.2.6. Optimización en la variedad de Stiefel

El gradiente euclídeo de la pérdida respecto a $W$ es inmediato:
$\nabla_W L = 2\, X W \operatorname{sym}(G)$. Pero un paso de descenso $W - \eta \nabla_W L$ deja de
cumplir $W^\top W = I$. Se aplica el esquema general de optimización riemanniana (Absil, Mahony &
Sepulchre, 2008), el mismo de "ir al espacio tangente, operar y volver" que se usa para calcular
la media riemanniana:

1. **Gradiente riemanniano.** Se proyecta $\nabla = \nabla_W L$ sobre el espacio tangente
   $T_W \operatorname{St} = \{ \xi : W^\top \xi + \xi^\top W = 0 \}$:
   $$
   \operatorname{grad} L(W) = \pi_W(\nabla) = \nabla - W \operatorname{sym}(W^\top \nabla).
   $$
2. **Paso en el espacio tangente:** $W - \eta \operatorname{grad} L(W)$, que en general ya no está
   en la variedad.
3. **Retracción.** Se vuelve a la variedad con la factorización QR,
   $W \leftarrow \operatorname{qf}\big(W - \eta \operatorname{grad} L(W)\big)$, donde
   $\operatorname{qf}$ es el factor $Q$ con signos tales que $\operatorname{diag}(R) > 0$ (lo que lo
   hace único).

Nuestro optimizador `StiefelSGD` añade momento. El vector de momento se transporta al nuevo
espacio tangente proyectándolo con $\pi_W$, una aproximación del transporte paralelo habitual en
la literatura. Los parámetros euclídeos (la capa lineal final) se actualizan con SGD ordinario.

La restricción reduce los grados de libertad de cada BiMap de $n_{\text{in}} n_{\text{out}}$ a
$n_{\text{in}} n_{\text{out}} - n_{\text{out}}(n_{\text{out}}+1)/2$. Para `SPDNet([22, 16, 8], 4)`
son $216 + 92$ parámetros efectivos en las BiMap y $148$ en la capa lineal, unos 450 en total: una
red muy pequeña, adecuada para los 288 ensayos de entrenamiento por sujeto.

### 3.2.7. Hiperparámetros

| Hiperparámetro | Papel | Valor de partida |
|---|---|---|
| Dimensiones de las capas | Número de bloques BiMap–ReEig y tamaño de cada uno (no creciente) | `[22, 16, 8]` |
| $\varepsilon$ (ReEig) | Suelo de los autovalores | $10^{-4}$, con covarianzas normalizadas |
| Batch normalization | Usar o no SPDNetBN (§3.3.1) | Probar ambas |
| Tasa de aprendizaje, momento | Optimizador `StiefelSGD` | $\eta \approx 10^{-2}$–$10^{-1}$, $\beta = 0.9$ |
| Tamaño de lote, épocas | Entrenamiento | 32; parada temprana |
| Estimador de covarianza y escala | Condicionamiento de la entrada | *Shrinkage* (OAS), división por la traza media |

El artículo original evalúa variantes con distinto número de bloques BiMap–ReEig sobre matrices
de hasta $400 \times 400$. En EEG, con matrices pequeñas ($22 \times 22$) y pocos ensayos, lo
habitual es usar **uno o dos** bloques. Todos los hiperparámetros deben elegirse con datos de la
sesión de entrenamiento únicamente; la sesión de test se reserva para la evaluación final.

---

## 3.3. Estado del arte

### 3.3.1. Batch normalization riemanniana (SPDNetBN)

En redes euclídeas, la batch normalization resta la media del lote y suma un sesgo aprendido, lo
que estabiliza y acelera el entrenamiento. Brooks et al. (2019) la trasladan a $\mathcal{S}^{++}_n$
sustituyendo cada operación por su análoga riemanniana:

- **Media del lote:** media de Karcher $\mathfrak{G} = \arg\min_M \sum_i \delta^2(M, X_i)$, que se
  aproxima con la iteración de punto fijo
  $\mathfrak{G} \leftarrow \mathfrak{G}^{1/2} \exp\big(\tfrac1N \sum_i \log(\mathfrak{G}^{-1/2} X_i \mathfrak{G}^{-1/2})\big)\, \mathfrak{G}^{1/2}$.
- **Centrado:** transporte paralelo de $\mathfrak{G}$ a $I$,
  $X_i \mapsto \mathfrak{G}^{-1/2} X_i\, \mathfrak{G}^{-1/2}$. Por la invariancia afín, el lote
  centrado tiene media de Karcher exactamente $I$.
- **Sesgo:** transporte de $I$ a una matriz SPD aprendida $B$, $X_i \mapsto B^{1/2} X_i B^{1/2}$.
- **Inferencia:** se usa una media móvil que avanza una fracción $t$ a lo largo de la geodésica,
  $\mathfrak{G}_{\text{run}} \leftarrow \mathfrak{G}_{\text{run}} \#_t \mathfrak{G}_{\text{lote}}$,
  con $A \#_t B = A^{1/2}(A^{-1/2} B A^{-1/2})^t A^{1/2}$.

El gradiente se propaga a través de la media de Karcher con las mismas funciones espectrales de
§3.2.5.

Tras el centrado, LogEig calcula $\log(\mathfrak{G}^{-1/2} X\, \mathfrak{G}^{-1/2})$, que es
exactamente la proyección al espacio tangente en la media del método clásico (Barachant et al.,
2012). SPDNetBN es así una generalización aprendible del clasificador en espacio tangente.

*Simplificación en nuestra implementación:* Brooks et al. optimizan $B$ directamente sobre la
variedad SPD. Nosotros la parametrizamos como $B = \exp(S)$ con $S$ simétrica, que es SPD por
construcción y se aprende con descenso de gradiente ordinario.

### 3.3.2. Adaptación de dominio: SPDDSMBN

La variabilidad entre sesiones y entre sujetos es el principal obstáculo de las BCI: lo aprendido
un día no se transfiere bien al siguiente. Kobler et al. (2022) proponen una batch normalization SPD
**específica de cada dominio** (sesión o sujeto). Cada dominio se recentra en su propia media, que
puede estimarse sin etiquetas, y además se normaliza su dispersión. Es la versión integrada en la
red del recentrado riemanniano para transferencia entre sesiones, y es especialmente relevante
para el protocolo de evaluación entre sesiones de este proyecto.

### 3.3.3. Arquitecturas específicas para imaginación motora

- **Tensor-CSPNet** (Ju & Guan, 2023): calcula covarianzas en varias bandas de frecuencia y
  ventanas temporales, las procesa en paralelo con bloques BiMap–ReEig y combina la información
  temporal con una convolución. Es la extensión profunda del *filter bank* CSP (FBCSP).
- **Graph-CSPNet** (Ju & Guan, 2024): sustituye esa agregación por una red neuronal de grafos
  sobre matrices SPD, cuyos nodos son pares (banda, ventana temporal).
- **MAtt** (Pan, Chou & Wei, 2022): extrae covarianzas de segmentos temporales con una CNN y las
  combina con un mecanismo de atención basado en la distancia Log-Euclídea.

### 3.3.4. Otras líneas

- **Redes euclídeas sobre la señal cruda**, como EEGNet (Lawhern et al., 2018) o ShallowConvNet
  (Schirrmeister et al., 2017). Son la alternativa no geométrica: aprenden filtros temporales y
  espaciales directamente. ShallowConvNet replica la cadena de FBCSP (filtrado, cuadrado, media,
  logaritmo), es decir, calcula implícitamente log-potencias de electrodos virtuales.
- **Redes para otros datos en variedades**, como ManifoldNet (Chakraborty et al., 2022), que
  generaliza la convolución mediante medias de Fréchet ponderadas. Sitúan SPDNet dentro del
  aprendizaje profundo geométrico.

---

## 3.4. Implementaciones

### 3.4.1. Librerías existentes

| Librería | Qué ofrece | Relación con el proyecto |
|---|---|---|
| **pyRiemann** | Estimación de covarianzas, métricas, medias, MDM y espacio tangente, compatibles con scikit-learn | Métodos riemannianos clásicos (apartado 4) y estimación de covarianzas |
| **geoopt** (Kochurov et al., 2020) | Parámetros sobre variedades (Stiefel, SPD, esfera…) y optimizadores riemannianos (SGD, Adam) en PyTorch | Alternativa a nuestro `StiefelSGD` |
| **Geomstats** (Miolane et al., 2020) | Geometría diferencial general: variedades, métricas, geodésicas | Referencia |
| Código de los autores | SPDNet (MATLAB), torchspdnet (Brooks et al.), Tensor-CSPNet, SPDDSMBN | Implementaciones de referencia de cada artículo |

### 3.4.2. Implementación propia

Hemos implementado SPDNet desde cero en PyTorch. Así cada elemento geométrico es explícito y
verificable, en lugar de quedar oculto en una librería:

| Concepto | Sección | Código (`spdnet.py`) |
|---|---|---|
| Funciones espectrales con gradiente de Daleckii–Krein | §3.2.5 | `_EigFunction`, `sym_log`, `sym_exp`, `sym_pow`, `sym_rectify` |
| BiMap con pesos en la variedad de Stiefel | §3.2.2 | `BiMap`, `StiefelParameter` |
| ReEig | §3.2.3 | `ReEig` |
| LogEig y vectorización isométrica | §3.2.4 | `LogEig`, `SymVectorize` |
| Optimización en Stiefel: proyección, retracción QR, momento | §3.2.6 | `stiefel_project`, `stiefel_retract`, `StiefelSGD` |
| Media de Karcher, geodésica, distancia AIRM | §3.3.1 | `karcher_mean`, `geodesic`, `airm_distance` |
| Batch normalization riemanniana | §3.3.1 | `SPDBatchNorm` |
| Red completa | §3.2.1 | `SPDNet(dims, n_classes, batchnorm)` |

**Verificación.** Los tests (`uv run pytest tests`) comprueban propiedades matemáticas y no solo
que el código se ejecuta:

- $\exp$ y $\log$ son inversas, y $X^{-1/2} X X^{-1/2} = I$;
- el gradiente de Daleckii–Krein coincide con las diferencias finitas para $\log$, $\exp$,
  $\sqrt{\cdot}$ y ReEig, y es finito cuando todos los autovalores son iguales;
- BiMap, ReEig y la batch normalization producen matrices SPD, y ReEig deja todos los autovalores
  $\ge \varepsilon$;
- la vectorización conserva la norma, y $\lVert \log X \rVert_F = \delta(I, X)$;
- la media de Karcher es equivariante afín y conmuta con la inversión, algo que la media
  aritmética no cumple;
- la batch normalization centra el lote exactamente en $I$;
- la red aprende un problema sintético de dos clases y sus pesos cumplen $W^\top W = I$ con error
  inferior a $10^{-10}$ al terminar el entrenamiento.

El entrenamiento sobre BNCI2014_001 y la comparación con los demás métodos se presentan en los
apartados 4 y 5.

### 3.4.3. Limitaciones

- **Coste computacional.** Cada capa espectral requiere una descomposición en autovalores
  ($O(n^3)$), que se paraleliza peor en GPU que un producto de matrices. Con $n = 22$ es
  despreciable, pero limita el tamaño de las matrices de entrada.
- **Precisión numérica.** Las covarianzas de EEG están mal condicionadas, por lo que trabajamos en
  doble precisión y conviene regularizar el estimador de covarianza y normalizar su escala.
- **Pocos datos.** Con 288 ensayos por sujeto y sesión, una red tiene más riesgo de
  sobreajuste que MDM o el espacio tangente. La elección de la arquitectura exige validación
  dentro de la sesión de entrenamiento, y no está garantizado que SPDNet supere a los métodos
  riemannianos clásicos.
- **Optimización específica.** Los pesos en la variedad de Stiefel no se integran directamente con
  los optimizadores estándar de PyTorch; hacen falta optimizadores riemannianos propios o una
  librería como geoopt.

---

## Referencias del apartado

- Absil, P.-A., Mahony, R., Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
- Arsigny, V., Fillard, P., Pennec, X., Ayache, N. (2007). Geometric means in a novel vector space structure on symmetric positive-definite matrices. *SIAM J. Matrix Anal. Appl.*, 29(1), 328–347.
- Barachant, A., Bonnet, S., Congedo, M., Jutten, C. (2012). Multiclass brain-computer interface classification by Riemannian geometry. *IEEE Trans. Biomed. Eng.*, 59(4), 920–928.
- Bhatia, R. (1997). *Matrix Analysis*. Springer (Graduate Texts in Mathematics 169), cap. V.
- Brooks, D., Schwander, O., Barbaresco, F., Schneider, J.-Y., Cord, M. (2019). Riemannian batch normalization for SPD neural networks. *NeurIPS 2019*.
- Chakraborty, R., Bouza, J., Manton, J. H., Vemuri, B. C. (2022). ManifoldNet: A deep neural network for manifold-valued data with applications. *IEEE Trans. Pattern Anal. Mach. Intell.*, 44(2).
- Huang, Z., Van Gool, L. (2017). A Riemannian network for SPD matrix learning. *AAAI 2017*.
- Ionescu, C., Vantzos, O., Sminchisescu, C. (2015). Matrix backpropagation for deep networks with structured layers. *ICCV 2015*.
- Ju, C., Guan, C. (2023). Tensor-CSPNet: A novel geometric deep learning framework for motor imagery classification. *IEEE Trans. Neural Netw. Learn. Syst.*, 34(12).
- Ju, C., Guan, C. (2024). Graph neural networks on SPD manifolds for motor imagery classification: A perspective from the time-frequency analysis. *IEEE Trans. Neural Netw. Learn. Syst.*
- Kobler, R., Hirayama, J., Zhao, Q., Kawanabe, M. (2022). SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG. *NeurIPS 2022*.
- Kochurov, M., Karimov, R., Kozlukov, S. (2020). Geoopt: Riemannian optimization in PyTorch. arXiv:2005.02819.
- Lawhern, V. J. et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *J. Neural Eng.*, 15(5).
- Miolane, N. et al. (2020). Geomstats: A Python package for Riemannian geometry in machine learning. *J. Mach. Learn. Res.*, 21(223).
- Pan, Y.-T., Chou, J.-L., Wei, C.-S. (2022). MAtt: A manifold attention network for EEG decoding. *NeurIPS 2022*.
- Schirrmeister, R. T. et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11).
