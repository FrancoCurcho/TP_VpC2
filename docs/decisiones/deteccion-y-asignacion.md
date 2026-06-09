# Decisiones — Detección y asignación de equipo (Etapa 1)

Cómo decidimos encarar las dos primeras tareas del pipeline y por qué. Cubre los
módulos `src/futbol_vpc/detection.py` y `src/futbol_vpc/team_assign.py`.

Última actualización: **2026-06-08**.

---

## 1. Detección (`detection.py`)

### Formato de detección neutro (`Detection`)

En vez de pasar los objetos crudos de ultralytics por todo el pipeline, cada
detección se traduce a un `dataclass` propio:

```
Detection(bbox, cls, conf, team=None, track_id=None)
```

**Por qué:**
- **Desacopla el pipeline del detector.** Las etapas siguientes (equipo, tracking,
  posesión) trabajan con `Detection`, sin saber si abajo hay un YOLO26, un YOLOv8
  o cualquier otro. Si cambiamos de modelo, solo cambia `detection.py`.
- Los campos `team` y `track_id` arrancan en `None` y los **completan etapas
  posteriores** (asignación y tracking), así la detección "viaja" y se va
  enriqueciendo.
- `Detection.centro` se agrega de una porque la posesión (Etapa 3) se calcula por
  **distancia del centro del jugador al centro de la pelota**.

### Umbrales

`conf=0.5`, `iou=0.5` por defecto (en `config.py`). El `conf=0.5` ayuda a filtrar
los falsos positivos de "player" sobre el fondo que vimos en la comparación de
modelos (ver [modelo.md](modelo.md)). Son ajustables por llamada.

### Device

El wrapper acepta `device` (`'mps'` en Mac, `0` en GPU, `'cpu'`). La idea es
desarrollar e inferir local en la Mac con MPS, sin atarse a una GPU.

---

## 2. Asignación de equipo (`team_assign.py`)

La tarea: a cada `player` decidir si es del **equipo A** o **B**, a partir del
color de la camiseta (clustering en HSV, como dice la propuesta).

### Decisión 1 — Recortar solo el torso, no todo el bbox

Se usa la franja del **10% al 50% de la altura** del bbox.

**Por qué:** el bbox completo de un jugador incluye cabeza, piernas, shorts,
medias y mucho césped alrededor. La **camiseta** (lo que define el equipo) está en
el torso. Recortar esa franja reduce el ruido y hace el color mucho más
discriminativo.

### Decisión 2 — Enmascarar el verde del césped

Antes de calcular el color se descartan los píxeles verdes (tono H 35-85 en escala
OpenCV) y los muy desaturados (sombras, líneas blancas).

**Por qué:** aunque recortemos el torso, siempre se cuela césped de fondo entre
brazos/cuerpo. Si no lo sacamos, el verde "tira" del color promedio y dos equipos
distintos pueden terminar pareciéndose. Se usa la **mediana** de los píxeles
restantes (más robusta a outliers que el promedio).

### Decisión 3 — Entrenar el clustering UNA vez, no por frame

Se implementa como clase `TeamAssigner` con `fit()` (aprende los 2 colores sobre
varios frames) y `predict()` (asigna en cada frame con esos colores fijos).

**Por qué — esto es lo más importante:** si corriéramos KMeans **frame a frame**,
las etiquetas de cluster (0/1 → A/B) **se invertirían arbitrariamente** entre
frames, porque KMeans no garantiza un orden estable. Resultado: el "equipo A"
cambiaría de color cada pocos frames y las métricas de posesión por equipo serían
basura. Fijando los centros una sola vez, "equipo_A" es siempre el mismo color en
todo el video.

**Sobre qué frames se hace el `fit`:** frames **muestreados a lo largo de todo el
clip** (ej. 1 por segundo), no solo los primeros. Si usáramos solo el arranque y
ahí un equipo casi no aparece (saque de arco, jugada en un sector), los centroides
quedarían sesgados. El muestreo asegura que ambos equipos estén representados.

**Es por video, y está bien:** el `fit` aprende los colores de *ese* partido, así
que se corre una vez por cada video nuevo. No es un re-entrenamiento: es KMeans en
CPU (segundos), no supervisado. Esto es justo lo que hace que el sistema
**generalice a cualquier partido sin re-entrenar nada** (el detector YOLO sí se
entrena una vez y se reusa; el color de equipo se adapta solo).

**Red de seguridad:** tras el `fit` se mide la distancia entre los 2 centroides
(`TeamAssigner.separacion`). Si quedan muy cerca, las camisetas son parecidas y se
emite un warning, porque la asignación va a ser poco confiable.

### Decisión 4 — No clusterizar arquero ni árbitro

Solo se clusteriza la clase `player`. El `goalkeeper` y el `referee`:
- el detector ya los distingue como clases propias,
- usan colores distintos a ambos equipos (a propósito), así que meterlos en el
  KMeans de 2 grupos ensuciaría los centros de los equipos.

---

## 2.bis. Experimentos sobre clips reales (2026-06-08) — y el cambio de enfoque

Probamos la asignación per-frame sobre dos clips de SoccerNet con esquemas de
color distintos: **SNMOT-108** (rojo vs gris oscuro, fondo de estadio rojo) y
**SNMOT-061** (verde vs blanco). Resultado:

| Enfoque de color | 061 (verde/blanco) | 108 (rojo/gris) |
|---|---|---|
| HSV + máscara de saturación | ❌ | ✅ |
| 2 niveles (KMeans por jugador + borde=fondo) | ✅ | ❌ |
| Recorte central + mediana BGR | ❌ | ❌ |
| **DBSCAN** (eps 10/15/25/40) | nunca 2 clusters | nunca 2 clusters |

**Conclusión (con datos, no intuición):** la asignación **per-frame por color
crudo es inviable**. La causa raíz es la **contaminación de fondo**: cuando un
jugador está lejos/chico, el recorte mezcla la camiseta con el césped o con el
fondo del estadio (en 108, butacas/publicidad rojas → grises se vuelven rojizos).
Como el fondo cambia por partido, ninguna heurística per-frame gana en todos.

DBSCAN no ayudó porque la contaminación crea un **continuo** entre los dos colores
(gris limpio → gris-rojizo → rojo), sin valle de densidad para separar: con `eps`
grande da 1 solo cluster, con `eps` chico fragmenta los bordes pero la masa
central nunca se parte en 2. Y `eps` necesitaría tuneo por video → mata la
generalización. **El cuello de botella es la representación de entrada, no el
algoritmo de clustering.**

**Intento siguiente — color agregado por track.** La idea: como el fondo varía
frame a frame y la camiseta es constante, agregar (mediana) el color de cada
**track** sobre todo el clip debería lavar la contaminación. Se implementó:

1. Tracking con ByteTrack → IDs estables (verificado: 0 fragmentación, buena vida
   media de los IDs).
2. `asignar_equipos_por_track`: mediana del color por track → KMeans en 2 equipos.

**Validación cuantitativa contra ground truth.** En vez de mirar a ojo (que
*engañó*: un frame se veía bien), se midió contra el `gameinfo.ini` de SoccerNet
(equipo real de cada track), matcheando cada detección con la caja del GT de mayor
IoU. Sobre 400 frames × 2 clips (~9.000 detecciones):

| Clip | Accuracy de equipo |
|---|---|
| SNMOT-108 (rojo/gris) | **59.5%** |
| SNMOT-061 (verde/blanco) | **62.1%** |

**~60% ≈ azar** (50% para 2 equipos). La agregación por track ayudó algo pero **no
alcanza**: el color crudo, aun promediado, sigue contaminado por el fondo (en 108
el estadio rojo tiñe los grises). **Lección de proceso:** el ojímetro sobre un
frame mintió; el harness cuantitativo reveló la verdad. Sin él, hubiéramos dado
por buena una asignación del 60%.

**Decisión final — EMBEDDINGS.** Se escaló a embeddings: recorte de jugador →
embedding visual **SigLIP** (`google/siglip-base-patch16-224`, vía la librería
`sports` de Roboflow) → UMAP → KMeans, agregado por track (mayoría de los crops
del jugador). El embedding es robusto al fondo *por diseño* (mira el objeto, no
los píxeles crudos). Validado con el **mismo harness** de ground truth:

| Clip | Color por track | **Embeddings (SigLIP)** |
|---|---|---|
| SNMOT-108 (rojo/gris) | 59.5% | **92.0%** |
| SNMOT-061 (verde/blanco) | 62.1% | **97.8%** |

De ~azar a **92-98%** sin cambiar nada más que la representación. Asignación de
equipo resuelta. (Costo: agrega `transformers`/`umap-learn`/`supervision` y un
modelo SigLIP; corre por crop, se agrega por track.)

**Qué se reusó:** el tracking, la agregación por track y el harness de validación
(`scripts/validar_equipos.py`) quedaron intactos; solo cambió la **representación**
(embedding en vez de color BGR). Implementación: `scripts/validar_equipos_embed.py`.

### Resumen del recorrido (todo medido, no intuido)

| # | Enfoque | Resultado |
|---|---|---|
| 1 | Color per-frame (HSV+máscara / 2-niveles / central+mediana) | whack-a-mole: cada variante arregla un clip y rompe el otro |
| 2 | DBSCAN (varios eps) | nunca separa 2 equipos (contaminación = continuo sin valle) |
| 3 | Color agregado por track + KMeans | **59-62%** vs GT (≈ azar) |
| 4 | **Embeddings (SigLIP) + track** | **92-98%** vs GT ✅ **elegido** |

### Consecuencia: cambio de orden del pipeline (desviación de la propuesta)

La propuesta original ubicaba la asignación de equipo en la Etapa 1, **antes** del
tracking:

```
propuesta:   detección → asignación equipo → tracking → posesión
ejecución:   detección → tracking → asignación equipo (por track) → posesión
```

**Por qué se reordenó (forzado por los datos):** la asignación robusta (tanto
color como embeddings) necesita **agregar sobre la vida de cada jugador** para
lavar el ruido del fondo. Y agregar por track requiere **tener los track IDs
primero** → el tracking pasa a ser un **prerrequisito** de la asignación, no un
paso posterior. La evidencia: per-frame ≈ 60% vs agregado-por-track-con-embeddings
92-98%.

Es una **desviación justificada de la propuesta**, descubierta experimentando, no
un cambio arbitrario. El árbitro y el arquero sí se distinguen per-frame (el
detector los separa como clases propias); lo que se mueve después del tracking es
solo el **equipo (A/B) de los jugadores de campo**.

## 3. Limitaciones conocidas / a validar con el clip real

- **Equipos con colores parecidos:** si las dos camisetas son similares, el KMeans
  de 2 grupos puede fallar. A validar con el clip de SoccerNet.
- **Arquero — DECIDIDO:** el `goalkeeper` se trata aparte y **su posesión no
  cuenta**. Que el arquero tenga la pelota (saque, descuelgue) no representa
  posesión en el sentido táctico que nos interesa. Consecuencia para la Etapa 3:
  el arquero se **excluye de los candidatos a posesión** (no entra en el cálculo
  del jugador más cercano a la pelota). Por eso tampoco hace falta asignarle
  equipo.
- **Robustez del recorte de torso:** el 10-50% es heurístico; con jugadores muy
  chicos o tapados puede recortar mal. A revisar sobre frames reales.
- **Validación cuantitativa:** comparar la asignación contra el `gameinfo.ini` de
  SoccerNet (que tiene el equipo real de cada track) — ground truth gratis.
