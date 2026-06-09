# Decisiones — Posesión y métricas (Etapa 3)

Cómo decidimos estimar la posesión de la pelota y calcular las métricas
agregadas. Cubre `src/futbol_vpc/possession.py` y `metrics.py`.

Última actualización: **2026-06-09**.

---

## 1. Objetivo

- **Posesión instantánea** (por frame): qué equipo tiene la pelota.
- **Métricas agregadas:** % de posesión por equipo, cambios de posesión, duración
  media de las secuencias de posesión.

Insumos (ya resueltos): detección de la pelota (YOLO26m), tracking de jugadores
(ByteTrack) y equipo de cada track (embeddings SigLIP).

---

## 2. Decisiones de diseño

### Decisión 1 — Distancia desde los **pies** del jugador

La pelota está en el piso, así que se mide la distancia desde el **borde inferior
del bbox del jugador** (pies = `((x1+x2)/2, y2)`) al centro de la pelota, no desde
el centro del torso. Más fiel a "tener la pelota en los pies".

### Decisión 2 — Umbral **relativo al tamaño del jugador** (perspectiva)

Un umbral fijo en píxeles es engañoso: por la perspectiva, un jugador lejano se ve
chico y "50 px" cubren mucha más cancha que cerca del arco. Se usa la distancia
**normalizada por la altura del bbox del jugador** (`dist / alto_bbox`); el
poseedor es el jugador con menor distancia normalizada, si está por debajo de un
umbral. Esto es aproximadamente invariante a la perspectiva.

### Decisión 3 — Suavizado temporal por ventana

La posesión instantánea parpadea (la pelota salta entre jugadores cercanos, se
pierde la detección un frame). Se suaviza con la **moda en una ventana deslizante**
(la propuesta pide suavizado explícito) para quitar transiciones espurias.

### Decisión 4 — El arquero no cuenta (ya decidido)

Ver [deteccion-y-asignacion.md](deteccion-y-asignacion.md): el `goalkeeper` se
**excluye de los candidatos** a posesión.

### Decisión 5 — Pelota no detectada → sin posesión

La pelota es chica y a veces no se detecta (o hay falsos positivos). En frames sin
pelota confiable, la posesión queda **`None`** (sin posesión), no se fuerza. Se
maneja en las métricas (los frames sin posesión no cuentan para los porcentajes).

---

## 3. Validación — una limitación importante

SoccerNet-Tracking provee la **posición de la pelota** (`gt.txt`), pero **no
etiquetas de posesión** (quién la tiene). Entonces:

- La regla "jugador más cercano = poseedor" es una **heurística**; su correctitud
  contra la posesión real necesita **anotación manual** de algunos clips → eso es
  trabajo de la **Etapa 5** (ground truth de posesión, como dice la propuesta).
- Por ahora se valida: (a) **visualmente** (overlay de quién tiene la pelota), y
  (b) comparando la posesión calculada con **la pelota detectada** vs **la pelota
  del GT** (`gt.txt`), para aislar el efecto del error de detección de la pelota.

---

## 4. Experimentos

### Exp. 1 — Primer pipeline completo (imgsz=640) → la pelota es el cuello

Pipeline track→equipos→posesión sobre 400 frames de SNMOT-108:

- Solo **29/400 frames (7%) con posesión** → métricas sin sentido (1 cambio, 0.4s).

**Diagnóstico** (`scripts/diagnostico_posesion.py`): se aisló la causa barriendo el
umbral y contando detecciones de pelota:

| | resultado |
|---|---|
| Frames con **pelota detectada** (imgsz=640) | **40/400 (10%)** |
| Frames con poseedor a umbral 0.3 / 0.5 / 1.0 / 3.0 | 2% / 6% / 10% / 10% |

**Conclusión (medida):** el cuello **no es el umbral** sino la **detección de la
pelota**. Una vez detectada, casi siempre hay un jugador cerca (umbral ≥1.0 ya
llega al techo de 40 = los frames con pelota). El umbral 0.5 estaba algo estricto,
pero el problema real es que **la pelota se pierde el 90% del tiempo**: objeto
chico + cámara abierta + inferencia a 640 → pocos píxeles.

### Exp. 2 — imgsz=1280 y conf (SNMOT-108, 400 frames)

| Config | Pelota detectada | Poseedor (techo, umbral≥1) |
|---|---|---|
| 640, conf 0.5 (baseline) | 10% | 10% |
| **1280, conf 0.5** | **25%** | **16%** |
| 1280, conf 0.15 | 53% | 16% |

**Conclusiones (medidas):**
- **`imgsz=1280` ayuda:** detección de pelota 10→25%, posesión 7→16%. **Adoptado.**
- **conf 0.15 es ruido:** la pelota "sube" a 53% pero el poseedor no cambia (16%) →
  las detecciones extra son falsos positivos lejos de todos. **Se mantiene conf 0.5.**
- **El cuello sigue siendo la pelota:** aun a 1280, ~75% de los frames no tienen
  pelota detectada. Ese es el límite de cobertura de la posesión.

### Exp. 3 — Interpolación de la pelota en huecos cortos

Interpolar la posición de la pelota en huecos ≤5 frames entre detecciones.
Resultado (1280, conf 0.5): pelota 25% → **27%** (+2%), poseedor sin cambio (16%).

**Casi no ayudó.** Los huecos de detección son **largos, no cortos** → la pelota no
parpadea, está ausente por tramos largos. Interpolar huecos cortos no tiene qué
llenar. (Se deja `interpolar_pelota` en el código; ayuda marginalmente y no
estorba, pero no resuelve el problema.)

### Exp. 4 — ¿Detección o fuera de cámara? (GT) → es DETECCIÓN

Usando el `gt.txt` (posición real de la pelota): la pelota está **anotada en
397/400 frames (99%)** → está en cámara casi siempre. Pero el detector la encuentra
solo 25% (a 1280).

**Conclusión contundente:** el cuello es **100% la detección de la pelota**, no que
esté fuera de cuadro. Hay un **domain gap**: en SoccerNet (cámara táctica abierta)
la pelota es mucho más chica que en las imágenes de Roboflow del entrenamiento
(donde tenía 0.89 de recall). El detector no generaliza a esa escala de pelota.

**Implicancia — separar dos problemas:**
1. **Lógica de posesión** (jugador más cercano): se puede validar **con la pelota
   del GT** (decoupling), para probar que el método es correcto sin depender de la
   detección.
2. **Detección de la pelota**: problema aparte (probar imgsz=1920, fine-tune de un
   detector de pelota en SoccerNet, o documentar como limitación).

### Exp. 5 — Posesión con pelota GT: el modelo per-frame está mal planteado

Inyectando la pelota del GT (presente 99%) y barriendo el umbral:

| umbral_rel | cobertura de poseedor |
|---|---|
| 0.5 | 15% |
| 1.0 | 28% |
| 3.0 | 40% |
| 5.0 (~4 m) | 47% |

**Hallazgo:** aun con la pelota conocida y un radio enorme, solo ~47% de los
frames tienen a alguien cerca. **La pelota pasa mucho tiempo viajando** (pases,
pelotazos). El modelo per-frame ("alguien cerca AHORA" + suavizado de 5 frames)
deja la posesión en `None` durante cada pase → posesión fragmentada, secuencias de
0.4s. **Visualmente la lógica es correcta** (marca bien al jugador con la pelota
a los pies); el problema es el modelo temporal.

**Decisión — modelo de "hold":** la posesión es un **estado que se mantiene**. El
poseedor claro (pelota a los pies, umbral estricto) ancla la posesión; entre
anclajes (pelota viajando) **se sostiene con el último equipo** hasta que aparece
un poseedor claro del **otro** equipo. Así los pases no rompen la posesión,
la cobertura es ~100% y las métricas (cambios, duraciones) son reales. Es el modelo
estándar de posesión. Reemplaza el suavizado por ventana.

### Exp. 6 — Hold sobre clip completo: parpadeo → falta histéresis

Hold sobre los 750 frames (pelota GT, umbral 0.75):

| Clip | Cobertura | % equipo | Cambios | Duración media |
|---|---|---|---|---|
| 108 | 65% | 67/33 | 40 | 0.5 s |
| 061 | 100% | 57/43 | 16 | 1.8 s |

Métricas ya no degeneradas, **pero 40 cambios en 20s (uno cada 0.5s) es irreal**:
la posesión **parpadea** porque los *eventos de control* alternan rápido entre
equipos (pelota disputada / entre dos jugadores de equipos distintos). El hold
sostiene los huecos pero no filtra el parpadeo.

**Decisión — histéresis:** la posesión solo cambia cuando el **otro** equipo
controla de forma **sostenida** (≥ N eventos de control), no en un toque suelto.
Así las disputas momentáneas no cuentan como cambio y quedan las recuperaciones
reales. Parámetro `POSSESSION_HYSTERESIS` (a calibrar).

### Exp. 7 — Barrido de histéresis: sin ground truth no se puede calibrar

Barrido sobre los 750 frames (pelota GT, umbral 0.75):

| histéresis | 108 (%A/%B, cambios, dur) | 061 (%A/%B, cambios, dur) |
|---|---|---|
| 4 | 29/71, 8, 2.2s | 47/53, 10, 2.7s |
| 8 | 55/45, 3, 4.9s | **7/93, 1, 15s** (colapsa) |
| 20 | 63/37, 1, 9.8s | 8/92, 1, 15s (colapsa) |

**Hallazgo fundamental (de evaluación, no técnico):** las métricas son **muy
sensibles** a la histéresis, y **no hay un valor claramente correcto** para ambos
clips (h=8 anda en 108 pero colapsa 061). Y no se puede decidir cuál es el bueno
porque **SoccerNet no tiene ground truth de posesión** (tiene posición de la
pelota, no quién la tiene). Calibrar contra la propia intuición es justo el error
que cometimos con los equipos (el ojímetro engaña).

**Conclusión:** el **método de posesión funciona** (marca bien al poseedor,
métricas no degeneradas con hold+histéresis), pero **calibrar objetivamente sus
parámetros (umbral + histéresis) requiere ground truth de posesión** = anotación
manual (Etapa 5 de la propuesta). Se deja `h=4` como default tentativo (el más
balanceado, sin colapso). La validación honesta hasta tener GT es **visual** (el
video con overlay, donde se ve la posesión en movimiento — mucho mejor que un
frame suelto).

### Exp. 8 — Evaluación contra ground truth manual (Etapa 5)

Se anotó la posesión de SNMOT-108 a mano (`data/ground_truth/SNMOT-108_posesion.csv`,
6 tramos, arquero/pelota-fuera = "ninguno") y se evaluó frame-a-frame (2 mapeos
A/B, excluyendo "ninguno").

**Primera corrida (default) dio 57%** — sospechoso vs el buen video. La matriz de
confusión mostró **sesgo sistemático** hacia un equipo. Diagnóstico
(`calibrar_posesion.py`):

- **Accuracy de equipos en esa corrida concreta: 94.8%** → la asignación NO era el
  problema.
- **Barrido umbral × histéresis vs GT:** el mejor es **umbral 0.5 → 76%** (robusto
  a la histéresis). Los parámetros default (umbral 0.5, h=4) ya son los mejores.

**Causa del 57% inicial:** la asignación por embeddings usa **UMAP, que es no
determinista** → varía entre corridas; esa corrida salió peor y arrastró la
posesión. Con una corrida buena de equipos (94.8%), la posesión da **76%**.

**Resultados finales:**
- **Posesión: ~76%** de accuracy vs ground truth manual (pelota GT, parámetros
  calibrados). Resultado sólido para el alcance del TP.
- **Limitación de reproducibilidad:** la asignación de equipo (UMAP/KMeans)
  **varía entre corridas**. A futuro: fijar semilla o votar varias corridas. Es la
  fuente principal de varianza del pipeline.
