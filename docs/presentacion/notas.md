# Notas para la presentación

Guion de apoyo, slide por slide. No leer textual — son los **puntos a tocar** y los
**números clave**. La presentación está dividida en 3 partes; cada uno prepara la suya.

> Tip general: la narrativa fuerte del trabajo es **"probamos, medimos, y los datos
> nos hicieron cambiar de enfoque"**. Cada vez que se pueda, contar el *por qué* de una
> decisión, no solo el *qué*.

---

## Parte 1 — Contexto y detección

### Portada
- Presentarse, título del trabajo, materia (VPC II, CEIA).

### El problema
- La **posesión** es una métrica táctica clave; hoy la dan sistemas caros (Stats, Opta) o se mide a mano.
- Automatizarla no es un solo problema: hay que **encadenar** detección → equipo → tracking → **quién tiene la pelota**.
- El desafío es **integrar** todo en un sistema robusto. Eso es lo que construimos.

### El pipeline
- Mostrar el flujo: Detección (YOLO26m) → Tracking (ByteTrack) → Equipo (embeddings) → Posesión → Overlay.
- **Adelantar el spoiler:** el orden **cambió respecto de la propuesta** (tracking antes que equipo). Lo explicamos en la Parte 2, pero dejarlo picando.

### Datasets
- **Dos datasets distintos:** se entrena el detector con uno de **Roboflow** (4 clases), y se **evalúa** con **SoccerNet-Tracking**.
- SoccerNet trae **ground truth** (tracking + equipo de cada jugador) → nos sirve para validar con números, no a ojo.
- Elegimos **5 clips** con **3 esquemas de color** distintos a propósito, para estresar la asignación de equipo (camisetas blancas, grises = casos difíciles).

### Detección + comparación
- Comparamos **3 modelos YOLO** (v8m, 11m, 26m), todos en tamaño **m** y mismos hiperparámetros → comparación justa (cambia solo la arquitectura).
- Mencionar que la **pelota** está empatada (~0.89) en los tres → no fue el criterio de desempate.
- **Elegimos YOLO26m** por el mejor **mAP50-95** (localización fina de las cajas). ¿Por qué importa eso? Porque la posesión se decide por **distancia jugador-pelota**, así que necesitamos cajas bien ubicadas.

---

## Parte 2 — Tracking y asignación de equipo

### Tracking
- **ByteTrack**: asocia las detecciones frame a frame por **IoU** (solapamiento de cajas), sin usar apariencia → liviano.
- Da un **ID estable** a cada jugador. En las pruebas: IDs estables, **0 fragmentación**.
- **Clave:** el tracking no es solo "para seguir la jugada" — es lo que **habilita agregar información por jugador a lo largo del tiempo**, que es lo que salva la asignación de equipo.

### Asignación de equipo (I) — el color falla
- La propuesta decía **clustering por color HSV** de la camiseta. Lo implementamos... y **no anduvo**.
- Contar el recorrido: HSV con variantes (whack-a-mole: arreglás un clip, rompés otro), **DBSCAN** (nunca separa 2 equipos), color promediado por track (**59-62%, casi azar**).
- **Por qué falla:** el recorte de la camiseta se **contamina con el fondo** (césped, butacas, publicidad), y el fondo cambia en cada partido.
- **La lección más importante del trabajo:** mirando frames sueltos *parecía* bien, pero la **evaluación cuantitativa** mostró que era azar. El ojímetro engaña.

### Asignación de equipo (II) — embeddings
- Solución: en vez del color crudo, usamos un **embedding visual (SigLIP)** del recorte del jugador → UMAP → KMeans, **agregado por track**.
- Idea intuitiva: el embedding "entiende" que es **un jugador de tal camiseta** y es robusto al fondo, en vez de promediar píxeles.
- Resultado: de ~60% a **92-98%** validado contra el ground truth. (Mostrar la tabla 108: 59.5→92, 061: 62.1→97.8.)

### Reorden del pipeline
- Acá se cierra el spoiler de la Parte 1.
- Para agregar por track necesitás **tener los IDs primero** → por eso el **tracking pasó antes** de la asignación de equipo.
- Es una **desviación de la propuesta justificada por los datos**, no un capricho. (Esto suma para la nota: muestra criterio.)

---

## Parte 3 — Posesión, evaluación y cierre

### Estimación de posesión
- **Poseedor** = jugador más cercano a la pelota, medido desde los **pies** (no el torso, la pelota está en el piso), con umbral **relativo a la altura** del jugador → robusto a la perspectiva (un jugador lejos se ve chico).
- El **arquero no cuenta** (decisión nuestra: su saque no es posesión táctica).
- **Problema:** la pelota viaja en los pases → la posesión instantánea **parpadea** y las métricas dan cualquier cosa.
- **Solución (hold + histéresis):** la posesión es un **estado que se mantiene** durante el pase, y solo **cambia** cuando el otro equipo controla de forma **sostenida**. Es el modelo estándar de posesión.

### El cuello: detección de la pelota
- Mostrar la tabla: a 640 la pelota se detecta 10%, a **1280 sube a 25%**, pero el GT dice que la pelota está **presente el 99%** del tiempo.
- **Conclusión:** el problema **no es que la pelota esté fuera de cámara** — es que el **detector no la ve** (objeto chico).
- **Domain gap:** se entrenó con Roboflow (planos cerrados, pelota grande) y se evalúa en SoccerNet (cámara abierta, pelota chica). El recall de pelota cae de **0.89 a ~0.25**.

### Métricas + Overlay
- Las 3 métricas: **% de posesión por equipo, cambios de posesión, duración media de secuencias**.
- Mostrar el frame de overlay (cajas por equipo, pelota, poseedor marcado, scoreboard).
- **Acá es el momento del demo del video** → abrir `outputs/overlay_SNMOT-108_gt.mp4` y dejar correr unos segundos.

### Evaluación contra ground truth manual
- Anotamos **a mano** la posesión de un clip (un asistente que armamos: mirás el video y apretás A/B/ninguno) → un CSV de tramos.
- Comparamos frame a frame contra esa anotación → **~76%** de accuracy con los mejores parámetros.
- **Anécdota fuerte:** la primera medición dio **57%** y nos asustó... resultó ser **varianza de UMAP** en la asignación de equipo (en una corrida mala). Con una corrida buena (94.8% de equipos) la posesión da 76%. Otra vez: medir > intuición.

### Desviaciones + Limitaciones
- **Desviaciones de la propuesta** (todas con respaldo experimental): HSV → embeddings, suavizado → hold/histéresis, reorden del pipeline.
- **Limitaciones honestas:** (1) detección de pelota (el cuello), (2) **no-determinismo de UMAP** (los equipos varían entre corridas), (3) calibración de posesión sobre un solo clip.

### Conclusiones
- Pipeline **completo, integrado en un video, y validado con métricas** (equipos 92-98%, posesión ~76%).
- El cuello del sistema es la **pelota** (domain gap) → ahí está la mejora más grande pendiente.
- **El mensaje de cierre:** la **evaluación cuantitativa fue esencial**; varias veces lo que se "veía bien" estaba mal.
- **Trabajo futuro:** fine-tune del detector de pelota / imgsz 1920, fijar la semilla de UMAP, anotar más clips.

---

## Por si preguntan

- **¿Por qué no SoccerNet para entrenar?** Roboflow ya tenía las 4 clases listas y fue más rápido; SoccerNet lo usamos para evaluar (tiene el ground truth).
- **¿Por qué el arquero no cuenta?** Su saque/descuelgue no es posesión en sentido táctico; lo excluimos de los candidatos.
- **¿Por qué embeddings y no entrenar un clasificador de equipos?** Un clasificador no generaliza a camisetas nuevas sin re-entrenar; el embedding + clustering es **no supervisado** y anda en cualquier partido.
- **¿Cómo sabe cuál es el equipo A y cuál el B?** Son etiquetas arbitrarias; la evaluación prueba los 2 mapeos y se queda con el mejor.
