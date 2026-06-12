# TP Visión por Computadora II (CEIA) — Grupo 7

**Detección de jugadores y estimación de posesión de pelota en video de fútbol**

Facundo Rivas · Franco Curcho · Juan Ignacio Teich

---

## Objetivo

Construir un pipeline de visión por computadora que, a partir de video de fútbol,
detecte jugadores/árbitro/pelota, asigne cada jugador a su equipo, mantenga
identidades estables en el tiempo y estime la **posesión de la pelota**,
produciendo métricas agregadas y un video con overlay que las visualiza.

## Pipeline

```
Video ─▶ [1] Detección ─▶ [2] Tracking ─▶ Asignación de equipo ─▶ [3] Posesión ─▶ Métricas + Overlay
            (YOLO26m)      (ByteTrack)      (embeddings, por track)   (jugador más    % posesión,
                            IDs estables                               cercano +       cambios, duración
                                                                       hold/histéresis)
```

| Etapa | Módulo | Estado |
|---|---|---|
| 1 · Detección por frame | `detection.py` | ✅ YOLO26m (mAP@50 ≈ 0.95) |
| 2 · Tracking (ByteTrack) | `tracking.py` | ✅ IDs estables |
| · Asignación de equipo | `team_assign.py` | ✅ embeddings SigLIP por track (92-98%) |
| 3 · Posesión | `possession.py` | ✅ jugador más cercano + hold/histéresis (~76%) |
| 3 · Métricas | `metrics.py` | ✅ % posesión, cambios, duración |
| 3 · Video con overlay | `video.py` | ✅ genera mp4 |

> **Desviaciones de la propuesta (con respaldo experimental — ver `docs/decisiones/`):**
> la asignación de equipo se hace con **embeddings** (no clustering HSV, que dio
> ~60%); el suavizado de posesión es **hold + histéresis** (no ventana); y el
> **tracking corre antes** de la asignación de equipo (la agregación por track lo
> requiere).

## Resultados

| Componente | Resultado | Validación |
|---|---|---|
| Detección | mAP@50 0.95 / mAP@50-95 0.56 | comparación de 3 modelos |
| Tracking | IDs estables, 0 fragmentación | conteo de IDs |
| **Asignación de equipo** | **92-98%** | vs `gameinfo.ini` de SoccerNet |
| **Posesión** | **~76%** | vs ground truth **manual** |
| Detección de pelota | 25% (limitación) | vs GT (pelota presente 99%) |

**Limitaciones conocidas:** (1) la **detección de la pelota** es el cuello del
sistema en vivo (objeto chico, domain gap); (2) la asignación de equipo (UMAP)
**varía entre corridas** — principal fuente de varianza. Detalle en
`docs/decisiones/`.

## Estructura del repo

```
src/futbol_vpc/      Paquete Python — un módulo por etapa del pipeline
notebooks/           00_train_*_legacy, 01_train_detector, 02_compare_models
models/              Modelos entrenados (best.pt + results.csv + gráficos)
scripts/             Descarga, pruebas, validación, anotación y evaluación
data/videos/         gol.mp4 (prueba visual)
data/ground_truth/   Anotaciones de posesión (CSV de tramos) — evaluación
data/soccernet/      Descarga de SoccerNet-Tracking (gitignored, NDA)
outputs/             Videos/imágenes generados (gitignored, NDA)
docs/                Propuesta + decisiones/ (registro de experimentos)
```

## Decisiones y experimentos

Todo el recorrido (enfoques probados, resultados medidos, decisiones) está en
**`docs/decisiones/`**:
- [`modelo.md`](docs/decisiones/modelo.md) — comparación de modelos, hiperparámetros.
- [`dataset.md`](docs/decisiones/dataset.md) — Roboflow (train) + SoccerNet (eval).
- [`deteccion-y-asignacion.md`](docs/decisiones/deteccion-y-asignacion.md) — color → DBSCAN → embeddings.
- [`posesion.md`](docs/decisiones/posesion.md) — modelo de posesión, hold/histéresis, evaluación.

## Datasets

- **Entrenamiento (detección):** dataset de Roboflow `footbaldetection-tiny4/finalv2`,
  4 clases (`ball`, `goalkeeper`, `player`, `referee`).
- **Evaluación:** **SoccerNet-Tracking** (clips de 30s con anotaciones de tracking
  + equipo). Bajo NDA → no se versiona; se reproduce con `scripts/download_soccernet.py`.

## Cómo correr

### Instalación local (Mac / Linux)

Dependencias con [`uv`](https://docs.astral.sh/uv/) (instala también el paquete
`futbol_vpc` editable y el stack de embeddings — torch, transformers, umap, sports):

```shell
uv sync
```

La Mac (Apple Silicon) sirve para **desarrollo, inferencia y el pipeline** (MPS).
El **entrenamiento** del detector se hace en Colab (GPU NVIDIA).

### Correr el pipeline / evaluación

```shell
# 1. Descargar un clip de SoccerNet-Tracking (requiere password del NDA)
export SOCCERNET_PASSWORD="..."
uv run python scripts/download_soccernet.py     # baja train.zip; extraer SNMOT-108

# 2. Generar el video con overlay (pipeline completo)
PYTHONPATH=src uv run python scripts/generar_overlay.py SNMOT-108

# 3. Evaluar posesión contra el ground truth anotado
PYTHONPATH=src uv run python scripts/evaluar_posesion.py SNMOT-108
```

Anotar el ground truth de posesión de un clip (asistente por teclado):

```shell
uv run python scripts/anotar_posesion.py SNMOT-108   # a/b/n + barra; q guarda
```

### Entrenamiento del detector en Google Colab

No hace falta clonar el repo: entrenar solo necesita `ultralytics` + el dataset de
Roboflow. Abrir `notebooks/01_train_detector.ipynb` en Colab (con GPU), que recorre
los 3 modelos (`yolo11m`, `yolo26m`, `yolov8m`) y guarda los resultados en Drive;
después se bajan a `models/<modelo>/`. La comparación se cierra con
`notebooks/02_compare_models.ipynb` (gana **YOLO26m**). Detalle y parámetros en
[`docs/decisiones/modelo.md`](docs/decisiones/modelo.md).

> La API key de Roboflow viene puesta en el notebook de entrenamiento. Para correr
> local, se lee de `ROBOFLOW_API_KEY`.

## Estado del trabajo

✅ Las 5 etapas implementadas, integradas en un video con overlay, y validadas con
métricas (equipos 92-98%, posesión ~76% vs ground truth manual). El recorrido
completo de experimentos está documentado en `docs/decisiones/`.
