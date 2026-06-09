# TP Visión por Computadora II (CEIA) — Grupo 7

**Detección de jugadores y estimación de posesión de pelota en video de fútbol**

Facundo Rivas · Franco Curcho · Juan Ignacio Teich

---

## Objetivo

Construir un pipeline de visión por computadora que, a partir de video de
fútbol, detecte jugadores/árbitro/pelota, asigne cada jugador a su equipo,
mantenga identidades estables en el tiempo y estime la **posesión de la pelota**,
produciendo métricas agregadas y un video con overlay que las visualiza.

## Pipeline (3 etapas)

```
Video ─▶ [1] Detección (YOLO)          ─▶ [2] Tracking (ByteTrack) ─▶ [3] Posesión ─▶ Métricas + Video overlay
            + Asignación de equipo (HSV)        IDs estables             jugador más      % posesión, cambios,
                                                                          cercano a la     duración de secuencias
                                                                          pelota + suavizado
```

| Etapa | Módulo | Estado |
|---|---|---|
| 1 · Detección por frame | `detection.py` | ✅ modelo entrenado (ver `models/`) |
| 1 · Asignación de equipo (HSV) | `team_assign.py` | ⏳ pendiente |
| 2 · Tracking (ByteTrack) | `tracking.py` | ⏳ pendiente |
| 3 · Posesión + suavizado | `possession.py` | ⏳ pendiente |
| 3 · Métricas agregadas | `metrics.py` | ⏳ pendiente |
| 3 · Video con overlay | `video.py` | ⏳ pendiente |

## Estructura del repo

```
src/futbol_vpc/      Paquete Python con la lógica del pipeline (un módulo por etapa)
notebooks/           Notebooks finos que orquestan y muestran resultados
  00_train_*_legacy  Entrenamientos previos (YOLO11x, YOLO26m) — referencia
models/              Modelos entrenados: best.pt + results.csv + gráficos por modelo
data/videos/         Videos de prueba (gol.mp4)
data/ground_truth/   Anotaciones de posesión frame a frame (evaluación)
outputs/             Videos con overlay y métricas generadas
docs/                Propuesta del trabajo
  decisiones/        Registro de decisiones (modelo, dataset, detección y asignación)
```

## Dataset

Se entrena sobre un dataset de fútbol alojado en **Roboflow**
(`footbaldetection-tiny4/finalv2`), con 4 clases: `ball`, `goalkeeper`,
`player`, `referee`. La descarga requiere una API key de Roboflow (ver abajo).

## Modelos y comparación

Se comparan tres arquitecturas de detección, **todas en tamaño `m`** para que la
comparación sea justa (mismo tamaño, mismo `data.yaml`, mismas épocas):

- **YOLO11m** (CNN)
- **YOLO26m** (CNN)
- **YOLOv8m** (CNN, baseline clásico)

Métricas de comparación: mAP@50 / mAP@50-95 (global y por clase, con foco en
`ball`) y FPS de inferencia. El modelo ganador se usa en el resto del pipeline;
si la pelota queda floja, el lever es subir `imgsz` (640 → 1280) antes que el
tamaño del modelo.

> Nota: en `models/yolo11x/` quedan los resultados de un entrenamiento previo en
> tamaño `x` (extra-large), conservado como referencia.

## Cómo correr

### Instalación local (Mac / Linux)

Dependencias gestionadas con [`uv`](https://docs.astral.sh/uv/):

```shell
uv sync
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=tp_vpc2
```

Esto instala también el paquete `futbol_vpc` en modo editable, así que desde
cualquier notebook podés `from futbol_vpc import detection, tracking, ...`.

La Mac (Apple Silicon) sirve para **desarrollo e inferencia** (backend MPS). El
**entrenamiento** conviene hacerlo en Colab (GPU NVIDIA, más rápido y consistente).

### Entrenamiento en Google Colab

El entrenamiento se hace en Colab (GPU NVIDIA). **No hace falta clonar el repo**:
entrenar solo necesita `ultralytics` + el dataset de Roboflow. El paquete
`futbol_vpc` (pipeline, etapas 2-3) se corre local en la Mac, no acá.

**1. Abrir el notebook en Colab.** Dos opciones:
- *Desde GitHub:* en Colab → `Archivo → Abrir cuaderno → pestaña GitHub` →
  autorizar el acceso → elegir el repo → abrir `notebooks/02_compare_models.ipynb`.
  (Funciona con repos privados vía OAuth, sin tokens.)
- *Subiéndolo:* bajar el `.ipynb` y en Colab → `Archivo → Subir cuaderno`.

**2. API key de Roboflow.** Ya viene puesta en el notebook (`ROBOFLOW_API_KEY`),
así que no hay que configurar nada. Si la regeneran en Roboflow, actualizar ese
valor en la celda.

**3. Bajar el dataset y entrenar.** El notebook recorre los 3 modelos
(`yolo11m.pt`, `yolo26m.pt`, `yolov8m.pt`) en un solo run:

```python
!pip install ultralytics roboflow

from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
dataset = rf.workspace("footbaldetection-tiny4").project("finalv2").version(1).download("yolo11")

from ultralytics import YOLO
model = YOLO("yolo11m.pt")          # <-- cambiar por modelo: yolo11m / yolo26m / yolov8m
model.train(
    data="FinalV2-1/data.yaml",
    epochs=80,
    imgsz=640,                       # subir a 1280 solo en el modelo ganador si la pelota queda floja
    hsv_s=0.5, hsv_v=0.5,
    translate=0.1, scale=0.6, perspective=0.0005, shear=40, fliplr=0.5,
    exist_ok=True,
)
```

**4. Guardar resultados.** Montar Drive y copiar ahí los artefactos del run:

```python
from google.colab import drive; drive.mount('/content/drive')
import shutil, os
destino = "/content/drive/MyDrive/TP_VpC2_models/yolo11m"   # cambiar por modelo
os.makedirs(destino, exist_ok=True)
for f in ["weights/best.pt", "results.csv", "results.png",
          "confusion_matrix.png", "BoxPR_curve.png"]:
    shutil.copy(f"runs/detect/train/{f}", destino)
```

**5. Versionar.** Desde la Mac, bajar esos archivos de Drive a
`models/<modelo>/` y hacer commit. (No se commitea desde Colab.)

> Parámetros fijos para que la comparación sea justa: mismo `imgsz=640`,
> `epochs=80` y mismas augmentations en los tres modelos. Lo único que cambia es
> el peso base (`yolo11m` / `yolo26m` / `yolov8m`).

### Credenciales (local)

Para correr local, la API key se lee de la variable de entorno `ROBOFLOW_API_KEY`
(nunca hardcodeada en el código):

```shell
export ROBOFLOW_API_KEY="..."
```

## Estado del trabajo

- ✅ Etapa 1: detector YOLO entrenado con buenas métricas (mAP@50 ≈ 0.95).
- ⏳ Comparación normalizada de los 3 modelos en tamaño `m`.
- ⏳ Etapas 2 y 3 (tracking, posesión, métricas, overlay).
- ⏳ Ground truth de posesión y evaluación final.
