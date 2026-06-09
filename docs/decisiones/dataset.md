# Decisiones — Dataset

Última actualización: **2026-06-09**.

---

## 1. Dataset de entrenamiento (detección)

- **Fuente:** Roboflow — workspace `footbaldetection-tiny4`, proyecto `finalv2`,
  versión `1`.
- **Formato de exportación:** `yolo11` (formato Ultralytics: `data.yaml` +
  labels `.txt` con cajas normalizadas). Los formatos YOLO de Roboflow
  (`yolov8`/`yolo11`/`yolo26`) son **intercambiables** para detección, así que
  con una sola descarga entrenamos los tres modelos comparados.
- **Clases (4):** `ball`, `goalkeeper`, `player`, `referee`.

**Nota sobre la propuesta:** la propuesta original mencionaba **SoccerNet** como
dataset. Para el *entrenamiento del detector* se usó este dataset de Roboflow
(`finalv2`), que ya provee las 4 clases necesarias. SoccerNet se usa para la
etapa de **evaluación** (abajo).

---

## 2. Dataset de evaluación: SoccerNet-Tracking

Para testear el pipeline completo y construir el ground truth se eligió
**SoccerNet-Tracking** (subset de SoccerNet, el dataset de la propuesta).

**Por qué SoccerNet-Tracking y no partidos completos ni clips sueltos:**

- Son **clips de ~30 s** desde la **cámara táctica** (la principal de
  transmisión), justo el largo y el tipo de toma que necesitamos para métricas de
  posesión testeables.
- **Vienen con anotaciones de tracking**: caja + ID por frame, y un
  `gameinfo.ini` que mapea cada track a su **equipo/rol**. Eso nos da ground
  truth gratis para validar tracking y **asignación de equipo**, y una base para
  el ground truth de posesión.
- Es la fuente **canónica** y la más defendible para el TP (está en la propuesta).

**Split elegido: `train`.** Tiene las anotaciones completas (`gt/gt.txt` +
`gameinfo.ini`); en `test`/`challenge` suelen estar ocultas. Son 57 clips (~9,6
GB); usamos **uno** para empezar, el resto queda disponible.

**Estructura por clip (`SNMOT-XXX/`):**

```
img1/000001.jpg ...   frames (25 fps, ~750 = 30 s)
gt/gt.txt             MOT: frame, id, left, top, w, h, conf, ...
gameinfo.ini          equipo/rol de cada track id
seqinfo.ini           tamaño, fps, cantidad de frames
```

Cada clip es **1920×1080, 25 fps, 750 frames (30 s)**. La descarga vive en
`data/soccernet/` (fuera de git por `.gitignore` — NDA + peso). Se reproduce con
`scripts/download_soccernet.py`. **No se versiona** ningún frame/clip de SoccerNet
(material bajo NDA, repo público); las salidas anotadas (`outputs/`) también están
gitignoreadas.

---

## 3. Clips elegidos

De los 57 clips se eligieron **5** (criterio: juego abierto desde la cámara
táctica, dos camisetas distinguibles, presencia de pelota y cambios de posesión),
cubriendo **3 esquemas de color** distintos para estresar la asignación de equipo:

| Clip | Partido / colores | Acción | Rol |
|---|---|---|---|
| **SNMOT-108** | Thun · rojo vs gris oscuro | Clearance | **primario** (mayor contraste) |
| SNMOT-169 | Xamax · rojo-negro vs blanco | Clearance | backup |
| SNMOT-061 | St. Gallen · verde vs blanco | Shots on target | backup (camiseta clara) |
| SNMOT-157 | Xamax · rojo-negro vs blanco-azul | Clearance | backup |
| SNMOT-065 | St. Gallen · verde vs blanco | Clearance | backup |

Extraídos y usados hasta ahora: **108** (rojo/gris) y **061** (verde/blanco), los
dos esquemas más difíciles. El resto se extrae del `train.zip` si hace falta.

## 4. Ground truth usado

- **Asignación de equipo — ✅ validado.** Se usa `gameinfo.ini` (equipo real de
  cada track) + `gt/gt.txt` (cajas por frame); se matchea cada detección con la
  caja del GT de mayor IoU. Resultado: ver
  [deteccion-y-asignacion.md](deteccion-y-asignacion.md) (92-98%).
- **Posesión — pendiente.** A partir de la posición de la pelota (`gt.txt`) + los
  tracks, derivar quién tiene la pelota frame a frame para la evaluación de la
  Etapa 3.

> `gol.mp4` (5,6 s, 848×384) queda solo como prueba visual rápida, no para métricas.
