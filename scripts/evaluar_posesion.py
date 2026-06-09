"""Evaluación de posesión contra el ground truth anotado (Etapa 5).

    PYTHONPATH=src uv run --no-project \
        --with ultralytics --with lapx --with opencv-python \
        --with "git+https://github.com/roboflow/sports.git" \
        --with transformers --with umap-learn --with supervision \
        python scripts/evaluar_posesion.py [SNMOT-XXX]

Corre el pipeline con pelota DETECTADA (imgsz=1280) y con pelota GT, y compara su
posesión frame-a-frame contra data/ground_truth/<clip>_posesion.csv. Prueba los 2
mapeos A/B, excluye los frames 'ninguno' del GT, y reporta cobertura + accuracy +
el % de posesión (sistema vs GT).
"""

import configparser
import sys

import cv2

sys.path.insert(0, "src")
from futbol_vpc import config, detection, metrics, possession, team_assign, tracking  # noqa: E402
from futbol_vpc.detection import Detection  # noqa: E402

DEVICE = "mps"
CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "SNMOT-108"
CLIP = config.DATA_DIR / "soccernet" / "tracking" / "train" / CLIP_NAME
GT_CSV = config.GROUND_TRUTH_DIR / f"{CLIP_NAME}_posesion.csv"

# --- Ground truth de posesión (CSV de tramos → serie por frame) ---
tramos = []
for line in open(GT_CSV):
    if line.startswith("frame_inicio"):
        continue
    a, b, eq = line.strip().split(",")
    tramos.append((int(a), int(b), eq))
N = max(b for _, b, _ in tramos)
gt = [None] * N
for a, b, eq in tramos:
    if eq in ("equipo_A", "equipo_B"):
        for f in range(a - 1, b):
            gt[f] = eq

# --- Pipeline: tracking (1280) + equipos ---
print(f"Trackeando {N} frames de {CLIP_NAME} (imgsz=1280)...")
modelo = detection.cargar_modelo()
frames = [cv2.imread(str(CLIP / "img1" / f"{i:06d}.jpg")) for i in range(1, N + 1)]
por_frame = tracking.trackear(modelo, frames, device=DEVICE, imgsz=1280)
team_assign.TeamAssigner(device=DEVICE).asignar(por_frame, frames)

# --- Posesión con pelota DETECTADA ---
serie_det = possession.secuencia_posesion(por_frame)

# --- Reemplazar por pelota GT y recomputar ---
cfg = configparser.ConfigParser()
cfg.read(CLIP / "gameinfo.ini")
ball_id = next(int(k.split("_")[1]) for k, v in cfg["Sequence"].items()
               if k.startswith("trackletid_") and v.strip().lower().startswith("ball"))
ball_pos = {}
for line in open(CLIP / "gt" / "gt.txt"):
    p = line.split(",")
    if int(p[1]) == ball_id:
        l, t, w, h = map(float, p[2:6])
        ball_pos[int(p[0])] = (l, t, l + w, t + h)
for i, dets in enumerate(por_frame):
    dets[:] = [d for d in dets if d.cls != "ball"]
    if (i + 1) in ball_pos:
        dets.append(Detection(bbox=ball_pos[i + 1], cls="ball", conf=1.0))
serie_gt = possession.secuencia_posesion(por_frame)


def evaluar(serie, nombre):
    idx = [i for i, g in enumerate(gt) if g is not None]          # frames con posesión en el GT
    total = len(idx)
    m1 = {"equipo_A": "equipo_A", "equipo_B": "equipo_B"}
    m2 = {"equipo_A": "equipo_B", "equipo_B": "equipo_A"}

    def aciertos(mapa):
        return sum(1 for i in idx if serie[i] is not None and mapa[serie[i]] == gt[i])

    mejor = m1 if aciertos(m1) >= aciertos(m2) else m2
    correctos = aciertos(mejor)
    predichos = sum(1 for i in idx if serie[i] is not None)
    print(f"\n=== Posesión con pelota {nombre} ===")
    print(f"  cobertura: {predichos}/{total} ({predichos/total:.0%}) de los frames con posesión GT")
    print(f"  accuracy (sobre predichos): {correctos/predichos:.1%}" if predichos else "  sin predicciones")
    print(f"  accuracy global (None = error): {correctos/total:.1%}")
    # % del sistema (mapeado) y matriz de confusión
    sa = sum(1 for i in idx if serie[i] is not None and mejor[serie[i]] == "equipo_A")
    sb = sum(1 for i in idx if serie[i] is not None and mejor[serie[i]] == "equipo_B")
    print(f"  % sistema (mapeado): A {sa/predichos:.0%} / B {sb/predichos:.0%}")
    conf = {("equipo_A", "equipo_A"): 0, ("equipo_A", "equipo_B"): 0,
            ("equipo_B", "equipo_A"): 0, ("equipo_B", "equipo_B"): 0}
    for i in idx:
        if serie[i] is not None:
            conf[(mejor[serie[i]], gt[i])] += 1
    print(f"  confusión (sys→gt): AA={conf[('equipo_A','equipo_A')]} AB={conf[('equipo_A','equipo_B')]} "
          f"BA={conf[('equipo_B','equipo_A')]} BB={conf[('equipo_B','equipo_B')]}")


# % de posesión del GT (referencia)
gt_a = sum(1 for g in gt if g == "equipo_A")
gt_b = sum(1 for g in gt if g == "equipo_B")
tot = gt_a + gt_b
print(f"\nGround truth — %A/%B: {gt_a/tot:.0%}/{gt_b/tot:.0%}  ({tot} frames con posesión)")
evaluar(serie_det, "DETECTADA (imgsz=1280)")
evaluar(serie_gt, "GT")
