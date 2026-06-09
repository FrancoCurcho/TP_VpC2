"""Diagnóstico + calibración de posesión contra el ground truth (Etapa 5).

Trackea una vez (pelota GT, para aislar la lógica) y:
  1. mide la accuracy de ASIGNACIÓN DE EQUIPO en esta corrida (vs gameinfo.ini),
  2. barre umbral × histéresis contra el GT de posesión y reporta la mejor combo.

    PYTHONPATH=src uv run --no-project \
        --with ultralytics --with lapx --with opencv-python \
        --with "git+https://github.com/roboflow/sports.git" \
        --with transformers --with umap-learn --with supervision \
        python scripts/calibrar_posesion.py [SNMOT-XXX]
"""

import configparser
import sys

import cv2

sys.path.insert(0, "src")
from futbol_vpc import config, detection, possession, team_assign, tracking  # noqa: E402
from futbol_vpc.detection import Detection  # noqa: E402

DEVICE = "mps"
CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "SNMOT-108"
CLIP = config.DATA_DIR / "soccernet" / "tracking" / "train" / CLIP_NAME


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


# --- GT de posesión ---
tramos = []
for line in open(config.GROUND_TRUTH_DIR / f"{CLIP_NAME}_posesion.csv"):
    if not line[0].isdigit():
        continue
    a, b, eq = line.strip().split(",")
    tramos.append((int(a), int(b), eq))
N = max(b for _, b, _ in tramos)
gt_pos = [None] * N
for a, b, eq in tramos:
    if eq.startswith("equipo"):
        for f in range(a - 1, b):
            gt_pos[f] = eq

# --- GT de equipos + pelota ---
cfg = configparser.ConfigParser()
cfg.read(CLIP / "gameinfo.ini")
gt_equipo, ball_id = {}, None
for k, v in cfg["Sequence"].items():
    if k.startswith("trackletid_"):
        r = v.lower()
        if "player team left" in r:
            gt_equipo[int(k.split("_")[1])] = "left"
        elif "player team right" in r:
            gt_equipo[int(k.split("_")[1])] = "right"
        elif r.strip().startswith("ball"):
            ball_id = int(k.split("_")[1])
gt_boxes, ball_pos = {}, {}
for line in open(CLIP / "gt" / "gt.txt"):
    p = line.split(",")
    f, tid = int(p[0]), int(p[1])
    l, t, w, h = map(float, p[2:6])
    box = (l, t, l + w, t + h)
    gt_boxes.setdefault(f, []).append((tid, box))
    if tid == ball_id:
        ball_pos[f] = box

# --- Pipeline ---
print(f"Trackeando {N} frames de {CLIP_NAME}...")
modelo = detection.cargar_modelo()
frames = [cv2.imread(str(CLIP / "img1" / f"{i:06d}.jpg")) for i in range(1, N + 1)]
por_frame = tracking.trackear(modelo, frames, device=DEVICE)
team_assign.TeamAssigner(device=DEVICE).asignar(por_frame, frames)

# (1) accuracy de equipos en esta corrida
from collections import Counter  # noqa: E402
conf = Counter()
for idx, dets in enumerate(por_frame):
    for d in dets:
        if d.cls != "player" or d.team is None:
            continue
        mt, mi = None, 0.3
        for tid, box in gt_boxes.get(idx + 1, []):
            if tid in gt_equipo and iou(d.bbox, box) > mi:
                mi, mt = iou(d.bbox, box), tid
        if mt is not None:
            conf[(d.team, gt_equipo[mt])] += 1
tot = sum(conf.values())
m1 = conf[("equipo_A", "left")] + conf[("equipo_B", "right")]
m2 = conf[("equipo_A", "right")] + conf[("equipo_B", "left")]
print(f"Accuracy de EQUIPOS en esta corrida: {max(m1, m2)/tot:.1%}")

# inyectar pelota GT
for i, dets in enumerate(por_frame):
    dets[:] = [d for d in dets if d.cls != "ball"]
    if (i + 1) in ball_pos:
        dets.append(Detection(bbox=ball_pos[i + 1], cls="ball", conf=1.0))

# (2) barrido umbral × histéresis vs GT de posesión
idx_pos = [i for i, g in enumerate(gt_pos) if g is not None]


def acc_pos(serie):
    ma = {"equipo_A": "equipo_A", "equipo_B": "equipo_B"}
    mb = {"equipo_A": "equipo_B", "equipo_B": "equipo_A"}
    f = lambda mp: sum(1 for i in idx_pos if serie[i] and mp[serie[i]] == gt_pos[i])
    return max(f(ma), f(mb)) / len(idx_pos)


print("\numbral \\ hist |   2      4      8")
for u in [0.5, 0.75, 1.0, 1.5]:
    fila = []
    for h in [2, 4, 8]:
        s = possession.secuencia_posesion(por_frame, umbral_rel=u, histeresis=h)
        fila.append(f"{acc_pos(s):.0%}")
    print(f"    {u:<8} | {fila[0]:>5}  {fila[1]:>5}  {fila[2]:>5}")
