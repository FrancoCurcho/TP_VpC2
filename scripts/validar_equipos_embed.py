"""Validación de asignación de equipo (módulo team_assign, SigLIP) vs ground truth.

    PYTHONPATH=src uv run --no-project \
        --with ultralytics --with lapx --with opencv-python \
        --with "git+https://github.com/roboflow/sports.git" \
        --with transformers --with umap-learn --with supervision \
        python scripts/validar_equipos_embed.py [SNMOT-XXX] [N_FRAMES] [device]

Trackea, asigna equipo con `team_assign.TeamAssigner` (embeddings SigLIP por
track) y mide la accuracy contra el `gameinfo.ini` de SoccerNet (equipo real por
track), matcheando cada detección con la caja del GT de mayor IoU. Corre por el
mismo código que el pipeline.
"""

import configparser
import sys
from collections import Counter, defaultdict

import cv2

sys.path.insert(0, "src")
from futbol_vpc import config, detection, team_assign, tracking  # noqa: E402

DEVICE_YOLO = "mps"
CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "SNMOT-108"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 400
DEVICE_EMB = sys.argv[3] if len(sys.argv) > 3 else "mps"
CLIP = config.DATA_DIR / "soccernet" / "tracking" / "train" / CLIP_NAME


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0


# --- Ground truth (gameinfo.ini + gt.txt) ---
cfg = configparser.ConfigParser()
cfg.read(CLIP / "gameinfo.ini")
gt_equipo = {}
for key, val in cfg["Sequence"].items():
    if key.startswith("trackletid_"):
        rol = val.lower()
        if "player team left" in rol:
            gt_equipo[int(key.split("_")[1])] = "left"
        elif "player team right" in rol:
            gt_equipo[int(key.split("_")[1])] = "right"

gt_boxes = defaultdict(list)
for line in open(CLIP / "gt" / "gt.txt"):
    p = line.split(",")
    f, tid = int(p[0]), int(p[1])
    l, t, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
    gt_boxes[f].append((tid, (l, t, l + w, t + h)))

# --- Pipeline: tracking + asignación de equipo (módulo) ---
print(f"Trackeando {N_FRAMES} frames de {CLIP_NAME}...")
modelo = detection.cargar_modelo()
frames = [cv2.imread(str(CLIP / "img1" / f"{i:06d}.jpg")) for i in range(1, N_FRAMES + 1)]
por_frame = tracking.trackear(modelo, frames, device=DEVICE_YOLO)

print(f"Asignando equipos (SigLIP, device={DEVICE_EMB})...")
team_assign.TeamAssigner(device=DEVICE_EMB).asignar(por_frame, frames)

# --- Validación vs GT ---
conf = Counter()
for idx, dets in enumerate(por_frame):
    gtb = gt_boxes.get(idx + 1, [])
    for d in dets:
        if d.cls != "player" or d.team is None:
            continue
        mejor_tid, mejor_iou = None, 0.3
        for tid, box in gtb:
            if tid in gt_equipo and iou(d.bbox, box) > mejor_iou:
                mejor_iou, mejor_tid = iou(d.bbox, box), tid
        if mejor_tid is not None:
            conf[(d.team, gt_equipo[mejor_tid])] += 1

total = sum(conf.values())
m1 = conf[("equipo_A", "left")] + conf[("equipo_B", "right")]
m2 = conf[("equipo_A", "right")] + conf[("equipo_B", "left")]
acc = max(m1, m2) / total if total else 0
print(f"\n=== Equipos (SigLIP) vs ground truth ({CLIP_NAME}) ===")
print(f"Detecciones matcheadas: {total}")
print(f"Accuracy de asignación de equipo: {acc:.1%}")
