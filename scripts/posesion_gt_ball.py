"""Validación de la LÓGICA de posesión usando la pelota del GROUND TRUTH.

Decoupling: la detección de la pelota es el cuello (ver posesion.md Exp 4). Acá
usamos nuestro tracking + equipos para los jugadores, pero **inyectamos la
posición real de la pelota** (gt.txt) en vez de la detectada, para validar que la
regla "jugador más cercano = poseedor" produce posesión y métricas sensatas.

    PYTHONPATH=src uv run --no-project \
        --with ultralytics --with lapx --with opencv-python \
        --with "git+https://github.com/roboflow/sports.git" \
        --with transformers --with umap-learn --with supervision \
        python scripts/posesion_gt_ball.py [SNMOT-XXX] [N_FRAMES] [FRAME_DIBUJO]
"""

import configparser
import sys

import cv2

sys.path.insert(0, "src")
from futbol_vpc import config, detection, metrics, possession, team_assign, tracking  # noqa: E402
from futbol_vpc.detection import Detection  # noqa: E402

DEVICE = "mps"
CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "SNMOT-108"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 400
FRAME_DIBUJO = int(sys.argv[3]) if len(sys.argv) > 3 else 375
UMBRAL = float(sys.argv[4]) if len(sys.argv) > 4 else None
CLIP = config.DATA_DIR / "soccernet" / "tracking" / "train" / CLIP_NAME

# --- Posición de la pelota del GT (gt.txt) ---
cfg = configparser.ConfigParser()
cfg.read(CLIP / "gameinfo.ini")
ball_id = next(int(k.split("_")[1]) for k, v in cfg["Sequence"].items()
               if k.startswith("trackletid_") and v.strip().lower().startswith("ball"))
ball_pos = {}
for line in open(CLIP / "gt" / "gt.txt"):
    p = line.split(",")
    f, tid = int(p[0]), int(p[1])
    if tid == ball_id:
        l, t, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        ball_pos[f] = (l, t, l + w, t + h)

# --- Tracking de jugadores + equipos ---
print(f"Trackeando {N_FRAMES} frames de {CLIP_NAME}...")
modelo = detection.cargar_modelo()
frames = [cv2.imread(str(CLIP / "img1" / f"{i:06d}.jpg")) for i in range(1, N_FRAMES + 1)]
por_frame = tracking.trackear(modelo, frames, device=DEVICE)
team_assign.TeamAssigner(device=DEVICE).asignar(por_frame, frames)

# --- Inyectar la pelota del GT (descartar la detectada) ---
con_ball_gt = 0
for i, dets in enumerate(por_frame):
    dets[:] = [d for d in dets if d.cls != "ball"]
    f = i + 1
    if f in ball_pos:
        dets.append(Detection(bbox=ball_pos[f], cls="ball", conf=1.0))
        con_ball_gt += 1
print(f"Frames con pelota GT inyectada: {con_ball_gt}/{N_FRAMES}")

# --- Barrido de umbral (¿el umbral es muy estricto, o la pelota viaja lejos?) ---
print("\numbral_rel | frames con poseedor (pelota GT)")
for u in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
    n = sum(1 for dets in por_frame if possession.poseedor_en_frame(dets, umbral_rel=u))
    print(f"   {u:<5} | {n}/{N_FRAMES} ({n/N_FRAMES:.0%})")

# --- Posesión + métricas: barrido de histéresis ---
print("\n=== Posesión con pelota GT (barrido de histéresis) ===")
print(" histeresis | cobertura | %A/%B          | cambios | dur.media")
for h in [4, 8, 12, 20, 30]:
    serie = possession.secuencia_posesion(por_frame, umbral_rel=UMBRAL, histeresis=h)
    met = metrics.resumen(serie, fps=25)
    cob = sum(1 for t in serie if t is not None) / N_FRAMES
    pa = met["porcentaje_posesion"].get("equipo_A", 0)
    pb = met["porcentaje_posesion"].get("equipo_B", 0)
    print(f"    {h:<7} | {cob:>6.0%}    | {pa:>4.0%}/{pb:<4.0%}      "
          f"| {met['cambios_de_posesion']:>5}   | {met['duracion_media_seg']:.1f}s")

# serie con histéresis default, para el dibujo
serie = possession.secuencia_posesion(por_frame, umbral_rel=UMBRAL)

# --- Dibujar frame con el poseedor marcado ---
idx = FRAME_DIBUJO - 1
COL = {"equipo_A": (0, 0, 255), "equipo_B": (255, 0, 0), None: (128, 128, 128)}
out = frames[idx].copy()
for d in por_frame[idx]:
    x1, y1, x2, y2 = map(int, d.bbox)
    c = COL[d.team] if d.cls == "player" else ((0, 255, 255) if d.cls == "ball" else (0, 255, 0))
    cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
pos = possession.poseedor_en_frame(por_frame[idx])
pelotas = [d for d in por_frame[idx] if d.cls == "ball"]
if pos and pelotas:
    px, py = map(int, possession._pies(pos.bbox))
    bx, by = map(int, pelotas[0].centro)
    cv2.circle(out, (px, py), 14, (0, 255, 255), 3)
    cv2.line(out, (px, py), (bx, by), (0, 255, 255), 2)
    cv2.putText(out, "POSESION", (px - 30, py + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
config.OUTPUTS_DIR.mkdir(exist_ok=True)
salida = config.OUTPUTS_DIR / f"posesion_gtball_{CLIP_NAME}_f{FRAME_DIBUJO}.jpg"
cv2.imwrite(str(salida), out)
print("\nFrame anotado guardado en:", salida)
