"""Genera el video con overlay del pipeline completo (Etapa 4).

    PYTHONPATH=src uv run --no-project \
        --with ultralytics --with lapx --with opencv-python \
        --with "git+https://github.com/roboflow/sports.git" \
        --with transformers --with umap-learn --with supervision \
        python scripts/generar_overlay.py [SNMOT-XXX] [N_FRAMES] [ball=gt|det]

Pipeline: detección → tracking → equipos (embeddings) → posesión. Dibuja overlay
(cajas por equipo, pelota, poseedor, scoreboard con % acumulado) y exporta mp4 a
outputs/. Por defecto usa la pelota del GT (para validar el método en movimiento).
"""

import configparser
import sys

import cv2

sys.path.insert(0, "src")
from futbol_vpc import config, detection, metrics, possession, team_assign, tracking, video  # noqa: E402
from futbol_vpc.detection import Detection  # noqa: E402

DEVICE = "mps"
CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "SNMOT-108"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 750
BALL = sys.argv[3] if len(sys.argv) > 3 else "gt"   # 'gt' o 'det'
CLIP = config.DATA_DIR / "soccernet" / "tracking" / "train" / CLIP_NAME
FPS = 25

print(f"Pipeline sobre {N_FRAMES} frames de {CLIP_NAME} (pelota={BALL})...")
modelo = detection.cargar_modelo()
frames = [cv2.imread(str(CLIP / "img1" / f"{i:06d}.jpg")) for i in range(1, N_FRAMES + 1)]
# imgsz=1280 si usamos la pelota detectada (mejor recall de pelota)
por_frame = tracking.trackear(modelo, frames, device=DEVICE, imgsz=1280 if BALL == "det" else None)
team_assign.TeamAssigner(device=DEVICE).asignar(por_frame, frames)

# Pelota del GT (decoupling de la detección) si corresponde
if BALL == "gt":
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

# Posesión + render frame a frame con % acumulado
serie = possession.secuencia_posesion(por_frame)
anotados = []
for i, (frame, dets) in enumerate(zip(frames, por_frame)):
    pct = metrics.porcentaje_posesion(serie[: i + 1])
    poseedor = possession.poseedor_en_frame(dets)
    anotados.append(video.anotar_frame(frame, dets, poseedor, serie[i], pct))

config.OUTPUTS_DIR.mkdir(exist_ok=True)
salida = config.OUTPUTS_DIR / f"overlay_{CLIP_NAME}_{BALL}.mp4"
video.exportar_video(anotados, salida, fps=FPS)
cv2.imwrite(str(config.OUTPUTS_DIR / f"overlay_{CLIP_NAME}_{BALL}_sample.jpg"), anotados[N_FRAMES // 2])

met = metrics.resumen(serie, fps=FPS)
print(f"\nMétricas: {met}")
print("Video guardado en:", salida)
