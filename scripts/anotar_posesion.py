"""Asistente de anotación de posesión (ground truth, Etapa 5).

Reproduce un clip y vos marcás con el teclado qué equipo tiene la pelota. El
estado se MANTIENE hasta que lo cambiás (la posesión es escalonada), así que solo
apretás una tecla cuando cambia. Guarda los tramos en un CSV en data/ground_truth/.

    uv run --no-project --with opencv-python \
        python scripts/anotar_posesion.py [SNMOT-XXX] [frame_inicio]

Teclas:
    a  → equipo A (izquierda en el GT)      b  → equipo B (derecha)
    n  → nadie / pelota fuera de juego      barra espaciadora → pausar / reanudar
    ,  → retroceder 1 frame                 .  → avanzar 1 frame (en pausa)
    + / -  → más / menos velocidad          q  → guardar y salir

Anotá mirando el CLIP CRUDO (no el overlay del sistema) para que el ground truth
sea independiente de lo que predice el pipeline.
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "SNMOT-108"
INICIO = int(sys.argv[2]) if len(sys.argv) > 2 else 1
FRAMES_DIR = ROOT / "data" / "soccernet" / "tracking" / "train" / CLIP_NAME / "img1"
SALIDA = ROOT / "data" / "ground_truth" / f"{CLIP_NAME}_posesion.csv"

archivos = sorted(FRAMES_DIR.glob("*.jpg"))
N = len(archivos)
if N == 0:
    raise SystemExit(f"No hay frames en {FRAMES_DIR}")

ETIQUETAS = {ord("a"): "equipo_A", ord("b"): "equipo_B", ord("n"): "ninguno"}
COLORES = {"equipo_A": (0, 0, 255), "equipo_B": (255, 0, 0), "ninguno": (160, 160, 160), None: (200, 200, 200)}

labels = [None] * N      # posesión por frame
estado = None
i = INICIO - 1
jugando = False
delay = 70               # ms entre frames (velocidad)
FONT = cv2.FONT_HERSHEY_SIMPLEX

print(f"Anotando {CLIP_NAME} ({N} frames). Barra=play/pausa, a/b/n=equipo, q=guardar.")

while 0 <= i < N:
    frame = cv2.imread(str(archivos[i]))
    labels[i] = estado

    # Overlay de estado e instrucciones
    barra = frame.copy()
    cv2.rectangle(barra, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
    cv2.addWeighted(barra, 0.6, frame, 0.4, 0, frame)
    et = estado if estado else "(sin marcar)"
    cv2.putText(frame, f"frame {i+1}/{N}  t={i/25:.1f}s", (12, 28), FONT, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"POSESION: {et}", (12, 58), FONT, 0.8, COLORES[estado], 2)
    estado_txt = "PLAY" if jugando else "PAUSA"
    cv2.putText(frame, estado_txt, (frame.shape[1] - 120, 28), FONT, 0.7, (0, 255, 255), 2)
    cv2.imshow("Anotacion de posesion", frame)

    k = cv2.waitKey(delay if jugando else 0) & 0xFF
    if k == ord("q"):
        break
    elif k == ord(" "):
        jugando = not jugando
    elif k in ETIQUETAS:
        estado = ETIQUETAS[k]
        labels[i] = estado
    elif k == ord(","):
        jugando = False
        i = max(0, i - 1)
        continue
    elif k == ord("."):
        jugando = False
        i = min(N - 1, i + 1)
        continue
    elif k == ord("+") or k == ord("="):
        delay = max(10, delay - 20)
    elif k == ord("-"):
        delay = min(300, delay + 20)

    if jugando:
        i += 1

cv2.destroyAllWindows()

# --- Comprimir a tramos y guardar ---
SALIDA.parent.mkdir(parents=True, exist_ok=True)
tramos = []
ini = 0
for j in range(1, N + 1):
    if j == N or labels[j] != labels[ini]:
        if labels[ini] is not None:
            tramos.append((ini + 1, j, labels[ini]))  # frames 1-based, fin inclusivo
        ini = j

with open(SALIDA, "w") as f:
    f.write("frame_inicio,frame_fin,equipo\n")
    for a, b, eq in tramos:
        f.write(f"{a},{b},{eq}\n")

print(f"\nGuardado: {SALIDA}")
print(f"{len(tramos)} tramos anotados sobre {sum(1 for x in labels if x is not None)}/{N} frames.")
