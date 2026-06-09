"""Etapa 4 — Render del video con overlay de resultados.

Dibuja sobre cada frame las cajas (jugadores coloreados por equipo, pelota,
árbitro/arquero), marca al poseedor, y un marcador (scoreboard) con el % de
posesión acumulado por equipo. Exporta el video a `outputs/`.
"""

from __future__ import annotations

import cv2

# BGR: equipo_A rojo, equipo_B azul, sin equipo gris.
COLORES_EQUIPO = {"equipo_A": (0, 0, 255), "equipo_B": (255, 0, 0), None: (160, 160, 160)}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _color(d):
    if d.cls == "player":
        return COLORES_EQUIPO[d.team]
    if d.cls == "ball":
        return (0, 255, 255)
    return (0, 255, 0)  # goalkeeper / referee


def _scoreboard(frame, equipo_actual, pct):
    """Barra superior con % de posesión por equipo y quién tiene la pelota."""
    h, w = frame.shape[:2]
    barra = frame.copy()
    cv2.rectangle(barra, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.addWeighted(barra, 0.55, frame, 0.45, 0, frame)

    a, b = pct.get("equipo_A", 0.0), pct.get("equipo_B", 0.0)
    cv2.putText(frame, f"Equipo A: {a:.0%}", (12, 30), _FONT, 0.8, COLORES_EQUIPO["equipo_A"], 2)
    cv2.putText(frame, f"Equipo B: {b:.0%}", (230, 30), _FONT, 0.8, COLORES_EQUIPO["equipo_B"], 2)
    if equipo_actual is not None:
        txt = f"POSESION: {'A' if equipo_actual == 'equipo_A' else 'B'}"
        cv2.putText(frame, txt, (w - 250, 30), _FONT, 0.8, COLORES_EQUIPO[equipo_actual], 2)


def anotar_frame(frame, detecciones, poseedor, equipo_actual, pct):
    """Devuelve el frame con el overlay dibujado (no modifica el original)."""
    out = frame.copy()
    for d in detecciones:
        x1, y1, x2, y2 = map(int, d.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), _color(d), 2)
    if poseedor is not None:
        px = int((poseedor.bbox[0] + poseedor.bbox[2]) / 2)
        py = int(poseedor.bbox[3])
        cv2.circle(out, (px, py), 14, (0, 255, 255), 3)
    _scoreboard(out, equipo_actual, pct)
    return out


def exportar_video(frames, salida, fps=25):
    """Escribe una secuencia de frames (ya anotados) a un archivo mp4."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return salida
