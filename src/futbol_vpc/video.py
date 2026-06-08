"""Etapa 3 — Render del video con overlay de resultados.

Dibuja sobre cada frame las cajas de jugadores (coloreadas por equipo), la
pelota, el ID de track, quién tiene la posesión y un marcador con las métricas
agregadas; luego exporta el video a outputs/.

TODO:
    - dibujar detecciones + equipo + ID + indicador de posesión por frame
    - panel/scoreboard con % de posesión por equipo
    - export del video final (cv2 / moviepy)
"""

from __future__ import annotations


def dibujar_overlay(frame, tracks, posesion_actual, metricas):
    """Dibuja sobre un frame las cajas, equipos, IDs, posesión y métricas."""
    raise NotImplementedError


def exportar_video(frames, salida_path, fps):
    """Escribe la secuencia de frames con overlay a un archivo de video."""
    raise NotImplementedError
