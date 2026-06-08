"""Etapa 2 — Seguimiento multi-objeto con ByteTrack.

Asocia temporalmente las detecciones para mantener identidades (IDs) estables
de cada jugador a lo largo del video. ultralytics trae ByteTrack integrado vía
`model.track(..., tracker="bytetrack.yaml")`, que es el camino más directo.

TODO:
    - integrar ByteTrack sobre el stream de detecciones
    - propagar el equipo asignado de forma estable por ID
"""

from __future__ import annotations


def trackear(modelo, video_path):
    """Corre detección + ByteTrack sobre el video y devuelve tracks con ID estable."""
    raise NotImplementedError
