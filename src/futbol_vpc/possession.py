"""Etapa 3 — Estimación de posesión instantánea de la pelota.

En cada frame se decide qué jugador tiene la pelota como el más cercano dentro
de un umbral de distancia (config.POSSESSION_DIST_THRESHOLD). Luego se aplica un
suavizado temporal (config.POSSESSION_SMOOTHING_WINDOW) para evitar transiciones
espurias frame a frame.

TODO:
    - distancia jugador-pelota por frame y selección del más cercano
    - suavizado temporal (moda en ventana deslizante / histéresis)
"""

from __future__ import annotations


def posesion_por_frame(tracks_frame, pelota_bbox, umbral):
    """Devuelve el ID del jugador en posesión en un frame, o None si nadie cumple."""
    raise NotImplementedError


def suavizar(serie_posesion, ventana):
    """Suaviza temporalmente la serie de posesión para quitar transiciones espurias."""
    raise NotImplementedError
