"""Etapa 1 — Asignación de equipo por color de camiseta (clustering HSV).

A cada detección de clase 'player' se le recorta la camiseta, se extrae el
color dominante en espacio HSV y se agrupan (clustering) en dos equipos. El
arquero suele quedar fuera por usar colores distintos: se maneja aparte.

TODO:
    - recorte de la región de la camiseta dentro del bbox
    - color dominante en HSV (evitando verde del césped)
    - clustering (KMeans, k=2) para separar equipo A / equipo B
"""

from __future__ import annotations


def color_dominante_hsv(frame, bbox):
    """Extrae el color HSV dominante de la camiseta dentro del bbox."""
    raise NotImplementedError


def asignar_equipos(detecciones, frame):
    """Asigna 'equipo_A' / 'equipo_B' a cada jugador según el color de camiseta."""
    raise NotImplementedError
