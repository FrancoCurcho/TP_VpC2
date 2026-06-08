"""Etapa 3 — Métricas agregadas de posesión.

A partir de la serie de posesión (ya suavizada) calcula las métricas que el TP
quiere reportar y comparar contra el ground truth.

TODO:
    - porcentaje de posesión por equipo
    - cantidad de cambios de posesión
    - duración media de las secuencias de posesión
    - (evaluación) comparación contra ground truth frame a frame
"""

from __future__ import annotations


def porcentaje_posesion(serie_posesion, equipos_por_id):
    """Porcentaje de posesión de cada equipo sobre el total de frames con posesión."""
    raise NotImplementedError


def cambios_de_posesion(serie_posesion, equipos_por_id):
    """Cuenta cuántas veces la posesión cambió de un equipo al otro."""
    raise NotImplementedError


def duracion_media_secuencias(serie_posesion):
    """Duración media (en frames) de las secuencias continuas de posesión."""
    raise NotImplementedError
