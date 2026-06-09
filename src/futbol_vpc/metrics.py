"""Etapa 3 — Métricas agregadas de posesión.

A partir de la serie de posesión (ya suavizada por `possession.secuencia_posesion`)
calcula las métricas que reporta el TP. Los frames sin posesión (`None`) no cuentan
para los porcentajes y cortan las secuencias.
"""

from __future__ import annotations

from collections import Counter
from itertools import groupby


def porcentaje_posesion(serie):
    """Porcentaje de posesión por equipo, sobre los frames CON posesión.

    Returns:
        dict {equipo: fracción [0,1]}.
    """
    cuenta = Counter(t for t in serie if t is not None)
    total = sum(cuenta.values())
    if total == 0:
        return {}
    return {equipo: n / total for equipo, n in cuenta.items()}


def cambios_de_posesion(serie):
    """Cantidad de veces que la posesión cambió de un equipo al otro.

    Ignora los frames sin posesión (no cuentan como cambio por sí mismos).
    """
    equipos = [t for t in serie if t is not None]
    return sum(1 for a, b in zip(equipos, equipos[1:]) if a != b)


def duracion_media_secuencias(serie, fps=25):
    """Duración media de las secuencias de posesión continua, en segundos.

    Una secuencia es una racha maximal de frames con el mismo equipo (los `None`
    la cortan).
    """
    duraciones = [len(list(g)) for k, g in groupby(serie) if k is not None]
    if not duraciones:
        return 0.0
    return (sum(duraciones) / len(duraciones)) / fps


def resumen(serie, fps=25):
    """Devuelve todas las métricas en un dict, listo para imprimir/loguear."""
    return {
        "porcentaje_posesion": porcentaje_posesion(serie),
        "cambios_de_posesion": cambios_de_posesion(serie),
        "duracion_media_seg": duracion_media_secuencias(serie, fps),
    }
