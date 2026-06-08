"""Etapa 1 — Detección por frame de jugadores, árbitro y pelota con YOLO.

Envuelve al modelo YOLO (fine-tuned sobre el dataset de fútbol) para exponer
una API simple al resto del pipeline, independiente de los detalles de
ultralytics.

TODO:
    - cargar el modelo elegido tras la comparación (models/<modelo>/best.pt)
    - devolver detecciones en un formato neutro (bbox, clase, conf) por frame
"""

from __future__ import annotations


def cargar_modelo(weights_path):
    """Carga los pesos YOLO entrenados y devuelve el modelo listo para inferir."""
    raise NotImplementedError


def detectar(modelo, frame):
    """Corre la detección sobre un frame y devuelve las detecciones.

    Returns:
        Lista de detecciones, cada una con bbox (xyxy), clase y confianza.
    """
    raise NotImplementedError
