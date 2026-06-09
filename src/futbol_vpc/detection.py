"""Etapa 1 — Detección por frame de jugadores, árbitro y pelota con YOLO.

Envuelve al modelo YOLO (fine-tuned sobre el dataset de fútbol) para exponer
una API simple al resto del pipeline, independiente de los detalles de
ultralytics. Las detecciones se devuelven en un formato neutro (`Detection`) que
el resto de las etapas consume sin saber de qué modelo vienen.
"""

from __future__ import annotations

from dataclasses import dataclass

from ultralytics import YOLO

from . import config


@dataclass
class Detection:
    """Una detección en un frame, en formato neutro.

    Attributes:
        bbox: caja en píxeles (x1, y1, x2, y2).
        cls: nombre de la clase ('ball', 'goalkeeper', 'player', 'referee').
        conf: confianza [0, 1].
        team: equipo asignado ('equipo_A' / 'equipo_B'); lo completa team_assign.
        track_id: ID estable del objeto; lo completa la etapa de tracking.
    """

    bbox: tuple[float, float, float, float]
    cls: str
    conf: float
    team: str | None = None
    track_id: int | None = None

    @property
    def centro(self) -> tuple[float, float]:
        """Centro de la caja (x, y) en píxeles."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def cargar_modelo(weights_path=None):
    """Carga los pesos YOLO entrenados y devuelve el modelo listo para inferir.

    Args:
        weights_path: ruta al .pt. Por defecto, el modelo elegido (yolo26m).
    """
    weights_path = weights_path or (config.MODELS_DIR / "yolo26m" / "best.pt")
    return YOLO(str(weights_path))


def detectar(modelo, frame, conf=None, iou=None, device=None) -> list[Detection]:
    """Corre la detección sobre un frame y devuelve las detecciones.

    Args:
        modelo: modelo cargado con `cargar_modelo`.
        frame: imagen BGR (numpy array) o ruta.
        conf: umbral de confianza (default: config.CONF_THRESHOLD).
        iou: umbral de IoU para NMS (default: config.IOU_THRESHOLD).
        device: 'mps' (Mac), 0 (GPU), 'cpu', o None (autodetecta).

    Returns:
        Lista de `Detection`.
    """
    conf = config.CONF_THRESHOLD if conf is None else conf
    iou = config.IOU_THRESHOLD if iou is None else iou

    resultado = modelo.predict(frame, conf=conf, iou=iou, device=device, verbose=False)[0]
    nombres = resultado.names

    detecciones = []
    for box in resultado.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_idx = int(box.cls[0])
        detecciones.append(
            Detection(bbox=(x1, y1, x2, y2), cls=nombres[cls_idx], conf=float(box.conf[0]))
        )
    return detecciones
