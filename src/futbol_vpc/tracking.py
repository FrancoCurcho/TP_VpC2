"""Etapa 2 — Seguimiento multi-objeto con ByteTrack.

Asocia temporalmente las detecciones para mantener identidades (IDs) estables de
cada objeto a lo largo del video. Usa el ByteTrack integrado de ultralytics vía
`model.track(..., persist=True)`, que corre detección + tracking en una pasada.

Devuelve, por frame, una lista de `Detection` con el campo `track_id` completo,
para que las etapas siguientes (asignación de equipo por track, posesión) tengan
identidades estables.
"""

from __future__ import annotations

from . import config
from .detection import Detection


def trackear(modelo, frames, conf=None, iou=None, device=None, tracker="bytetrack.yaml"):
    """Corre detección + ByteTrack sobre una secuencia de frames.

    Args:
        modelo: modelo cargado con `detection.cargar_modelo`.
        frames: iterable de frames BGR (numpy arrays), en orden temporal.
        conf, iou: umbrales (default: config.CONF_THRESHOLD / IOU_THRESHOLD).
        device: 'mps' (Mac), 0 (GPU), 'cpu', o None.
        tracker: config YAML de ultralytics ('bytetrack.yaml' o 'botsort.yaml').

    Returns:
        Lista (una entrada por frame) de listas de `Detection` con `track_id`.
    """
    conf = config.CONF_THRESHOLD if conf is None else conf
    iou = config.IOU_THRESHOLD if iou is None else iou

    por_frame = []
    for frame in frames:
        resultado = modelo.track(
            frame,
            persist=True,        # mantiene el estado del tracker entre frames
            tracker=tracker,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )[0]
        nombres = resultado.names

        detecciones = []
        for box in resultado.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_idx = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else None
            detecciones.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    cls=nombres[cls_idx],
                    conf=float(box.conf[0]),
                    track_id=track_id,
                )
            )
        por_frame.append(detecciones)
    return por_frame
