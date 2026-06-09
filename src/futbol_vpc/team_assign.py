"""Etapa 1/2 — Asignación de equipo por embeddings visuales (SigLIP), a nivel track.

**Por qué embeddings y no color** (ver `docs/decisiones/deteccion-y-asignacion.md`):
el color crudo de la camiseta da ~60% de accuracy (≈ azar) porque el fondo
(césped, publicidad del estadio) contamina el recorte y no hay heurística que gane
en todos los partidos. Los embeddings de SigLIP "miran el objeto" y son robustos
al fondo por diseño → **92-98%** validado contra ground truth.

Pipeline:
  1. Recorte de cada jugador (bbox del tracking).
  2. Embedding SigLIP del recorte (vía `TeamClassifier` de la librería `sports`,
     que hace SigLIP → UMAP → KMeans en 2 equipos).
  3. Agregación **por track**: se asigna el equipo por mayoría de los crops del
     jugador a lo largo del clip → un equipo estable por track.

Se entrena (`fit`) una vez por video (los colores cambian de partido a partido,
pero es no supervisado). El arquero y el árbitro no se asignan (clases propias).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _recortar(frame, bbox):
    """Recorta el bbox del jugador (clamp a coordenadas válidas)."""
    x1, y1, x2, y2 = map(int, bbox)
    return frame[max(0, y1):y2, max(0, x1):x2]


class TeamAssigner:
    """Asigna jugadores a 'equipo_A'/'equipo_B' por embeddings, estable por track.

    Uso:
        ta = TeamAssigner(device="mps").fit(por_frame, frames)
        ta.predict(por_frame)   # setea d.team in-place
    o, en una sola llamada:
        TeamAssigner(device="mps").asignar(por_frame, frames)
    """

    def __init__(self, device="cpu", min_apariciones=5, max_muestras_fit=400,
                 muestras_por_track=15):
        # Import diferido: la dependencia pesada (sports/transformers) solo se
        # necesita al instanciar, no al importar el módulo.
        from sports.common.team import TeamClassifier

        self._clf = TeamClassifier(device=device)
        self.min_apariciones = min_apariciones
        self.max_muestras_fit = max_muestras_fit
        self.muestras_por_track = muestras_por_track
        self.equipo_por_track: dict[int, str] = {}

    def fit(self, por_frame, frames):
        """Aprende el equipo de cada track agregando los embeddings de sus crops.

        Args:
            por_frame: lista (por frame) de listas de Detection con track_id
                (salida de `tracking.trackear`).
            frames: lista de frames BGR alineada con `por_frame`.
        """
        crops_por_track: dict[int, list] = defaultdict(list)
        for detecciones, frame in zip(por_frame, frames):
            for d in detecciones:
                if d.cls != "player" or d.track_id is None:
                    continue
                crop = _recortar(frame, d.bbox)
                if crop.size > 0:
                    crops_por_track[d.track_id].append(crop)
        if not crops_por_track:
            raise ValueError("No hay crops de jugador para asignar equipos.")

        # fit del clasificador (SigLIP+UMAP+KMeans) sobre una muestra de crops.
        todos = [c for cs in crops_por_track.values() for c in cs]
        paso = max(1, len(todos) // self.max_muestras_fit)
        self._clf.fit(todos[::paso])

        # equipo por track = mayoría de las predicciones de sus crops.
        self.equipo_por_track = {}
        for tid, cs in crops_por_track.items():
            if len(cs) < self.min_apariciones:
                continue
            sub = cs[:: max(1, len(cs) // self.muestras_por_track)]
            etiqueta = int(round(float(np.mean(self._clf.predict(sub)))))
            self.equipo_por_track[tid] = "equipo_A" if etiqueta == 0 else "equipo_B"
        return self

    def predict(self, por_frame):
        """Setea `d.team` in-place según el equipo aprendido por track."""
        for detecciones in por_frame:
            for d in detecciones:
                if d.track_id in self.equipo_por_track:
                    d.team = self.equipo_por_track[d.track_id]
        return por_frame

    def asignar(self, por_frame, frames):
        """Conveniencia: `fit` + `predict` en una llamada."""
        return self.fit(por_frame, frames).predict(por_frame)
