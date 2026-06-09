"""Etapa 3 — Estimación de posesión instantánea de la pelota.

En cada frame, el poseedor es el **jugador más cercano a la pelota** medido desde
sus **pies** (borde inferior del bbox) y normalizado por su altura (robusto a la
perspectiva — ver `docs/decisiones/posesion.md`). Luego se aplica un **suavizado
temporal** (moda en ventana deslizante) para quitar transiciones espurias.

El arquero y el árbitro no son candidatos (solo `player`). Si no hay pelota
detectada o nadie está dentro del umbral, la posesión del frame es `None`.
"""

from __future__ import annotations

from . import config
from .detection import Detection

# Tamaño (px) del bbox sintético de la pelota interpolada.
_BALL_BOX = 12


def interpolar_pelota(por_frame, max_gap=5):
    """Rellena la posición de la pelota en huecos CORTOS por interpolación lineal.

    Agrega una detección sintética de `ball` (conf=0) en cada frame sin pelota que
    esté entre dos detecciones separadas por <= `max_gap` frames. Solo huecos
    cortos: la pelota no se mueve lineal en patadas/picadas, interpolar huecos
    largos daría posiciones falsas. Modifica `por_frame` in-place.
    """
    centros = []
    for detecciones in por_frame:
        pelotas = [d for d in detecciones if d.cls == "ball"]
        centros.append(max(pelotas, key=lambda d: d.conf).centro if pelotas else None)

    detectados = [k for k, c in enumerate(centros) if c is not None]
    r = _BALL_BOX / 2
    for a, b in zip(detectados, detectados[1:]):
        if 1 < (b - a) <= max_gap + 1:
            (ax, ay), (bx, by) = centros[a], centros[b]
            for k in range(a + 1, b):
                f = (k - a) / (b - a)
                cx, cy = ax + (bx - ax) * f, ay + (by - ay) * f
                por_frame[k].append(
                    Detection(bbox=(cx - r, cy - r, cx + r, cy + r), cls="ball", conf=0.0)
                )
    return por_frame


def _pies(bbox):
    """Punto de los pies del jugador: centro del borde inferior del bbox."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def poseedor_en_frame(detecciones, umbral_rel=None):
    """Devuelve la Detection del jugador que tiene la pelota, o None.

    Poseedor = jugador (`player`) con menor distancia pies→pelota normalizada por
    su altura, si esa distancia normalizada está por debajo de `umbral_rel`.
    """
    umbral_rel = config.POSSESSION_DIST_REL if umbral_rel is None else umbral_rel

    pelotas = [d for d in detecciones if d.cls == "ball"]
    if not pelotas:
        return None
    bx, by = max(pelotas, key=lambda d: d.conf).centro  # la pelota más confiable

    mejor, mejor_score = None, umbral_rel
    for d in detecciones:
        if d.cls != "player":  # excluye goalkeeper y referee
            continue
        x1, y1, x2, y2 = d.bbox
        alto = y2 - y1
        if alto <= 0:
            continue
        px, py = _pies(d.bbox)
        dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
        score = dist / alto
        if score < mejor_score:
            mejor_score, mejor = score, d
    return mejor


def _mantener(controles, histeresis):
    """Modelo de 'hold' con histéresis.

    Sostiene el equipo actual durante los huecos (None). Solo cambia al otro
    equipo cuando este acumula `histeresis` eventos de control (filtra disputas y
    parpadeo); un toque suelto del otro equipo no cambia la posesión. Antes del
    primer control sostenido, es None.
    """
    serie, actual = [], None
    candidato, racha = None, 0
    for c in controles:
        if c is not None and c != actual:
            racha = racha + 1 if c == candidato else 1
            candidato = c
            if actual is None or racha >= histeresis:
                actual, candidato, racha = c, None, 0
        elif c == actual:
            candidato, racha = None, 0  # el actual se reafirma
        serie.append(actual)
    return serie


def secuencia_posesion(por_frame, umbral_rel=None, histeresis=None):
    """Serie temporal del equipo en posesión por frame (hold + histéresis).

    Cada frame con un **poseedor claro** (jugador dentro de `umbral_rel` de la
    pelota) es un *evento de control*. La posesión se **sostiene** entre eventos
    (los pases no la rompen) y solo **cambia** cuando el otro equipo controla de
    forma sostenida (`histeresis` eventos). Ver `docs/decisiones/posesion.md`.

    Args:
        por_frame: lista (por frame) de listas de Detection con `team` asignado.
        umbral_rel: umbral estricto de control (default: config.POSSESSION_DIST_REL).
        histeresis: eventos para cambiar de equipo (default: config.POSSESSION_HYSTERESIS).

    Returns:
        Lista (por frame) de 'equipo_A' / 'equipo_B' / None.
    """
    histeresis = config.POSSESSION_HYSTERESIS if histeresis is None else histeresis
    controles = []
    for detecciones in por_frame:
        poseedor = poseedor_en_frame(detecciones, umbral_rel)
        controles.append(poseedor.team if poseedor is not None else None)
    return _mantener(controles, histeresis)
