"""futbol_vpc — Detección de jugadores y estimación de posesión de pelota.

Paquete del TP de Visión por Computadora II (CEIA, Grupo 7).

El pipeline se estructura en tres etapas:
    1. Detección por frame (detection) + asignación de equipo (team_assign)
    2. Tracking multi-objeto con ByteTrack (tracking)
    3. Estimación de posesión (possession) + métricas (metrics) + overlay (video)

Uso típico desde un notebook:
    from futbol_vpc import detection, tracking, possession
"""

__version__ = "0.1.0"
