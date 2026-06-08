"""Configuración central del proyecto: rutas, clases, umbrales y credenciales.

Centralizar esto evita tener constantes mágicas y API keys repartidas por los
notebooks. Las credenciales se leen de variables de entorno (nunca hardcodeadas).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Rutas del proyecto ---------------------------------------------------
# ROOT apunta a la raíz del repo (config.py vive en src/futbol_vpc/).
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
OUTPUTS_DIR = ROOT / "outputs"

# --- Clases del detector --------------------------------------------------
# Orden según el data.yaml del dataset (Roboflow finalv2).
CLASSES = ["ball", "goalkeeper", "player", "referee"]

# --- Umbrales del pipeline ------------------------------------------------
# Distancia (en px) máxima jugador-pelota para considerar posesión.
POSSESSION_DIST_THRESHOLD = 50
# Ventana (en frames) para el suavizado temporal de la posesión.
POSSESSION_SMOOTHING_WINDOW = 5
# Confianza e IoU por defecto para la inferencia del detector.
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# --- Credenciales (vía entorno) -------------------------------------------
# Definir antes de correr:  export ROBOFLOW_API_KEY="..."
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
