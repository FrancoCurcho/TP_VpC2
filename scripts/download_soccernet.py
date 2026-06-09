"""Descarga clips de SoccerNet-Tracking (30 s, con anotaciones de tracking + equipo).

Bajamos el split `train` del task `tracking`: son clips de 30 s desde la cámara
táctica, cada uno con frames + anotaciones MOT (track ID por caja) + un
`gameinfo.ini` que mapea cada track a su equipo/rol. Eso nos sirve de ground
truth para tracking y asignación de equipo.

Estructura que queda al descomprimir (por clip `SNMOT-XXX/`):
    img1/000001.jpg ...      → frames (25 fps, ~750 = 30 s)
    gt/gt.txt                → MOT: frame, id, left, top, w, h, conf, ...
    gameinfo.ini             → equipo/rol de cada track id
    seqinfo.ini              → tamaño, fps, cantidad de frames

Uso (sin tocar el entorno del proyecto, así evitamos el tema de `lap`):
    export SOCCERNET_PASSWORD="la-password-del-mail"
    uv run --no-project --with SoccerNet python scripts/download_soccernet.py

Ojo: el split `train` son 57 clips → la descarga pesa varios GB. Solo vamos a
usar UNO; el resto queda disponible por si queremos más adelante.
"""

import os
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader

password = os.environ.get("SOCCERNET_PASSWORD")
if not password:
    raise SystemExit(
        "Falta la password. Corré:\n"
        '  export SOCCERNET_PASSWORD="..."  (la que te llegó por mail)\n'
        "y volvé a ejecutar este script."
    )

# Carpeta destino dentro del repo (queda fuera de git por .gitignore).
dest = Path(__file__).resolve().parents[1] / "data" / "soccernet"
dest.mkdir(parents=True, exist_ok=True)

dl = SoccerNetDownloader(LocalDirectory=str(dest))
dl.password = password

print(f"Descargando SoccerNet-Tracking (split train) en: {dest}")
print("Esto puede tardar (varios GB)...")
dl.downloadDataTask(task="tracking", split=["train"])

print("\nListo. Los clips quedaron en:", dest / "tracking" / "train")
print("Cada subcarpeta SNMOT-XXX es un clip de 30 s con sus anotaciones.")
