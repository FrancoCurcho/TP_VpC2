# Decisiones — Modelo de detección

Última actualización: **2026-06-08**.

---

## 1. Modelos comparados y por qué

Se comparan **tres arquitecturas de detección, todas de Ultralytics**:

| Modelo | Rol en la comparación |
|---|---|
| **YOLOv8m** | Baseline clásico y más citado de la comunidad — punto de referencia |
| **YOLO11m** | Generación intermedia |
| **YOLO26m** | Generación más reciente |

**Por qué estos tres:** son todos nativos de `ultralytics` (mismo flujo de
entrenamiento, mismo `data.yaml`, integración sin fricción) y cubren la evolución
reciente de la familia YOLO, lo que da una comparación con sustancia sin salir de
un ecosistema homogéneo.

### Por qué tamaño `m` (y no `x`)

Originalmente había entrenamientos sueltos en tamaños distintos (YOLO11**x**,
YOLO26**m**), lo que **no es una comparación justa** (el tamaño domina sobre la
arquitectura). Se normalizó todo a **`m`** por:

- **Es video:** la estimación de posesión necesita procesar frames a buen ritmo;
  un `x` es ~3× más lento sin ganar tanto mAP.
- **`m` es el punto dulce** precisión/velocidad para objetos grandes (jugadores).
- **Reentrenar es barato** (~18-20 min por modelo en GPU T4), así que normalizar
  no tuvo costo relevante.

> El YOLO11**x** previo se conserva en `models/yolo11x/` solo como referencia
> histórica; no entra en la comparación.

---

## 2. Hiperparámetros de entrenamiento y por qué

Idénticos para los tres modelos (condición para una comparación justa — lo único
que cambia es el peso base):

| Parámetro | Valor | Justificación |
|---|---|---|
| `epochs` | 80 | Las curvas convergen y se estabilizan antes de la época 80 |
| `imgsz` | 640 | Default; suficiente (ver punto 3, la pelota se detecta bien) |
| `hsv_s` / `hsv_v` | 0.5 / 0.5 | Robustez a variaciones de color/iluminación entre transmisiones |
| `translate` | 0.1 | Robustez a la posición de los objetos en el frame |
| `scale` | 0.6 | Robustez a la escala (zoom de cámara, distancia) |
| `perspective` | 0.0005 | Simula leves cambios de ángulo de cámara |
| `shear` | 40 | Aumenta variabilidad geométrica |
| `fliplr` | 0.5 | El fútbol es simétrico izquierda/derecha → duplica variabilidad útil |

**Idea general:** las augmentations de color (`hsv_*`) cubren la variación de
iluminación/transmisión, y las geométricas (`translate`, `scale`, `perspective`,
`shear`, `fliplr`) cubren los cambios de cámara y posición. No se tocó el `batch`
(default de Ultralytics).

---

## 3. Resultados de la comparación

### Métricas globales (última época, set de validación)

| Modelo | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| yolo11m | 0.9409 | 0.5570 | 0.9191 | 0.9067 |
| **yolo26m** | 0.9541 | **0.5947** | 0.9216 | **0.9344** |
| yolov8m | **0.9568** | 0.5575 | **0.9552** | 0.9201 |

### Recall por clase (diagonal de la matriz de confusión normalizada)

| Clase | yolo11m | yolo26m | yolov8m |
|---|---|---|---|
| **ball** | 0.88 | 0.89 | 0.89 |
| goalkeeper | 0.90 | **0.95** | 0.87 |
| player | 0.98 | 0.98 | **0.99** |
| referee | 0.90 | 0.93 | **0.95** |
| *background→player (FP)* | 0.48 | 0.71 | 0.58 |

**Hallazgo clave:** la **pelota** (el cuello de botella esperado del pipeline)
está **empatada** (~0.89 en los tres). No es un factor de decisión, y por ahora
**no hace falta subir `imgsz` a 1280**.

---

## 4. Modelo elegido: **YOLO26m**

**Decisión:** se usa **YOLO26m** como detector del pipeline.

**Por qué:**
- **Mejor mAP50-95 (0.595):** la métrica más exigente, mide localización fina de
  las cajas. Es lo más relevante para *este* problema porque la posesión se
  decide por **distancia jugador-pelota** → cajas mejor localizadas = mejor
  estimación de quién tiene la pelota.
- **Mejor recall global (0.934)** y mejor detección de `goalkeeper` → pierde menos
  objetos.
- La pelota está empatada, así que no se sacrifica nada en la clase crítica.

**Contrapartida asumida:** YOLO26m genera más falsos positivos de `player` sobre
el fondo (0.71 vs 0.58 de yolov8m). Se mitiga con el umbral de confianza
(`conf=0.5`, ajustable).

**Alternativa considerada — YOLOv8m:** mejor precision (0.955) y mAP50, menos
jugadores fantasma. Se descartó como principal porque su mAP50-95 (localización)
es más bajo, que es lo que pesa para la posesión. Queda como segunda opción si en
la práctica los falsos positivos de YOLO26m molestan.

---

## 5. Pendiente

- **`imgsz=1280`:** solo si en el video real la pelota se pierde más de lo que
  sugiere la validación. Por ahora descartado.
