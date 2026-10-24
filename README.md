# MVP Etapa 1: subida de video, YOLO26, recortes y mejora de brillo

Implementa la Etapa 1 (Extracción) del Documento Técnico TT 2027-A048 (secciones 5.1 y 6.2):

1. Muestreo de 1 de cada N frames con OpenCV.
2. Mejora de brillo: ganancia en espacio lineal estimada con el percentil 99 de la luminancia. Solo se aplica si la ganancia supera el umbral; CLAHE suave y reducción de ruido son opcionales.
3. Detección de personas con YOLO26 y seguimiento con ByteTrack o BoT-SORT.
4. Un solo recorte JPG por persona (el de mayor confianza × área de toda su trayectoria) y `metadata.json` con el esquema del Anexo B.1, que será la entrada de la Etapa 2 (FastVLM).

## Instalación

Requiere Python 3.10 (con el que se probó) y Windows, Linux o macOS. Desde la raíz del proyecto:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
```

En Linux/macOS usa `python3.10 -m venv .venv` y `.venv/bin/python`. Con GPU NVIDIA, cambia `whl/cpu` por `whl/cu124`.

Descarga los pesos `yolo26s.pt` (modelo por defecto) y, para comparar en la evaluación, `yolo26n.pt`, y colócalos en `models/`. El tamaño de entrada por defecto es 1280 px: fue la mejor relación entre recall (96.8 %) y costo en la evaluación del dataset.

## Uso

```powershell
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

Abre http://127.0.0.1:8000, sube un video y espera a que termine. Aparece un recorte por persona.

Por línea de comandos:

```powershell
.venv\Scripts\python stage1\extract.py --video ruta\video.mp4 --out data\salidas\prueba
```

## Salidas (`data/salidas/<id>/`)

- `metadata.json`: video, fps, intervalo_muestreo y un registro por persona (ruta_jpg, num_frame, timestamp_s, caja, confianza, id_trayectoria, mejorado, más primera_aparicion_s, ultima_aparicion_s y num_detecciones).
- `recortes/<video>/id12_f000370.jpg`
- `stats.json`: frames muestreados y mejorados, recortes, trayectorias y tiempo.
- `ejemplo_antes.jpg` / `ejemplo_despues.jpg`: el primer frame al que se le mejoró el brillo.
- `job.json`, `progress.json`, `log.txt`

## Evaluación de detección con el dataset

```powershell
.venv\Scripts\python eval\evaluar_deteccion.py --modelos models\yolo26n.pt models\yolo26s.pt --imgsz 640 1024 1280 1920
```

El dataset no se versiona (`data/` está en `.gitignore`): colócalo en `data/videos/DatasetPersonasActividades/` con un `.mp4`, un `.png` (el frame etiquetado) y un `.txt` por video.

Corre YOLO26 sobre el `.png` de cada video de `data/videos/DatasetPersonasActividades`. Cada punto `bbx` del `.txt` se empareja con a lo más una caja de persona. Métrica principal: **recall** = personas del `.txt` detectadas / personas del `.txt`. También reporta la comparación de conteos YOLO vs `.txt`; esa comparación no sirve como métrica porque los `.txt` solo anotan a una parte de las personas visibles. Los `.txt` con cajas `x1,y1,x2,y2` o sin coordenadas se marcan como inválidos.

Salidas en `data/evaluacion/<fecha>/`: `resumen.json` (métricas por configuración), `detalle.csv` (por archivo, con los ids no detectados) y `fallos_<modelo>_<imgsz>/` (imágenes de los archivos con personas no detectadas: punto verde = detectada, rojo = no detectada).
