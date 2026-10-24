"""API del MVP: subida de video, cola de procesamiento y consulta de resultados.

    .venv\\Scripts\\python -m uvicorn app.main:app --port 8000

La Etapa 1 corre como proceso externo (stage1/extract.py) y solo se comunica
por archivos en disco, como en la arquitectura del Documento Tecnico (4.5.1).
"""

import csv
import json
import queue
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "videos"
SALIDAS = ROOT / "data" / "salidas"
EVALUACION = ROOT / "data" / "evaluacion"
EXTRACT = ROOT / "stage1" / "extract.py"
EXTENSIONES = {".mp4", ".avi", ".mov", ".mkv"}

VIDEOS.mkdir(parents=True, exist_ok=True)
SALIDAS.mkdir(parents=True, exist_ok=True)
EVALUACION.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TT 2027-A048 · Etapa 1")
cola: "queue.Queue[str]" = queue.Queue()
lock = threading.Lock()


def leer_json(path: Path, defecto=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return defecto


def guardar_job(job: dict):
    with lock:
        (SALIDAS / job["id"] / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def worker():
    while True:
        job_id = cola.get()
        job = leer_json(SALIDAS / job_id / "job.json")
        job["estado"] = "procesando"
        guardar_job(job)
        p = job["parametros"]
        cmd = [sys.executable, str(EXTRACT), "--video", job["video_path"], "--out", str(SALIDAS / job_id),
               "--interval", str(p["intervalo"]), "--conf", str(p["confianza"]), "--imgsz", str(p["imgsz"]),
               "--tracker", p["rastreador"], "--gain-threshold", str(p["umbral_ganancia"])]
        if p["clahe"]:
            cmd.append("--clahe")
        if p["denoise"]:
            cmd.append("--denoise")
        with open(SALIDAS / job_id / "log.txt", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        job["estado"] = "terminado" if rc == 0 else "error"
        job["fin"] = datetime.now().isoformat(timespec="seconds")
        guardar_job(job)
        cola.task_done()


@app.on_event("startup")
def iniciar():
    # Los trabajos que quedaron a medias al apagar el servidor se vuelven a encolar
    for d in sorted(SALIDAS.iterdir(), key=lambda d: d.stat().st_mtime):
        job = leer_json(d / "job.json")
        if job and job["estado"] in ("en_cola", "procesando"):
            job["estado"] = "en_cola"
            guardar_job(job)
            cola.put(job["id"])
    threading.Thread(target=worker, daemon=True).start()


def vista(job: dict, completo=False) -> dict:
    d = SALIDAS / job["id"]
    out = {**job, "progreso": leer_json(d / "progress.json"), "stats": leer_json(d / "stats.json")}
    if job["estado"] == "error":
        out["log"] = (d / "log.txt").read_text(encoding="utf-8", errors="replace")[-3000:] if (d / "log.txt").exists() else ""
    if completo and job["estado"] == "terminado":
        out["metadata"] = leer_json(d / "metadata.json")
        out["ejemplo"] = (d / "ejemplo_antes.jpg").exists()
    return out


@app.post("/api/videos")
async def subir(
    archivo: UploadFile = File(...),
    intervalo: int = Form(10),
    confianza: float = Form(0.25),
    imgsz: int = Form(1280),
    rastreador: str = Form("bytetrack.yaml"),
    umbral_ganancia: float = Form(1.5),
    clahe: bool = Form(False),
    denoise: bool = Form(False),
):
    nombre = Path(archivo.filename or "").name
    if Path(nombre).suffix.lower() not in EXTENSIONES:
        raise HTTPException(400, f"Formato no soportado. Usa: {', '.join(sorted(EXTENSIONES))}")
    if rastreador not in ("bytetrack.yaml", "botsort.yaml"):
        raise HTTPException(400, "Rastreador no valido")
    if intervalo < 1 or not 0 < confianza < 1 or not 64 <= imgsz <= 1920:
        raise HTTPException(400, "Parametros fuera de rango")

    job_id = uuid.uuid4().hex[:10]
    destino = VIDEOS / f"{job_id}_{nombre}"
    with open(destino, "wb") as f:
        while chunk := await archivo.read(1 << 20):
            f.write(chunk)

    (SALIDAS / job_id).mkdir()
    job = {
        "id": job_id, "video": nombre, "video_path": str(destino), "estado": "en_cola",
        "inicio": datetime.now().isoformat(timespec="seconds"),
        "parametros": {"intervalo": intervalo, "confianza": confianza, "imgsz": imgsz, "rastreador": rastreador,
                       "umbral_ganancia": umbral_ganancia, "clahe": clahe, "denoise": denoise},
    }
    guardar_job(job)
    cola.put(job_id)
    return job


@app.get("/api/jobs")
def listar():
    jobs = [leer_json(d / "job.json") for d in SALIDAS.iterdir() if (d / "job.json").exists()]
    return [vista(j) for j in sorted(jobs, key=lambda j: j["inicio"], reverse=True)]


@app.get("/api/jobs/{job_id}")
def detalle(job_id: str):
    job = leer_json(SALIDAS / Path(job_id).name / "job.json")
    if not job:
        raise HTTPException(404, "No existe")
    return vista(job, completo=True)


# ---------------------------------------------------------------- evaluacion
# Corridas de eval/evaluar_deteccion.py, una carpeta por corrida en data/evaluacion
@app.get("/api/evaluaciones")
def evaluaciones():
    corridas = []
    for d in sorted(EVALUACION.iterdir(), reverse=True):
        resumen = leer_json(d / "resumen.json")
        if resumen:
            corridas.append({"id": d.name, "archivos_validos": resumen["archivos_validos"],
                             "configuraciones": resumen["configuraciones"]})
    return corridas


@app.get("/api/evaluaciones/{corrida}")
def evaluacion(corrida: str):
    d = EVALUACION / Path(corrida).name
    resumen = leer_json(d / "resumen.json")
    if not resumen:
        raise HTTPException(404, "No existe")
    with open(d / "detalle.csv", encoding="utf-8-sig", newline="") as f:
        detalle = list(csv.DictReader(f))
    for fila in detalle:
        img = d / f"fallos_{fila['configuracion']}" / f"{fila['archivo']}.jpg"
        fila["imagen"] = f"/evaluacion/{d.name}/{img.parent.name}/{img.name}" if img.exists() else None
    return {**resumen, "id": d.name, "detalle": detalle}


app.mount("/salidas", StaticFiles(directory=SALIDAS), name="salidas")
app.mount("/evaluacion", StaticFiles(directory=EVALUACION), name="evaluacion")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
