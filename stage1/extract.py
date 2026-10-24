"""Etapa 1: Extraccion (Documento Tecnico, secciones 5.1 y 6.2).

Muestrea 1 de cada N frames, mejora los oscuros, detecta personas con YOLO26,
las sigue con ByteTrack/BoT-SORT, guarda UN recorte JPG por persona (el de mayor
confianza x area de toda su trayectoria) y
escribe metadata.json (Anexo B.1), que es el contrato con la Etapa 2.

Uso:
    python stage1/extract.py --video data/videos/x.mp4 --out data/salidas/x
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enhance import enhance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def escribir_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def miniatura(frame, ancho=640):
    h, w = frame.shape[:2]
    return frame if w <= ancho else cv2.resize(frame, (ancho, int(h * ancho / w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--model", default=str(ROOT / "models" / "yolo26s.pt"))
    ap.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    ap.add_argument("--gain-threshold", type=float, default=1.5)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--denoise", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    video = Path(args.video)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    progreso = out / "progress.json"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir el video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_muestreados = max(1, -(-total // args.interval))

    model = YOLO(args.model)
    recortes = []
    mejores = {}  # id_trayectoria -> mejor recorte visto hasta ahora
    frames_mejorados = 0
    muestreados = 0
    ejemplo_guardado = False
    t0 = time.perf_counter()
    escribir_json(progreso, {"procesados": 0, "total": total_muestreados})

    n = -1
    while cap.grab():
        n += 1
        if n % args.interval:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        muestreados += 1

        mejorado_frame, mejorado, ganancia = enhance(
            frame, args.gain_threshold, clahe=args.clahe, denoise=args.denoise)
        if mejorado:
            frames_mejorados += 1
            if not ejemplo_guardado:
                cv2.imwrite(str(out / "ejemplo_antes.jpg"), miniatura(frame))
                cv2.imwrite(str(out / "ejemplo_despues.jpg"), miniatura(mejorado_frame))
                ejemplo_guardado = True

        res = model.track(mejorado_frame, persist=True, classes=[0], conf=args.conf,
                          imgsz=args.imgsz, tracker=args.tracker, verbose=False)[0]
        boxes = res.boxes
        if boxes is not None and boxes.id is not None:
            h, w = mejorado_frame.shape[:2]
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.int().cpu().numpy()
            t = round(n / fps, 3)
            for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                tid = int(tid)
                # Un solo recorte por persona: el de mayor confianza x area (5.2.1)
                puntaje = float(c) * (x2 - x1) * (y2 - y1)
                pista = mejores.setdefault(tid, {"puntaje": -1.0, "primera": t, "detecciones": 0})
                pista["ultima"] = t
                pista["detecciones"] += 1
                if puntaje > pista["puntaje"]:
                    pista.update(puntaje=puntaje, imagen=mejorado_frame[y1:y2, x1:x2].copy(), registro={
                        "num_frame": n,
                        "timestamp_s": t,
                        "caja": {"x": x1, "y": y1, "ancho": x2 - x1, "alto": y2 - y1},
                        "confianza": round(float(c), 3),
                        "id_trayectoria": tid,
                        "mejorado": mejorado,
                    })

        if muestreados % 5 == 0:
            escribir_json(progreso, {"procesados": muestreados, "total": total_muestreados})

    cap.release()

    carpeta = out / "recortes" / video.stem
    carpeta.mkdir(parents=True, exist_ok=True)
    for tid, pista in sorted(mejores.items()):
        reg = pista["registro"]
        jpg = carpeta / f"id{tid}_f{reg['num_frame']:06d}.jpg"
        cv2.imwrite(str(jpg), pista["imagen"])
        recortes.append({
            "ruta_jpg": jpg.relative_to(out).as_posix(),
            **reg,
            "primera_aparicion_s": pista["primera"],
            "ultima_aparicion_s": pista["ultima"],
            "num_detecciones": pista["detecciones"],
        })
    duracion = time.perf_counter() - t0
    escribir_json(out / "metadata.json", {
        "video": video.name,
        "fps": round(fps, 3),
        "intervalo_muestreo": args.interval,
        "recortes": recortes,
    })
    escribir_json(out / "stats.json", {
        "modelo": Path(args.model).name,
        "imgsz": args.imgsz,
        "frames_muestreados": muestreados,
        "frames_mejorados": frames_mejorados,
        "recortes": len(recortes),
        "trayectorias": len({r["id_trayectoria"] for r in recortes}),
        "tiempo_s": round(duracion, 2),
        "frames_por_segundo": round(muestreados / duracion, 2) if duracion else None,
    })
    escribir_json(progreso, {"procesados": muestreados, "total": muestreados})
    print(f"{len(recortes)} recortes de {muestreados} frames en {duracion:.1f} s")


if __name__ == "__main__":
    main()
