"""Evalua si YOLO26 detecta a todas las personas anotadas en el dataset.

Cada video del dataset trae un .png (un frame del video) y un .txt con una
fila por persona; la primera columna (bbx) es un punto "x,y" sobre la persona
en ese .png. El .txt no anota a todas las personas visibles, solo a un
subconjunto, asi que la metrica principal es la exhaustividad (recall):

    recall = personas del .txt que quedan dentro de una caja de YOLO / personas del .txt

Cada caja se empareja con a lo mas una persona (emparejamiento bipartito), de
modo que dos personas dentro de una misma caja cuentan como una sola deteccion.
Tambien se reporta la comparacion directa de conteos (YOLO vs .txt), que solo
coincide si el .txt anoto a todas las personas visibles.

Los .txt con cajas "x1,y1,x2,y2" o sin coordenadas se marcan como invalidos y
no entran en las metricas.

Uso:
    python eval/evaluar_deteccion.py
    python eval/evaluar_deteccion.py --imgsz 640 1024 1280 1920 --modelos models/yolo26n.pt models/yolo26s.pt
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
PUNTO = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*;")
CAJA = re.compile(r"^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*;")


def leer_anotaciones(path: Path):
    """Devuelve (puntos [(x, y, id)], motivo_invalido | None, filas_persona)."""
    lineas = [l for l in path.read_text(encoding="utf-8-sig", errors="replace").splitlines() if l.strip()]
    puntos, cajas, sin_coord = [], 0, 0
    for l in lineas:
        m = PUNTO.match(l)
        if m:
            puntos.append((int(m[1]), int(m[2]), l.split(";")[1].strip()))
        elif CAJA.match(l):
            cajas += 1
        elif re.match(r"^\s*;\s*\d+\s*;", l):
            sin_coord += 1
    if cajas:
        return [], "cajas x1,y1,x2,y2 en lugar de puntos", cajas
    if sin_coord and not puntos:
        return [], "filas sin coordenadas", sin_coord
    if not puntos:
        return [], "sin personas anotadas", 0
    return puntos, None, len(puntos)


def emparejar(puntos, cajas):
    """Emparejamiento maximo punto -> caja que lo contiene (Kuhn). Devuelve {i_punto: i_caja}."""
    adj = [[j for j, (x1, y1, x2, y2) in enumerate(cajas) if x1 <= x <= x2 and y1 <= y <= y2]
           for x, y, _ in puntos]
    dueno = {}

    def aumentar(i, vistos):
        for j in adj[i]:
            if j in vistos:
                continue
            vistos.add(j)
            if j not in dueno or aumentar(dueno[j], vistos):
                dueno[j] = i
                return True
        return False

    for i in range(len(puntos)):
        aumentar(i, set())
    return {i: j for j, i in dueno.items()}


def dibujar(img, puntos, cajas, match, path):
    img = img.copy()
    for x1, y1, x2, y2 in cajas:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
    for i, (x, y, pid) in enumerate(puntos):
        color = (0, 200, 0) if i in match else (0, 0, 255)
        cv2.circle(img, (x, y), 14, color, -1)
        cv2.putText(img, pid, (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    h, w = img.shape[:2]
    cv2.imwrite(str(path), cv2.resize(img, (1280, int(h * 1280 / w))) if w > 1280 else img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "data" / "videos" / "DatasetPersonasActividades"))
    ap.add_argument("--modelos", nargs="+", default=[str(ROOT / "models" / "yolo26n.pt")])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[1024])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "data" / "evaluacion" / datetime.now().strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--sin-imagenes", action="store_true", help="no guardar imagenes de los fallos")
    args = ap.parse_args()

    from ultralytics import YOLO

    dataset, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    validos, invalidos = [], []
    for txt in sorted(dataset.glob("*.txt")):
        png = txt.with_suffix(".png")
        puntos, motivo, filas = leer_anotaciones(txt)
        if motivo or not png.exists():
            invalidos.append({"archivo": txt.stem, "motivo": motivo or "falta el .png", "filas": filas})
        else:
            validos.append((txt.stem, png, puntos))
    print(f"{len(validos)} archivos validos, {len(invalidos)} invalidos")

    resumen = {"dataset": str(dataset), "conf": args.conf, "archivos_validos": len(validos),
               "archivos_invalidos": invalidos, "configuraciones": []}
    filas_csv = []

    for modelo_path in args.modelos:
        modelo = YOLO(modelo_path)
        nombre_modelo = Path(modelo_path).stem
        for imgsz in args.imgsz:
            etiqueta = f"{nombre_modelo}_{imgsz}"
            dir_fallos = out / f"fallos_{etiqueta}"
            tot_gt = tot_ok = tot_yolo = completos = conteo_igual = 0
            err_abs = 0
            for stem, png, puntos in validos:
                img = cv2.imread(str(png))
                r = modelo.predict(img, classes=[0], conf=args.conf, imgsz=imgsz, verbose=False)[0]
                cajas = r.boxes.xyxy.cpu().numpy().astype(int).tolist()
                match = emparejar(puntos, cajas)
                n_gt, n_ok, n_yolo = len(puntos), len(match), len(cajas)
                no_detectadas = [puntos[i][2] for i in range(n_gt) if i not in match]
                tot_gt += n_gt
                tot_ok += n_ok
                tot_yolo += n_yolo
                completos += n_ok == n_gt
                conteo_igual += n_yolo == n_gt
                err_abs += abs(n_yolo - n_gt)
                filas_csv.append({
                    "configuracion": etiqueta, "archivo": stem, "personas_txt": n_gt, "detecciones_yolo": n_yolo,
                    "personas_txt_detectadas": n_ok, "recall": round(n_ok / n_gt, 4),
                    "diferencia_conteo": n_yolo - n_gt, "ids_no_detectados": " ".join(no_detectadas),
                    "detecciones_sin_anotar": n_yolo - n_ok,
                })
                if no_detectadas and not args.sin_imagenes:
                    dir_fallos.mkdir(exist_ok=True)
                    dibujar(img, puntos, cajas, match, dir_fallos / f"{stem}.jpg")
            n = len(validos)
            conf_res = {
                "configuracion": etiqueta, "modelo": nombre_modelo, "imgsz": imgsz,
                "personas_anotadas": tot_gt, "personas_detectadas": tot_ok,
                "recall_global": round(tot_ok / tot_gt, 4) if tot_gt else None,
                "archivos_con_100pct": completos, "pct_archivos_con_100pct": round(completos / n, 4) if n else None,
                "detecciones_yolo": tot_yolo,
                "archivos_con_mismo_conteo": conteo_igual,
                "error_absoluto_medio_conteo": round(err_abs / n, 2) if n else None,
            }
            resumen["configuraciones"].append(conf_res)
            print(f"{etiqueta}: recall {conf_res['recall_global']:.1%} ({tot_ok}/{tot_gt}) · "
                  f"100% en {completos}/{n} archivos · mismo conteo en {conteo_igual}/{n} · "
                  f"error medio de conteo {conf_res['error_absoluto_medio_conteo']}")

    with open(out / "detalle.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_csv[0]))
        w.writeheader()
        w.writerows(filas_csv)
    (out / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultados en {out}")


if __name__ == "__main__":
    main()
