"""Mejora de imagen en baja iluminacion (Documento Tecnico, secciones 2.2.3 y 5.1.2).

Ganancia en espacio lineal estimada con el percentil 99 de la luminancia:
  1. sRGB -> lineal (ec. 3)
  2. k = lin(240/255) / Y99 (ec. 4); si k < umbral el frame pasa sin cambios
  3. canales lineales * k, recorte a [0, 1], lineal -> sRGB (ec. 5)
  4. opcional: CLAHE suave sobre el canal L de LAB y/o medias no locales
"""

import cv2
import numpy as np


def srgb_a_lineal(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lineal_a_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


# LUT uint8 -> lineal (evita la potencia por pixel)
_LUT_LINEAL = srgb_a_lineal(np.arange(256) / 255.0).astype(np.float32)
_OBJETIVO = float(srgb_a_lineal(240 / 255.0))
# Pesos BT.709 en orden BGR (OpenCV)
_PESOS_Y = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)


def estimar_ganancia(frame_bgr) -> float:
    lin = _LUT_LINEAL[frame_bgr[::2, ::2]]  # submuestreo: el percentil casi no cambia
    y99 = float(np.percentile(lin @ _PESOS_Y, 99))
    return _OBJETIVO / max(y99, 1e-6)


def enhance(frame_bgr, gain_threshold=1.5, clahe=False, clahe_clip=1.75, denoise=False):
    """Devuelve (frame, mejorado: bool, ganancia: float)."""
    k = estimar_ganancia(frame_bgr)
    if k < gain_threshold:
        return frame_bgr, False, k

    # Como la ganancia es la misma para todo el frame, la cadena
    # sRGB -> lineal -> *k -> sRGB se reduce a una LUT de 256 entradas
    lut = (lineal_a_srgb(np.clip(_LUT_LINEAL * k, 0.0, 1.0)) * 255.0 + 0.5).astype(np.uint8)
    out = cv2.LUT(frame_bgr, lut)

    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
    return out, True, k
