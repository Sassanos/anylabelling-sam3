#!/usr/bin/env python3
"""Recalage visible -> thermique d'un vol : où tombe chaque frame IR dans le RGB.

Pendant une rafale thermique, la piste VisibleVideo (4K) et la piste
ThermalVideo (640x512) filment en même temps, à ~40 ms près. Les deux caméras
sont solidaires : un pixel IR (u, v) correspond au pixel visible

    x = (u + ox) * s        y = (v + oy) * s

avec s = f_visible / f_thermique (focales en pixels : le champ du visible est
dans la télémétrie, `hfov_deg` ; celui du thermique n'y est pas, mesuré à 32°)
et (ox, oy) un décalage que la corrélation des contours mesure frame par
frame, dans une fenêtre bornée autour de la coïncidence des axes optiques.

L'écart à cette coïncidence suit une loi en 1/s (AXE) : un écart angulaire
constant entre les deux axes, plus un zoom numérique du visible qui n'est pas
centré sur l'image. Ajustée sur 3 818 frames bien corrélées des vols 0000011,
0000012 et 0000018 : résidu médian 0,6 px IR, 2,3 px au 90e centile, biais
par vol sous 1 px. C'est elle qui recale les scènes sans contour (prairie,
vue haute), où la corrélation donne un pic quelconque.

Sortie : `<sortie>/<vol>/<vol>_recalage.csv`, une ligne par frame thermique
valide appariée (plus proche frame visible par `utc_us`, tolérance 70 ms) :
samples, écart, champ visible, s, décalage mesuré (ox, oy), score de
corrélation, et décalage retenu (ox_l, oy_l) = loi AXE + médiane des résidus
des frames bien corrélées voisines (à moins de --voisinage-s secondes).
`couvert` vaut 0 quand le visible, zoomé, ne couvre pas toute l'image IR.

    X-AnyLabeling-Server/.venv/bin/python recalage-ir.py --vol 0000011
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pistes_io import (RACINE, SORTIES, SORTIES_CAMPAGNE3,  # noqa: E402
                       VOLS_ANAFIUKR, decode_frames, en_tache_de_fond)

_spec = importlib.util.spec_from_file_location(
    "annote_video_sam3", HERE / "annote-video-sam3.py")
thermique = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(thermique)

HFOV_IR_DEG = 32.0
# Écart du centre IR au centre du visible, en px IR : (A + B / s) par axe.
AXE = {"x": (12.15, -46.84), "y": (-27.11, 25.50)}
IR_W, IR_H = thermique.THERMAL_W, thermique.THERMAL_H
COLONNES = ["thermal_sample", "thermal_ticks", "visible_sample", "visible_ticks",
            "dt_ms", "hfov_visible_deg", "s", "ox", "oy", "ncc", "ox_l", "oy_l",
            "couvert"]


def dossier_ir(vol: str) -> Path:
    base = SORTIES if vol in VOLS_ANAFIUKR else SORTIES_CAMPAGNE3
    return base.with_name("ir") / vol


def lignes_index(chemin: Path):
    with open(chemin, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["valid"] == "1"]


def apparie(thermiques, visibles, tol_ms: float):
    """[(ligne thermique, ligne visible, dt_ms)] par plus proche utc_us."""
    vis = sorted(visibles, key=lambda r: int(r["utc_us"]))
    cles = [int(r["utc_us"]) for r in vis]
    paires = []
    for t in thermiques:
        u = int(t["utc_us"])
        k = bisect_left(cles, u)
        j = min((j for j in (k - 1, k) if 0 <= j < len(cles)),
                key=lambda j: abs(cles[j] - u), default=None)
        if j is not None and abs(cles[j] - u) <= tol_ms * 1000:
            paires.append((t, vis[j], (u - cles[j]) / 1000.0))
    return paires


def gradient(image):
    g = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.5)
    g = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return g / (g.mean() + 1e-6)


def echelle(hfov_visible_deg: float, largeur: int) -> float:
    """Pixels visibles par pixel IR."""
    return (largeur / 2 / math.tan(math.radians(hfov_visible_deg) / 2)) / (
        IR_W / 2 / math.tan(math.radians(HFOV_IR_DEG) / 2))


def recale(gris, ir8, s, marge=0.3, fenetre=60):
    """(ox, oy, ncc) ou None si le visible ramené à l'échelle IR est trop petit."""
    h, w = gris.shape
    w2, h2 = int(round(w / s)), int(round(h / s))
    if w2 < 200 or h2 < 160:
        return None
    gr = gradient(cv2.resize(gris, (w2, h2), interpolation=cv2.INTER_AREA))
    tw, th = int(min(IR_W, w2) * (1 - marge)), int(min(IR_H, h2) * (1 - marge))
    x0, y0 = (IR_W - tw) // 2, (IR_H - th) // 2
    ax, ay = round(w2 / 2 - tw / 2), round(h2 / 2 - th / 2)
    xa, ya = max(0, ax - fenetre), max(0, ay - fenetre)
    xb, yb = min(w2, ax + tw + fenetre), min(h2, ay + th + fenetre)
    res = cv2.matchTemplate(gr[ya:yb, xa:xb],
                            gradient(ir8)[y0:y0 + th, x0:x0 + tw],
                            cv2.TM_CCOEFF_NORMED)
    _, ncc, _, loc = cv2.minMaxLoc(res)
    return xa + loc[0] - x0, ya + loc[1] - y0, float(ncc)


def axe(s: float):
    """Écart (dx, dy) du centre IR au centre du visible prédit par la loi AXE."""
    return AXE["x"][0] + AXE["x"][1] / s, AXE["y"][0] + AXE["y"][1] / s


def lisse(lignes, ncc_min, demi_fenetre, voisinage_s, timescale):
    """Décalage retenu : loi AXE, corrigée de la médiane des résidus des frames
    bien corrélées voisines. Sans voisine proche dans le temps, la loi seule."""
    recalables = [l for l in lignes if "s" in l]
    residus = np.array([[l["ox"] - (l["w2"] / 2 - IR_W / 2) - axe(l["s"])[0],
                         l["oy"] - (l["h2"] / 2 - IR_H / 2) - axe(l["s"])[1]]
                        for l in recalables
                        if l["ncc"] is not None and l["ncc"] >= ncc_min]
                       ).reshape(-1, 2)
    ticks = np.array([l["thermal_ticks"] for l in recalables
                      if l["ncc"] is not None and l["ncc"] >= ncc_min])
    for l in recalables:
        dx, dy = axe(l["s"])
        k = np.searchsorted(ticks, l["thermal_ticks"])
        v = np.arange(max(0, k - demi_fenetre), min(len(ticks), k + demi_fenetre))
        v = v[np.abs(ticks[v] - l["thermal_ticks"]) <= voisinage_s * timescale]
        if len(v):
            dx += float(np.median(residus[v, 0]))
            dy += float(np.median(residus[v, 1]))
        l["ox_l"] = dx + l["w2"] / 2 - IR_W / 2
        l["oy_l"] = dy + l["h2"] / 2 - IR_H / 2


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vol", required=True)
    ap.add_argument("--racine", type=Path, default=RACINE)
    ap.add_argument("--sortie", type=Path, default=None,
                    help="défaut : AnafiUKR/ir/<vol>/ ou CAMPAGNE3/ir/<vol>/")
    ap.add_argument("--tol-ms", type=float, default=70.0)
    ap.add_argument("--ncc-min", type=float, default=0.15,
                    help="corrélation minimale pour qu'une frame vote au lissage")
    ap.add_argument("--fenetre-lissage", type=int, default=15,
                    help="demi-fenêtre de la médiane, en frames bien corrélées")
    ap.add_argument("--voisinage-s", type=float, default=3.0,
                    help="écart de temps maximal d'une frame voisine")
    ap.add_argument("--no-hwaccel", action="store_true")
    args = ap.parse_args()

    index = args.racine / "campagne3_index" / "annotation" / args.vol / "index"
    videos = sorted((args.racine / "campagne3_index").glob(f"*/{args.vol}_video.MP4"))
    if len(videos) != 1:
        raise SystemExit(f"[!] {len(videos)} vidéo(s) {args.vol}_video.MP4")
    thermiques = lignes_index(index / "samples_thermal.csv")
    visibles = lignes_index(index / "samples_visible.csv")
    paires = apparie(thermiques, visibles, args.tol_ms)
    print(f"[i] {args.vol} : {len(thermiques)} frames thermiques valides, "
          f"{len(paires)} appariées à {args.tol_ms:.0f} ms", flush=True)

    lignes = []
    t0 = dernier = time.time()
    with open(videos[0], "rb") as f:
        clamp, _ = thermique.clamp_destripage(f, thermiques)
        ticks = [int(v["dts_ticks"]) for _t, v, _d in paires]
        for k, image in en_tache_de_fond(decode_frames(
                videos[0], ticks, not args.no_hwaccel, piste=1)):
            t, v, dt = paires[k]
            ligne = {"thermal_sample": int(t["sample_index"]),
                     "thermal_ticks": int(t["dts_ticks"]),
                     "visible_sample": int(v["sample_index"]),
                     "visible_ticks": int(v["dts_ticks"]),
                     "dt_ms": round(dt, 1),
                     "hfov_visible_deg": float(v["hfov_deg"]),
                     "ncc": None, "ox_l": None, "oy_l": None}
            lignes.append(ligne)
            if image is None:
                continue
            brute = thermique.lit_thermique(f, t)
            if brute.min() == brute.max():
                continue
            ir8, _lo, _hi = thermique.thermique_8bits(brute, clamp)
            h, w = image.shape[:2]
            s = echelle(ligne["hfov_visible_deg"], w)
            ligne.update(s=s, w2=round(w / s), h2=round(h / s),
                         couvert=int(w / s >= IR_W and h / s >= IR_H))
            r = recale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), ir8, s)
            if r is not None:   # sinon visible trop zoomé : loi AXE seule
                ligne["ox"], ligne["oy"], ligne["ncc"] = r
            if time.time() - dernier > 30:
                dernier = time.time()
                print(f"  {k + 1}/{len(paires)} | {(k + 1) / (dernier - t0):.1f} "
                      "frames/s", flush=True)
    lisse(lignes, args.ncc_min, args.fenetre_lissage, args.voisinage_s,
          int(thermiques[0]["timescale"]))

    dossier = args.sortie or dossier_ir(args.vol)
    dossier.mkdir(parents=True, exist_ok=True)
    sortie = dossier / f"{args.vol}_recalage.csv"
    with open(sortie, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES, extrasaction="ignore")
        w.writeheader()
        for l in lignes:
            w.writerow({c: ("" if l.get(c) is None else
                            round(l[c], 4) if isinstance(l[c], float) else l[c])
                        for c in COLONNES})
    ncc = np.array([l["ncc"] for l in lignes if l["ncc"] is not None])
    bons = int((ncc >= args.ncc_min).sum())
    print(f"[i] {len(lignes)} paires en {time.time() - t0:.0f} s : {len(ncc)} "
          f"recalées, {bons} à ncc >= {args.ncc_min} (médiane {np.median(ncc):.2f}), "
          f"{sum(1 for l in lignes if l.get('couvert'))} où le visible couvre "
          f"toute l'image IR -> {sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
