#!/usr/bin/env python3
"""Planche des pistes : une ligne par piste, vignettes réparties sur sa durée.

Le contrôle le plus direct de la pureté : une piste qui passe d'un objet à un
autre change d'aspect au milieu de sa ligne. Les frames nécessaires sont
décodées en streaming (flux 0), rien n'est extrait sur disque.

    X-AnyLabeling-Server/.venv/bin/python planche-pistes.py --vol 0000011 \\
        --n 40 --k 10 --tri longueur --out runs/pistes/planche-longues.jpg
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pistes_io import (RACINE, chemins_vol, decode_frames,  # noqa: E402
                       dernier_jeu, en_tache_de_fond, ticks_index)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vol", required=True)
    ap.add_argument("--racine", type=Path, default=RACINE)
    ap.add_argument("--pistes", type=Path, default=None,
                    help="jeu de pistes (défaut : le plus récent du vol)")
    ap.add_argument("--gid", type=int, nargs="+", default=None,
                    help="group_id précis (sinon --n pistes choisies par --tri)")
    ap.add_argument("--n", type=int, default=30, help="pistes sur la planche")
    ap.add_argument("--k", type=int, default=10, help="vignettes par piste")
    ap.add_argument("--tri", choices=["longueur", "aleatoire", "debut"],
                    default="longueur")
    ap.add_argument("--classes", nargs="+", default=None)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--debut", type=int, default=None, help="sample_index inclus")
    ap.add_argument("--fin", type=int, default=None, help="sample_index exclu")
    ap.add_argument("--vignette", type=int, default=96, help="côté en pixels")
    ap.add_argument("--marge", type=float, default=0.6,
                    help="contexte autour de la boîte, en fraction de sa taille")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import cv2

    chemins = chemins_vol(args.vol, args.racine)
    jeu = args.pistes or dernier_jeu(chemins["pistes"], args.vol)
    pistes = []
    with open(jeu, encoding="utf-8") as fh:
        for ligne in fh:
            p = json.loads(ligne)
            if args.gid:
                if p["group_id"] in args.gid:
                    pistes.append(p)
                continue
            if p["n_detected"] < args.min_len:
                continue
            if args.classes and p["coarse"] not in args.classes:
                continue
            if (args.debut is not None and p["sample_max"] < args.debut) or (
                    args.fin is not None and p["sample_min"] >= args.fin):
                continue
            pistes.append(p)
    if not args.gid:
        if args.tri == "longueur":
            pistes.sort(key=lambda p: -p["n_detected"])
        elif args.tri == "aleatoire":
            random.Random(args.seed).shuffle(pistes)
        pistes = pistes[:args.n]
        pistes.sort(key=lambda p: p["group_id"])
    if not pistes:
        raise SystemExit("[!] aucune piste retenue")

    # k frames détectées (pas interpolées) réparties sur chaque piste
    besoins = {}
    choix = []
    for p in pistes:
        det = [f for f in p["frames"] if not f.get("interpolated")]
        idx = np.unique(np.linspace(0, len(det) - 1, args.k).round().astype(int))
        choix.append([det[i] for i in idx])
        for f in choix[-1]:
            besoins.setdefault(f["i"], []).append(f)

    ticks = ticks_index(chemins["index"])
    samples = sorted(besoins)
    vignettes = {}
    v = args.vignette
    for pos, image in en_tache_de_fond(
            decode_frames(chemins["video"], [ticks[s] for s in samples])):
        if image is None:
            continue
        h, w = image.shape[:2]
        for f in besoins[samples[pos]]:
            x1, y1, x2, y2 = f["bbox"]
            cote = max(x2 - x1, y2 - y1) * (1 + 2 * args.marge)
            cote = max(cote, 24)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            a = int(max(0, cx - cote / 2)), int(max(0, cy - cote / 2))
            b = int(min(w, cx + cote / 2)), int(min(h, cy + cote / 2))
            crop = image[a[1]:b[1], a[0]:b[0]]
            ex, ey = v / crop.shape[1], v / crop.shape[0]
            vignette = cv2.resize(crop, (v, v), interpolation=cv2.INTER_CUBIC
                                  if crop.shape[0] < v else cv2.INTER_AREA)
            # la boîte elle-même, tracée après réduction pour rester visible
            cv2.rectangle(vignette, (int((x1 - a[0]) * ex), int((y1 - a[1]) * ey)),
                          (int((x2 - a[0]) * ex), int((y2 - a[1]) * ey)),
                          (0, 255, 255), 1)
            vignettes[(samples[pos], tuple(f["bbox"]))] = vignette

    entete = 260
    hauteur_ligne = v + 22
    planche = np.full((len(pistes) * hauteur_ligne + 8,
                       entete + args.k * (v + 4) + 8, 3), 28, np.uint8)
    for r, (p, frames) in enumerate(zip(pistes, choix)):
        y = 4 + r * hauteur_ligne
        votes = ", ".join(f"{k} {int(100 * x)}%"
                          for k, x in list(p["labels"].items())[:2])
        duree = p["t_max_s"] - p["t_min_s"]
        for ligne, t in enumerate((
                f"#{p['group_id']}  {p['label']}",
                f"{p['n_detected']} det. / {duree:.1f} s",
                f"{votes}",
                f"samples {p['sample_min']}-{p['sample_max']}",
                f"~{p['size_median_px']:.0f} px, score {p['score_mean']:.2f}")):
            cv2.putText(planche, t, (8, y + 18 + 19 * ligne),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 if ligne else 0.6,
                        (235, 235, 235), 1, cv2.LINE_AA)
        for c, f in enumerate(frames):
            x = entete + c * (v + 4)
            img = vignettes.get((f["i"], tuple(f["bbox"])))
            if img is not None:
                planche[y:y + v, x:x + v] = img
            cv2.putText(planche, str(f["i"]), (x + 2, y + v + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1,
                        cv2.LINE_AA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), planche, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[i] {len(pistes)} pistes x {args.k} vignettes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
