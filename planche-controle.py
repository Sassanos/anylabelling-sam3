#!/usr/bin/env python3
"""Planche de contrôle : frames décodées en streaming + boîtes des JSONL.

Extrait les frames du flux DefaultVideo (PyAV, rien écrit sur disque), charge
les annotations des chunks JSONL (compressés .jsonl.zst ou non) couvrant la
plage, dessine les boîtes (une couleur par label, score affiché) et assemble
une planche JPEG — le moyen le plus rapide de contrôler visuellement ce que
l'annotation à grande échelle produit, sans extraire les frames d'un vol.

  planche-controle.py --video <vol>_video.MP4 --index samples_default.csv \
      --jsonl annots/0000011/ --start 21287 --end 21317 --n 6 --out planche.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# Couleurs BGR stables par label fin (taxonomie Anafi UKR)
LABEL_COULEURS = {
    "person": (66, 133, 244),
    "car": (52, 168, 83),
    "van": (251, 188, 5),
    "truck": (244, 180, 0),
    "bus": (255, 167, 14),
    "motorcycle": (255, 87, 34),
    "bicycle": (171, 71, 188),
    "agri_vehicle": (0, 172, 193),
    "mil_tank": (0, 0, 255),
    "mil_apc_ifv": (120, 0, 255),
    "mil_truck": (180, 0, 200),
    "mil_other": (200, 0, 120),
    "vehicle_unknown": (128, 128, 128),
}
DEFAUT = (200, 200, 200)


def charger_annotations(chemins: list[Path]):
    """{sample_index: shapes} depuis des .jsonl / .jsonl.zst / dossiers."""
    fichiers: list[Path] = []
    for c in chemins:
        if c.is_dir():
            fichiers += sorted(p for p in c.iterdir()
                               if p.name.endswith((".jsonl", ".jsonl.zst")))
        else:
            fichiers.append(c)
    ann = {}
    for f in fichiers:
        if f.name.endswith(".zst"):
            lignes = subprocess.run(["zstd", "-dc", str(f)], capture_output=True,
                                    check=True).stdout.decode("utf-8").splitlines()
        else:
            lignes = f.read_text(encoding="utf-8").splitlines()
        for ligne in lignes:
            try:
                r = json.loads(ligne)
            except Exception:
                continue
            if "sample_index" in r:
                ann[int(r["sample_index"])] = r
    return ann


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--index", required=True, help="samples_default.csv (build_index.py)")
    ap.add_argument("--jsonl", required=True, nargs="+",
                    help="fichiers .jsonl/.jsonl.zst ou dossiers de chunks")
    ap.add_argument("--start", type=int, required=True, help="premier sample_index (inclus)")
    ap.add_argument("--end", type=int, required=True, help="dernier sample_index (exclu)")
    ap.add_argument("--n", type=int, default=6, help="frames sur la planche (défaut 6)")
    ap.add_argument("--cols", type=int, default=3, help="colonnes de la planche (défaut 3)")
    ap.add_argument("--largeur", type=int, default=1280, help="largeur d'une vignette (défaut 1280)")
    ap.add_argument("--out", required=True, help="JPEG de sortie")
    args = ap.parse_args()

    import av
    import cv2
    import numpy as np

    ann = charger_annotations([Path(p) for p in args.jsonl])
    presents = sorted(i for i in ann if args.start <= i < args.end)
    if not presents:
        print(f"[!] aucune frame annotée dans [{args.start},{args.end}) "
              f"sur {len(args.jsonl)} source(s)", file=sys.stderr)
        return 2

    # index : tick -> sample_index et sample_index -> tick (seek + décodage)
    tick_de_sample, sample_de_tick = {}, {}
    timescale = None
    with open(args.index, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["stream"] == "default":
                i, t = int(r["sample_index"]), int(r["dts_ticks"])
                tick_de_sample[i] = t
                sample_de_tick[t] = i
                timescale = int(r["timescale"])

    # n frames réparties uniformément sur la plage annotée
    pas = max(1, len(presents) // args.n)
    choisis = presents[::pas][:args.n]

    container = av.open(args.video)
    stream = container.streams.video[0]
    container.seek(tick_de_sample[choisis[0]], stream=stream, backward=True)

    vignettes, vus = [], set()
    for packet in container.demux(stream):
        for frame in packet.decode():
            if frame.pts is None:
                continue
            i = sample_de_tick.get(int(frame.pts))
            if i in choisis and i not in vus:
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                for sh in ann[i].get("shapes", []):
                    pts = sh["points"]
                    x1, y1 = int(min(p[0] for p in pts)), int(min(p[1] for p in pts))
                    x2, y2 = int(max(p[0] for p in pts)), int(max(p[1] for p in pts))
                    coul = LABEL_COULEURS.get(sh["label"], DEFAUT)
                    cv2.rectangle(img, (x1, y1), (x2, y2), coul, 2)
                    txt = f"{sh['label']}"
                    if sh.get("score") is not None:
                        txt += f" {sh['score']:.2f}"
                    cv2.rectangle(img, (x1, y1 - 26), (x1 + 10 * len(txt) + 8, y1), coul, -1)
                    cv2.putText(img, txt, (x1 + 4, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                legende = (f"sample {i} | {len(ann[i].get('shapes', []))} shapes"
                           + (" | ERREUR" if ann[i].get("error") else ""))
                cv2.putText(img, legende, (12, h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                cv2.putText(img, legende, (12, h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 1)
                # réduction
                ech = args.largeur / w
                vignettes.append(cv2.resize(img, (args.largeur, int(h * ech)),
                                            interpolation=cv2.INTER_AREA))
                vus.add(i)
            if vus == set(choisis):
                break
        if vus == set(choisis):
            break
    container.close()
    if len(vignettes) < len(choisis):
        print(f"[!] seulement {len(vignettes)}/{len(choisis)} frames décodées", file=sys.stderr)

    cols = min(args.cols, len(vignettes))
    lignes = (len(vignettes) + cols - 1) // cols
    vh = max(v.shape[0] for v in vignettes)
    planche = np.full((lignes * vh + (lignes + 1) * 8,
                       cols * args.largeur + (cols + 1) * 8, 3),
                      (24, 24, 24), np.uint8)
    for k, v in enumerate(vignettes):
        y, x = divmod(k, cols)
        planche[8 + y * (vh + 8): 8 + y * (vh + 8) + v.shape[0],
                8 + x * (args.largeur + 8): 8 + x * (args.largeur + 8) + v.shape[1]] = v
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    p = Path(args.out)
    tmp = p.with_name(p.stem + "_tmp.jpg")   # imwrite exige une extension .jpg
    cv2.imwrite(str(tmp), planche, [cv2.IMWRITE_JPEG_QUALITY, 90])
    tmp.replace(args.out)
    print(f"[i] planche : {args.out} ({len(vignettes)} frames, "
          f"{len(ann)} frames annotées chargées)")


if __name__ == "__main__":
    sys.exit(main())
