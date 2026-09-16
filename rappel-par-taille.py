#!/usr/bin/env python3
"""Calcule le rappel par tranche de taille a partir des predictions gardees.

Prend autant de fichiers que voulu et les met sur des lignes comparables, ce
qui est la forme utile pour arbitrer un reglage (image_size, tuilage, modele).
Les tranches sont en pixels natifs ; pour comparer d'un jeu de donnees a
l'autre, convertir en espace modele : d * image_size / largeur_source.
"""
import argparse
import json
import sys

BINS = [(0, 16), (16, 32), (32, 64), (64, 96), (96, 128), (128, 10 ** 6)]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def evaluate(data, score_thresh, iou_thresh):
    n_gt = {b: 0 for b in BINS}
    n_hit = {b: 0 for b in BINS}
    n_pred = n_tp = 0

    for im in data["images"]:
        preds = sorted(
            (p for p in im["pred"] if p["score"] >= score_thresh),
            key=lambda p: -p["score"],
        )
        gts = im["gt"]
        taken = [False] * len(gts)
        n_pred += len(preds)

        for p in preds:
            best, best_i = 0.0, -1
            for j, g in enumerate(gts):
                if taken[j]:
                    continue
                v = iou(p["box"], g["box"])
                if v > best:
                    best, best_i = v, j
            if best >= iou_thresh and best_i >= 0:
                taken[best_i] = True
                n_tp += 1

        for j, g in enumerate(gts):
            side = min(g["box"][2] - g["box"][0], g["box"][3] - g["box"][1])
            for b in BINS:
                if b[0] <= side < b[1]:
                    n_gt[b] += 1
                    if taken[j]:
                        n_hit[b] += 1
                    break

    return n_gt, n_hit, n_pred, n_tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="fichiers de predictions")
    ap.add_argument("--score", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()

    labels = [(f"{b[0]}-{b[1] if b[1] < 10 ** 6 else 'inf'}") for b in BINS]
    header = f"{'run':>10s} " + " ".join(f"{l:>9s}" for l in labels)
    header += f" {'rappel':>8s} {'precis.':>8s} {'pred':>6s}"
    print(f"# seuil score {args.score}, IoU {args.iou}")
    print(header)

    for path in args.runs:
        data = json.load(open(path))
        n_gt, n_hit, n_pred, n_tp = evaluate(data, args.score, args.iou)
        name = data.get("image_size", path.split("/")[-1].replace(".json", ""))
        cells = []
        for b in BINS:
            if n_gt[b]:
                cells.append(f"{n_hit[b]}/{n_gt[b]}")
            else:
                cells.append("-")
        tot_gt = sum(n_gt.values())
        tot_hit = sum(n_hit.values())
        rec = tot_hit / tot_gt if tot_gt else 0
        prec = n_tp / n_pred if n_pred else 0
        print(f"{str(name):>10s} " + " ".join(f"{c:>9s}" for c in cells)
              + f" {rec:8.3f} {prec:8.3f} {n_pred:6d}")


if __name__ == "__main__":
    sys.exit(main())
