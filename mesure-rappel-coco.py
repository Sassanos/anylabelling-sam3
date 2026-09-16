#!/usr/bin/env python3
"""Fait tourner un modele du serveur sur un echantillon COCO et garde tout.

Ne juge rien : ecrit les predictions brutes avec leur score, pour que le
rappel puisse etre recalcule hors ligne a n'importe quel seuil sans relancer
l'inference. C'est ce qui rend le lot L0 rejouable : une passe d'inference,
autant d'analyses qu'on veut.

L'echantillonnage est regulier et non pas les N premieres images : les jeux
aeriens sont des sequences de vol, ou N images consecutives sont la meme scene.

    ./mesure-rappel-coco.py --dataset Datasets/release/rgb/vtuav_rgb \
        --n 120 --category person --prompt person --conf 0.15 --out run.json
    ./rappel-par-taille.py run.json --score 0.25 --iou 0.3
"""
import argparse
import base64
import json
import os
import statistics
import sys
import time

import requests

BASE = "http://127.0.0.1:8000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="segment_anything_3")
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--category", default=None,
                    help="ne garder que cette classe dans la verite "
                         "terrain (nom COCO)")
    ap.add_argument("--with-empty", action="store_true",
                    help="inclure aussi des images sans objet annote")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = args.dataset
    coco = json.load(
        open(os.path.join(root, "annotations", f"{args.split}.json"))
    )
    keep_ids = None
    if args.category:
        keep_ids = {
            c["id"] for c in coco["categories"] if c["name"] == args.category
        }
        if not keep_ids:
            print(f"[!] classe inconnue : {args.category}", file=sys.stderr)
            return 2

    by_image = {}
    for a in coco["annotations"]:
        if keep_ids is not None and a["category_id"] not in keep_ids:
            continue
        by_image.setdefault(a["image_id"], []).append(a)

    images = sorted(coco["images"], key=lambda im: im["id"])
    if not args.with_empty:
        images = [im for im in images if by_image.get(im["id"])]
    # Echantillon regulier plutot que les N premieres : les jeux aeriens sont
    # des sequences de vol, N images consecutives sont la meme scene.
    step = max(1, len(images) // args.n)
    images = images[::step][: args.n]

    print(f"[i] {len(images)} images | {args.model} | prompt '{args.prompt}' "
          f"| conf {args.conf}", flush=True)

    sess = requests.Session()
    params = {
        "text_prompt": args.prompt,
        "conf_threshold": args.conf,
        "show_boxes": True,
        "show_masks": False,
    }

    out = {"model": args.model, "prompt": args.prompt, "conf": args.conf,
           "dataset": root, "split": args.split, "images": []}
    lat = []
    for i, im in enumerate(images):
        path = os.path.join(root, "images", im["file_name"])
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        t = time.time()
        try:
            r = sess.post(f"{BASE}/v1/predict", timeout=900, json={
                "model": args.model,
                "image": f"data:image/jpeg;base64,{b64}",
                "params": params})
            r.raise_for_status()
            shapes = r.json()["data"]["shapes"]
        except Exception as exc:
            print(f"[!] {im['file_name']} : {exc}", file=sys.stderr)
            return 1
        lat.append((time.time() - t) * 1000)

        preds = []
        for s in shapes:
            xs = [p[0] for p in s["points"]]
            ys = [p[1] for p in s["points"]]
            preds.append({"box": [min(xs), min(ys), max(xs), max(ys)],
                          "score": s.get("score", 1.0),
                          "label": s["label"]})
        gt = [
            {"box": [a["bbox"][0], a["bbox"][1],
                     a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]}
            for a in by_image.get(im["id"], [])
        ]
        out["images"].append({"file_name": im["file_name"],
                              "width": im["width"], "height": im["height"],
                              "gt": gt, "pred": preds})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(images)} : {len(preds)} pred / "
                  f"{len(gt)} gt, {lat[-1]:.0f} ms", flush=True)

    out["latence_ms_mediane"] = round(statistics.median(lat), 1)
    out["latence_ms_p90"] = round(
        sorted(lat)[int(0.9 * (len(lat) - 1))], 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"[i] latence mediane {out['latence_ms_mediane']} ms | ecrit dans "
          f"{args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
