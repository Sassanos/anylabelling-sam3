#!/usr/bin/env python3
"""Fait tourner SAM 3 frame par frame sur des sequences UAVDT et garde tout.

Produit les detections reelles que mesure-tracker.py (--detections) donne
ensuite a chaque tracker : les pertes de SAM 3 sont correlees d'une frame a
l'autre, ce que la simulation --drop ne reproduit pas.

Ne juge rien : boites, scores et labels bruts, seuil bas, pour que le seuil
se rejoue hors ligne. Serveur lance (./run-server.sh) avec segment_anything_3.

    ./detecte-uavdt.py --sequences M0403,M0601,M0801 --prompt vehicle \\
        --out runs/tracker/sam3-uavdt.json
"""
import argparse
import base64
import glob
import json
import os
import statistics
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
UAVDT = os.path.join(HERE, "..", "..", "Datasets", "opensource", "raw",
                     "UAVDT")
BASE = "http://127.0.0.1:8000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uavdt", default=UAVDT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--model", default="segment_anything_3")
    ap.add_argument("--prompt", default="vehicle")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {"model": args.model, "prompt": args.prompt, "conf": args.conf,
           "split": args.split, "sequences": {}}
    if os.path.exists(args.out):
        out = json.load(open(args.out))
    sess = requests.Session()
    params = {"text_prompt": args.prompt, "conf_threshold": args.conf,
              "show_boxes": True, "show_masks": False}

    for seq in args.sequences.split(","):
        if seq in out["sequences"]:
            print(f"{seq} deja fait, saute", flush=True)
            continue
        images = sorted(glob.glob(os.path.join(
            args.uavdt, args.split, "img", f"{seq}_img*.jpg")))
        frames, lat = {}, []
        for i, path in enumerate(images):
            index = int(os.path.basename(path).split("_img")[1][:6])
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            t = time.time()
            r = sess.post(f"{BASE}/v1/predict", timeout=900, json={
                "model": args.model,
                "image": f"data:image/jpeg;base64,{b64}",
                "params": params})
            r.raise_for_status()
            lat.append((time.time() - t) * 1000)
            dets = []
            for s in r.json()["data"]["shapes"]:
                xs = [p[0] for p in s["points"]]
                ys = [p[1] for p in s["points"]]
                dets.append([min(xs), min(ys), max(xs), max(ys),
                             float(s.get("score") or 1.0), s["label"]])
            frames[index] = dets
            if (i + 1) % 100 == 0:
                print(f"  {seq} {i + 1}/{len(images)}  "
                      f"{statistics.median(lat[-100:]):.0f} ms", flush=True)
        out["sequences"][seq] = {
            "frames": frames,
            "latence_ms_mediane": round(statistics.median(lat), 1),
        }
        # Ecriture apres chaque sequence : une coupure ne perd qu'une sequence.
        with open(args.out, "w") as fh:
            json.dump(out, fh)
        print(f"{seq} : {len(images)} frames, "
              f"{sum(len(v) for v in frames.values())} detections, "
              f"{statistics.median(lat):.0f} ms", flush=True)


if __name__ == "__main__":
    sys.exit(main())
