#!/usr/bin/env python3
"""Mesure l'association des group_id contre de vraies identites (UAVDT).

Le flux vise : SAM 3 frame par frame en classes grossieres, association des
group_id, puis une classe fine choisie **par piste**. Deux erreurs n'ont pas
le meme cout :
- une piste **fragmentee** (un objet reel coupe en plusieurs pistes) coute un
  clic de plus a la revue ;
- une piste **impure** (deux objets reels dans la meme piste) donne une classe
  fausse a une partie de ses frames, sans que rien ne le signale.

Les sequences MOT d'UAVDT (drone, vehicules, 30 fps, 1024x540) portent un
`target id` par objet. On donne au tracker les boites de verite terrain sans
leur identite, eventuellement degradees (detections perdues, bruit), a une
cadence choisie, et on compare les pistes rendues aux vraies.

    X-AnyLabeling/.venv/bin/python mesure-tracker.py --stride 1 3 30 \\
        --drop 0 0.2 --out runs/tracker/uavdt.json
"""
import argparse
import collections
import glob
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "X-AnyLabeling"))

from anylabeling.views.labeling.utils.group_id_association import (  # noqa
    Det,
    TrackingConfig,
    associate,
    build_gmc,
    merge_fragments,
)

UAVDT = os.path.join(
    HERE, "..", "..", "Datasets", "opensource", "raw", "UAVDT"
)
# Sequences MOT variees (jour/nuit/brouillard, altitudes). Regler sur TRAIN,
# rapporter sur TEST : sinon on mesure le reglage, pas le tracker.
DEFAUT = {
    "test": "M0203,M0205,M0208,M0209,M0403,M0601,M0602,M0606,M0701,M0801",
    "train": "M0101,M0202,M0204,M0206,M0301,M0402,M0501,M0702,M0902,M1005",
}


def load_sequence(root, split, seq):
    """Frames d'une sequence : [(indice, chemin image, [(box, target id)])]."""
    frames = []
    for path in sorted(glob.glob(os.path.join(root, split, "ann",
                                              f"{seq}_img*.jpg.json"))):
        name = os.path.basename(path)[: -len(".json")]
        index = int(name.split("_img")[1].split(".")[0])
        data = json.load(open(path))
        objects = []
        for obj in data["objects"]:
            (x1, y1), (x2, y2) = obj["points"]["exterior"]
            tid = next((t["value"] for t in obj["tags"]
                        if t["name"] == "target id"), None)
            if tid is None or x2 <= x1 or y2 <= y1:
                continue
            objects.append((np.array([x1, y1, x2, y2], float), int(tid)))
        frames.append((index, os.path.join(root, split, "img", name),
                       objects))
    frames.sort(key=lambda f: f[0])
    return frames


def degrade(box, jitter, rng):
    """Bruit gaussien proportionnel a la taille, sur centre et dimensions."""
    if jitter <= 0:
        return box
    w, h = box[2] - box[0], box[3] - box[1]
    cx = (box[0] + box[2]) / 2 + rng.gauss(0, jitter * w)
    cy = (box[1] + box[3]) / 2 + rng.gauss(0, jitter * h)
    w *= max(0.3, 1 + rng.gauss(0, jitter))
    h *= max(0.3, 1 + rng.gauss(0, jitter))
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def run_one(frames, stride, drop, jitter, config, seed, cache_gmc):
    """Associe une sequence sous-echantillonnee et rend les metriques."""
    rng = random.Random(seed)
    kept = frames[::stride]
    dets, truth = [], {}
    tuples = []
    for number, (_, image, objects) in enumerate(kept):
        # Chemin factice : build_gmc retrouve l'image par le nom sans extension.
        tuples.append((number, os.path.splitext(image)[0] + ".json", {}))
        for idx, (box, tid) in enumerate(objects):
            if rng.random() < drop:
                continue
            dets.append(Det(frame=number, shape_idx=idx, label="vehicle",
                            score=1.0, box=degrade(box, jitter, rng)))
            truth[(number, idx)] = tid

    key = (id(frames), stride)
    if config.use_gmc:
        if key not in cache_gmc:
            images_dir = os.path.dirname(kept[0][1])
            cache_gmc[key] = build_gmc(tuples, images_dir, config.gmc_scale)[0]
        gmc = cache_gmc[key]
    else:
        gmc = {}

    tracks = associate(dets, gmc, config)
    tracks = merge_fragments(tracks, config)
    tracks = [t for t in tracks if len(t.members) >= config.min_len]

    # Purete : part des detections d'une piste qui appartiennent a son objet
    # majoritaire. Fragmentation : pistes rendues par objet reel.
    n_det = len(dets)
    in_tracks = 0
    majority = 0
    impure_det = 0
    impure_tracks = 0
    pieces = collections.defaultdict(set)
    for n, t in enumerate(tracks):
        ids = collections.Counter(truth[(d.frame, d.shape_idx)]
                                  for d in t.members)
        top = ids.most_common(1)[0][1]
        in_tracks += len(t.members)
        majority += top
        if top < len(t.members):
            impure_tracks += 1
            impure_det += len(t.members)
        for tid, c in ids.items():
            # Un objet n'est compte dans une piste que s'il y pese vraiment,
            # sinon une seule detection volee gonflerait la fragmentation.
            if c >= max(2, 0.1 * len(t.members)):
                pieces[tid].add(n)
    gt_ids = {tid for tid in truth.values()}
    frag = [len(pieces[tid]) for tid in gt_ids if pieces[tid]]
    return {
        "frames": len(kept),
        "detections": n_det,
        "gt_tracks": len(gt_ids),
        "tracks": len(tracks),
        "coverage": in_tracks / n_det if n_det else 0.0,
        "purity": majority / in_tracks if in_tracks else 0.0,
        "impure_tracks": impure_tracks,
        "impure_det_share": impure_det / in_tracks if in_tracks else 0.0,
        "pieces_per_object": float(np.mean(frag)) if frag else 0.0,
        "objects_in_one_piece": (sum(1 for f in frag if f == 1) / len(frag)
                                 if frag else 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uavdt", default=UAVDT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sequences", default=None,
                    help="defaut : 10 sequences MOT du split choisi")
    ap.add_argument("--stride", type=int, nargs="+", default=[1],
                    help="1 = 30 fps, 3 = 10 fps, 30 = 1 fps")
    ap.add_argument("--drop", type=float, nargs="+", default=[0.0],
                    help="part des detections retirees au hasard")
    ap.add_argument("--jitter", type=float, nargs="+", default=[0.0],
                    help="ecart-type du bruit, en fraction de la taille")
    ap.add_argument("--no-gmc", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="CLE=VAL",
                    help="surcharge un champ de TrackingConfig, ex. "
                         "--set max_dist=1.0 (repetable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = TrackingConfig(use_gmc=not args.no_gmc)
    if args.no_merge:
        config.merge_gap = 0
    for item in args.set:
        key, value = item.split("=", 1)
        current = getattr(config, key)
        setattr(config, key, type(current)(value))

    results = []
    cache_gmc = {}
    for seq in (args.sequences or DEFAUT[args.split]).split(","):
        t0 = time.time()
        frames = load_sequence(args.uavdt, args.split, seq)
        for stride in args.stride:
            for drop in args.drop:
                for jitter in args.jitter:
                    m = run_one(frames, stride, drop, jitter, config,
                                args.seed, cache_gmc)
                    m.update(seq=seq, stride=stride, drop=drop, jitter=jitter)
                    results.append(m)
                    print(f"{seq} fps={30 / stride:5.1f} drop={drop:.2f} "
                          f"jit={jitter:.2f}  pistes {m['tracks']:4}/"
                          f"{m['gt_tracks']:3} objets  purete "
                          f"{m['purity']:.3f}  impures {m['impure_tracks']:3}"
                          f"  morceaux/objet {m['pieces_per_object']:.2f}  "
                          f"couverture {m['coverage']:.3f}", flush=True)
        print(f"  ({time.time() - t0:.0f} s)", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"args": vars(args), "config": vars(config),
               "results": results}, open(args.out, "w"), indent=1)

    print("\n== Moyenne sur les sequences (ponderee par detections)")
    groups = collections.defaultdict(list)
    for r in results:
        groups[(r["stride"], r["drop"], r["jitter"])].append(r)
    for (stride, drop, jitter), rs in sorted(groups.items()):
        w = np.array([r["detections"] for r in rs], float)
        avg = lambda k: float(np.average([r[k] for r in rs], weights=w))
        print(f"  fps={30 / stride:5.1f} drop={drop:.2f} jit={jitter:.2f}  "
              f"purete {avg('purity'):.3f}  detections en piste impure "
              f"{100 * avg('impure_det_share'):4.1f} %  morceaux/objet "
              f"{avg('pieces_per_object'):.2f}  objets d'un seul tenant "
              f"{100 * avg('objects_in_one_piece'):4.1f} %  couverture "
              f"{avg('coverage'):.3f}")


if __name__ == "__main__":
    main()
