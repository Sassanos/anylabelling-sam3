#!/usr/bin/env python3
"""Mesure la purete des pistes contre de vraies identites (UAVDT).

Le flux vise : SAM 3 frame par frame en classes grossieres, association des
group_id, puis une classe fine choisie **par piste**. Deux erreurs n'ont pas
le meme cout :
- une piste **fragmentee** (un objet reel coupe en plusieurs pistes) coute un
  clic de plus a la revue ;
- une piste **impure** (deux objets reels dans la meme piste) donne une classe
  fausse a une partie de ses frames, sans que rien ne le signale.

Les sequences MOT d'UAVDT (drone, vehicules, 30 fps, 1024x540) portent un
`target id` par objet. Deux sources de detections :
- la verite terrain sans identite, eventuellement degradee (--drop, --jitter) ;
- de vraies detections SAM 3 (--detections, produites par detecte-uavdt.py),
  appariees a la verite terrain par IoU pour connaitre leur identite ; celles
  qui ne s'apparient a rien sont des faux positifs, comptes a part.

Trackers : `maison` (group_id_association.py du client, charge par chemin pour
ne pas importer Qt) et ceux de boxmot (bytetrack, ocsort, botsort...), dans le
venv .venv-tracking qui l'a installe.

    .venv-tracking/bin/python mesure-tracker.py --tracker maison bytetrack \\
        ocsort botsort --detections runs/tracker/sam3-uavdt.json \\
        --out runs/tracker/comparaison.json
"""
import argparse
import collections
import glob
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "group_id_association",
    os.path.join(HERE, "X-AnyLabeling", "anylabeling", "views", "labeling",
                 "utils", "group_id_association.py"),
)
gia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gia)

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


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def build_inputs(frames, stride, source, rng):
    """Detections par frame renumerotee, et identite vraie de chacune.

    Retourne ``kept`` (frames gardees), ``per_frame`` (liste de tableaux Nx5
    x1 y1 x2 y2 score) et ``truth`` {(frame, indice): target id ou None}.
    """
    kept = frames[::stride]
    per_frame, truth = [], {}
    n_gt = 0
    for number, (index, _image, objects) in enumerate(kept):
        n_gt += len(objects)
        if source["kind"] == "gt":
            rows = []
            for box, tid in objects:
                if rng.random() < source["drop"]:
                    continue
                truth[(number, len(rows))] = tid
                rows.append([*degrade(box, source["jitter"], rng), 1.0])
        else:
            raw = source["frames"].get(str(index), [])
            rows = [r[:5] for r in raw if r[4] >= source["score"]]
            gt_boxes = np.array([b for b, _ in objects]) if objects \
                else np.zeros((0, 4))
            dets = np.array([r[:4] for r in rows]) if rows \
                else np.zeros((0, 4))
            iou = iou_matrix(dets, gt_boxes)
            pairs = []
            if iou.size:
                from scipy.optimize import linear_sum_assignment
                r_idx, c_idx = linear_sum_assignment(-iou)
                pairs = [(r, c) for r, c in zip(r_idx, c_idx)
                         if iou[r, c] >= source["iou"]]
            matched = {r: objects[c][1] for r, c in pairs}
            for i in range(len(rows)):
                truth[(number, i)] = matched.get(i)
        per_frame.append(np.array(rows, float).reshape(-1, 5))
    return kept, per_frame, truth, n_gt


def track_maison(kept, per_frame, config, cache_gmc, key):
    dets = []
    tuples = []
    for number, (_, image, _) in enumerate(kept):
        tuples.append((number, os.path.splitext(image)[0] + ".json", {}))
        for i, row in enumerate(per_frame[number]):
            dets.append(gia.Det(frame=number, shape_idx=i, label="vehicle",
                                score=float(row[4]), box=row[:4]))
    if config.use_gmc:
        if key not in cache_gmc:
            images_dir = os.path.dirname(kept[0][1])
            cache_gmc[key] = gia.build_gmc(tuples, images_dir,
                                           config.gmc_scale)[0]
        gmc = cache_gmc[key]
    else:
        gmc = {}
    tracks = gia.associate(dets, gmc, config)
    tracks = gia.merge_fragments(tracks, config)
    return [[(d.frame, d.shape_idx) for d in t.members] for t in tracks]


def make_boxmot(name, fps, device):
    import boxmot
    classes = {"bytetrack": "ByteTrack", "ocsort": "OcSort",
               "botsort": "BotSort", "deepocsort": "DeepOcSort",
               "strongsort": "StrongSort", "boosttrack": "BoostTrack"}
    cls = getattr(boxmot, classes[name])
    kwargs = {}
    if name in ("botsort", "deepocsort", "strongsort", "boosttrack"):
        kwargs = dict(reid_weights="osnet_x0_25_msmt17.pt", device=device,
                      half=False)
    if name in ("bytetrack", "botsort"):
        kwargs["frame_rate"] = int(round(fps))
    return cls(**kwargs)


def track_boxmot(name, kept, per_frame, fps, device):
    import cv2
    tracker = make_boxmot(name, fps, device)
    members = collections.defaultdict(list)
    for number, (_, image, _) in enumerate(kept):
        rows = per_frame[number]
        dets = np.zeros((len(rows), 6))
        if len(rows):
            dets[:, :5] = rows
        img = cv2.imread(image)
        out = tracker.update(dets, img)
        for row in np.asarray(out).reshape(-1, 8):
            members[int(row[4])].append((number, int(row[7])))
    return list(members.values())


def metrics(tracks, truth, n_gt, min_len):
    tracks = [t for t in tracks if len(t) >= min_len]
    n_det = len(truth)
    n_true = sum(1 for v in truth.values() if v is not None)
    in_tracks = sum(len(t) for t in tracks)
    true_in = majority = fp_in = impure = 0
    pieces = collections.defaultdict(set)
    for n, t in enumerate(tracks):
        ids = collections.Counter(truth[m] for m in t
                                  if truth.get(m) is not None)
        fp_in += sum(1 for m in t if truth.get(m) is None)
        if not ids:
            continue
        count = sum(ids.values())
        top = ids.most_common(1)[0][1]
        true_in += count
        majority += top
        if top < count:
            impure += 1
        for tid, c in ids.items():
            # Un objet ne compte dans une piste que s'il y pese vraiment,
            # sinon une detection volee gonflerait la fragmentation.
            if c >= max(2, 0.1 * count):
                pieces[tid].add(n)
    gt_ids = {v for v in truth.values() if v is not None}
    frag = [len(pieces[tid]) for tid in gt_ids if pieces[tid]]
    return {
        "gt_boxes": n_gt,
        "detections": n_det,
        "recall_det": n_true / n_gt if n_gt else 0.0,
        "fp_share": (n_det - n_true) / n_det if n_det else 0.0,
        "gt_objects": len(gt_ids),
        "tracks": len(tracks),
        "coverage": in_tracks / n_det if n_det else 0.0,
        "purity": majority / true_in if true_in else 0.0,
        "impure_tracks": impure,
        "fp_in_tracks": fp_in / in_tracks if in_tracks else 0.0,
        "pieces_per_object": float(np.mean(frag)) if frag else 0.0,
        "objects_in_one_piece": (sum(1 for f in frag if f == 1) / len(frag)
                                 if frag else 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uavdt", default=UAVDT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sequences", default=None,
                    help="defaut : 10 sequences MOT du split, ou celles du "
                         "fichier --detections")
    ap.add_argument("--tracker", nargs="+", default=["maison"])
    ap.add_argument("--stride", type=int, nargs="+", default=[1],
                    help="1 = 30 fps, 3 = 10 fps, 30 = 1 fps")
    ap.add_argument("--drop", type=float, nargs="+", default=[0.0],
                    help="verite terrain : part des detections retirees")
    ap.add_argument("--jitter", type=float, nargs="+", default=[0.0],
                    help="verite terrain : bruit en fraction de la taille")
    ap.add_argument("--detections", default=None,
                    help="detections SAM 3 de detecte-uavdt.py")
    ap.add_argument("--score", type=float, nargs="+", default=[0.3],
                    help="SAM 3 : seuil de score")
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--no-gmc", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="CLE=VAL",
                    help="surcharge un champ de TrackingConfig du tracker "
                         "maison, ex. --set max_dist=1.0 (repetable)")
    ap.add_argument("--device", default="cpu",
                    help="ReID des trackers boxmot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = gia.TrackingConfig(use_gmc=not args.no_gmc, min_len=1)
    if args.no_merge:
        config.merge_gap = 0
    for item in args.set:
        key, value = item.split("=", 1)
        current = getattr(config, key)
        setattr(config, key, type(current)(value))

    if args.detections:
        det_file = json.load(open(args.detections))
        sequences = args.sequences or ",".join(det_file["sequences"])
        sources = [{"kind": "sam3", "score": s, "iou": args.match_iou}
                   for s in args.score]
    else:
        det_file = None
        sequences = args.sequences or DEFAUT[args.split]
        sources = [{"kind": "gt", "drop": d, "jitter": j}
                   for d in args.drop for j in args.jitter]

    results = []
    cache_gmc = {}
    for seq in sequences.split(","):
        t0 = time.time()
        frames = load_sequence(args.uavdt, args.split, seq)
        for stride in args.stride:
            for source in sources:
                if source["kind"] == "sam3":
                    source = dict(source,
                                  frames=det_file["sequences"][seq]["frames"])
                kept, per_frame, truth, n_gt = build_inputs(
                    frames, stride, source, random.Random(args.seed))
                for name in args.tracker:
                    t1 = time.time()
                    if name == "maison":
                        tracks = track_maison(kept, per_frame, config,
                                              cache_gmc, (seq, stride))
                    else:
                        tracks = track_boxmot(name, kept, per_frame,
                                              30 / stride, args.device)
                    m = metrics(tracks, truth, n_gt, args.min_len)
                    label = {k: v for k, v in source.items()
                             if k != "frames"}
                    m.update(seq=seq, stride=stride, tracker=name,
                             source=label, seconds=time.time() - t1)
                    results.append(m)
                    what = (f"score>={source['score']}"
                            if source["kind"] == "sam3" else
                            f"drop={source['drop']:.2f} "
                            f"jit={source['jitter']:.2f}")
                    print(f"{seq} {name:10} fps={30 / stride:4.1f} {what}  "
                          f"rappel {m['recall_det']:.2f}  pistes "
                          f"{m['tracks']:4}/{m['gt_objects']:3} objets  "
                          f"purete {m['purity']:.3f}  morceaux/objet "
                          f"{m['pieces_per_object']:.2f}  FP en piste "
                          f"{100 * m['fp_in_tracks']:4.1f} %", flush=True)
        print(f"  ({time.time() - t0:.0f} s)", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"args": vars(args), "config": vars(config),
               "results": results}, open(args.out, "w"), indent=1)

    print("\n== Moyenne sur les sequences (ponderee par detections)")
    groups = collections.defaultdict(list)
    for r in results:
        groups[(r["tracker"], r["stride"],
                json.dumps(r["source"], sort_keys=True))].append(r)
    for (name, stride, source), rs in sorted(groups.items()):
        w = np.array([max(r["detections"], 1) for r in rs], float)
        avg = lambda k: float(np.average([r[k] for r in rs], weights=w))
        src = json.loads(source)
        what = (f"score>={src['score']}" if src["kind"] == "sam3" else
                f"drop={src['drop']:.2f} jit={src['jitter']:.2f}")
        print(f"  {name:10} fps={30 / stride:4.1f} {what:20} purete "
              f"{avg('purity'):.3f}  morceaux/objet "
              f"{avg('pieces_per_object'):.2f}  objets d'un seul tenant "
              f"{100 * avg('objects_in_one_piece'):4.1f} %  FP en piste "
              f"{100 * avg('fp_in_tracks'):4.1f} %  couverture "
              f"{avg('coverage'):.3f}  rappel det {avg('recall_det'):.2f}")


if __name__ == "__main__":
    main()
