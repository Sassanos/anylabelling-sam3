#!/usr/bin/env python3
"""Classe au VLM des objets annotes et garde les probabilites de chaque choix.

Ne juge rien, comme mesure-rappel-coco.py : ecrit, pour chaque decoupage, la
probabilite de chaque choix, pour que vlm-par-classe.py puisse tout recalculer
hors ligne (seuils, vote par piste, tranches de taille) sans relancer le GPU.

Deux sources :
- Campagne_1 (defaut) : annotations lues directement dans les zips, sans
  extraction. Les sequences sont des vols autour d'un meme vehicule : un
  Griffon vu 400 fois reste un seul Griffon. L'echantillonnage se fait donc
  par piste (sequence, label, group_id), au plus K frames espacees par piste.
- un jeu COCO (--coco) : chaque annotation est sa propre piste, et
  l'echantillon est equilibre par tranche de taille pour trouver ou
  l'exactitude s'effondre. Sert aux sous-classes civiles, absentes de
  Campagne_1.

A lancer avec le venv du serveur, serveur SAM 3 arrete (8,7 Gio de crete) :

    X-AnyLabeling-Server/.venv/bin/python mesure-vlm.py --out run.json
    X-AnyLabeling-Server/.venv/bin/python mesure-vlm.py \\
        --coco ../../Datasets/opensource/release/rgb/auair_rgb --out auair.json
    ./vlm-par-classe.py run.json auair.json
"""
import argparse
import collections
import io
import json
import os
import random
import sys
import time
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "X-AnyLabeling-Server"))

from app.utils.crop_classifier import (  # noqa: E402
    Choice,
    OpenAIBackend,
    TransformersBackend,
    build_prompt,
    crop_square,
)

CAMPAGNE = os.path.join(
    HERE, "..", "..", "Datasets", "real", "Campagne_1"
)

# Descriptions visuelles, pas seulement des noms : un 4B connait mal les
# materiels francais. Sources : fiches defense.gouv.fr et Wikipedia.
MILITAIRES = [
    Choice("griffon", "VBMR Griffon: large 6x6 wheeled armoured vehicle, "
           "three axles, tall angular box-shaped hull, often a remote "
           "weapon station on the roof"),
    Choice("vab", "VAB: 4x4 wheeled armoured personnel carrier, two axles, "
           "boat-shaped sloping front, flat armoured sides, smaller and "
           "lower than a Griffon"),
    Choice("gbc", "GBC 180: 6x6 military cargo truck, cab in front and a "
           "canvas-covered cargo bed"),
    Choice("vt4", "VT4: military 4x4 SUV based on the Ford Everest, closed "
           "five-door body, olive green or camouflage"),
    Choice("masstech", "Masstech T4: militarised Toyota Land Cruiser 70, "
           "boxy 4x4 station wagon or pickup with roll bars and military "
           "equipment"),
]
# v2 (2026-09-16) : dans les sequences B, le Griffon est bache sous un filet
# et la v1 l'envoyait vers le GBC (« canvas-covered cargo bed »). On decrit ce
# qui survit a une bache : essieux, cabine separee ou non, gabarit.
MILITAIRES_V2 = [
    Choice("griffon", "VBMR Griffon: large 6x6 armoured vehicle, three evenly "
           "spaced axles with big wheels, one-piece armoured hull with no "
           "separate cab, often covered by a camouflage net or tarpaulin "
           "draped over the whole hull"),
    Choice("vab", "VAB: 4x4 armoured personnel carrier, only two axles, "
           "boat-shaped sloping front, one-piece hull with no separate cab, "
           "shorter and lower than a Griffon"),
    Choice("gbc", "GBC 180: 6x6 military cargo truck with a distinct cab "
           "(windscreen, doors) in front and a separate cargo bed behind it "
           "under a canvas tarp; cab and bed are clearly separate"),
    Choice("vt4", "VT4: light military 4x4 SUV (Ford Everest), car-sized, "
           "much smaller than an armoured vehicle, closed five-door body with "
           "windows"),
    Choice("masstech", "Masstech T4: light militarised Toyota Land Cruiser, "
           "car-sized boxy 4x4 station wagon or pickup with roll bars, much "
           "smaller than an armoured vehicle"),
]
CIVILS = [
    Choice("voiture", "civilian passenger car (sedan, hatchback, estate or "
           "civilian SUV)"),
    Choice("pickup", "civilian pickup truck with an open cargo bed"),
    Choice("camionnette", "civilian van or minibus"),
    Choice("camion", "civilian truck, lorry, semi-trailer or tanker"),
    Choice("bus", "bus or coach"),
    Choice("deux_roues", "motorcycle, scooter or bicycle"),
    Choice("engin", "tractor, agricultural or construction machinery"),
]
AUTRES = [
    Choice("incertain", "a vehicle, but its type cannot be determined"),
    Choice("pas_vehicule", "not a vehicle (person, building, vegetation, "
           "shadow, road marking or anything else)"),
]
BINAIRE = [
    Choice("vehicule", "a vehicle (military or civilian, possibly covered by "
           "a net or tarpaulin)"),
    Choice("pas_vehicule", "not a vehicle"),
]
# Trois questions possibles :
# - complet : sous-classe et rejet dans la meme liste (le rejet y aspire la
#   moitie des vrais vehicules, mesure le 2026-09-16) ;
# - sous_classe : le detecteur a deja dit « vehicule », on ne demande que le
#   type ;
# - binaire : vehicule ou non, pour filtrer les faux positifs a part.
QUESTIONS = {
    "complet": MILITAIRES + CIVILS + AUTRES,
    "sous_classe": MILITAIRES + CIVILS,
    "binaire": BINAIRE,
}
DESCRIPTIONS = {"v1": MILITAIRES, "v2": MILITAIRES_V2}

# Casse et accents varient d'une sequence a l'autre dans les annotations.
NORMALISE = {
    "person": "Person",
    "Véhicule civil": "Véhicule Civil",
}


def box_of(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def collect(campagne, classes):
    """Liste les formes de tous les zips lisibles, sans doublon de sequence."""
    shapes = []
    seen_seq = {}
    for zpath in sorted(
        os.path.join(campagne, f) for f in os.listdir(campagne)
        if f.endswith(".zip")
    ):
        try:
            zf = zipfile.ZipFile(zpath)
        except zipfile.BadZipFile:
            print(f"ignore (illisible) : {os.path.basename(zpath)}",
                  file=sys.stderr)
            continue
        zname = os.path.basename(zpath)
        names = set(zf.namelist())
        for name in sorted(n for n in names if n.endswith(".json")):
            seq, frame = name.rsplit("/", 1)
            if seen_seq.setdefault(seq, zname) != zname:
                continue
            img = name[:-5] + ".jpg"
            if img not in names:
                continue
            data = json.loads(zf.read(name))
            w, h = data["imageWidth"], data["imageHeight"]
            ir = (w, h) == (640, 512) or "_T_" in seq or "_T[" in seq
            for i, s in enumerate(data["shapes"]):
                label = NORMALISE.get(s["label"], s["label"])
                if label not in classes or not s.get("points"):
                    continue
                gid = s.get("group_id")
                track = [seq, label, gid] if gid is not None \
                    else [seq, label, None, frame, i]
                box = box_of(s["points"])
                shapes.append({
                    "zip": zname, "seq": seq, "frame": frame, "image": img,
                    "label": label, "group_id": gid, "track": track,
                    "box": box, "modality": "IR" if ir else "RGB",
                    "width": w, "height": h,
                    "side_min": min(box[2] - box[0], box[3] - box[1]),
                    "side_max": max(box[2] - box[0], box[3] - box[1]),
                })
    return shapes


def collect_coco(dataset, split, classes):
    """Liste les annotations d'un jeu COCO, une piste par annotation."""
    info = json.load(open(os.path.join(dataset, "dataset.json")))
    root = os.path.join(dataset, info.get("image_root", "images"))
    coco = json.load(open(os.path.join(dataset, "annotations", f"{split}.json")))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    ir = info.get("modality", "RGB").upper() != "RGB"
    shapes = []
    for ann in coco["annotations"]:
        label = names[ann["category_id"]]
        if classes and label not in classes:
            continue
        im = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        shapes.append({
            "zip": None, "seq": os.path.basename(dataset.rstrip("/")),
            "frame": im["file_name"], "image": os.path.join(root, im["file_name"]),
            "label": label, "group_id": None, "track": [ann["id"]],
            "box": [x, y, x + w, y + h], "modality": "IR" if ir else "RGB",
            "width": im["width"], "height": im["height"],
            "side_min": min(w, h), "side_max": max(w, h),
        })
    return shapes


def sample_by_size(shapes, cap, rng):
    """Au plus cap decoupages par classe, repartis entre tranches de taille."""
    bins = [(0, 16), (16, 32), (32, 64), (64, 96), (96, 128), (128, 10 ** 6)]
    groups = collections.defaultdict(list)
    for s in shapes:
        b = next(i for i, (lo, hi) in enumerate(bins) if s["side_min"] < hi)
        groups[(s["label"], b)].append(s)
    picked = []
    for label in sorted({s["label"] for s in shapes}):
        cells = [groups[(label, b)] for b in range(len(bins))
                 if groups[(label, b)]]
        per_cell = max(1, cap // len(cells))
        for cell in cells:
            picked.extend(rng.sample(cell, min(per_cell, len(cell))))
    return picked


def sample(shapes, k, cap, rng):
    """Au plus k frames espacees par piste, au plus cap decoupages par classe."""
    by_class = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in shapes:
        by_class[s["label"]][json.dumps(s["track"])].append(s)

    picked = []
    for label, tracks in sorted(by_class.items()):
        keys = sorted(tracks)
        per_track = max(1, min(k, cap // max(1, len(keys))))
        if len(keys) * per_track > cap:
            keys = sorted(rng.sample(keys, cap // per_track))
        for key in keys:
            items = sorted(tracks[key], key=lambda s: s["frame"])
            if len(items) > per_track:
                idx = np.linspace(0, len(items) - 1, per_track).round()
                items = [items[int(i)] for i in idx]
            picked.extend(items)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campagne", default=CAMPAGNE)
    ap.add_argument("--coco", default=None,
                    help="jeu COCO (dossier avec dataset.json) au lieu de "
                         "Campagne_1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--prefixe", default=None,
                    help="ne garder que les sequences dont le nom commence "
                         "par ce prefixe (ex. B_)")
    ap.add_argument("--classes", default=None,
                    help="defaut : Griffon,VAB,GBC,Véhicule Civil,Person pour "
                         "Campagne_1, toutes pour un jeu COCO")
    ap.add_argument("--k", type=int, default=6,
                    help="frames par piste au plus")
    ap.add_argument("--cap", type=int, default=300,
                    help="decoupages par classe au plus")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--margin-min", type=float, default=16.0)
    ap.add_argument("--min-side", type=int, default=448)
    ap.add_argument("--max-side", type=int, default=896)
    ap.add_argument("--no-marker", action="store_true")
    ap.add_argument("--question", choices=sorted(QUESTIONS), default="complet")
    ap.add_argument("--descriptions", choices=sorted(DESCRIPTIONS),
                    default="v1", help="descriptions des classes militaires")
    ap.add_argument("--backend", choices=["transformers", "openai"],
                    default="transformers")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-crops", default=None,
                    help="dossier ou ecrire les decoupages, pour relecture")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.coco:
        classes = set(args.classes.split(",")) if args.classes else None
        shapes = collect_coco(args.coco, args.split, classes)
        items = sample_by_size(shapes, args.cap, rng)
    else:
        classes = set((args.classes or
                       "Griffon,VAB,GBC,Véhicule Civil,Person").split(","))
        shapes = collect(args.campagne, classes)
        if args.prefixe:
            shapes = [s for s in shapes if s["seq"].startswith(args.prefixe)]
        items = sample(shapes, args.k, args.cap, rng)
    counts = collections.Counter(s["label"] for s in items)
    tracks = collections.Counter(
        s["label"] for s in {json.dumps(s["track"]): s for s in items}.values()
    )
    print(f"{len(shapes)} formes, {len(items)} decoupages retenus")
    for label in sorted(counts):
        print(f"  {label:15} {counts[label]:4} decoupages, "
              f"{tracks[label]:3} pistes")

    marker = not args.no_marker
    choices = [DESCRIPTIONS[args.descriptions][MILITAIRES.index(c)]
               if c in MILITAIRES else c for c in QUESTIONS[args.question]]
    prompt = build_prompt(choices, marker=marker)

    t0 = time.time()
    if args.backend == "transformers":
        import torch
        backend = TransformersBackend(args.model, dtype=args.dtype)
        torch.cuda.reset_peak_memory_stats()
    else:
        torch = None
        backend = OpenAIBackend(args.api_base, model_name=args.model)
    load_s = time.time() - t0

    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)

    zips = {}
    by_image = collections.defaultdict(list)
    for s in items:
        by_image[(s["zip"], s["image"])].append(s)

    done = 0
    t_start = time.time()
    for (zname, img_name), group in by_image.items():
        if zname is None:
            source = img_name
        else:
            zf = zips.setdefault(
                zname, zipfile.ZipFile(os.path.join(args.campagne, zname))
            )
            source = io.BytesIO(zf.read(img_name))
        image = np.array(Image.open(source).convert("RGB"))
        for s in group:
            crop = crop_square(
                image, s["box"], margin_ratio=args.margin,
                margin_min_px=args.margin_min, min_side_px=args.min_side,
                max_side_px=args.max_side, marker=marker,
            )
            if torch is not None:
                torch.cuda.synchronize()
            t = time.time()
            probs, mass = backend.choice_probs(crop, prompt, len(choices))
            if torch is not None:
                torch.cuda.synchronize()
            s["ms"] = (time.time() - t) * 1000.0
            s["probs"] = [round(float(p), 5) for p in probs]
            s["letter_mass"] = round(float(mass), 4)
            if args.save_crops:
                top = choices[int(np.argmax(probs))].key
                crop.save(os.path.join(
                    args.save_crops,
                    f"{s['label']}_{done:04d}_{top}.jpg".replace(" ", "_"),
                ))
            done += 1
        if done % 50 < len(group):
            rate = done / (time.time() - t_start)
            print(f"  {done}/{len(items)}  {rate:.1f} decoupages/s",
                  file=sys.stderr)

    meta = {
        "args": vars(args),
        "choices": [[c.key, c.description] for c in choices],
        "prompt": prompt,
        "load_s": round(load_s, 1),
        "wall_s": round(time.time() - t_start, 1),
    }
    if torch is not None:
        meta["peak_vram_gib"] = round(
            torch.cuda.max_memory_allocated() / 2 ** 30, 2
        )
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "items": items}, f, ensure_ascii=False)
    print(f"ecrit {args.out} ({meta['wall_s']} s, "
          f"crete {meta.get('peak_vram_gib', '?')} Gio)")


if __name__ == "__main__":
    main()
