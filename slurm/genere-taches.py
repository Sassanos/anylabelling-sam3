#!/usr/bin/env python3
"""Découpe les vols annotables en tronçons de frames et écrit tasks.json.

Chaque tâche = un tronçon d'un vol : plage de sample_index du flux DefaultVideo,
ne contenant que des frames valides en mode RGB (le DefaultVideo est un rendu
d'affichage gelé pendant les rafales thermiques — on ne l'annote pas).

tasks.json alimente le job array Slurm : la tâche N traite l'entrée N du
fichier. Les tronçons sont idempotents (marqueur .done côté client), un
relancement ne refait que ce qui manque.

  genere-taches.py --flights-root .../campagne3_index --out .../slurm/tasks.json

--stream visible découpe à la place la piste VisibleVideo (RGB 4K des rafales
thermiques, 8,57 fps, frames valid=1) ; les tâches portent "stream": "visible",
que le client lit. À soumettre avec un ANNOTS_OUT_ROOT distinct.
--stream thermal fait de même pour la piste thermique brute ; --tile-size
inscrit le côté de tuile dans les tâches (320 pour le thermique).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flights-root", required=True,
                    help="racine du miroir (contenant CodeAnnotationPrep, annotation/, <date>/...)")
    ap.add_argument("--out", required=True, help="tasks.json de sortie")
    ap.add_argument("--chunk-size", type=int, default=2500,
                    help="frames valides par tronçon (défaut 2500 : ~12 Mo de JSONL, "
                         "limite le nombre de fichiers sur les espaces quota en inodes)")
    ap.add_argument("--flights", nargs="*", help="limiter à ces vols (ex. 0000011)")
    ap.add_argument("--stream", choices=("default", "visible", "thermal"), default="default",
                    help="flux à découper (visible = RGB des rafales thermiques, "
                         "thermal = piste thermique brute)")
    ap.add_argument("--tile-size", type=int, default=None,
                    help="côté de tuile inscrit dans les tâches (défaut : celui du sbatch)")
    args = ap.parse_args()

    root = Path(args.flights_root).resolve()
    sys.path.insert(0, str(root / "CodeAnnotationPrep"))
    import anafi_common  # noqa: E402

    flights = anafi_common.discover_flights(root)
    if args.flights:
        flights = [f for f in flights if f.flight_id in set(args.flights)]
    if not flights:
        print("[!] aucun vol découvert", file=sys.stderr)
        return 2

    tasks = []
    for fl in flights:
        index_csv = fl.index_csv(args.stream)
        if not index_csv.exists():
            print(f"[{fl.flight_id}] PAS d'index ({index_csv}), ignoré", file=sys.stderr)
            continue
        with open(index_csv, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if r.get("valid") == "1" and (args.stream != "default"
                                                  or r.get("mode") == "RGB")]
        rows.sort(key=lambda r: int(r["sample_index"]))
        if not rows:
            print(f"[{fl.flight_id}] aucune frame {args.stream} valide, ignoré")
            continue
        n_chunks = 0
        for k in range(0, len(rows), args.chunk_size):
            block = rows[k:k + args.chunk_size]
            tasks.append({
                "flight": fl.flight_id,
                "video": str(fl.original),
                "index": str(index_csv),
                "start": int(block[0]["sample_index"]),
                "end": int(block[-1]["sample_index"]) + 1,
                "n_frames": len(block),
            })
            if args.stream != "default":
                tasks[-1]["stream"] = args.stream
            if args.tile_size is not None:
                tasks[-1]["tile_size"] = args.tile_size
            n_chunks += 1
        print(f"[{fl.flight_id}] {len(rows)} frames {args.stream} valides -> {n_chunks} tronçons")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(tasks, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out)
    total = sum(t["n_frames"] for t in tasks)
    print(f"[i] {len(tasks)} tâches, {total} frames -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
