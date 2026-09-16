#!/usr/bin/env python3
"""Associe les `group_id` d'annotations X-AnyLabeling produites frame par frame.

Façade en ligne de commande sur
`anylabeling/views/labeling/utils/group_id_association.py`, le module que le
client utilise aussi derrière **Tool > Group ID Tracker**. Une seule
implémentation, deux points d'entrée.

Le flux complet :

    1. dans le client, ouvrir le dossier de frames ;
    2. choisir le modèle `segment_anything_3`, saisir le prompt texte
       (plusieurs classes separees par des virgules), puis « Auto Run » ;
    3. Tool > Group ID Tracker — ou ce script sur le meme dossier.

Exemples
--------
    ./associe-group-ids.py --dir mes_frames --dry-run
    ./associe-group-ids.py --dir mes_frames --backup mes_frames_avant
    ./associe-group-ids.py --dir sorties --images mes_frames --min-len 3
"""

import argparse
import os.path as osp
import sys

CLIENT = osp.join(osp.dirname(osp.abspath(__file__)), "X-AnyLabeling")
if CLIENT not in sys.path:
    sys.path.insert(0, CLIENT)

try:
    from anylabeling.views.labeling.utils.group_id_association import (
        TrackingConfig,
        associate_folder,
    )
except ImportError as e:
    sys.exit(
        f"Import impossible depuis {CLIENT} : {e}\n"
        "Lancer ce script avec un interpreteur disposant de numpy, scipy et "
        "opencv, par exemple X-AnyLabeling/.venv/bin/python."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir", required=True,
                    help="dossier des JSON X-AnyLabeling (un par frame)")
    ap.add_argument("--images", default=None,
                    help="dossier des images (defaut : --dir)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="ne traiter que ces labels (defaut : tous)")
    ap.add_argument("--no-gmc", action="store_true",
                    help="desactiver la compensation de mouvement camera")
    ap.add_argument("--gmc-scale", type=float, default=0.5)

    g = ap.add_argument_group("association")
    g.add_argument("--buffer1", type=float, default=0.3,
                   help="tampon IoU, 1re passe, en fraction de la taille")
    g.add_argument("--buffer2", type=float, default=1.0)
    g.add_argument("--min-iou", type=float, default=0.05)
    g.add_argument("--max-dist", type=float, default=2.5,
                   help="deplacement max du centroide, en diagonales d'objet")
    g.add_argument("--max-age", type=int, default=30,
                   help="frames sans detection avant de clore une piste")
    g.add_argument("--high-thresh", type=float, default=0.5)
    g.add_argument("--new-thresh", type=float, default=0.5)
    g.add_argument("--sigma-p", type=float, default=0.5)
    g.add_argument("--sigma-m", type=float, default=0.25)
    g.add_argument("--size-alpha", type=float, default=0.5)

    m = ap.add_argument_group("recollement et filtrage")
    m.add_argument("--merge-gap", type=int, default=30)
    m.add_argument("--merge-dist", type=float, default=3.0)
    m.add_argument("--merge-size-ratio", type=float, default=2.5)
    m.add_argument("--min-len", type=int, default=2)
    m.add_argument("--interpolate", action="store_true",
                   help="creer des boites dans les trous (flag interpolated)")

    o = ap.add_argument_group("sortie")
    o.add_argument("--dry-run", action="store_true")
    o.add_argument("--backup", default=None,
                   help="copier les JSON dans ce dossier avant reecriture")
    o.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    config = TrackingConfig(
        labels=args.labels or None,
        use_gmc=not args.no_gmc,
        gmc_scale=args.gmc_scale,
        buffer1=args.buffer1,
        buffer2=args.buffer2,
        min_iou=args.min_iou,
        max_dist=args.max_dist,
        max_age=args.max_age,
        high_thresh=args.high_thresh,
        new_thresh=args.new_thresh,
        sigma_p=args.sigma_p,
        sigma_m=args.sigma_m,
        size_alpha=args.size_alpha,
        merge_gap=args.merge_gap,
        merge_dist=args.merge_dist,
        merge_size_ratio=args.merge_size_ratio,
        min_len=args.min_len,
        interpolate=args.interpolate,
        dry_run=args.dry_run,
        backup_dir=args.backup,
    )

    last = {"stage": None}

    def progress(current, total, stage):
        if not args.quiet and stage != last["stage"]:
            print(f"[i] {stage}...", flush=True)
            last["stage"] = stage
        return True

    stats = associate_folder(args.dir, args.images, config, progress)

    if not stats.n_files:
        print(f"[!] aucun JSON exploitable dans {args.dir}", file=sys.stderr)
        return 1
    print(f"[i] {stats.n_files} fichiers | {stats.n_detections} detections | "
          f"labels : {', '.join(stats.labels) or 'aucun'}")
    if not stats.n_detections:
        return 0
    if stats.n_pairs:
        print(f"[i] mouvement camera estime sur {stats.n_gmc}/{stats.n_pairs} "
              "paires de frames")
    for label, (raw, merged, kept) in stats.per_label.items():
        print(f"[i] {label:20s} : {raw} pistes brutes -> {merged} apres "
              f"recollement -> {kept} retenues (>= {config.min_len} frames)")
    if stats.lengths:
        s = sorted(stats.lengths)
        print(f"    longueurs : min {s[0]}, mediane {s[len(s) // 2]}, "
              f"max {s[-1]}")
    print(f"[i] {stats.kept_tracks} objets distincts | "
          f"{stats.n_assigned} formes etiquetees"
          + (f" | {stats.n_interpolated} boites interpolees"
             if stats.n_interpolated else ""))
    if config.dry_run:
        print("[i] --dry-run : rien n'est ecrit")
    else:
        if config.backup_dir:
            print(f"[i] sauvegarde dans {config.backup_dir}")
        print(f"[i] {stats.n_written} fichiers reecrits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
