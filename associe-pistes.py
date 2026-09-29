#!/usr/bin/env python3
"""Pistes (group_id) d'un vol annoté par SAM 3 frame par frame : BoT-SORT hors ligne.

Lit les tronçons JSONL du lot Slurm (`annots/<vol>/`, tous terminés), écrit
une piste par ligne dans `<dossier_pistes(vol)>/<vol>_pistes_<tag>.jsonl`
(local) et le bilan dans le `.json` de même nom. Les détections ne sont jamais
modifiées.

Deux étapes, la première mise en cache :
1. indices (coûteuse, une fois par vol) : décodage streaming du flux 0 (NVDEC
   si disponible), compensation du mouvement caméra entre frames annotées
   successives, descripteur OSNet de chaque détection -> `indices/*.npz` ;
2. association (quelques dizaines de secondes, rejouable à volonté) :
   - les doublons inter-prompts sont fusionnés par classe grossière (SAM 3
     répond une fois par prompt : un même véhicule sort en `car` ET en `van`),
     la boîte au meilleur score représente la grappe ;
   - BoT-SORT (pistes_botsort.py) par classe grossière et par tronçon continu ;
   - classe fine de la piste = vote des scores sur toutes ses frames ;
   - trous internes interpolés (marqués `interpolated`).

Trous IR : un écart de plus de --coupure frames entre deux frames annotées
coupe le suivi (toutes les pistes s'arrêtent). Rien n'est recollé à travers
un trou IR : la caméra a bougé, la scène n'est plus la même.

    X-AnyLabeling-Server/.venv/bin/python associe-pistes.py --vol 0000011
    X-AnyLabeling-Server/.venv/bin/python associe-pistes.py --vol 0000011 \\
        --debut 19830 --fin 22330 --tag essai      # zone dense seule
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pistes_botsort import (BotSort, ParamsBotSort,  # noqa: E402
                            gmc_flot_optique, iou_xyxy)
from pistes_io import (RACINE, charger_frames, chemins_vol,  # noqa: E402
                       decode_frames, en_tache_de_fond)

VERSION_INDICES = 1

# Les seuils de BoT-SORT (0,5 / 0,6) supposent un détecteur aux scores de type
# YOLOX. SAM 3 note les personnes entre 0,25 et 0,6 : aucune piste n'y
# naîtrait. Mesuré sur UAVDT (vraies détections SAM 3, score >= 0,25) :
# pureté 0,986 dans les deux cas, 1,03 -> 1,06 morceau par objet, couverture
# 0,84 -> 0,96 en passant à 0,3 / 0,3.
SEUILS_SAM3 = {"track_high_thresh": 0.3, "new_track_thresh": 0.3}


# ---------------------------------------------------------------------------
# Étape 1 : indices (GMC + ReID), mis en cache
# ---------------------------------------------------------------------------


def signature_detections(frames) -> str:
    """Empreinte du contenu, insensible à la compression des tronçons."""
    h = hashlib.sha1()
    for f in frames:
        h.update(np.array([f.sample, f.tick, len(f.scores)], np.int64)
                 .tobytes())
        h.update(np.round(f.boites).astype(np.int32).tobytes())
    return h.hexdigest()[:16]


def calcule_indices(frames, video, encodeur, gmc_largeur, hwaccel,
                    gmc_ecart_max):
    import cv2
    n = len(frames)
    offsets = np.zeros(n + 1, np.int64)
    offsets[1:] = np.cumsum([len(f.scores) for f in frames])
    emb = np.zeros((int(offsets[-1]), encodeur.dim), np.float16)
    gmc = np.full((n, 2, 3), np.nan, np.float32)
    lot_decoupes, lot_lignes, taille_lot = [], [], 0
    manquantes = 0

    def vide_lot():
        nonlocal taille_lot
        if lot_lignes:
            d = encodeur.descripteurs(np.concatenate(lot_decoupes))
            emb[np.concatenate(lot_lignes)] = d.astype(np.float16)
            lot_decoupes.clear()
            lot_lignes.clear()
            taille_lot = 0

    # Les frames en erreur côté client (mur de corruption en fin de 0000004)
    # n'ont ni détection ni image : ne pas les redemander au décodeur.
    lisibles = [p for p, f in enumerate(frames) if not f.erreur]
    prec_gris, prec_sample = None, None
    t0 = dernier = time.time()
    for k, image in en_tache_de_fond(
            decode_frames(video, [frames[p].tick for p in lisibles], hwaccel)):
        pos = lisibles[k]
        f = frames[pos]
        if image is None:
            manquantes += 1
            prec_gris = None
            continue
        h, w = image.shape[:2]
        echelle = gmc_largeur / w
        gris = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                          (gmc_largeur, round(h * echelle)),
                          interpolation=cv2.INTER_AREA)
        if prec_gris is not None and f.sample - prec_sample <= gmc_ecart_max:
            m = gmc_flot_optique(prec_gris, gris, echelle)
            if m is not None:
                gmc[pos] = m
        prec_gris, prec_sample = gris, f.sample
        if len(f.scores):
            lot_decoupes.append(encodeur.decoupes(image, f.boites))
            lot_lignes.append(np.arange(offsets[pos], offsets[pos + 1]))
            taille_lot += len(f.scores)
            if taille_lot >= 1024:
                vide_lot()
        maintenant = time.time()
        if maintenant - dernier > 30:
            fait = pos + 1
            vitesse = fait / (maintenant - t0)
            print(f"  indices {fait}/{n} frames | {vitesse:.1f} frames/s | "
                  f"reste ~{(n - fait) / vitesse / 60:.1f} min", flush=True)
            dernier = maintenant
    vide_lot()
    return offsets, emb, gmc, manquantes


def indices_en_cache(frames, chemins, args):
    signature = signature_detections(frames)
    meta_attendue = {
        "version": VERSION_INDICES, "signature": signature,
        "reid_poids": Path(args.reid_poids).name,
        "gmc_largeur": args.gmc_largeur, "gmc_ecart_max": args.gmc_ecart_max,
    }
    dossier = Path(args.indices) if args.indices else chemins["pistes"] / "indices"
    plage = "" if args.debut is None and args.fin is None else \
        f"_{args.debut or 0}_{args.fin or 'fin'}"
    base = dossier / f"{args.vol}_indices{plage}"
    npz, meta_json = base.with_suffix(".npz"), base.with_suffix(".json")
    if npz.exists() and meta_json.exists() and not args.refaire_indices:
        meta = json.loads(meta_json.read_text())
        if all(meta.get(k) == v for k, v in meta_attendue.items()):
            print(f"[i] indices en cache : {npz}")
            d = np.load(npz)
            return d["offsets"], d["emb"], d["gmc"], meta
        print(f"[i] cache {npz.name} périmé, recalcul")

    from reid_osnet import EncodeurOSNet
    encodeur = EncodeurOSNet(args.reid_poids, device=args.device)
    print(f"[i] indices : {len(frames)} frames, "
          f"{sum(len(f.scores) for f in frames)} détections, "
          f"vidéo {chemins['video'].name}", flush=True)
    t0 = time.time()
    offsets, emb, gmc, manquantes = calcule_indices(
        frames, chemins["video"], encodeur, args.gmc_largeur,
        not args.no_hwaccel, args.gmc_ecart_max)
    duree = time.time() - t0
    meta = dict(meta_attendue, frames=len(frames), detections=int(offsets[-1]),
                frames_manquantes=manquantes,
                gmc_estimees=int(np.isfinite(gmc[:, 0, 0]).sum()),
                duree_s=round(duree, 1), date=datetime.now().isoformat(
                    timespec="seconds"))
    dossier.mkdir(parents=True, exist_ok=True)
    tmp = base.with_name(base.name + ".tmp.npz")
    np.savez(tmp, offsets=offsets, emb=emb, gmc=gmc)
    os.replace(tmp, npz)
    meta_json.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[i] indices calculés en {duree / 60:.1f} min "
          f"({len(frames) / duree:.1f} frames/s), {manquantes} frame(s) "
          f"introuvable(s) dans le flux -> {npz}", flush=True)
    return offsets, emb, gmc, meta


# ---------------------------------------------------------------------------
# Étape 2 : association
# ---------------------------------------------------------------------------


def regroupe(boites, scores, seuil):
    """Grappes de doublons (NMS glouton sans classe fine), tête en premier."""
    n = len(scores)
    if n == 0:
        return []
    ordre = np.argsort(-scores, kind="stable")
    iou = iou_xyxy(boites.astype(float), boites.astype(float))
    libre = np.ones(n, bool)
    grappes = []
    for i in ordre:
        if not libre[i]:
            continue
        membres = ordre[libre[ordre] & (iou[i, ordre] >= seuil)]
        libre[membres] = False
        grappes.append(membres)
    return grappes


def segments_continus(frames, coupure):
    segments, courant = [], []
    for pos, f in enumerate(frames):
        if f.erreur:
            continue
        if courant and f.sample - frames[courant[-1]].sample > coupure:
            segments.append(courant)
            courant = []
        courant.append(pos)
    if courant:
        segments.append(courant)
    return segments


def associe(frames, offsets, emb, gmc, params, nms_iou, coupure, classes):
    segments = segments_continus(frames, coupure)
    ids = [0]
    pistes = []
    compte = {"detections": {}, "grappes": {}}
    for seg in segments:
        trackers = {c: BotSort(params, ids) for c in classes}
        for k, pos in enumerate(seg):
            f = frames[pos]
            matrice = None
            if k > 0 and np.isfinite(gmc[pos]).all():
                matrice = gmc[pos].astype(float)
            par_classe = {}
            for j, c in enumerate(f.grossieres):
                par_classe.setdefault(c, []).append(j)
            for c, tracker in trackers.items():
                js = np.array(par_classe.get(c, []), int)
                grappes = [js[g] for g in regroupe(f.boites[js], f.scores[js],
                                                   nms_iou)] if len(js) else []
                compte["detections"][c] = compte["detections"].get(c, 0) + len(js)
                compte["grappes"][c] = compte["grappes"].get(c, 0) + len(grappes)
                tetes = np.array([g[0] for g in grappes], int)
                descs = (emb[offsets[pos] + tetes].astype(np.float32)
                         if len(tetes) else None)
                tracker.update(f.sample, f.boites[tetes], f.scores[tetes],
                               descs, matrice, [(pos, g) for g in grappes])
        for c, tracker in trackers.items():
            pistes += [(c, t) for t in tracker.pistes()]
    return pistes, segments, compte


def ligne_piste(frames, grossiere, piste, interpole):
    votes = {}
    sorties, tailles, scores = [], [], []
    precedent = None
    for sample, (pos, grappe), boite, score in piste.membres:
        f = frames[pos]
        for j in grappe:
            votes[f.labels[j]] = votes.get(f.labels[j], 0.0) + float(f.scores[j])
        if interpole and precedent is not None and sample - precedent[0] > 1:
            s0, b0 = precedent
            for s in range(s0 + 1, sample):
                a = (s - s0) / (sample - s0)
                b = (1 - a) * b0 + a * boite
                sorties.append({"i": s, "bbox": [round(float(v), 1) for v in b],
                                "interpolated": True})
        sorties.append({"i": sample,
                        "bbox": [round(float(v), 1) for v in boite],
                        "score": round(score, 3),
                        "label": f.labels[grappe[0]],
                        "shapes": [int(j) for j in grappe]})
        tailles.append(float(max(boite[2] - boite[0], boite[3] - boite[1])))
        scores.append(score)
        precedent = (sample, boite)
    total = sum(votes.values())
    votes = dict(sorted(((k, round(v / total, 3)) for k, v in votes.items()),
                        key=lambda kv: -kv[1]))
    premier, dernier = piste.membres[0], piste.membres[-1]
    return {
        "coarse": grossiere,
        "label": next(iter(votes)),
        "labels": votes,
        "n_frames": len(sorties),
        "n_detected": len(piste.membres),
        "sample_min": int(premier[0]),
        "sample_max": int(dernier[0]),
        "t_min_s": round(frames[premier[1][0]].t, 3),
        "t_max_s": round(frames[dernier[1][0]].t, 3),
        "score_mean": round(float(np.mean(scores)), 3),
        "size_median_px": round(float(np.median(tailles)), 1),
        "frames": sorties,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vol", required=True)
    ap.add_argument("--racine", type=Path, default=RACINE)
    ap.add_argument("--sortie", type=Path, default=None,
                    help="défaut : dossier_pistes(vol), AnafiUKR/pistes/<vol>/ ou "
                    "CAMPAGNE3/pistes/<vol>/")
    ap.add_argument("--indices", default=None,
                    help="dossier du cache d'indices (défaut : <sortie>/indices)")
    ap.add_argument("--tag", default=None,
                    help="nom du jeu de pistes (défaut : botsort-<empreinte "
                         "des réglages>)")
    ap.add_argument("--debut", type=int, default=None,
                    help="premier sample_index (inclus), pour un essai")
    ap.add_argument("--fin", type=int, default=None,
                    help="dernier sample_index (exclu)")
    ap.add_argument("--partiel", action="store_true",
                    help="ignorer les tronçons sans .done au lieu de refuser")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="classes grossières à suivre (défaut : toutes)")
    ap.add_argument("--nms-iou", type=float, default=0.6,
                    help="IoU à partir de laquelle deux boîtes de même classe "
                         "grossière sont le même objet (doublons de prompts)")
    ap.add_argument("--coupure", type=int, default=5,
                    help="écart (frames) au-delà duquel le suivi repart à zéro")
    ap.add_argument("--min-len", type=int, default=5,
                    help="détections minimales pour garder une piste")
    ap.add_argument("--no-interpolate", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="CLE=VAL",
                    help="surcharge un champ de ParamsBotSort (répétable)")
    ap.add_argument("--reid-poids",
                    default=str(HERE / "poids" / "osnet_x0_25_msmt17.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gmc-largeur", type=int, default=960,
                    help="largeur des images de la compensation de mouvement")
    ap.add_argument("--gmc-ecart-max", type=int, default=30,
                    help="écart max (frames) pour estimer le mouvement caméra")
    ap.add_argument("--no-hwaccel", action="store_true",
                    help="décodage logiciel au lieu de NVDEC")
    ap.add_argument("--refaire-indices", action="store_true")
    args = ap.parse_args()

    params = ParamsBotSort(**SEUILS_SAM3)
    for item in args.set:
        cle, valeur = item.split("=", 1)
        courant = getattr(params, cle)
        setattr(params, cle, valeur.lower() in ("1", "true", "oui")
                if isinstance(courant, bool) else type(courant)(valeur))
    reglages = {"botsort": vars(params), "nms_iou": args.nms_iou,
                "coupure": args.coupure, "min_len": args.min_len,
                "interpolate": not args.no_interpolate,
                "classes": args.classes}
    empreinte = hashlib.sha1(json.dumps(reglages, sort_keys=True).encode()
                             ).hexdigest()[:6]
    tag = args.tag or f"botsort-{empreinte}"

    chemins = chemins_vol(args.vol, args.racine)
    if args.sortie:
        chemins["pistes"] = args.sortie
    t0 = time.time()
    frames, sources = charger_frames(chemins["annots"], args.vol, args.debut,
                                     args.fin, args.partiel)
    n_err = sum(1 for f in frames if f.erreur)
    print(f"[i] {len(frames)} frames annotées ({n_err} en erreur), "
          f"{sum(len(f.scores) for f in frames)} détections, "
          f"{len(sources)} tronçon(s), lus en {time.time() - t0:.0f} s",
          flush=True)
    if not frames:
        return 2

    offsets, emb, gmc, meta_indices = indices_en_cache(frames, chemins, args)

    t1 = time.time()
    classes = args.classes or sorted({c for f in frames for c in f.grossieres})
    pistes, segments, compte = associe(frames, offsets, emb, gmc, params,
                                       args.nms_iou, args.coupure, classes)
    lignes = [ligne_piste(frames, c, t, not args.no_interpolate)
              for c, t in pistes if len(t.membres) >= args.min_len]
    lignes.sort(key=lambda l: (l["sample_min"], l["coarse"],
                               l["frames"][0]["bbox"][0],
                               l["frames"][0]["bbox"][1]))
    for gid, ligne in enumerate(lignes, start=1):
        ligne["group_id"] = gid
    duree_assoc = time.time() - t1

    # Bilan
    longueurs = np.array([l["n_detected"] for l in lignes]) if lignes else \
        np.zeros(1)
    par_classe = {}
    for c in classes:
        mes = [l for l in lignes if l["coarse"] == c]
        dans = sum(l["n_detected"] for l in mes)
        par_classe[c] = {
            "detections": compte["detections"].get(c, 0),
            "apres_fusion_doublons": compte["grappes"].get(c, 0),
            "pistes": len(mes),
            "couverture": round(dans / max(compte["grappes"].get(c, 0), 1), 3),
            "labels": dict(sorted(
                {lab: sum(1 for l in mes if l["label"] == lab)
                 for lab in {l["label"] for l in mes}}.items(),
                key=lambda kv: -kv[1])),
        }
    bilan = {
        "vol": args.vol, "tag": tag, "empreinte": empreinte,
        "date": datetime.now().isoformat(timespec="seconds"),
        "reglages": reglages, "sources": sources, "video": str(chemins["video"]),
        "indices": meta_indices,
        "frames": len(frames), "frames_erreur": n_err,
        "segments": [[frames[s[0]].sample, frames[s[-1]].sample, len(s)]
                     for s in segments],
        "pistes": len(lignes),
        "par_classe": par_classe,
        "longueur_detections": {
            f"p{q}": float(np.percentile(longueurs, q))
            for q in (10, 25, 50, 75, 90, 99)} | {"max": int(longueurs.max())},
        "pistes_plus_de_1s": int(sum(1 for l in lignes
                                     if l["sample_max"] - l["sample_min"] >= 30)),
        "duree_association_s": round(duree_assoc, 1),
    }

    chemins["pistes"].mkdir(parents=True, exist_ok=True)
    base = chemins["pistes"] / f"{args.vol}_pistes_{tag}"
    tmp = base.with_name(base.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for ligne in lignes:
            fh.write(json.dumps({"group_id": ligne.pop("group_id"), **ligne},
                                ensure_ascii=False) + "\n")
    os.replace(tmp, base.with_suffix(".jsonl"))
    base.with_suffix(".json").write_text(
        json.dumps(bilan, indent=1, ensure_ascii=False) + "\n")

    print(f"[i] association en {duree_assoc:.0f} s : {len(segments)} tronçon(s) "
          f"continu(s), {len(lignes)} pistes >= {args.min_len} détections "
          f"({bilan['pistes_plus_de_1s']} de plus d'une seconde)")
    for c, d in par_classe.items():
        print(f"    {c:8} {d['detections']:7} détections -> "
              f"{d['apres_fusion_doublons']:7} après fusion des doublons -> "
              f"{d['pistes']:5} pistes, couverture {d['couverture']:.2f}, "
              f"labels {d['labels']}")
    lg = bilan["longueur_detections"]
    print(f"    longueur (détections) : médiane {lg['p50']:.0f}, p90 "
          f"{lg['p90']:.0f}, max {lg['max']}")
    print(f"[i] -> {base.with_suffix('.jsonl')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
