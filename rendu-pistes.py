#!/usr/bin/env python3
"""Vidéo de contrôle des pistes : une couleur et un numéro par group_id.

Décode le flux 0 en streaming (rien n'est extrait sur disque), dessine les
pistes d'un `<vol>_pistes_<tag>.jsonl` (associe-pistes.py) et encode un MP4
réduit, lisible dans n'importe quel lecteur (mpv, VLC : `.` / `,` avancent
d'une frame, ce qui suffit à repérer un changement d'objet dans une piste).

- boîte pleine : détection SAM 3 retenue dans la piste ; fine et pointillée :
  frame interpolée (objet non détecté, position déduite) ;
- étiquette : `group_id classe` (classe fine votée sur la piste) ;
- traîne : centres des dernières frames de la piste (repère image, non
  compensé du mouvement caméra) ;
- `--detections` ajoute en gris les détections restées hors de toute piste ;
- `--zone auto` recadre sur les pistes (objets de 20 px en 4K : illisibles
  dans l'image entière réduite en 1920) ;
- `--vlm p5` étiquette avec les réponses du VLM (verifie-pistes-vlm.py) :
  classe fine du VLM, pistes rejetées en gris (`--masquer-rejets` pour les
  cacher), `partie`, `impure` et `?` (objet incertain) en suffixe, `mil` pour
  une personne militaire.

Seules les frames annotées (RGB) sont rendues : la vidéo saute les rafales IR,
le bandeau donne le sample_index et le temps.

    X-AnyLabeling-Server/.venv/bin/python rendu-pistes.py --vol 0000011 \\
        --debut 21100 --fin 21700 --zone auto --detections
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pistes_io import (RACINE, charger_frames, chemins_vol,  # noqa: E402
                       decode_frames, dernier_jeu, en_tache_de_fond,
                       ticks_index)


def couleur(gid: int):
    """Teintes espacées par le nombre d'or : voisins de numéro bien distincts."""
    r, g, b = colorsys.hsv_to_rgb((gid * 0.6180339887) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def rectangle_pointille(img, p1, p2, coul, pas=6):
    import cv2
    (x1, y1), (x2, y2) = p1, p2
    for xa in range(x1, x2, 2 * pas):
        xb = min(xa + pas, x2)
        cv2.line(img, (xa, y1), (xb, y1), coul, 1)
        cv2.line(img, (xa, y2), (xb, y2), coul, 1)
    for ya in range(y1, y2, 2 * pas):
        yb = min(ya + pas, y2)
        cv2.line(img, (x1, ya), (x1, yb), coul, 1)
        cv2.line(img, (x2, ya), (x2, yb), coul, 1)


def texte(img, t, org, echelle, coul, epais=1):
    import cv2
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, echelle, (0, 0, 0),
                epais + 2, cv2.LINE_AA)
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, echelle, coul, epais,
                cv2.LINE_AA)


GRIS_REJET = (140, 140, 140)


def etiquette(gid, piste, reponse, avec_vlm):
    """Texte d'une piste : classe fine votée par SAM 3, ou verdict du VLM."""
    if not avec_vlm:
        return f"{gid} {piste['label']}"
    if reponse is None:
        return f"{gid} {piste['label']} (sans VLM)"
    if reponse["real_object"] == "no":
        return f"{gid} rejete"
    classe = reponse.get("fine_class") or "person"
    suffixes = []
    if piste["coarse"] == "person" and reponse.get("affiliation") == "military":
        suffixes.append("mil")
    if reponse["real_object"] == "unsure":
        suffixes.append("?")
    if reponse.get("box_covers") == "part":
        suffixes.append("partie")
    if reponse.get("same_object") == "no":
        suffixes.append("impure")
    return " ".join([str(gid), classe] + suffixes)


def ouvre_encodeur(chemin: Path, largeur, hauteur, debit: Fraction, qualite):
    import av
    sortie = av.open(str(chemin), "w")
    for nom, options in (("h264_nvenc", {"preset": "p5", "rc": "vbr",
                                         "cq": str(qualite)}),
                         ("libx264", {"preset": "veryfast",
                                      "crf": str(qualite)})):
        try:
            flux = sortie.add_stream(nom, rate=debit)
            flux.width, flux.height, flux.pix_fmt = largeur, hauteur, "yuv420p"
            flux.options = options
            flux.codec_context.open()
            return sortie, flux, nom
        except Exception:
            continue
    raise SystemExit("[!] ni h264_nvenc ni libx264 dans ce PyAV")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vol", required=True)
    ap.add_argument("--racine", type=Path, default=RACINE)
    ap.add_argument("--pistes", type=Path, default=None,
                    help="jeu de pistes (défaut : le plus récent du vol)")
    ap.add_argument("--debut", type=int, default=None, help="sample_index inclus")
    ap.add_argument("--fin", type=int, default=None, help="sample_index exclu")
    ap.add_argument("--pas", type=int, default=1,
                    help="une frame sur N (3 = survol en accéléré x3)")
    ap.add_argument("--largeur", type=int, default=1920)
    ap.add_argument("--zone", nargs="+", default=None,
                    metavar="auto | X1 Y1 X2 Y2",
                    help="recadrage en pixels source : 'auto' suit les pistes "
                         "de chaque frame (lissé), 4 nombres = zone fixe")
    ap.add_argument("--zoom-max", type=float, default=4.0,
                    help="--zone auto : fenêtre d'au moins largeur/zoom-max")
    ap.add_argument("--traine", type=int, default=20,
                    help="frames de traîne (0 = aucune)")
    ap.add_argument("--min-len", type=int, default=0,
                    help="n'afficher que les pistes d'au moins N détections")
    ap.add_argument("--classes", nargs="+", default=None)
    ap.add_argument("--detections", action="store_true",
                    help="détections hors piste en gris")
    ap.add_argument("--sans-etiquette", action="store_true")
    ap.add_argument("--qualite", type=int, default=24,
                    help="cq (NVENC) ou crf (x264) : plus bas = plus fidèle")
    ap.add_argument("--sortie", type=Path, default=None,
                    help="MP4 (défaut : <pistes>/rendus/<vol>_<tag>_<début>_<fin>.mp4)")
    ap.add_argument("--no-hwaccel", action="store_true")
    ap.add_argument("--vlm", default=None, metavar="PROMPT",
                    help="étiqueter avec les réponses du VLM (ex. p5)")
    ap.add_argument("--mode-vlm", default="rapide")
    ap.add_argument("--masquer-rejets", action="store_true",
                    help="avec --vlm : ne pas dessiner les pistes rejetées")
    args = ap.parse_args()

    import av
    import cv2

    chemins = chemins_vol(args.vol, args.racine)
    jeu = args.pistes or dernier_jeu(chemins["pistes"], args.vol)
    bilan = json.loads(jeu.with_suffix(".json").read_text())
    tag = bilan["tag"]
    reponses = {}
    if args.vlm:
        f = jeu.parent / "vlm" / f"{args.vol}_vlm_{args.mode_vlm}_{args.vlm}.jsonl"
        if not f.exists():
            raise SystemExit(f"[!] pas de réponses VLM : {f}")
        for ligne in open(f, encoding="utf-8"):
            r = json.loads(ligne)
            if r.get("reponse"):
                reponses[r["group_id"]] = r["reponse"]   # la dernière fait foi
        print(f"[i] {len(reponses)} pistes jugées par le VLM ({f.name})")

    # Frames à rendre : celles des tronçons continus annotés, dans la plage.
    samples = []
    for debut, fin, _n in bilan["segments"]:
        samples += range(debut, fin + 1)
    samples = [s for s in samples
               if (args.debut is None or s >= args.debut)
               and (args.fin is None or s < args.fin)][::args.pas]
    if not samples:
        raise SystemExit("[!] aucune frame annotée dans la plage demandée")
    lo, hi = samples[0], samples[-1]

    # Pistes -> {sample: [(gid, bbox, interpolée, piste)]}
    par_frame = defaultdict(list)
    centres = {}
    dans_piste = defaultdict(set)
    pistes = {}
    with open(jeu, encoding="utf-8") as fh:
        for ligne in fh:
            p = json.loads(ligne)
            if p["sample_max"] < lo or p["sample_min"] > hi:
                continue
            if p["n_detected"] < args.min_len or (
                    args.classes and p["coarse"] not in args.classes):
                continue
            if args.masquer_rejets and (reponses.get(p["group_id"]) or {}).get(
                    "real_object") == "no":
                continue
            gid = p["group_id"]
            pistes[gid] = p
            pts = {}
            for f in p["frames"]:
                b = f["bbox"]
                pts[f["i"]] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                if lo - args.traine <= f["i"] <= hi:
                    par_frame[f["i"]].append((gid, b, f.get("interpolated",
                                                            False)))
                for j in f.get("shapes", ()):
                    dans_piste[f["i"]].add(j)
            centres[gid] = pts

    hors_piste = {}
    if args.detections:
        frames, _ = charger_frames(chemins["annots"], args.vol, lo, hi + 1)
        for f in frames:
            hors_piste[f.sample] = [
                f.boites[j] for j in range(len(f.scores))
                if j not in dans_piste.get(f.sample, ())
                and (not args.classes or f.grossieres[j] in args.classes)]

    ticks = ticks_index(chemins["index"])
    manquants = [s for s in samples if s not in ticks]
    if manquants:
        raise SystemExit(f"[!] {len(manquants)} samples absents de l'index")

    suffixe = f"_vlm-{args.vlm}" if args.vlm else ""
    sortie = args.sortie or (jeu.parent / "rendus" /
                             f"{args.vol}_{tag}_{lo}_{hi + 1}{suffixe}.mp4")
    sortie.parent.mkdir(parents=True, exist_ok=True)

    zone_fixe = None
    if args.zone and args.zone != ["auto"]:
        if len(args.zone) != 4:
            raise SystemExit("[!] --zone auto, ou --zone X1 Y1 X2 Y2")
        zone_fixe = [float(v) for v in args.zone]
    lissee = None

    t0 = time.time()
    conteneur = flux = None
    n = 0
    for pos, image in en_tache_de_fond(
            decode_frames(chemins["video"], [ticks[s] for s in samples],
                          not args.no_hwaccel)):
        s = samples[pos]
        if image is None:
            continue
        h, w = image.shape[:2]
        if conteneur is None:
            hauteur = int(round(args.largeur * h / w / 2)) * 2
            conteneur, flux, codec = ouvre_encodeur(
                sortie, args.largeur, hauteur, Fraction(30000, 1001),
                args.qualite)

        # Fenêtre source (cx, cy, largeur) : image entière, zone fixe, ou zone
        # qui suit l'enveloppe des pistes de la frame, lissée.
        if zone_fixe:
            x1, y1, x2, y2 = zone_fixe
            fenetre = ((x1 + x2) / 2, (y1 + y2) / 2,
                       max(x2 - x1, (y2 - y1) * w / h))
        elif args.zone:
            boites = np.array([b for _g, b, _i in par_frame.get(s, ())])
            cible = lissee
            if len(boites):
                # Enveloppe robuste : une piste isolée dans un coin ne doit pas
                # dézoomer toute la vue.
                q = (10, 90) if len(boites) >= 5 else (0, 100)
                u = [np.percentile(boites[:, 0], q[0]),
                     np.percentile(boites[:, 1], q[0]),
                     np.percentile(boites[:, 2], q[1]),
                     np.percentile(boites[:, 3], q[1])]
                marge = 0.1 * max(u[2] - u[0], u[3] - u[1]) + 60
                cible = ((u[0] + u[2]) / 2, (u[1] + u[3]) / 2,
                         max(u[2] - u[0] + 2 * marge,
                             (u[3] - u[1] + 2 * marge) * w / h,
                             w / args.zoom_max))
            if cible is None:
                cible = (w / 2, h / 2, w)
            lissee = cible if lissee is None else tuple(
                a + 0.08 * (b - a) for a, b in zip(lissee, cible))
            fenetre = lissee
        else:
            fenetre = (w / 2, h / 2, w)
        zw = min(fenetre[2], w)
        zh = zw * h / w
        zx = min(max(fenetre[0] - zw / 2, 0), w - zw)
        zy = min(max(fenetre[1] - zh / 2, 0), h - zh)
        ech = args.largeur / zw
        img = cv2.resize(
            image[int(zy):int(zy + zh), int(zx):int(zx + zw)],
            (args.largeur, hauteur),
            interpolation=cv2.INTER_AREA if ech < 1 else cv2.INTER_LINEAR)

        def vers_sortie(x, y):
            return int((x - zx) * ech), int((y - zy) * ech)

        for b in hors_piste.get(s, ()):
            cv2.rectangle(img, vers_sortie(b[0], b[1]),
                          vers_sortie(b[2], b[3]), (150, 150, 150), 1)

        actives = defaultdict(int)
        rejetees = 0
        for gid, b, interp in par_frame.get(s, ()):
            p = pistes[gid]
            reponse = reponses.get(gid)
            rejet = reponse is not None and reponse["real_object"] == "no"
            rejetees += rejet
            coul = GRIS_REJET if rejet else couleur(gid)
            p1 = vers_sortie(b[0], b[1])
            p2 = vers_sortie(b[2], b[3])
            p2 = (max(p2[0], p1[0] + 2), max(p2[1], p1[1] + 2))
            if args.traine:
                pts = [vers_sortie(*centres[gid][k])
                       for k in range(s - args.traine, s + 1)
                       if k in centres[gid]]
                if len(pts) > 1:
                    cv2.polylines(img, [np.int32(pts)], False, coul, 1,
                                  cv2.LINE_AA)
            if interp:
                rectangle_pointille(img, p1, p2, coul)
            else:
                cv2.rectangle(img, p1, p2, coul, 1 if rejet else 2)
            if not args.sans_etiquette:
                texte(img, etiquette(gid, p, reponse, bool(args.vlm)),
                      (p1[0], p1[1] - 4), 0.42 if ech < 1 else 0.5, coul)
            actives[p["coarse"]] += 1

        # bandeau sur fond assombri : lisible sur une image claire (sable, ciel)
        img[:42] = (img[:42] * 0.35).astype(img.dtype)
        detail = " ".join(f"{c} {k}" for c, k in sorted(actives.items()))
        verdict = (f"  |  VLM {args.vlm} : {rejetees} rejetee(s)"
                   if args.vlm else "")
        texte(img, f"{args.vol}  sample {s}  |  {sum(actives.values())} "
                   f"pistes ({detail})  |  {tag}{verdict}", (12, 28), 0.7,
              (255, 255, 255), 2)

        cadre = av.VideoFrame.from_ndarray(img, format="bgr24")
        for paquet in flux.encode(cadre):
            conteneur.mux(paquet)
        n += 1
        if n % 500 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(samples)} frames | {n / el:.1f} frames/s",
                  flush=True)

    if conteneur is None:
        raise SystemExit("[!] aucune frame décodée")
    for paquet in flux.encode():
        conteneur.mux(paquet)
    conteneur.close()
    el = time.time() - t0
    print(f"[i] {n} frames ({codec}, {n / el:.1f} frames/s), "
          f"{len(pistes)} pistes -> {sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
