#!/usr/bin/env python3
"""Jeu de détection 3 classes (person, civilian_vehicle, military_vehicle)
à partir des pistes SAM 3 + BoT-SORT et des verdicts du VLM.

Les verdicts sont par piste (p5 : verifie-pistes-vlm.py --prompt p5 ; m1 :
2e passe sur les candidats militaires) ; ils sont propagés à chaque frame de
la piste. Chaque boîte reçoit un niveau :

- garder  : objet sûr, classe fixée ;
- ignorer : objet douteux (partie sans véhicule entier, VLM incertain,
            affiliation inconnue, frame interpolée, détection SAM 3 hors
            piste...) ; iscrowd=1 en COCO, zone grisée dans l'image YOLO ;
- retirer : rejet net (faux positif, partie contenue dans la boîte d'un
            véhicule entier) ; rien n'est écrit en COCO,
            la boîte n'est comptée que dans le bilan.

Frames : une par seconde, plus des frames bonus (une par tiers de seconde)
là où une boîte gardée est militaire ou une personne. Les frames sont
décodées de la vidéo (NVDEC) et écrites en JPEG 4K intactes ; rien n'est
écrit ni supprimé sous /media.

    PY=X-AnyLabeling-Server/.venv/bin/python
    $PY exporte-detection.py seuils              # règles contre la revue humaine
    $PY exporte-detection.py vol 0000001         # frames + par-vol/0000001.json
    $PY exporte-detection.py controle 0000001    # controle.html
    $PY exporte-detection.py assemble            # annotations/{train,val,test}.json
    $PY exporte-detection.py yolo                # tuiles 1024 (yolo/)
"""
from __future__ import annotations

import argparse
import collections
import html
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pistes_io import (VOLS_JOUR, charger_frames, chemins_vol,  # noqa: E402
                       decode_frames, dernier_jeu, en_tache_de_fond,
                       telemetrie_index)


def _module(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, HERE / fichier)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


vlm = _module("verifie_pistes_vlm", "verifie-pistes-vlm.py")

SORTIE = Path("/home/cbarbier/Documents/Geolocalisation/Datasets/real/CAMPAGNE3"
              "/detection-p5")
CATEGORIES = ["person", "civilian_vehicle", "military_vehicle"]
CIVILS = {"car", "van", "truck", "bus", "motorcycle", "bicycle", "agri_vehicle"}
# Splits par vol, jamais par image. 0000018 : vol test du jeu AnafiUKR annoté
# à la main (Datasets/real/AnafiUKR/annotation/splits.json).
SPLITS = {"0000018": "test", "0000005": "val"}
REVUE_MILITAIRES = Path("/home/cbarbier/Documents/Geolocalisation/Datasets/real"
                        "/CAMPAGNE3/revue-militaires/annotations.jsonl")
TELEMETRIE = ("rel_alt_m", "agl_m", "hfov_deg", "vfov_deg", "cam_pitch_deg",
              "lat", "lon")

REGLES = {
    # Les P du VLM valent presque toujours 0 ou 1 : ces seuils changent peu de
    # choses (revue humaine, `seuils`).
    "p_rejet": 0.5,          # P(real_object=no) >= : retirer
    "p_objet": 0.5,          # P(real_object=yes) < : ignorer (incertain)
    "p_partie": 0.5,         # P(box_covers=part) >= : partie (véhicules seulement)
    # same_object n'est pas utilisé : sur la revue, il écarterait 7 pistes pures
    # pour 2 impures, et une piste impure garde des boîtes justes frame à frame.
    "m1_equipement": 0.5,    # P(towed_equipment) + P(other) >= : ignorer
    "m1_vehicule": 0.5,      # P(vehicle) >= et military != no : militaire
    "contenance": vlm.CONTENANCE,   # part d'aire 0,8, rapport d'aire 1,5
    "iou_shape_piste": 0.5,  # shape SAM 3 recouverte par une boîte de piste
    "score_hors_piste": 0.3,  # shapes hors piste plus faibles : ignorées du tout
    "interpolees": "ignorer",   # ou "garder"
    # Recollage des véhicules coupés aux coutures des tuiles de SAM 3 tuilé
    # (tuiles 1008, recouvrement 0,2 : .done des tronçons, configs/auto_labeling/
    # segment_anything_3_tiled.yaml). La fusion SAHI garde la boîte au
    # meilleur score, souvent le morceau d'une tuile plutôt que le véhicule
    # entier de la passe pleine image : sur 0000001, 1 715 des 2 135 parties
    # contenues et 13 % des boîtes gardées étaient coupées à une couture.
    "recoller": True,
    "tuile_sam3": 1008,
    "recouvrement_sam3": 0.2,
    "tol_couture": 8,          # px entre un bord de boîte et la couture
    "iou_autre_axe": 0.5,      # recouvrement des deux morceaux sur l'autre axe
    "pas_s": 1.0,
    "pas_bonus_s": 1 / 3,
}


# --------------------------------------------------------------------------
# Verdict d'une piste

def lire_verdicts(chemin):
    """{group_id: ligne} ; la dernière ligne sans erreur fait foi."""
    d = {}
    if chemin.exists():
        for l in open(chemin, encoding="utf-8"):
            r = json.loads(l)
            if r.get("reponse") and not r.get("erreur"):
                d[r["group_id"]] = r
    return d


def _p(v, champ, option):
    d = (v.get("p") or {}).get(champ)
    return None if not d else d["p"].get(option, 0.0)


def verdict_piste(coarse, v5, vm1, humain=None, regles=REGLES):
    """(niveau, catégorie ou groupe d'ignorance, raison, infos).

    niveau : garder | ignorer | retirer | partie (décidé frame par frame :
    retirée si contenue dans un véhicule entier, ignorée sinon).
    humain : verdict de revue-militaires.py (militaire, civil, non,
    incertain), qui remplace p5/m1 pour la famille du véhicule.
    """
    infos = {}
    if humain == "non":
        return "retirer", coarse, "humain_non", {"humain": humain}
    if v5 is None and humain is None:
        return "ignorer", coarse, "sans_verdict", infos
    if v5 is None:   # relue à la main sans verdict p5
        v5 = {"reponse": {}, "p": {}}
    rep = v5["reponse"]
    p_non, p_oui = _p(v5, "real_object", "no"), _p(v5, "real_object", "yes")
    if p_non is None:   # logprobs illisibles : on prend la réponse
        p_non = 1.0 if rep.get("real_object") == "no" else 0.0
        p_oui = 1.0 if rep.get("real_object") == "yes" else 0.0
    infos.update(p_objet=round(p_oui, 3), fine_class=rep.get("fine_class"))
    if humain in ("militaire", "civil", "incertain") and coarse == "vehicle":
        p_non, p_oui = 0.0, 1.0   # la revue humaine dit « véhicule »
    if p_non >= regles["p_rejet"]:
        return "retirer", coarse, "rejet_vlm", infos
    if p_oui < regles["p_objet"]:
        return "ignorer", coarse, "objet_incertain", infos
    if coarse == "person":
        return "garder", "person", "vlm", infos
    p_partie = _p(v5, "box_covers", "part")
    if p_partie is None:
        p_partie = 1.0 if rep.get("box_covers") == "part" else 0.0
    infos["p_partie"] = round(p_partie, 3)
    if humain:
        infos["humain"] = humain
    if p_partie >= regles["p_partie"]:
        return "partie", "vehicle", "partie", infos
    if humain == "militaire":
        return "garder", "military_vehicle", "humain", infos
    if humain == "civil":
        return "garder", "civilian_vehicle", "humain", infos
    if humain == "incertain":
        return "ignorer", "vehicle", "humain_incertain", infos
    fine = rep.get("fine_class")
    if fine in vlm.MILITAIRES:
        if vm1 is None:
            return "ignorer", "vehicle", "militaire_non_verifie", infos
        r1 = vm1["reponse"]
        genre = {g: _p(vm1, "kind", g) for g in vlm.GENRES_M1}
        if genre["vehicle"] is None:
            genre = {g: float(r1.get("kind") == g) for g in vlm.GENRES_M1}
        p_mil_non = _p(vm1, "military", "no")
        if p_mil_non is None:
            p_mil_non = float(r1.get("military") == "no")
        infos.update(m1_kind=r1.get("kind"), m1_military=r1.get("military"),
                     p_vehicule=round(genre["vehicle"], 3))
        # p5 dit véhicule, m1 dit équipement : conflit, donc ignoré et non
        # retiré (revue : 5/5 groupes électrogènes, mais aussi le robot UGV et
        # un VT4 vu de dessus pris pour des équipements).
        if genre["towed_equipment"] + genre["other"] >= regles["m1_equipement"]:
            return "ignorer", "vehicle", "equipement_m1", infos
        if genre["vehicle"] >= regles["m1_vehicule"] and p_mil_non < 0.5:
            return "garder", "military_vehicle", "vlm_m1", infos
        return "ignorer", "vehicle", "militaire_incertain", infos
    if fine in CIVILS:
        return "garder", "civilian_vehicle", "vlm", infos
    return "ignorer", "vehicle", "affiliation_inconnue", infos


# --------------------------------------------------------------------------
# Boîtes d'une frame

def _iou_matrice(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix = np.maximum(0, np.minimum(a[:, None, 2], b[None, :, 2])
                    - np.maximum(a[:, None, 0], b[None, :, 0]))
    iy = np.maximum(0, np.minimum(a[:, None, 3], b[None, :, 3])
                    - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter, aa, ab


def debuts_tuiles(n, tuile, recouvrement):
    """Débuts des tuiles sur un axe, comme get_slice_bboxes de SAHI (utilisé
    par segment_anything_3_tiled) : pas de tuile - int(recouvrement * tuile),
    dernière tuile recalée sur le bord."""
    pas, x, d = tuile - int(recouvrement * tuile), 0, []
    while x < n:
        if x + tuile > n:
            d.append(max(n - tuile, 0))
            break
        d.append(x)
        x += pas
    return d


def coutures(n, tuile, recouvrement):
    """[(début de la tuile i+1, fin de la tuile i)] : bandes de recouvrement."""
    d = debuts_tuiles(n, tuile, recouvrement)
    return [(d[i + 1], d[i] + tuile) for i in range(len(d) - 1)]


def recoller(boites, largeur, hauteur, regles=REGLES):
    """Groupes (listes d'indices, taille >= 2) de boîtes véhicule qui sont les
    morceaux d'un même véhicule coupé par les coutures des tuiles SAM 3 : l'un
    finit sur la fin de la tuile i, l'autre commence au début de la tuile
    i+1, même étendue sur l'autre axe. Union-find : coins et véhicules à
    cheval sur plusieurs coutures."""
    t, r, tol = regles["tuile_sam3"], regles["recouvrement_sam3"], regles["tol_couture"]
    b = np.asarray(boites, float).reshape(-1, 4)
    parent = list(range(len(b)))

    def racine(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for axe, n in ((0, largeur), (1, hauteur)):
        lo, hi = b[:, axe], b[:, axe + 2]
        olo, ohi = b[:, 1 - axe], b[:, 3 - axe]   # autre axe
        for debut, fin in coutures(n, t, r):
            gauche = np.where((np.abs(hi - fin) <= tol) & (lo < debut))[0]
            droite = np.where((np.abs(lo - debut) <= tol) & (hi > fin))[0]
            for i in gauche:
                for j in droite:
                    inter = min(ohi[i], ohi[j]) - max(olo[i], olo[j])
                    union = max(ohi[i], ohi[j]) - min(olo[i], olo[j])
                    if union > 0 and inter / union >= regles["iou_autre_axe"]:
                        parent[racine(i)] = racine(j)
    groupes = collections.defaultdict(list)
    for i in range(len(b)):
        groupes[racine(i)].append(i)
    return [g for g in groupes.values() if len(g) >= 2]


def famille(x):
    """Famille d'un morceau : militaire vérifié, militaire non vérifié, civil,
    ou None ; p5 donne la classe du véhicule entier même sur une partie."""
    if x["niveau"] == "retirer" or x["source"] != "piste":
        return None
    if x["cat"] == "military_vehicle" or x.get("humain") == "militaire":
        return "mil"
    if x["cat"] == "civilian_vehicle" or x.get("humain") == "civil":
        return "civ"
    fine = x.get("fine_class")
    if fine in vlm.MILITAIRES and x["raison"] not in ("equipement_m1",):
        return "mil?"
    if fine in CIVILS:
        return "civ"
    return None


def verdict_recolle(morceaux):
    """(niveau, cat, raison) d'un véhicule recollé, d'après ses morceaux,
    pondérés par l'aire."""
    if all(m["niveau"] == "retirer" for m in morceaux):
        return "retirer", "vehicle", "rejet_vlm"
    poids = collections.Counter()
    for m in morceaux:
        f = famille(m)
        if f:
            x1, y1, x2, y2 = m["bbox"]
            poids[f] += (x2 - x1) * (y2 - y1)
    mil, civ = poids["mil"] + poids["mil?"], poids["civ"]
    if not mil and not civ:
        return "ignorer", "vehicle", "recolle_inconnu"
    if civ >= mil:
        return "garder", "civilian_vehicle", "recolle"
    if poids["mil"]:
        return "garder", "military_vehicle", "recolle"
    return "ignorer", "vehicle", "recolle_mil_non_verifie"


def boites_frame(entrees, frame, regles=REGLES):
    """entrees : [(piste, entrée de frame, verdict)] des pistes présentes.

    Rend [{bbox, niveau, cat, raison, ...}] ; niveau garder | ignorer |
    retirer ; cat : catégorie gardée, ou « person » / « vehicle » pour une
    boîte ignorée ou retirée. Étapes : boîtes des pistes et shapes hors
    piste, recollage des véhicules coupés aux coutures des tuiles SAM 3,
    contenance, puis niveaux des parties et des frames interpolées.
    """
    items = []
    for p, f, (niveau, cat, raison, infos) in entrees:
        items.append({"bbox": [round(float(x), 1) for x in f["bbox"]],
                      "niveau": niveau, "cat": cat, "raison": raison,
                      "source": "piste", "group_id": p["group_id"],
                      "coarse": p["coarse"], "label_sam3": p["label"],
                      "interpolated": bool(f.get("interpolated")), **infos})
    # shapes SAM 3 hors de toute piste (pistes courtes, grappes non associées)
    if frame is not None and len(frame.boites):
        garde = frame.scores >= regles["score_hors_piste"]
        if items:
            inter, a, b = _iou_matrice(frame.boites, [x["bbox"] for x in items])
            iou = inter / np.maximum(a[:, None] + b[None, :] - inter, 1)
            garde &= iou.max(1) < regles["iou_shape_piste"]
        # doublons entre prompts : NMS simple entre shapes hors piste
        retenues = []
        for j in np.argsort(-frame.scores):
            if not garde[j]:
                continue
            if retenues:
                inter, a, b = _iou_matrice(frame.boites[j:j + 1],
                                           frame.boites[retenues])
                if (inter / np.maximum(a[:, None] + b[None, :] - inter, 1)).max() >= 0.6:
                    continue
            retenues.append(int(j))
        for j in retenues:
            g = frame.grossieres[j]
            items.append({"bbox": [round(float(x), 1) for x in frame.boites[j]],
                          "niveau": "ignorer",
                          "cat": "person" if g == "person" else "vehicle",
                          "raison": "hors_piste", "source": "sam3",
                          "coarse": g, "label_sam3": frame.labels[j],
                          "interpolated": False,
                          "score": round(float(frame.scores[j]), 3)})
    # recollage des véhicules coupés par les coutures des tuiles SAM 3
    if regles["recoller"] and frame is not None:
        veh = [k for k, x in enumerate(items)
               if x["coarse"] == "vehicle" and not x["interpolated"]]
        remplaces = set()
        for g in recoller([items[k]["bbox"] for k in veh], frame.largeur,
                          frame.hauteur, regles):
            morceaux = [items[veh[i]] for i in g]
            b = np.array([m["bbox"] for m in morceaux])
            niveau, cat, raison = verdict_recolle(morceaux)
            items.append({"bbox": [float(b[:, 0].min()), float(b[:, 1].min()),
                                   float(b[:, 2].max()), float(b[:, 3].max())],
                          "niveau": niveau, "cat": cat, "raison": raison,
                          "source": "recolle", "coarse": "vehicle",
                          "morceaux": [m.get("group_id") for m in morceaux],
                          "label_sam3": morceaux[0]["label_sam3"],
                          "fine_class": collections.Counter(
                              m.get("fine_class") for m in morceaux
                              if m.get("fine_class")).most_common(1)[0][0]
                          if any(m.get("fine_class") for m in morceaux) else None,
                          "interpolated": False})
            remplaces.update(veh[i] for i in g)
        items = [x for k, x in enumerate(items) if k not in remplaces]
    # contenance frame par frame entre véhicules non rejetés
    veh = [k for k, x in enumerate(items)
           if x["coarse"] == "vehicle" and x["niveau"] != "retirer"]
    contenue = set()
    if len(veh) >= 2:
        b = [items[k]["bbox"] for k in veh]
        inter, aire, _ = _iou_matrice(b, b)
        c = regles["contenance"]
        m = ((inter / np.maximum(aire[:, None], 1) >= c["part_aire"])
             & (aire[None, :] >= c["rapport"] * aire[:, None]))
        np.fill_diagonal(m, False)
        contenue = {veh[r] for r in np.where(m.any(1))[0]}
    for k, x in enumerate(items):
        if x["niveau"] == "partie":
            if k in contenue:
                x["niveau"], x["raison"] = "retirer", "partie_contenue"
            else:
                x["niveau"] = "ignorer"
        elif x["niveau"] == "garder" and x["coarse"] == "vehicle" and k in contenue:
            x["niveau"], x["cat"], x["raison"] = "ignorer", "vehicle", "entier_contenu"
        elif (x["niveau"] == "ignorer" and x["source"] == "sam3"
              and x["coarse"] == "vehicle" and k in contenue):
            x["niveau"], x["raison"] = "retirer", "hors_piste_contenue"
        if (x["niveau"] == "garder" and x["interpolated"]
                and regles["interpolees"] == "ignorer"):
            x["niveau"], x["raison"] = "ignorer", "interpolee"
            x["cat"] = "person" if x["coarse"] == "person" else "vehicle"
    return items


# --------------------------------------------------------------------------
# Un vol

def charger_vol(vol):
    chemins = chemins_vol(vol)
    jeu = dernier_jeu(chemins["pistes"], vol)
    pistes = [json.loads(l) for l in open(jeu, encoding="utf-8")]
    resume = json.load(open(jeu.with_suffix(".json"), encoding="utf-8"))
    d = chemins["pistes"] / "vlm"
    v5 = lire_verdicts(d / f"{vol}_vlm_rapide_p5.jsonl")
    vm1 = lire_verdicts(d / f"{vol}_vlm_rapide_m1.jsonl")
    humain = {}
    if REVUE_MILITAIRES.exists():
        for l in open(REVUE_MILITAIRES, encoding="utf-8"):
            r = json.loads(l)
            if r["vol"] == vol:
                humain[r["group_id"]] = r["verdict"]   # la dernière fait foi
    return chemins, jeu, pistes, resume, v5, vm1, humain


def choisir_frames(frames, par_frame, segments, regles=REGLES):
    """Une frame par pas_s, plus une par pas_bonus_s là où une boîte gardée
    (non interpolée) est militaire ou une personne."""
    def dans_segment(i):
        return any(a <= i <= b for a, b, _n in segments)

    def rare(i):
        return any(v[0] == "garder" and v[1] in ("person", "military_vehicle")
                   and not f.get("interpolated")
                   for _p, f, v in par_frame.get(i, ()))

    choisies, t_der = [], -1e9
    for fr in frames:
        if fr.erreur or not dans_segment(fr.sample):
            continue
        dt = fr.t - t_der
        if dt >= regles["pas_s"] - 1e-3 or (
                dt >= regles["pas_bonus_s"] - 1e-3 and rare(fr.sample)):
            choisies.append(fr)
            t_der = fr.t
    return choisies


def nom_frame(vol, sample):
    return f"frames/{vol}/{vol}_rgb_{sample:07d}.jpg"


def cmd_vol(args):
    import cv2
    for vol in args.vols:
        t0 = time.time()
        chemins, jeu, pistes, resume, v5, vm1, humain = charger_vol(vol)
        verdicts = {p["group_id"]: verdict_piste(p["coarse"], v5.get(p["group_id"]),
                                                 vm1.get(p["group_id"]),
                                                 humain.get(p["group_id"]))
                    for p in pistes}
        par_frame = collections.defaultdict(list)
        for p in pistes:
            for f in p["frames"]:
                par_frame[f["i"]].append((p, f, verdicts[p["group_id"]]))
        frames, _src = charger_frames(chemins["annots"], vol)
        choisies = choisir_frames(frames, par_frame, resume["segments"])
        telem = telemetrie_index(chemins["index"])
        print(f"[i] {vol} : {len(pistes)} pistes ({len(v5)} p5, {len(vm1)} m1, "
              f"{len(humain)} relues), "
              f"{len(frames)} frames annotées -> {len(choisies)} retenues",
              flush=True)
        images = []
        for fr in choisies:
            boites = boites_frame(par_frame.get(fr.sample, []), fr)
            ligne = telem.get(fr.sample, {})
            images.append({
                "file_name": nom_frame(vol, fr.sample), "vol": vol,
                "sample_index": fr.sample, "dts_ticks": fr.tick,
                "dts_s": round(fr.t, 3), "width": fr.largeur, "height": fr.hauteur,
                "telemetrie": {k: (float(ligne[k]) if ligne.get(k) not in (None, "")
                                   else None) for k in TELEMETRIE},
                "boites": boites})
        # décodage : seulement les frames absentes (reprise)
        a_ecrire = [im for im in images if not (SORTIE / im["file_name"]).exists()]
        (SORTIE / "frames" / vol).mkdir(parents=True, exist_ok=True)
        if a_ecrire:
            ticks = [im["dts_ticks"] for im in a_ecrire]
            n = 0
            for pos, img in en_tache_de_fond(decode_frames(chemins["video"], ticks)):
                im = a_ecrire[pos]
                if img is None:
                    im["absente"] = True
                    continue
                if img.shape[1] != im["width"] or img.shape[0] != im["height"]:
                    raise SystemExit(f"[!] {im['file_name']} : {img.shape} décodé, "
                                     f"{im['width']}x{im['height']} attendu")
                dest = SORTIE / im["file_name"]
                tmp = dest.with_suffix(".tmp.jpg")
                cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                tmp.rename(dest)
                n += 1
                if n % 100 == 0:
                    print(f"[i] {vol} : {n}/{len(a_ecrire)} frames écrites "
                          f"({time.time() - t0:.0f} s)", flush=True)
        manquantes = [im["file_name"] for im in images if im.get("absente")]
        images = [im for im in images if not im.get("absente")]
        (SORTIE / "par-vol").mkdir(parents=True, exist_ok=True)
        with open(SORTIE / "par-vol" / f"{vol}.json", "w", encoding="utf-8") as fh:
            json.dump({"vol": vol, "jeu": jeu.name, "regles": REGLES,
                       "verdicts": {"p5": len(v5), "m1": len(vm1)},
                       "manquantes": manquantes, "images": images}, fh,
                      ensure_ascii=False)
        bilan = bilan_vol(images)
        print(f"[i] {vol} : {len(images)} frames en {time.time() - t0:.0f} s "
              f"({len(manquantes)} absentes du flux)")
        afficher_bilan({vol: bilan})
    return 0


# --------------------------------------------------------------------------
# Bilan

def bilan_vol(images):
    b = {"frames": len(images), "niveaux": collections.Counter(),
         "raisons": collections.Counter(), "classes": collections.Counter(),
         "cote_px": collections.defaultdict(list), "frac_ignoree": []}
    for im in images:
        a_garde = a_ign = 0.0
        for x in im["boites"]:
            b["niveaux"][x["niveau"]] += 1
            b["raisons"][f"{x['niveau']}:{x['raison']}"] += 1
            w, h = x["bbox"][2] - x["bbox"][0], x["bbox"][3] - x["bbox"][1]
            if x["niveau"] == "garder":
                b["classes"][x["cat"]] += 1
                b["cote_px"][x["cat"]].append(float(np.sqrt(max(w * h, 0))))
                a_garde += w * h
            elif x["niveau"] == "ignorer":
                a_ign += w * h
        b["frac_ignoree"].append(a_ign / (a_garde + a_ign) if a_garde + a_ign else 0.0)
    return b


def resumer(b):
    fi = np.array(b["frac_ignoree"] or [0.0])
    return {"frames": b["frames"], "niveaux": dict(b["niveaux"]),
            "classes": dict(b["classes"]),
            "raisons": dict(sorted(b["raisons"].items())),
            "cote_median_px": {c: round(float(np.median(v)), 1)
                               for c, v in b["cote_px"].items() if v},
            "frames_aire_ignoree": {"> 25 %": int((fi > 0.25).sum()),
                                    "> 50 %": int((fi > 0.5).sum()),
                                    "> 75 %": int((fi > 0.75).sum())}}


def afficher_bilan(bilans):
    for vol, b in bilans.items():
        r = resumer(b)
        print(f"  {vol} : {r['frames']} frames, niveaux {r['niveaux']}")
        print(f"    gardées : {r['classes']}, côté médian {r['cote_median_px']}")
        print(f"    frames dont l'aire ignorée dépasse : {r['frames_aire_ignoree']}")
        for k, n in r["raisons"].items():
            print(f"    {k:32s} {n:7d}")


# --------------------------------------------------------------------------
# Contrôle visuel

COULEURS = {"person": "#ff8c00", "civilian_vehicle": "#1e90ff",
            "military_vehicle": "#32cd32"}
CSS_CONTROLE = """
body{font:14px system-ui,sans-serif;margin:16px;background:#111;color:#ddd}
h1{font-size:18px} .leg span{display:inline-block;margin-right:14px}
.leg i{display:inline-block;width:14px;height:10px;margin-right:4px;vertical-align:middle}
button{margin:2px;padding:4px 10px;background:#333;color:#ddd;border:1px solid #555;cursor:pointer}
button.on{background:#1e90ff;color:#fff}
figure{margin:18px 0;border-top:1px solid #333;padding-top:8px}
figcaption{margin-bottom:6px;color:#aaa}
.cadre{position:relative;overflow-x:auto}
.cadre .zone{position:relative;width:100%}
.cadre.grand .zone{width:3840px}
.cadre img{width:100%;display:block}
.cadre svg{position:absolute;left:0;top:0;width:100%;height:100%}
.cadre rect{fill:none;stroke-width:5}
.cadre rect.ignorer{stroke:#bbb;fill:url(#hachure);stroke-width:3}
.cadre rect.retirer{stroke:#ff3030;stroke-dasharray:14 10;stroke-width:3}
"""
JS_CONTROLE = """
function filtre(s){document.querySelectorAll('figure').forEach(f=>{
 f.style.display=(s==='tout'||f.dataset.s.split(' ').includes(s))?'':'none'});
 document.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.s===s));}
document.addEventListener('click',e=>{const c=e.target.closest('.cadre');
 if(c) c.classList.toggle('grand');});
"""


def strates_image(im):
    s = set()
    niv = collections.Counter((b["niveau"], b["cat"]) for b in im["boites"])
    raisons = {b["raison"] for b in im["boites"]}
    if niv[("garder", "military_vehicle")]:
        s.add("militaire")
    if niv[("garder", "person")]:
        s.add("personnes")
    if {"partie", "partie_contenue", "entier_contenu"} & raisons:
        s.add("parties")
    if {"equipement_m1", "militaire_incertain", "militaire_non_verifie"} & raisons:
        s.add("militaire_douteux")
    if "rejet_vlm" in raisons:
        s.add("rejets")
    a = collections.Counter()
    for b in im["boites"]:
        x1, y1, x2, y2 = b["bbox"]
        a[b["niveau"]] += (x2 - x1) * (y2 - y1)
    if a["ignorer"] > 0.5 * (a["ignorer"] + a["garder"]):
        s.add("surtout_ignoree")
    return s


def cmd_controle(args):
    rnd = random.Random(args.seed)
    figures, strates = [], collections.Counter()
    for vol in args.vols:
        d = json.load(open(SORTIE / "par-vol" / f"{vol}.json", encoding="utf-8"))
        par_strate = collections.defaultdict(list)
        for im in d["images"]:
            for s in strates_image(im) | {"hasard"}:
                par_strate[s].append(im)
        choisies = {}
        for s, liste in sorted(par_strate.items()):
            for im in rnd.sample(liste, min(args.n, len(liste))):
                choisies[im["file_name"]] = im
        for im in sorted(choisies.values(), key=lambda im: im["sample_index"]):
            st = strates_image(im)
            strates.update(st)
            rects = []
            for b in sorted(im["boites"], key=lambda b: b["niveau"] == "garder"):
                x1, y1, x2, y2 = b["bbox"]
                attrs = (f'x="{x1:.0f}" y="{y1:.0f}" width="{x2 - x1:.0f}" '
                         f'height="{y2 - y1:.0f}"')
                titre = html.escape(
                    f"{b['niveau']} {b['cat']} ({b['raison']}) "
                    f"#{b.get('group_id', '-')} SAM3 {b.get('label_sam3')} "
                    f"VLM {b.get('fine_class') or ''} {b.get('m1_kind') or ''}")
                if b["niveau"] == "garder":
                    rects.append(f'<rect {attrs} style="stroke:{COULEURS[b["cat"]]}">'
                                 f"<title>{titre}</title></rect>")
                else:
                    rects.append(f'<rect class="{b["niveau"]}" {attrs}>'
                                 f"<title>{titre}</title></rect>")
            n = collections.Counter(b["niveau"] for b in im["boites"])
            figures.append(
                f'<figure data-s="{" ".join(sorted(st))} hasard">'
                f"<figcaption>{vol} · sample {im['sample_index']} · "
                f"t = {im['dts_s']:.1f} s · gardées {n['garder']}, ignorées "
                f"{n['ignorer']}, retirées {n['retirer']} · "
                f"{' '.join(sorted(st))}</figcaption>"
                f'<div class="cadre"><div class="zone"><img loading="lazy" '
                f'src="{im["file_name"]}">'
                f'<svg viewBox="0 0 {im["width"]} {im["height"]}">'
                + "".join(rects) + "</svg></div></div></figure>")
    boutons = "".join(f'<button data-s="{s}" onclick="filtre(\'{s}\')">{s}</button>'
                      for s in ["tout", "hasard"] + sorted(set(strates) - {"hasard"}))
    legende = "".join(f'<span><i style="background:{c}"></i>{k}</span>'
                      for k, c in COULEURS.items())
    legende += ('<span><i style="background:#bbb"></i>ignorée (hachurée)</span>'
                '<span><i style="background:#ff3030"></i>retirée (pointillés)</span>')
    page = (
        "<!doctype html><meta charset=utf-8><title>Contrôle export détection</title>"
        f"<style>{CSS_CONTROLE}</style><script>{JS_CONTROLE}</script>"
        '<svg width="0" height="0"><defs><pattern id="hachure" width="16" '
        'height="16" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<line x1="0" y1="0" x2="0" y2="16" stroke="#bbb" stroke-width="4" '
        'opacity=".6"/></pattern></defs></svg>'
        f"<h1>Contrôle de l'export détection — {', '.join(args.vols)}</h1>"
        f"<p class=leg>{legende}</p><p>Survoler une boîte : niveau, raison, piste. "
        "Cliquer une image : taille native (défilement horizontal).</p>"
        f"<p>{boutons}</p>" + "".join(figures))
    sortie = SORTIE / "controle.html"
    sortie.write_text(page, encoding="utf-8")
    print(f"[i] {len(figures)} frames -> {sortie}")
    return 0


# --------------------------------------------------------------------------
# Seuils : règles contre la revue humaine

def cmd_seuils(args):
    ev = _module("evalue_prompts_vlm", "evalue-prompts-vlm.py")
    cas = ev.charger_cas()
    rep5 = ev.lire_reponses("rapide", "p5")
    m1 = {}
    for vol in VOLS_JOUR:
        d = chemins_vol(vol)["pistes"] / "vlm"
        for g, r in lire_verdicts(d / f"{vol}_vlm_rapide_m1.jsonl").items():
            m1[f"{vol}/{g}"] = r

    def verite(h, coarse):
        if h["real_object"] == "no":
            return "faux positif"
        if h["real_object"] != "yes":
            return "incertain"
        if coarse == "person":
            return "personne"
        if h.get("box_covers") == "part":
            return "partie"
        f = h.get("fine_class") or ""
        if f in vlm.MILITAIRES or h.get("affiliation") == "military":
            return "véh. militaire"
        if f == "vehicle_unknown" and h.get("affiliation") != "civil":
            return "véh. inconnu"
        return "véh. civil"

    def regle(v, cle, coarse):
        niveau, cat, raison, _i = verdict_piste(coarse, v, m1.get(cle))
        if niveau == "garder":
            return {"person": "person", "civilian_vehicle": "civil",
                    "military_vehicle": "military"}[cat]
        if niveau == "partie":
            return "partie"
        return niveau

    lignes = ["personne", "véh. civil", "véh. militaire", "véh. inconnu",
              "partie", "incertain", "faux positif"]
    cols = ["person", "civil", "military", "partie", "ignorer", "retirer"]
    m = collections.Counter()
    detail = collections.defaultdict(list)
    for p, h in cas:
        cle = f"{p['vol']}/{p['group_id']}"
        v = rep5.get(cle)
        if v is None:
            continue
        a, b = verite(h, p["coarse"]), regle(v, cle, p["coarse"])
        m[(a, b)] += 1
        detail[(a, b)].append(cle)
    avec_m1 = sum(1 for p, _h in cas if f"{p['vol']}/{p['group_id']}" in m1)
    print(f"{len(cas)} pistes de jour relues ; m1 disponible pour {avec_m1}")
    print("humain \\ règle".ljust(16) + "".join(f"{c:>10s}" for c in cols))
    for a in lignes:
        print(f"{a:16s}" + "".join(f"{m[(a, c)] or '.':>10}" for c in cols))
    if args.detail:
        for (a, b), cles in sorted(detail.items()):
            print(f"  {a} -> {b} : {' '.join(cles)}")
    return 0


# --------------------------------------------------------------------------
# Assemblage COCO

def categories_ignorees(cat):
    """Catégories neutralisées par une boîte ignorée : pycocotools n'ignore
    une détection que si la zone iscrowd est de la même catégorie."""
    if cat == "person":
        return ["person"]
    if cat in CATEGORIES:
        return [cat]
    return ["civilian_vehicle", "military_vehicle"]


def lire_par_vol(vols=None):
    fichiers = sorted((SORTIE / "par-vol").glob("*.json"))
    return {f.stem: json.load(open(f, encoding="utf-8")) for f in fichiers
            if vols is None or f.stem in vols}


def cmd_assemble(args):
    donnees = lire_par_vol(args.vols)
    splits = {**SPLITS, **dict(x.split("=") for x in args.split)}
    ids = {c: i + 1 for i, c in enumerate(CATEGORIES)}
    coco = {s: {"images": [], "annotations": []} for s in ("train", "val", "test")}
    n_img = n_ann = n_grp = 0
    bilans = {}
    for vol, d in donnees.items():
        split = splits.get(vol, "train")
        bilans[vol] = resumer(bilan_vol(d["images"])) | {"split": split}
        for im in d["images"]:
            n_img += 1
            coco[split]["images"].append({
                "id": n_img, "file_name": im["file_name"], "width": im["width"],
                "height": im["height"], "vol": vol,
                "sample_index": im["sample_index"], "dts_s": im["dts_s"],
                "telemetrie": im["telemetrie"]})
            for b in im["boites"]:
                if b["niveau"] == "retirer":
                    continue
                x1, y1, x2, y2 = b["bbox"]
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                attrs = {k: v for k, v in b.items() if k not in ("bbox", "cat")}
                if b["niveau"] == "garder":
                    cats, crowd = [b["cat"]], 0
                else:
                    n_grp += 1
                    cats, crowd = categories_ignorees(b["cat"]), 1
                    attrs["groupe_ignore"] = n_grp
                for c in cats:
                    n_ann += 1
                    coco[split]["annotations"].append({
                        "id": n_ann, "image_id": n_img, "category_id": ids[c],
                        "bbox": [x1, y1, w, h], "area": round(w * h, 1),
                        "iscrowd": crowd, "ignore": crowd, "attributes": attrs})
    (SORTIE / "annotations").mkdir(exist_ok=True)
    cats = [{"id": ids[c], "name": c, "supercategory": "person" if c == "person"
             else "vehicle"} for c in CATEGORIES]
    info = {"description": "CAMPAGNE3, 9 vols de jour RGB 4K : pseudo-labels SAM 3 "
            "+ BoT-SORT vérifiés par Qwen3.8-27B (p5, m1 sur les militaires)",
            "version": "p5-m1", "date_created": time.strftime("%Y-%m-%d")}
    for s, c in coco.items():
        if not c["images"]:
            continue
        with open(SORTIE / "annotations" / f"{s}.json", "w", encoding="utf-8") as fh:
            json.dump({"info": info, "categories": cats, **c}, fh, ensure_ascii=False)
        n_g = sum(1 for a in c["annotations"] if not a["iscrowd"])
        print(f"[i] {s:5s} : {len(c['images'])} images, {n_g} boîtes gardées, "
              f"{len(c['annotations']) - n_g} annotations iscrowd")
    json.dump({"info": info, "categories": CATEGORIES, "splits": {
                  s: sorted(v for v in donnees if splits.get(v, "train") == s)
                  for s in coco},
               "regles": REGLES, "niveaux": {
                  "garder": "iscrowd=0", "ignorer": "iscrowd=1 (une annotation par "
                  "catégorie neutralisée, même groupe_ignore)",
                  "retirer": "absent du COCO, compté dans bilan.json"}},
              open(SORTIE / "dataset.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(bilans, open(SORTIE / "bilan.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for vol, b in bilans.items():
        print(f"  {vol} {b['split']:5s} {b['frames']:6d} frames  {b['classes']}")
    return 0


# --------------------------------------------------------------------------
# Tuiles YOLO

def positions(n, tuile, recouvrement):
    if n <= tuile:
        return [0]
    pas = int(tuile * (1 - recouvrement))
    xs = list(range(0, n - tuile, pas))
    return xs + [n - tuile]


def cmd_yolo(args):
    import cv2
    donnees = lire_par_vol(args.vols)
    splits = {**SPLITS, **dict(x.split("=") for x in args.split)}
    racine = args.sortie or SORTIE / "yolo"
    rnd = random.Random(0)
    cls = {c: i for i, c in enumerate(CATEGORIES)}
    table, stats = [], collections.Counter()
    for vol, d in donnees.items():
        split = splits.get(vol, "train")
        (racine / "images" / split).mkdir(parents=True, exist_ok=True)
        (racine / "labels" / split).mkdir(parents=True, exist_ok=True)
        for im in d["images"]:
            img = None
            gardees = [b for b in im["boites"] if b["niveau"] == "garder"]
            ignorees = [b for b in im["boites"] if b["niveau"] == "ignorer"]
            W, H = im["width"], im["height"]
            s = args.echelle
            Wt, Ht = round(W * s), round(H * s)
            for y0 in positions(Ht, args.tuile, args.recouvrement):
                for x0 in positions(Wt, args.tuile, args.recouvrement):
                    x1t, y1t = x0 / s, y0 / s
                    x2t, y2t = (x0 + args.tuile) / s, (y0 + args.tuile) / s
                    lignes, masque, dedans = [], [], []
                    for b in gardees:
                        bx1, by1, bx2, by2 = b["bbox"]
                        ix1, iy1 = max(bx1, x1t), max(by1, y1t)
                        ix2, iy2 = min(bx2, x2t), min(by2, y2t)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        part = ((ix2 - ix1) * (iy2 - iy1)
                                / max((bx2 - bx1) * (by2 - by1), 1e-6))
                        if part >= args.part_min:
                            dedans.append((ix1, iy1, ix2, iy2))
                            cx = ((ix1 + ix2) / 2 - x1t) / (x2t - x1t)
                            cy = ((iy1 + iy2) / 2 - y1t) / (y2t - y1t)
                            lignes.append(f"{cls[b['cat']]} {cx:.6f} {cy:.6f} "
                                          f"{(ix2 - ix1) / (x2t - x1t):.6f} "
                                          f"{(iy2 - iy1) / (y2t - y1t):.6f}")
                        else:   # moignon : grisé, ni objet ni fond
                            masque.append((ix1, iy1, ix2, iy2))
                            stats["moignons"] += 1
                    for b in ignorees:
                        bx1, by1, bx2, by2 = b["bbox"]
                        if bx2 > x1t and bx1 < x2t and by2 > y1t and by1 < y2t:
                            masque.append((bx1, by1, bx2, by2))
                    if not lignes:   # tuile de fond : gardée avec la proba args.vides
                        if rnd.random() > args.vides:
                            stats["tuiles_vides_ecartees"] += 1
                            continue
                    if img is None:
                        img = cv2.imread(str(SORTIE / im["file_name"]))
                        if s != 1:
                            img = cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_AREA)
                    t = img[y0:y0 + args.tuile, x0:x0 + args.tuile].copy()
                    if masque:
                        m = np.zeros(t.shape[:2], bool)
                        marge = args.marge
                        for bx1, by1, bx2, by2 in masque:
                            a = [int(np.floor((bx1 - marge) * s - x0)),
                                 int(np.floor((by1 - marge) * s - y0)),
                                 int(np.ceil((bx2 + marge) * s - x0)),
                                 int(np.ceil((by2 + marge) * s - y0))]
                            m[max(a[1], 0):max(a[3], 0), max(a[0], 0):max(a[2], 0)] = True
                        # jamais sur une boîte gardée
                        for bx1, by1, bx2, by2 in dedans:
                            m[max(int(by1 * s - y0), 0):max(int(np.ceil(by2 * s - y0)), 0),
                              max(int(bx1 * s - x0), 0):max(int(np.ceil(bx2 * s - x0)), 0)] = False
                        t[m] = 114
                        stats["tuiles_grisees"] += 1
                    stem = f"{Path(im['file_name']).stem}_x{x0:04d}_y{y0:04d}"
                    cv2.imwrite(str(racine / "images" / split / f"{stem}.jpg"), t,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (racine / "labels" / split / f"{stem}.txt").write_text(
                        "".join(l + "\n" for l in lignes))
                    table.append({"tuile": f"{split}/{stem}", "frame": im["file_name"],
                                  "x0": x0, "y0": y0, "echelle": s,
                                  "objets": len(lignes)})
                    stats[f"tuiles_{split}"] += 1
                    stats[f"objets_{split}"] += len(lignes)
        print(f"[i] {vol} ({split}) : {dict(stats)}", flush=True)
    with open(racine / "data.yaml", "w", encoding="utf-8") as fh:
        fh.write("# CAMPAGNE3 detection-p5 : tuiles "
                 f"{args.tuile} px, recouvrement {args.recouvrement:.0%}, échelle "
                 f"{args.echelle}, zones ignorées grisées (114)\n")
        fh.write(f"path: {racine}\n")
        for sp in ("train", "val", "test"):
            if stats[f"tuiles_{sp}"]:
                fh.write(f"{sp}: images/{sp}\n")
        fh.write(f"nc: {len(CATEGORIES)}\nnames:\n")
        for i, c in enumerate(CATEGORIES):
            fh.write(f"  {i}: {c}\n")
    json.dump({"reglage": {k: v for k, v in vars(args).items() if k != "f"},
               "stats": stats,
               "tuiles": table}, open(racine / "tuiles.json", "w"), default=str)
    print(f"[i] {dict(stats)} -> {racine}")
    return 0


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sous = ap.add_subparsers(dest="cmd", required=True)
    s = sous.add_parser("seuils", help="règles contre la revue humaine")
    s.add_argument("--detail", action="store_true")
    s.set_defaults(f=cmd_seuils)
    s = sous.add_parser("vol", help="frames + par-vol/<vol>.json")
    s.add_argument("vols", nargs="+")
    s.set_defaults(f=cmd_vol)
    s = sous.add_parser("controle", help="page HTML de contrôle")
    s.add_argument("vols", nargs="+")
    s.add_argument("--n", type=int, default=8, help="frames par strate et par vol")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(f=cmd_controle)
    for nom, f, aide in (("assemble", cmd_assemble, "annotations/{train,val,test}.json"),
                         ("yolo", cmd_yolo, "tuiles YOLO (yolo/)")):
        s = sous.add_parser(nom, help=aide)
        s.add_argument("--vols", nargs="+", default=None)
        s.add_argument("--split", nargs="*", default=[],
                       help="vol=split en plus de SPLITS (ex. 0000231=val)")
        s.set_defaults(f=f)
        if nom == "yolo":
            s.add_argument("--tuile", type=int, default=1024)
            s.add_argument("--recouvrement", type=float, default=0.2)
            s.add_argument("--echelle", type=float, default=1.0)
            s.add_argument("--part-min", type=float, default=0.4,
                           help="part d'aire d'une boîte coupée pour la garder")
            s.add_argument("--vides", type=float, default=0.1,
                           help="probabilité de garder une tuile sans objet")
            s.add_argument("--sortie", type=Path, default=None,
                           help="dossier YOLO (défaut : <SORTIE>/yolo)")
            s.add_argument("--marge", type=float, default=4,
                           help="marge (px natifs) autour des zones grisées")
    args = ap.parse_args()
    return args.f(args)


if __name__ == "__main__":
    sys.exit(main())
