#!/usr/bin/env python3
"""Vérification des pistes au VLM (Qwen3.8-27B servi par vLLM sur Slurm).

Pour chaque piste : K vues réparties sur sa durée (boîte tracée en rouge) et
une vue large de la scène, envoyées ensemble au VLM, qui répond en JSON :

- same_object / different_views : la piste reste-t-elle sur le même objet
  (pureté), et quelles vues s'en écartent ;
- real_object : est-ce vraiment un véhicule / une personne (faux positifs) ;
- fine_class, affiliation (véhicules) : classe fine de la taxonomie AnafiUKR.

Le rejet est une question à part, jamais une option de fine_class (lot V0 :
« pas un véhicule » dans la liste des sous-classes aspire la moitié des vrais
véhicules). Probabilités lues sur les logprobs bruts de vLLM (avant la
grammaire JSON) : P(yes/no/unsure) au premier jeton de la valeur, P(classe
choisie) sur tous ses jetons. Seuiller ces probabilités, pas l'argmax.

Deux étapes, chacune reprise là où elle s'était arrêtée :
1. vues : frames décodées sur le PC (NVDEC, flux 0, streaming), vues rangées
   dans un zip par vol, `<SORTIES>/<vol>/vlm/<jeu>_vues-<empreinte>.zip` ;
2. questions : requêtes parallèles au serveur (tunnel SSH), une ligne JSONL
   par piste, `<SORTIES>/<vol>/vlm/<vol>_vlm_<mode>_<prompt>.jsonl`.

    PY=X-AnyLabeling-Server/.venv/bin/python
    $PY verifie-pistes-vlm.py --vol 0000011 --selection pilote --vues-seules
    ssh -N -L 13620:gpu02:13620 master.slurm.troie.ia &     # voir le journal
    $PY verifie-pistes-vlm.py --vol 0000011 --selection pilote \\
        --serveur http://localhost:13620/v1 --mode rapide
"""
from __future__ import annotations

import argparse
import base64
import bisect
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pistes_io import (RACINE, chemins_vol, decode_frames,  # noqa: E402
                       dernier_jeu, en_tache_de_fond, largeur_sol_m,
                       telemetrie_index, ticks_index, timescale_index)

# Réglages des vues par version de prompt : changent l'empreinte du zip (un
# zip par réglage). p2 : un peu plus de marge (voir le reste d'un véhicule
# quand la boîte n'en couvre qu'une partie) et une vue large resserrée — en
# p1, le modèle jugeait sur la vue large, où les panneaux passaient pour des
# personnes parmi d'autres personnes.
VUES_PROMPT = {
    "p1": {"k": 8, "marge": 0.75, "cote_min": 48, "sortie": 320,
           "ctx_facteur": 8.0, "ctx_min": 720, "ctx_sortie": 512},
    "p2": {"k": 8, "marge": 1.0, "cote_min": 48, "sortie": 320,
           "ctx_facteur": 5.0, "ctx_min": 400, "ctx_sortie": 448},
}
# p3 : questions de p2 sans « le détecteur se trompe souvent » ni « only if
# clearly », qui faisaient rejeter de vrais objets (4/23 sur les cas connus).
VUES_PROMPT["p3"] = VUES_PROMPT["p2"]
# p4 : mêmes vues, questions réglées sur la revue humaine (texte_question_p4).
VUES_PROMPT["p4"] = VUES_PROMPT["p2"]
# p5 : p3 retouché d'après les erreurs de p4 (texte_question_p5), mêmes vues.
VUES_PROMPT["p5"] = VUES_PROMPT["p2"]
# p6 : p5 + largeur au sol estimée par la télémétrie (taille_piste_m), pour
# l'échelle que les vues agrandies font perdre (chaises, bidons, mallettes
# pris pour des personnes ou des véhicules ; statue de 2,9 m). Une boîte peut
# être plus petite que l'objet (occlusion, partie) : un indice, jamais une
# raison de rejet à elle seule.
VUES_PROMPT["p6"] = VUES_PROMPT["p2"]
# p7 : taille pour les véhicules seulement, phrase neutre (sans parler de
# rejet : en p6 elle attirait les rejets, véhicules pris pour un arbre ou un
# oiseau). Rien pour les personnes : vues à la verticale, seules la tête et
# les épaules paraissent, le VLM y voyait un déchet (remarque utilisateur).
VUES_PROMPT["p7"] = VUES_PROMPT["p2"]
PROMPTS_TAILLE = {"p6", "p7"}
# Contenance géométrique (sans VLM) : une boîte véhicule est « contenue » sur
# une frame quand >= 80 % de son aire tombe dans la boîte d'une autre piste
# véhicule au moins 1,5 fois plus grande — une partie d'un véhicule déjà suivi.
CONTENANCE = {"part_aire": 0.8, "rapport": 1.5}

# Cas relus sur planche (README, « Pistes BoT-SORT sur un vol entier ») :
# objet réel ou non, piste pure ou non ; None = pas d'attendu.
CAS_CONNUS = {
    "0000011": {
        89: ("oui", True, "soldats devant le bâtiment"),
        314: ("oui", True, "soldats devant le bâtiment"),
        315: ("oui", True, "soldats devant le bâtiment"),
        316: ("oui", True, "soldats devant le bâtiment"),
        317: ("oui", True, "soldats devant le bâtiment"),
        933: ("oui", True, "personne de 8 px"),
        443: ("non", None, "clôture, score 0,39"),
        531: ("non", None, "faux positif, score 0,33"),
        276: (None, False, "change d'objet (zoom rapide)"),
        865: (None, False, "change d'objet (zoom rapide)"),
        386: (None, None, "ambiguë (zoom rapide)"),
        483: ("oui", True, "pure à travers un zoom"),
    },
    "0000012": {
        **{g: ("oui", True, "voiture de parking suivie 29 s")
           for g in range(507, 515)},
        1215: (None, False, "saut d'objet, caméra rapide"),
        1661: (None, False, "saut d'objet, caméra rapide"),
        1085: ("non", None, "panneau jaune"),
        1182: ("non", None, "panneau rond"),
        135: ("oui", None, "vraie personne"),
        1400: ("oui", None, "vraie personne"),
    },
    "0000018": {
        83: ("oui", True, "voiture floue de 12 px"),
        102: ("oui", True, "voiture floue de 11 px"),
        111: ("oui", True, "voiture floue de 10 px"),
        368: ("oui", True, "soldat"),
        369: ("oui", True, "soldat"),
        373: ("oui", True, "soldat"),
        41: ("non", None, "ombre de 668 px"),
        370: (None, None, "boîte contre une portière"),
        383: (None, None, "boîte contre une portière"),
        677: ("non", None, "tache de 14 px"),
        680: ("non", None, "tache de 14 px"),
    },
}
# Parties de véhicule déjà couvertes par une autre piste (contenance >= 80 %,
# vérifiées sur les vues) : attendu box_covers = part.
PARTIES = {
    "0000011": {74: "flanc haut d'une camionnette blanche",
                109: "bas d'une camionnette blanche",
                333: "arrière d'un SUV beige"},
    "0000012": {90: "flanc d'une camionnette blanche",
                140: "flanc d'une camionnette blanche",
                371: "coin d'un véhicule blanc (zoom)"},
    "0000018": {162: "bas de l'avant d'un camion sous les arbres (dans #159)",
                5: "roues d'un SUV beige",
                282: "bas d'une camionnette blanche"},
}
# Strates tirées au hasard (pistes >= 10 détections), indépendamment des cas
# connus : elles seules servent à estimer des taux.
QUOTAS = {"personne": 30, "vehicule": 40, "militaire": 15, "zoom": 0}
QUOTAS_VOL = {"0000011": {"zoom": 15}, "0000012": {"personne": 40}}
ZOOMS = {"0000011": (10000, 12489)}   # tronçon de zooms rapides et de flou

CLASSES_VEHICULE = [
    ("car", "civilian passenger car (sedan, hatchback, estate, SUV)"),
    ("van", "civilian van or minibus"),
    ("truck", "civilian truck, lorry, semi-trailer or tanker"),
    ("bus", "bus or coach"),
    ("motorcycle", "motorcycle or scooter"),
    ("bicycle", "bicycle"),
    ("agri_vehicle", "tractor, agricultural or construction machinery"),
    ("mil_tank", "tracked armoured vehicle with a turret and a large main "
     "gun (tank)"),
    ("mil_apc_ifv", "armoured personnel carrier or infantry fighting vehicle "
     "(wheeled or tracked armoured hull, no large main gun)"),
    ("mil_truck", "military cargo truck (military paint or camouflage, often "
     "a canvas-covered cargo bed)"),
    ("mil_other", "other military vehicle (military 4x4, engineering, "
     "artillery, ...)"),
    ("vehicle_unknown", "a vehicle whose type cannot be determined"),
]
# p4 (2026-09-29), d'après la revue humaine de 134 pistes de jour
# (revue-pistes.py) : faux positifs vus = groupes électrogènes, blocs de béton,
# bidon, mallettes, chaises, sacs, poubelle, statue, cuves ; véhicules
# militaires d'allure civile (VT4, 4x4 sur base Ford Ranger) étiquetés car/van
# par SAM 3 ; beaucoup de boîtes sur une partie (portière, coffre, moteur).
NON_OBJETS_P4 = {
    "vehicle": "an electric generator or other equipment on wheels or skids, a "
               "container, water tank or cistern, a fuel can, concrete blocks, a "
               "bin, a case, crates or bags, a chair or other furniture, a "
               "statue, a road or traffic sign, a pole, a shadow, a bush, a rock, "
               "part of a building, a road marking, debris",
    "person": "a chair, a bag, a case, a bin, a post or pole, a sign, a statue, "
              "a shadow, a bush, a rock, debris, part of a vehicle",
}
PARTIES_P4 = {
    "vehicle": "only a door, the bonnet or engine, the boot or rear, the cab, "
               "the cargo bed, a trailer, the roof or the wheels",
    "person": "only the legs, an arm or the head",
}
CLASSES_VEHICULE_P4 = [
    ("car", "civilian passenger car (sedan, hatchback, estate, SUV, civilian "
     "pickup)"),
    ("van", "civilian van: panel van, minivan, minibus or small van (e.g. "
     "Renault Trafic, Citroen Berlingo)"),
    ("truck", "civilian truck, lorry, semi-trailer or tanker"),
    ("bus", "bus or coach"),
    ("motorcycle", "motorcycle or scooter"),
    ("bicycle", "bicycle"),
    ("agri_vehicle", "tractor, agricultural or construction machinery"),
    ("mil_tank", "tracked armoured vehicle with a turret and a large main "
     "gun (tank)"),
    ("mil_apc_ifv", "armoured personnel carrier or infantry fighting vehicle "
     "(wheeled or tracked armoured hull, no large main gun)"),
    ("mil_truck", "military truck, e.g. Renault GBC (military paint, often a "
     "canvas-covered cargo bed)"),
    ("mil_other", "other military vehicle: military car, 4x4 or pickup (e.g. "
     "the French VT4, a Ford Ranger-based 4x4), engineering vehicle, unmanned "
     "ground robot, artillery, ..."),
    ("vehicle_unknown", "a vehicle whose type cannot be determined"),
]
assert [c for c, _ in CLASSES_VEHICULE_P4] == [c for c, _ in CLASSES_VEHICULE]
# p5 : p4 rejetait des véhicules vrais (« generator or other equipment » attirait
# les morceaux de véhicule : portière, roue de secours, arrière de VT4) et
# poussait les voitures blanches en van (exemple Berlingo). On garde de p4 les
# leurres concrets, VT4/GBC et la règle de la peinture militaire.
NON_OBJETS_P5 = {
    "vehicle": "road or traffic sign, pole, post, shadow, bush, rock, part of a "
               "building, road marking, debris, container, water tank, crate, "
               "case, bag, chair, bin, concrete block, statue",
    "person": "vehicle part, chair, bag, case, bin, post, pole, sign, statue, "
              "shadow, bush, rock, debris",
}
PARTIES_P5 = {
    "vehicle": "only a door, a wheel, the bonnet or front, the engine, the boot "
               "or rear, the cab, the cargo bed, a trailer or the roof",
    "person": "only the legs, an arm or the head",
}
CLASSES_VEHICULE_P5 = [
    (c, dict(CLASSES_VEHICULE_P4)[c] if c in ("mil_truck", "mil_other") else d)
    for c, d in CLASSES_VEHICULE]
OUI_NON = ["yes", "no", "unsure"]
COUVERTURE = ["whole", "part", "unsure"]
AFFILIATIONS = ["civil", "military", "unknown"]

SYSTEME = ("You check the output of an automatic object detector and tracker "
           "on aerial drone video. Judge only what is visible in the images.")

MODES = {
    # thinking coupé, glouton : une passe courte
    "rapide": {"chat_template_kwargs": {"enable_thinking": False},
               "temperature": 0.0, "max_tokens": 600},
    # thinking : échantillonnage recommandé (generation_config.json)
    "reflexion-low": {"chat_template_kwargs": {"enable_thinking": True,
                                               "reasoning_effort": "low"},
                      "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                      "max_tokens": 6000},
    "reflexion-medium": {"chat_template_kwargs": {"enable_thinking": True,
                                                  "reasoning_effort": "medium"},
                         "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                         "max_tokens": 12000},
}


# --------------------------------------------------------------------------
# Sélection

def selection_pilote(vol, pistes, seed=0):
    """{group_id: [catégories]} : cas connus + strates tirées au hasard."""
    rnd = random.Random(f"{seed}-{vol}")
    choisies = {}
    for gid in CAS_CONNUS.get(vol, {}):
        choisies.setdefault(gid, []).append("connu")
    quotas = {**QUOTAS, **QUOTAS_VOL.get(vol, {})}
    eligibles = [p for p in pistes if p["n_detected"] >= 10]
    zoom = ZOOMS.get(vol)
    filtres = {
        "personne": lambda p: p["coarse"] == "person",
        "vehicule": lambda p: p["coarse"] == "vehicle",
        "militaire": lambda p: p["label"].startswith("mil_"),
        "zoom": lambda p: (zoom is not None and p["coarse"] == "vehicle"
                           and p["sample_min"] < zoom[1]
                           and p["sample_max"] >= zoom[0]),
    }
    for cat in ("personne", "vehicule", "militaire", "zoom"):
        cands = [p["group_id"] for p in eligibles if filtres[cat](p)]
        rnd.shuffle(cands)
        for gid in cands[:quotas[cat]]:
            choisies.setdefault(gid, []).append(cat)
    return choisies


# --------------------------------------------------------------------------
# Vues

def empreinte_vues(reglage):
    return hashlib.sha1(json.dumps(reglage, sort_keys=True).encode()).hexdigest()[:6]


def contenance(pistes):
    """{group_id: {"frac", "par"}} pour les pistes véhicules : part de leurs
    frames détectées où la boîte est contenue dans celle d'une autre piste
    véhicule plus grande (CONTENANCE), et la piste contenante la plus
    fréquente."""
    import collections
    par_frame = collections.defaultdict(list)
    for p in pistes:
        if p["coarse"] != "vehicle":
            continue
        for f in p["frames"]:
            if not f.get("interpolated"):
                par_frame[f["i"]].append((p["group_id"], f["bbox"]))
    compte = collections.Counter()
    qui = collections.defaultdict(collections.Counter)
    for liste in par_frame.values():
        if len(liste) < 2:
            continue
        g = np.array([x[0] for x in liste])
        b = np.array([x[1] for x in liste], float)
        aire = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        ix = np.maximum(0, np.minimum(b[:, None, 2], b[None, :, 2])
                        - np.maximum(b[:, None, 0], b[None, :, 0]))
        iy = np.maximum(0, np.minimum(b[:, None, 3], b[None, :, 3])
                        - np.maximum(b[:, None, 1], b[None, :, 1]))
        m = ((ix * iy / np.maximum(aire[:, None], 1) >= CONTENANCE["part_aire"])
             & (aire[None, :] >= CONTENANCE["rapport"] * aire[:, None]))
        np.fill_diagonal(m, False)
        for r in np.where(m.any(1))[0]:
            compte[g[r]] += 1
            qui[g[r]][int(g[np.argmax(np.where(m[r], aire, 0))])] += 1
    return {p["group_id"]: {
                "frac": round(compte[p["group_id"]] / p["n_detected"], 3),
                "par": (qui[p["group_id"]].most_common(1)[0][0]
                        if qui[p["group_id"]] else None)}
            for p in pistes if p["coarse"] == "vehicle"}


def carre(bbox, marge, cote_min, largeur, hauteur):
    """Carré (x, y, côté) centré sur la boîte, recalé dans l'image sans
    déformer (plutôt que tronqué)."""
    x1, y1, x2, y2 = bbox
    cote = max(x2 - x1, y2 - y1) * (1 + 2 * marge)
    cote = min(max(cote, cote_min), min(largeur, hauteur))
    x = min(max((x1 + x2) / 2 - cote / 2, 0), largeur - cote)
    y = min(max((y1 + y2) / 2 - cote / 2, 0), hauteur - cote)
    return x, y, cote


def vue(image, bbox, marge, cote_min, sortie, epaisseur):
    import cv2
    h, w = image.shape[:2]
    x, y, cote = carre(bbox, marge, cote_min, w, h)
    xi, yi, ci = int(round(x)), int(round(y)), max(int(round(cote)), 1)
    crop = image[yi:yi + ci, xi:xi + ci]
    img = cv2.resize(crop, (sortie, sortie),
                     interpolation=cv2.INTER_CUBIC if ci < sortie
                     else cv2.INTER_AREA)
    e = sortie / ci
    x1, y1, x2, y2 = bbox
    # rectangle tracé un peu plus grand que la boîte : il ne cache pas l'objet
    a = (int(max(0, (x1 - xi) * e - 3)), int(max(0, (y1 - yi) * e - 3)))
    b = (int(min(sortie - 1, (x2 - xi) * e + 3)),
         int(min(sortie - 1, (y2 - yi) * e + 3)))
    cv2.rectangle(img, a, b, (0, 0, 255), epaisseur)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return jpg.tobytes()


def gids_du_zip(chemin):
    if not chemin.exists():
        return set()
    with zipfile.ZipFile(chemin) as z:
        return {int(n.split("/")[0]) for n in z.namelist()
                if n.endswith("/meta.json")}


def construire_vues(chemins, pistes, chemin_zip, reglage):
    """Ajoute au zip les vues des pistes qui n'y sont pas encore."""
    k = reglage["k"]
    plan = {}          # sample -> [(gid, rôle, frame)]
    metas = {}
    for p in pistes:
        det = [f for f in p["frames"] if not f.get("interpolated")]
        idx = np.unique(np.linspace(0, len(det) - 1, k).round().astype(int))
        choix = [det[i] for i in idx]
        milieu = choix[len(choix) // 2]
        metas[p["group_id"]] = {"group_id": p["group_id"], "vues": choix,
                                "contexte": milieu, "vue_contexte":
                                len(choix) // 2 + 1}
        for j, f in enumerate(choix, 1):
            plan.setdefault(f["i"], []).append((p["group_id"], str(j), f))
        plan.setdefault(milieu["i"], []).append(
            (p["group_id"], "contexte", milieu))

    ticks = ticks_index(chemins["index"])
    timescale = timescale_index(chemins["index"])
    samples = sorted(plan)
    images = {gid: {} for gid in metas}
    t0 = time.time()
    for n, (pos, image) in enumerate(en_tache_de_fond(
            decode_frames(chemins["video"], [ticks[s] for s in samples]))):
        if image is None:
            continue
        for gid, role, f in plan[samples[pos]]:
            if role == "contexte":
                images[gid][role] = vue(image, f["bbox"], 0.5 * (
                    reglage["ctx_facteur"] - 1), reglage["ctx_min"],
                    reglage["ctx_sortie"], 3)
            else:
                images[gid][role] = vue(image, f["bbox"], reglage["marge"],
                                        reglage["cote_min"], reglage["sortie"], 2)
        if (n + 1) % 500 == 0:
            print(f"[i] {n + 1}/{len(samples)} frames décodées "
                  f"({time.time() - t0:.0f} s)", flush=True)

    # écriture atomique : copie du zip existant, ajout, renommage
    chemin_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin_zip.with_suffix(".zip.tmp")
    if chemin_zip.exists():
        shutil.copyfile(chemin_zip, tmp)
    with zipfile.ZipFile(tmp, "a", compression=zipfile.ZIP_STORED) as z:
        for gid, meta in metas.items():
            p = next(q for q in pistes if q["group_id"] == gid)
            base = ticks[meta["vues"][0]["i"]]
            vues = []
            for j, f in enumerate(meta["vues"], 1):
                if str(j) not in images[gid]:
                    continue
                z.writestr(f"{gid}/{j}.jpg", images[gid][str(j)])
                x1, y1, x2, y2 = f["bbox"]
                vues.append({"n": j, "i": f["i"], "bbox": f["bbox"],
                             "dt_s": round((ticks[f["i"]] - base) / timescale, 2),
                             "taille_px": [round(x2 - x1), round(y2 - y1)]})
            if "contexte" in images[gid]:
                z.writestr(f"{gid}/contexte.jpg", images[gid]["contexte"])
            z.writestr(f"{gid}/meta.json", json.dumps({
                "group_id": gid, "coarse": p["coarse"], "vues": vues,
                "contexte": "contexte" in images[gid],
                "vue_contexte": meta["vue_contexte"],
                "duree_s": round(p["t_max_s"] - p["t_min_s"], 2)}))
    os.replace(tmp, chemin_zip)
    print(f"[i] {len(metas)} pistes -> {chemin_zip} "
          f"({time.time() - t0:.0f} s)")


# --------------------------------------------------------------------------
# Questions

def schema_reponse(coarse, k, prompt):
    props = {}
    if prompt != "p1":
        props["appearance"] = {"type": "string", "maxLength": 300}
    props.update({
        "description": {"type": "string", "maxLength": 250},
        "same_object": {"enum": OUI_NON},
        "different_views": {"type": "array", "maxItems": k,
                            "items": {"type": "integer", "minimum": 1,
                                      "maximum": k}},
        "real_object": {"enum": OUI_NON},
    })
    if prompt != "p1":
        props["box_covers"] = {"enum": COUVERTURE}
    if coarse == "vehicle":
        props["fine_class"] = {"enum": [c for c, _ in CLASSES_VEHICULE]}
    if coarse == "vehicle" or prompt not in ("p1", "p2", "p3"):
        props["affiliation"] = {"enum": AFFILIATIONS}
    if prompt in ("p5", "p6", "p7"):  # entier ou partie avant « objet réel »
        ordre = ["appearance", "description", "same_object", "different_views",
                 "box_covers", "real_object", "fine_class", "affiliation"]
        props = {k: props[k] for k in ordre if k in props}
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


def texte_question(coarse, prompt):
    if prompt == "p4":
        return texte_question_p4(coarse)
    if prompt in ("p5", "p6", "p7"):
        return texte_question_p5(coarse)
    objet = "vehicle" if coarse == "vehicle" else "person"
    lignes = ["Answer with a JSON object:"]
    if prompt == "p1":
        lignes.append('- "description": one short sentence describing what is '
                      "inside the red rectangle in most views.")
    else:
        # décrire avant de nommer : en p1, le modèle partait du mot « person »
        lignes += [
            '- "appearance": describe only what is inside the red rectangle in '
            "the close-up views: shape, colours, size compared with nearby "
            "things, shadow. Do not name the object yet.",
            '- "description": one short sentence saying what the object in the '
            "red rectangle most likely is, given its appearance.",
        ]
    lignes += [
        '- "same_object": "yes" if every view shows the same physical object; '
        '"no" if in some views the red rectangle is on a different object '
        "(another vehicle or person, or background). Changes of scale, viewing "
        "angle, blur, lighting or partial occlusion of the same object are "
        'normal and do not count as a change of object; "unsure" if you '
        "cannot tell.",
        '- "different_views": the numbers of the views whose red rectangle is '
        "not on the object seen in most views (empty list if none).",
    ]
    if prompt == "p1":
        lignes.append(
            f'- "real_object": "yes" if the object in the red rectangle (in most '
            f'views) really is a {objet}; "no" if it is something else (sign, '
            "pole, post, shadow, bush, rock, part of a building, road marking, "
            'debris, ...); "unsure" if it cannot be decided.')
    else:
        condition = "only if" if prompt == "p2" else "if"
        nette = "clearly is" if prompt == "p2" else "is"
        lignes += [
            f'- "real_object": "yes" {condition} the object in the red rectangle (in '
            f"most views) {nette} a {objet} or a part of one; \"no\" if it is "
            "something else (road or traffic sign, pole, post, shadow, bush, "
            "rock, part of a building, road marking, debris, ...); \"unsure\" if "
            "it cannot be decided.",
            '- "box_covers": "whole" if the red rectangle covers the whole visible '
            'object; "part" if it covers only a part of a larger object whose '
            "rest is clearly visible just outside the rectangle (for a vehicle: "
            "only its cab, front, rear, cargo bed, trailer, roof or wheels; for a "
            'person: only the legs, an arm or the head); "unsure" if you cannot '
            "tell. An object partly hidden by trees, another object or the image "
            'border is "whole" when the rectangle covers all of its visible part.',
        ]
    if coarse == "vehicle":
        lignes.append('- "fine_class": the vehicle type, one of:')
        lignes += [f"    {c}: {d};" for c, d in CLASSES_VEHICULE]
        lignes.append("  If it is not a vehicle, still give the closest type: "
                      '"real_object" carries the rejection.')
        lignes.append('- "affiliation": "civil", "military" or "unknown".')
    return "\n".join(lignes)


def texte_question_p4(coarse):
    objet = "vehicle" if coarse == "vehicle" else "person"
    lignes = [
        "Answer with a JSON object:",
        '- "appearance": describe only what is inside the red rectangle in '
        "the close-up views: shape, colours, size compared with nearby "
        "things, shadow. Do not name the object yet.",
        '- "description": one short sentence saying what the object in the '
        "red rectangle most likely is, given its appearance.",
        '- "same_object": "yes" if every view shows the same physical object; '
        '"no" if in some views the red rectangle is on a different object '
        "(another vehicle or person, or background). Changes of scale, viewing "
        "angle, blur, lighting or partial occlusion of the same object are "
        'normal and do not count as a change of object; "unsure" if you '
        "cannot tell.",
        '- "different_views": the numbers of the views whose red rectangle is '
        "not on the object seen in most views (empty list if none).",
        f'- "real_object": "yes" if the object in the red rectangle (in most '
        f'views) is a {objet} or a part of one; "no" if it is something else, '
        f'for example {NON_OBJETS_P4[coarse]}; "unsure" if it cannot be '
        "decided.",
        '- "box_covers": "whole" if the red rectangle covers the whole visible '
        f'object; "part" if it covers only a part of a larger {objet} whose '
        "rest is clearly visible just outside the rectangle ("
        f'{PARTIES_P4[coarse]}); "unsure" if you cannot tell. An object partly '
        "hidden by trees, another object or the image border is \"whole\" "
        "when the rectangle covers all of its visible part.",
    ]
    if coarse == "vehicle":
        lignes.append('- "fine_class": the type of the whole vehicle (also when '
                      "the box covers only a part of it), one of:")
        lignes += [f"    {c}: {d};" for c, d in CLASSES_VEHICULE_P4]
        lignes += [
            "  Military paint (matte olive green, khaki, sand or camouflage) or "
            "military markings make a vehicle military whatever its shape: a "
            "military car or 4x4 is mil_other, not car.",
            "  If it is not a vehicle, still give the closest type: "
            '"real_object" carries the rejection.',
            '- "affiliation": "military" if the vehicle has military paint or '
            'markings or is a military type; "civil" if it is a civilian '
            'vehicle; "unknown" if you cannot tell.',
        ]
    else:
        lignes.append(
            '- "affiliation": "military" if the person wears a uniform, '
            "camouflage, a helmet or a plate carrier, or carries a weapon; "
            '"civil" if the clothing is visible and civilian; "unknown" if '
            "you cannot tell.")
    return "\n".join(lignes)


def texte_question_p5(coarse):
    objet = "vehicle" if coarse == "vehicle" else "person"
    lignes = [
        "Answer with a JSON object:",
        '- "appearance": describe only what is inside the red rectangle in '
        "the close-up views: shape, colours, size compared with nearby "
        "things, shadow. Do not name the object yet.",
        '- "description": one short sentence saying what the object in the '
        "red rectangle most likely is, given its appearance.",
        '- "same_object": "yes" if every view shows the same physical object; '
        '"no" if in some views the red rectangle is on a different object '
        "(another vehicle or person, or background). Changes of scale, viewing "
        "angle, blur, lighting or partial occlusion of the same object are "
        'normal and do not count as a change of object; "unsure" if you '
        "cannot tell.",
        '- "different_views": the numbers of the views whose red rectangle is '
        "not on the object seen in most views (empty list if none).",
        '- "box_covers": "whole" if the red rectangle covers the whole visible '
        f'object; "part" if it covers only a part of a larger {objet} whose '
        "rest is clearly visible just outside the rectangle "
        f'({PARTIES_P5[coarse]}); "unsure" if you cannot tell. An object partly '
        "hidden by trees, another object or the image border is \"whole\" "
        "when the rectangle covers all of its visible part.",
        f'- "real_object": "yes" if the object in the red rectangle (in most '
        f'views) is a {objet}, or a part of one: a rectangle on a part of a '
        f'{objet} is still "yes". "no" if it is something else '
        f'({NON_OBJETS_P5[coarse]}, ...); "unsure" if it cannot be decided.',
    ]
    if coarse == "vehicle":
        lignes.append('- "fine_class": the type of the whole vehicle (also when '
                      "the rectangle covers only a part of it), one of:")
        lignes += [f"    {c}: {d};" for c, d in CLASSES_VEHICULE_P5]
        lignes += [
            "  Military paint (matte olive green, khaki, sand or camouflage) or "
            "military markings make a vehicle military whatever its shape: a "
            "military car or 4x4 is mil_other, not car.",
            "  If it is not a vehicle, still give the closest type: "
            '"real_object" carries the rejection.',
            '- "affiliation": "military" if the vehicle has military paint or '
            'markings or is a military type; "civil" if it is a civilian '
            'vehicle; "unknown" if you cannot tell.',
        ]
    else:
        lignes.append(
            '- "affiliation": "military" if the person wears a uniform, '
            "camouflage, a helmet or a plate carrier, or carries a weapon; "
            '"civil" if none of these is visible; "unknown" only if the person '
            "is too small or blurred to see the clothing.")
    return "\n".join(lignes)


def taille_piste_m(telemetrie, meta):
    """Médiane des largeurs au sol (m) des vues de la piste, ou None."""
    largeurs = [largeur_sol_m(telemetrie[v["i"]], v["bbox"])
                for v in meta["vues"] if v["i"] in telemetrie]
    largeurs = [x for x in largeurs if x]
    return float(np.median(largeurs)) if largeurs else None


def phrase_taille(taille_m, prompt="p6"):
    arrondi = round(taille_m, 1) if taille_m < 3 else round(taille_m)
    if prompt == "p7":
        return (f"Scale: from the drone altitude and camera angle, the tracked "
                f"box is roughly {arrondi:g} m wide on the ground (a car is "
                "about 1.8 m wide and 4.5 m long, a truck about 2.5 m wide and "
                "10 m long). A box on a partly hidden vehicle or on a part of "
                "one is smaller than the vehicle. ")
    return (f"From the drone altitude and camera angle, the tracked box is "
            f"about {arrondi:g} m wide on the ground (rough estimate, can be "
            "off by half). For scale, seen from above: a person is about 0.5 m "
            "wide, a car about 1.8 m wide and 4.5 m long, a van 2 m by 5 m, a "
            "truck 2.5 m by 8 to 12 m, a bus 2.5 m by 12 m. The box can be "
            "smaller than the object when the object is partly hidden (trees, "
            "another vehicle, image border) or when the box covers only a part "
            "of it: use the size as a hint, never as the only reason to reject "
            "an object. ")


def construire_messages(z, gid, prompt, telemetrie=None):
    meta = json.loads(z.read(f"{gid}/meta.json"))
    coarse, vues = meta["coarse"], meta["vues"]
    objet = "vehicle" if coarse == "vehicle" else "person"
    b64 = lambda nom: ("data:image/jpeg;base64,"
                       + base64.b64encode(z.read(f"{gid}/{nom}")).decode())
    fin_intro = (f"Views 1-{len(vues)} are crops centred on the tracked box at "
                 "different times, in time order; the tracked box is the red "
                 "rectangle (drawn slightly larger than the box). The crops are "
                 "enlarged, so small objects look blurry.")
    if prompt == "p1":
        intro = (f"A tracker followed one {objet} through a drone video for "
                 f"{meta['duree_s']:.1f} s. ")
        legende_ctx = (f"Wider view of the scene at the time of view "
                       f"{meta['vue_contexte']} (same red rectangle):")
    else:
        # formulation neutre : le détecteur propose, il se trompe souvent
        intro = (f"An automatic detector flagged a possible {objet} in a drone "
                 f"video and a tracker followed that box for "
                 f"{meta['duree_s']:.1f} s. ")
        if prompt == "p2":
            intro += ("The detector is often wrong: it also fires on signs, "
                      "poles, shadows, bushes, parts of buildings and parts of "
                      "vehicles. ")
        legende_ctx = (f"Wider view around the object at the time of view "
                       f"{meta['vue_contexte']} (same red rectangle), only to "
                       "understand the surroundings; judge the object itself from "
                       "the close-up views:")
    if (prompt in PROMPTS_TAILLE and telemetrie is not None
            and (prompt != "p7" or coarse == "vehicle")):
        taille = taille_piste_m(telemetrie, meta)
        if taille is not None:
            fin_intro += " " + phrase_taille(taille, prompt).rstrip()
    contenu = [{"type": "text", "text": intro + fin_intro}]
    for v in vues:
        w, h = v["taille_px"]
        contenu.append({"type": "text", "text":
                        f"View {v['n']} (t = +{v['dt_s']:.1f} s, box {w}x{h} "
                        "px in the 3840x2160 frame):"})
        contenu.append({"type": "image_url",
                        "image_url": {"url": b64(f"{v['n']}.jpg")}})
    if meta["contexte"]:
        contenu.append({"type": "text", "text": legende_ctx})
        contenu.append({"type": "image_url",
                        "image_url": {"url": b64("contexte.jpg")}})
    contenu.append({"type": "text", "text": texte_question(coarse, prompt)})
    return ([{"role": "system", "content": SYSTEME},
             {"role": "user", "content": contenu}],
            schema_reponse(coarse, len(vues), prompt))


def _texte_jetons(lp):
    debuts, n = [], 0
    for t in lp:
        debuts.append(n)
        n += len(t["token"])
    return "".join(t["token"] for t in lp), debuts


def _debut_valeur(lp, champ):
    """(indice du jeton, décalage dans le jeton, position) du 1er caractère
    de la valeur chaîne de `champ` (dernière occurrence : la réflexion peut
    citer le champ avant la réponse)."""
    texte, debuts = _texte_jetons(lp)
    occ = list(re.finditer(r'"%s"\s*:\s*"' % re.escape(champ), texte))
    if not occ:
        return None
    pos = occ[-1].end()
    k = bisect.bisect_right(debuts, pos) - 1
    if k >= len(lp) or pos >= len(texte):
        return None
    return k, pos - debuts[k], pos, texte


def proba_options(lp, champ, options):
    """Distribution sur `options` au premier jeton de la valeur (logprobs
    bruts, top-k), renormalisée ; `masse` = part du top-k qui y correspond."""
    d = _debut_valeur(lp, champ)
    if d is None:
        return None
    k, decal, _pos, _texte = d
    prefixe = lp[k]["token"][:decal]
    masses = dict.fromkeys(options, 0.0)
    for alt in lp[k].get("top_logprobs") or []:
        s = alt["token"]
        if not s.startswith(prefixe):
            continue
        reste = s[decal:].strip().lower()
        if not reste:
            continue
        cands = [o for o in options if o.startswith(reste) or reste.startswith(o)]
        if len(cands) == 1:
            masses[cands[0]] += math.exp(alt["logprob"])
    total = sum(masses.values())
    if total <= 0:
        return None
    return {"p": {o: round(m / total, 4) for o, m in masses.items()},
            "masse": round(total, 4)}


def proba_valeur(lp, champ):
    """P(valeur choisie) : produit des probabilités de ses jetons."""
    d = _debut_valeur(lp, champ)
    if d is None:
        return None
    k, _decal, pos, texte = d
    fin = texte.find('"', pos)
    if fin < 0:
        return None
    _t, debuts = _texte_jetons(lp)
    s = 0.0
    for j in range(k, len(lp)):
        if debuts[j] >= fin:
            break
        s += lp[j]["logprob"]
    return round(math.exp(s), 4)


def interroger(serveur, modele, mode, messages, schema, seed, essais=3):
    import requests
    corps = {"model": modele, "messages": messages,
             "response_format": {"type": "json_schema", "json_schema": {
                 "name": "verification", "schema": schema, "strict": True}},
             "logprobs": True, "top_logprobs": 10, "seed": seed,
             **MODES[mode]}
    for essai in range(essais):
        try:
            t0 = time.time()
            r = requests.post(f"{serveur}/chat/completions", json=corps,
                              timeout=900)
            r.raise_for_status()
            return r.json(), time.time() - t0
        except Exception as e:  # réseau, 5xx : on réessaie
            if essai == essais - 1:
                raise
            print(f"[!] essai {essai + 1} : {e}", file=sys.stderr)
            time.sleep(5 * (essai + 1))


def depouiller(reponse, coarse):
    choix = reponse["choices"][0]
    msg = choix["message"]
    lp = (choix.get("logprobs") or {}).get("content") or []
    sortie = {"raisonnement": msg.get("reasoning_content") or msg.get("reasoning"),
              "fin": choix.get("finish_reason"),
              "jetons": reponse.get("usage")}
    brut = msg.get("content") or ""
    try:
        sortie["reponse"] = json.loads(brut)
    except ValueError:
        sortie["reponse"], sortie["brut"] = None, brut
        return sortie
    p = {"same_object": proba_options(lp, "same_object", OUI_NON),
         "real_object": proba_options(lp, "real_object", OUI_NON)}
    if "box_covers" in sortie["reponse"]:
        p["box_covers"] = proba_options(lp, "box_covers", COUVERTURE)
    if "affiliation" in sortie["reponse"]:
        p["affiliation"] = proba_options(lp, "affiliation", AFFILIATIONS)
    if coarse == "vehicle":
        p["fine_class"] = proba_valeur(lp, "fine_class")
    sortie["p"] = p
    return sortie


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vol", required=True)
    ap.add_argument("--racine", type=Path, default=RACINE)
    ap.add_argument("--pistes", type=Path, default=None,
                    help="jeu de pistes (défaut : le plus récent du vol)")
    ap.add_argument("--selection", choices=["connus", "pilote", "toutes"],
                    default="pilote",
                    help="connus : les cas relus (CAS_CONNUS) ; pilote : "
                    "+ strates au hasard ; toutes : tout le vol")
    ap.add_argument("--gid", type=int, nargs="+", default=None,
                    help="group_id précis (remplace --selection)")
    ap.add_argument("--min-len", type=int, default=5,
                    help="--selection toutes : détections minimum")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vues-seules", action="store_true",
                    help="préparer les vues sans interroger le serveur")
    ap.add_argument("--serveur", nargs="+", default=None,
                    help="http://localhost:<port>/v1 (tunnel SSH)")
    ap.add_argument("--mode", choices=sorted(MODES), default="rapide")
    ap.add_argument("--prompt", choices=sorted(VUES_PROMPT), default="p2")
    ap.add_argument("--paralleles", type=int, default=16,
                    help="requêtes simultanées par serveur")
    ap.add_argument("--limite", type=int, default=None,
                    help="au plus N pistes interrogées (essai)")
    args = ap.parse_args()

    chemins = chemins_vol(args.vol, args.racine)
    jeu = args.pistes or dernier_jeu(chemins["pistes"], args.vol)
    pistes = [json.loads(l) for l in open(jeu, encoding="utf-8")]
    if args.gid:
        categories = {g: ["demande"] for g in args.gid}
    elif args.selection == "connus":
        categories = {g: ["connu"] for g in {**CAS_CONNUS.get(args.vol, {}),
                                              **PARTIES.get(args.vol, {})}}
    elif args.selection == "pilote":
        categories = selection_pilote(args.vol, pistes, args.seed)
    else:
        categories = {p["group_id"]: ["toutes"] for p in pistes
                      if p["n_detected"] >= args.min_len}
    retenues = [p for p in pistes if p["group_id"] in categories]
    print(f"[i] {jeu.name} : {len(retenues)} pistes retenues")

    dossier = chemins["pistes"] / "vlm"
    reglage = VUES_PROMPT[args.prompt]
    chemin_zip = dossier / f"{jeu.stem}_vues-{empreinte_vues(reglage)}.zip"
    manquantes = [p for p in retenues if p["group_id"] not in gids_du_zip(chemin_zip)]
    if manquantes:
        print(f"[i] vues à préparer : {len(manquantes)} pistes")
        construire_vues(chemins, manquantes, chemin_zip, reglage)
    if args.vues_seules:
        return 0
    if not args.serveur:
        raise SystemExit("[!] --serveur requis (ou --vues-seules)")

    import requests
    # plusieurs serveurs (un par GPU) : pistes réparties par group_id
    serveurs = [u.rstrip("/") for u in args.serveur]
    modeles = {requests.get(f"{u}/models", timeout=30).json()["data"][0]["id"]
               for u in serveurs}
    if len(modeles) != 1:
        raise SystemExit(f"[!] modèles différents selon les serveurs : {modeles}")
    modele = modeles.pop()
    sortie = dossier / f"{args.vol}_vlm_{args.mode}_{args.prompt}.jsonl"
    faites = set()
    if sortie.exists():
        for l in open(sortie, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("erreur"):
                faites.add(r["group_id"])
    a_faire = [p for p in retenues if p["group_id"] not in faites]
    if args.limite:
        a_faire = a_faire[:args.limite]
    print(f"[i] {modele}, mode {args.mode} : {len(a_faire)} pistes à "
          f"interroger ({len(faites)} déjà faites) -> {sortie}")

    verrou = threading.Lock()
    zf = zipfile.ZipFile(chemin_zip)
    zverrou = threading.Lock()
    connus = CAS_CONNUS.get(args.vol, {})
    parties = PARTIES.get(args.vol, {})
    contenues = contenance(pistes)
    telemetrie = telemetrie_index(chemins["index"])

    def une(p):
        gid = p["group_id"]
        with zverrou:
            messages, schema = construire_messages(zf, gid, args.prompt,
                                                   telemetrie)
        ligne = {"vol": args.vol, "group_id": gid, "coarse": p["coarse"],
                 "label_sam3": p["label"], "labels_sam3": p["labels"],
                 "n_detected": p["n_detected"],
                 "size_median_px": p["size_median_px"],
                 "score_mean": p["score_mean"],
                 "duree_s": round(p["t_max_s"] - p["t_min_s"], 2),
                 "selection": categories[gid], "mode": args.mode,
                 "prompt": args.prompt, "modele": modele,
                 "vues": chemin_zip.name, "contenue": contenues.get(gid)}
        if gid in connus:
            objet, pure, note = connus[gid]
            ligne["attendu"] = {"real_object": objet, "pure": pure, "note": note}
        if gid in parties:
            ligne["attendu"] = {"real_object": "oui", "pure": None,
                                "partie": True, "note": parties[gid]}
        try:
            rep, duree = interroger(serveurs[gid % len(serveurs)], modele,
                                    args.mode, messages,
                                    schema, seed=gid)
            ligne.update(depouiller(rep, p["coarse"]))
            ligne["latence_s"] = round(duree, 2)
        except Exception as e:
            ligne["erreur"] = repr(e)
        return ligne

    t0, n, erreurs = time.time(), 0, 0
    with open(sortie, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(args.paralleles * len(serveurs)) as pool:
        for fut in as_completed([pool.submit(une, p) for p in a_faire]):
            ligne = fut.result()
            n += 1
            erreurs += bool(ligne.get("erreur"))
            with verrou:
                fh.write(json.dumps(ligne, ensure_ascii=False) + "\n")
                fh.flush()
            if n % 20 == 0 or n == len(a_faire):
                print(f"[i] {n}/{len(a_faire)} ({erreurs} erreurs, "
                      f"{time.time() - t0:.0f} s)", flush=True)
    zf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
