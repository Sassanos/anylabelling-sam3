#!/usr/bin/env python3
"""Évaluer un prompt du VLM contre la revue humaine (revue-pistes.py).

Pistes : celles de la revue, annotées, des vols de jour seulement
(VOLS_JOUR). Deux parts fixes, par rang : « reglage » (2/3), où l'on lit les
erreurs pour retoucher le prompt, et « controle » (1/3, rang multiple de 3),
qu'on ne lit qu'à la fin pour savoir si le réglage se généralise.

Réponses du VLM dans <REVUE>/vlm/<mode>_<prompt>.jsonl (reprise piste par
piste), mêmes vues et mêmes messages que verifie-pistes-vlm.py.

    PY=X-AnyLabeling-Server/.venv/bin/python
    $PY evalue-prompts-vlm.py --serveur http://localhost:13863/v1 --prompt p4
    $PY evalue-prompts-vlm.py --score p3 p4                 # sans serveur
    $PY evalue-prompts-vlm.py --score p4 --erreurs          # désaccords (réglage)
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pistes_io import VOLS_JOUR, chemins_vol, telemetrie_index  # noqa: E402


def _module(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, HERE / fichier)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


revue = _module("revue_pistes", "revue-pistes.py")
vlm = revue.vlm
SORTIE = revue.REVUE / "vlm"
MILITAIRES = {"mil_tank", "mil_apc_ifv", "mil_truck", "mil_other"}


def charger_cas():
    """[(piste de la sélection, annotation humaine)] des vols de jour."""
    sel = revue.charger_selection()["pistes"]
    humain = revue.lire_annotations()
    cas = []
    for p in sel:
        h = humain.get(f"{p['vol']}/{p['group_id']}")
        if p["vol"] in VOLS_JOUR and h and h.get("real_object"):
            p = dict(p, part="controle" if p["rang"] % 3 == 0 else "reglage")
            cas.append((p, h))
    return cas


def fichier(mode, prompt):
    return SORTIE / f"{mode}_{prompt}.jsonl"


def lire_reponses(mode, prompt):
    f = fichier(mode, prompt)
    rep = {}
    if f.exists():
        for ligne in open(f, encoding="utf-8"):
            r = json.loads(ligne)
            if not r.get("erreur") and r.get("reponse"):
                rep[f"{r['vol']}/{r['group_id']}"] = r
    return rep


def interroger_tout(args, cas):
    import requests
    serveur = args.serveur.rstrip("/")
    modele = requests.get(f"{serveur}/models", timeout=30).json()["data"][0]["id"]
    faites = lire_reponses(args.mode, args.prompt)
    a_faire = [(p, h) for p, h in cas if f"{p['vol']}/{p['group_id']}" not in faites]
    print(f"[i] {modele}, {args.mode} {args.prompt} : {len(a_faire)} pistes à "
          f"interroger ({len(faites)} déjà faites)", flush=True)
    zips, telem = {}, {}
    zverrou, verrou = threading.Lock(), threading.Lock()

    def une(p):
        with zverrou:
            if p["vol"] not in zips:
                zips[p["vol"]] = zipfile.ZipFile(revue.chemin_zip(p["vol"], p["jeu"]))
                telem[p["vol"]] = telemetrie_index(chemins_vol(p["vol"])["index"])
            messages, schema = vlm.construire_messages(
                zips[p["vol"]], p["group_id"], args.prompt, telem[p["vol"]])
        ligne = {"vol": p["vol"], "group_id": p["group_id"], "rang": p["rang"],
                 "coarse": p["coarse"], "label_sam3": p["label"],
                 "mode": args.mode, "prompt": args.prompt, "modele": modele}
        try:
            r, duree = vlm.interroger(serveur, modele, args.mode, messages, schema,
                                      seed=p["group_id"])
            ligne.update(vlm.depouiller(r, p["coarse"]))
            ligne["latence_s"] = round(duree, 2)
        except Exception as e:
            ligne["erreur"] = repr(e)
        return ligne

    SORTIE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(fichier(args.mode, args.prompt), "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(args.paralleles) as pool:
        for n, fut in enumerate(as_completed(
                [pool.submit(une, p) for p, _h in a_faire]), 1):
            ligne = fut.result()
            with verrou:
                fh.write(json.dumps(ligne, ensure_ascii=False) + "\n")
                fh.flush()
            if n % 20 == 0 or n == len(a_faire):
                print(f"[i] {n}/{len(a_faire)} ({time.time() - t0:.0f} s)", flush=True)


# --------------------------------------------------------------------------
# Scores

def mesures(cas, rep):
    """{nom: (réussis, total)} ; les réponses humaines « incertain » ou vides
    ne comptent pas."""
    m = collections.defaultdict(lambda: [0, 0])

    def compte(nom, ok):
        m[nom][0] += bool(ok)
        m[nom][1] += 1

    for p, h in cas:
        v = rep.get(f"{p['vol']}/{p['group_id']}")
        if v is None:
            continue
        r = v["reponse"]
        objet = "véh" if p["coarse"] == "vehicle" else "pers"
        if h["real_object"] == "yes":
            compte(f"vrais objets gardés ({objet})", r["real_object"] != "no")
        elif h["real_object"] == "no":
            compte(f"faux positifs rejetés ({objet})", r["real_object"] == "no")
        if h["real_object"] != "yes":
            continue
        if h.get("same_object") == "no":
            compte("pistes impures trouvées", r["same_object"] == "no")
        elif h.get("same_object") == "yes":
            compte("pistes pures gardées", r["same_object"] != "no")
        if h.get("box_covers") == "part":
            compte("parties trouvées", r.get("box_covers") == "part")
        elif h.get("box_covers") == "whole":
            compte("objets entiers gardés", r.get("box_covers") != "part")
        if p["coarse"] == "vehicle" and h.get("fine_class"):
            compte("classe fine exacte", r["fine_class"] == h["fine_class"])
            compte("famille civil/militaire",
                   (r["fine_class"] in MILITAIRES) == (h["fine_class"] in MILITAIRES))
        if h.get("affiliation") in ("civil", "military") and "affiliation" in r:
            compte(f"affiliation ({objet})", r["affiliation"] == h["affiliation"])
    return m


def mesures_sam3(cas):
    m = collections.defaultdict(lambda: [0, 0])
    for p, h in cas:
        if p["coarse"] == "vehicle" and h["real_object"] == "yes" and h.get("fine_class"):
            for nom, ok in (("classe fine exacte", p["label"] == h["fine_class"]),
                            ("famille civil/militaire", (p["label"] in MILITAIRES)
                             == (h["fine_class"] in MILITAIRES))):
                m[nom][0] += ok
                m[nom][1] += 1
    return m


def tableau(cas, mode, prompts):
    colonnes = [("SAM 3", mesures_sam3(cas))]
    colonnes += [(p, mesures(cas, lire_reponses(mode, p))) for p in prompts]
    noms = []
    for _t, m in colonnes[1:] + colonnes[:1]:
        noms += [n for n in m if n not in noms]
    print(f"{'':32s}" + "".join(f"{t:>12s}" for t, _m in colonnes))
    for n in noms:
        cellules = []
        for _t, m in colonnes:
            ok, tot = m.get(n, (0, 0))
            cellules.append(f"{ok:>5d}/{tot:<5d}" if tot else f"{'—':>12s}")
        print(f"{n:32s}" + "".join(f"{c:>12s}" for c in cellules))


def erreurs(cas, mode, prompt):
    rep = lire_reponses(mode, prompt)
    for p, h in sorted(cas, key=lambda c: c[0]["rang"]):
        v = rep.get(f"{p['vol']}/{p['group_id']}")
        if v is None:
            continue
        r = v["reponse"]
        ecarts = []
        if h["real_object"] in ("yes", "no") and (r["real_object"] == "no") != (
                h["real_object"] == "no"):
            ecarts.append(f"réel {h['real_object']}→{r['real_object']}")
        if h["real_object"] == "yes":
            if h.get("same_object") in ("yes", "no") and (
                    r["same_object"] == "no") != (h["same_object"] == "no"):
                ecarts.append(f"pure {h['same_object']}→{r['same_object']}")
            if h.get("box_covers") in ("whole", "part") and (
                    r.get("box_covers") == "part") != (h["box_covers"] == "part"):
                ecarts.append(f"couvre {h['box_covers']}→{r.get('box_covers')}")
            if h.get("fine_class") and r.get("fine_class") and r["fine_class"] != h["fine_class"]:
                ecarts.append(f"classe {h['fine_class']}→{r['fine_class']}")
            if h.get("affiliation") in ("civil", "military") and r.get(
                    "affiliation") not in (None, h["affiliation"]):
                ecarts.append(f"affil {h['affiliation']}→{r['affiliation']}")
        if ecarts:
            print(f"#{p['rang']:<4d} {p['vol']}/{p['group_id']:<6d} {p['coarse']:7s} "
                  f"SAM3 {p['label']:12s} {', '.join(ecarts)}")
            print(f"      VLM : {r.get('description', '')}")
            if h.get("comment") or h.get("nature"):
                print(f"      humain : {h.get('nature') or ''} {h.get('comment') or ''}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serveur", default=None)
    ap.add_argument("--prompt", choices=sorted(vlm.VUES_PROMPT), default=None)
    ap.add_argument("--mode", choices=sorted(vlm.MODES), default="rapide")
    ap.add_argument("--paralleles", type=int, default=16)
    ap.add_argument("--score", nargs="+", default=None,
                    help="prompts à comparer (réponses déjà en fichier)")
    ap.add_argument("--part", choices=["reglage", "controle", "tout"],
                    default="reglage")
    ap.add_argument("--erreurs", action="store_true",
                    help="lister les désaccords du premier prompt de --score")
    args = ap.parse_args()

    cas = charger_cas()
    if args.serveur:
        if not args.prompt:
            raise SystemExit("[!] --prompt requis avec --serveur")
        interroger_tout(args, cas)
    prompts = args.score or ([args.prompt] if args.prompt else [])
    if not prompts:
        return
    choisis = [c for c in cas if args.part == "tout" or c[0]["part"] == args.part]
    print(f"\n[i] part {args.part} : {len(choisis)} pistes annotées de jour "
          f"({sum(c[0]['coarse'] == 'vehicle' for c in choisis)} véhicules)")
    tableau(choisis, args.mode, prompts)
    if args.erreurs:
        print()
        erreurs(choisis, args.mode, prompts[0])


if __name__ == "__main__":
    main()
