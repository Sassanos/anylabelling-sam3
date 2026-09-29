#!/usr/bin/env python3
"""Revue humaine de pistes, vérité terrain pour régler le prompt du VLM.

200 pistes variées tirées des 19 vols de CAMPAGNE3, montrées avec les vues
mêmes que reçoit le VLM (réglage p3 : 8 vues + vue large) dans une page
locale. On y répond aux questions posées au VLM (objet réel, pureté, entier
ou partie, classe fine, affiliation), plus la nature des faux positifs et la
difficulté à juger sur ces vues. Réponses rangées sous les noms de champs du
VLM (yes/no/unsure, whole/part, fine_class...) pour les comparer directement.

Trois étapes :
1. selection : tirage diversifié -> <REVUE>/selection.json ;
2. vues : vues VLM des pistes tirées, ajoutées au zip de chaque vol, celui
   que verifie-pistes-vlm.py réutilisera
   (<dossier_pistes(vol)>/vlm/<jeu>_vues-<empreinte>.zip) ;
3. serveur : page sur http://127.0.0.1:8765, chaque réponse ajoutée à
   <REVUE>/annotations.jsonl (la dernière ligne d'une piste fait foi).

    PY=X-AnyLabeling-Server/.venv/bin/python
    $PY revue-pistes.py selection
    $PY revue-pistes.py vues
    $PY revue-pistes.py serveur
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import sys
import threading
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pistes_io import RACINE, chemins_vol, dernier_jeu  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "verifie_pistes_vlm", HERE / "verifie-pistes-vlm.py")
vlm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vlm)

REVUE = Path("/home/cbarbier/Documents/Geolocalisation/Datasets/real/CAMPAGNE3"
             "/revue-vlm")
PROMPT_VUES = "p3"

# Strates remplies dans cet ordre, une piste ne sert qu'une fois. Les deux
# premières visent des cas que le VLM gère mal (README, pilote p3) ; les
# autres suivent le label SAM 3, dont les rares (bus, mil_apc_ifv) sont
# sur-représentés exprès : on cherche des échecs, pas des taux.
STRATES = [
    ("partie", 7, lambda t: t["coarse"] == "vehicle" and t["contenue"] >= 0.8),
    ("zoom", 5, lambda t: t["echelle"] >= 3),
    ("person", 55, lambda t: t["label"] == "person"),
    ("car", 40, lambda t: t["label"] == "car"),
    ("van", 18, lambda t: t["label"] == "van"),
    ("truck", 18, lambda t: t["label"] == "truck"),
    ("bus", 8, lambda t: t["label"] == "bus"),
    ("motorcycle", 10, lambda t: t["label"] == "motorcycle"),
    ("mil_tank", 20, lambda t: t["label"] == "mil_tank"),
    ("mil_truck", 12, lambda t: t["label"] == "mil_truck"),
    ("mil_apc_ifv", 7, lambda t: t["label"] == "mil_apc_ifv"),
]
# Bornes des classes de diversité (px du plus grand côté, détections, score).
BORNES = {"taille": [15, 25, 45, 100], "longueur": [10, 30, 100],
          "score": [0.4, 0.6, 0.8]}


def vols():
    return sorted(p.name for p in (RACINE / "annots").iterdir() if p.is_dir())


def deja_vues_du_vlm(dossier):
    """group_id déjà passés au VLM (réglage des prompts) : exclus."""
    gids = set()
    for f in (dossier / "vlm").glob("*_vlm_*.jsonl"):
        for ligne in open(f, encoding="utf-8"):
            gids.add(json.loads(ligne)["group_id"])
    return gids


def traits(vol, jeu, p, cont):
    det = [f for f in p["frames"] if not f.get("interpolated")]
    cotes = np.array([max(b[2] - b[0], b[3] - b[1])
                      for b in (f["bbox"] for f in det)])
    c = cont.get(p["group_id"]) or {}
    t = {"vol": vol, "group_id": p["group_id"], "jeu": jeu.name,
         "coarse": p["coarse"], "label": p["label"], "labels": p["labels"],
         "n_detected": p["n_detected"], "size_median_px": p["size_median_px"],
         "score_mean": p["score_mean"],
         "duree_s": round(p["t_max_s"] - p["t_min_s"], 2),
         "contenue": c.get("frac", 0.0), "par": c.get("par"),
         "echelle": round(float(np.percentile(cotes, 90)
                                / max(np.percentile(cotes, 10), 1.0)), 2)}
    for nom, x in (("taille", t["size_median_px"]),
                   ("longueur", t["n_detected"]), ("score", t["score_mean"])):
        t["b_" + nom] = int(np.searchsorted(BORNES[nom], x, side="right"))
    return t


def tirer(cands, quota, rnd, par_vol):
    """Tirage glouton : chaque piste minimise les doublons de vol, de taille,
    de longueur et de score parmi celles déjà prises dans la strate (ex æquo
    départagés au hasard par le mélange initial)."""
    cands = list(cands)
    rnd.shuffle(cands)
    n = collections.Counter()
    choisis = []
    for _ in range(min(quota, len(cands))):
        t = min(cands, key=lambda t: (
            2 * n["vol", t["vol"]] + n["taille", t["b_taille"]]
            + n["longueur", t["b_longueur"]] + n["score", t["b_score"]]
            + 0.5 * par_vol[t["vol"]]))
        cands.remove(t)
        choisis.append(t)
        par_vol[t["vol"]] += 1
        for nom in ("taille", "longueur", "score"):
            n[nom, t["b_" + nom]] += 1
        n["vol", t["vol"]] += 1
    return choisis


def cmd_selection(args):
    tous = []
    for vol in vols():
        dossier = chemins_vol(vol)["pistes"]
        jeu = dernier_jeu(dossier, vol)
        pistes = [json.loads(l) for l in open(jeu, encoding="utf-8")]
        exclues = deja_vues_du_vlm(dossier)
        cont = vlm.contenance(pistes)
        tous += [traits(vol, jeu, p, cont) for p in pistes
                 if p["group_id"] not in exclues]
        print(f"[i] {vol} : {len(pistes)} pistes, {len(exclues)} déjà vues "
              f"du VLM exclues", flush=True)
    rnd = random.Random(args.graine)
    pris, par_vol, strates = set(), collections.Counter(), {}
    selection = []
    for nom, quota, filtre in STRATES:
        cands = [t for t in tous if filtre(t) and (t["vol"], t["group_id"]) not in pris]
        strates[nom] = {"quota": quota, "population": dict(
            collections.Counter(t["vol"] for t in cands))}
        for t in tirer(cands, quota, rnd, par_vol):
            pris.add((t["vol"], t["group_id"]))
            selection.append({**{k: v for k, v in t.items()
                                 if not k.startswith("b_")}, "strate": nom})
        print(f"[i] {nom:12s} {quota:3d} sur {len(cands)} candidates")
    rnd.shuffle(selection)
    for rang, t in enumerate(selection, 1):
        t["rang"] = rang
    REVUE.mkdir(parents=True, exist_ok=True)
    sortie = REVUE / "selection.json"
    if sortie.exists() and not args.force:
        raise SystemExit(f"[!] {sortie} existe déjà (--force pour le remplacer)")
    sortie.write_text(json.dumps({
        "date": datetime.now().isoformat(timespec="seconds"),
        "graine": args.graine, "prompt_vues": PROMPT_VUES,
        "strates": strates, "pistes": selection}, ensure_ascii=False, indent=1))
    print(f"[i] {len(selection)} pistes, "
          f"{len({t['vol'] for t in selection})} vols -> {sortie}")


def charger_selection():
    return json.loads((REVUE / "selection.json").read_text())


def chemin_zip(vol, jeu_nom):
    dossier = chemins_vol(vol)["pistes"]
    stem = Path(jeu_nom).stem
    return dossier / "vlm" / (f"{stem}_vues-"
                              f"{vlm.empreinte_vues(vlm.VUES_PROMPT[PROMPT_VUES])}.zip")


def cmd_vues(_args):
    sel = charger_selection()
    par_vol = collections.defaultdict(list)
    for t in sel["pistes"]:
        par_vol[t["vol"], t["jeu"]].append(t["group_id"])
    for (vol, jeu_nom), gids in sorted(par_vol.items()):
        chemins = chemins_vol(vol)
        z = chemin_zip(vol, jeu_nom)
        manquants = set(gids) - vlm.gids_du_zip(z)
        if not manquants:
            print(f"[i] {vol} : {len(gids)} pistes, vues déjà prêtes")
            continue
        pistes = [p for p in (json.loads(l) for l in
                              open(chemins["pistes"] / jeu_nom, encoding="utf-8"))
                  if p["group_id"] in manquants]
        print(f"[i] {vol} : vues de {len(pistes)} pistes", flush=True)
        vlm.construire_vues(chemins, pistes, z, vlm.VUES_PROMPT[PROMPT_VUES])


# --------------------------------------------------------------------------
# Serveur

def lire_annotations():
    faites = {}
    f = REVUE / "annotations.jsonl"
    if f.exists():
        for ligne in open(f, encoding="utf-8"):
            r = json.loads(ligne)
            faites[f"{r['vol']}/{r['group_id']}"] = r
    return faites


def cmd_serveur(args):
    sel = charger_selection()
    zips = {}
    for t in sel["pistes"]:
        zips.setdefault(t["vol"], zipfile.ZipFile(chemin_zip(t["vol"], t["jeu"])))
    pistes = []
    for t in sel["pistes"]:
        meta = json.loads(zips[t["vol"]].read(f"{t['group_id']}/meta.json"))
        pistes.append({**t, "vues": meta["vues"], "contexte": meta["contexte"],
                       "vue_contexte": meta["vue_contexte"]})
    page = (HERE / "revue-pistes.html").read_bytes()
    verrou = threading.Lock()
    journal = REVUE / "annotations.jsonl"

    class Gestion(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def envoyer(self, code, corps, type_):
            self.send_response(code)
            self.send_header("Content-Type", type_)
            self.send_header("Content-Length", str(len(corps)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corps)

        def do_GET(self):
            chemin = unquote(self.path.split("?")[0])
            if chemin == "/":
                return self.envoyer(200, page, "text/html; charset=utf-8")
            if chemin == "/api/pistes":
                with verrou:
                    faites = lire_annotations()
                corps = json.dumps({"pistes": pistes, "annotations": faites,
                                    "classes": vlm.CLASSES_VEHICULE},
                                   ensure_ascii=False).encode()
                return self.envoyer(200, corps, "application/json")
            morceaux = chemin.strip("/").split("/")
            if len(morceaux) == 4 and morceaux[0] == "img" and morceaux[1] in zips:
                _, vol, gid, nom = morceaux
                try:
                    with verrou:
                        data = zips[vol].read(f"{int(gid)}/{nom}")
                except (KeyError, ValueError):
                    return self.envoyer(404, b"", "text/plain")
                return self.envoyer(200, data, "image/jpeg")
            self.envoyer(404, b"", "text/plain")

        def do_POST(self):
            if self.path != "/api/annotation":
                return self.envoyer(404, b"", "text/plain")
            r = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            r["date"] = datetime.now().isoformat(timespec="seconds")
            with verrou, open(journal, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            self.envoyer(200, b"{}", "application/json")

    serveur = ThreadingHTTPServer(("127.0.0.1", args.port), Gestion)
    print(f"[i] {len(pistes)} pistes, {len(lire_annotations())} déjà annotées "
          f"-> http://127.0.0.1:{args.port}  (réponses : {journal})", flush=True)
    serveur.serve_forever()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sous = ap.add_subparsers(dest="cmd", required=True)
    s = sous.add_parser("selection")
    s.add_argument("--graine", type=int, default=0)
    s.add_argument("--force", action="store_true")
    sous.add_parser("vues")
    s = sous.add_parser("serveur")
    s.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    {"selection": cmd_selection, "vues": cmd_vues, "serveur": cmd_serveur}[args.cmd](args)


if __name__ == "__main__":
    main()
