#!/usr/bin/env python3
"""Analyse hors ligne les sorties de mesure-vlm.py.

Pour chaque classe annotee : exactitude par decoupage (par modalite et par
tranche de taille), exactitude par piste apres vote, erreurs dominantes,
couverture a seuil de confiance, et repartition des sous-classes civiles
(sans verite terrain, a relire). Plusieurs runs se comparent ligne a ligne.

    ./vlm-par-classe.py runs/vlm/v0-base.json [runs/vlm/v0-autre.json ...]
    ./vlm-par-classe.py --prefixe B_ runs/vlm/v0-base.json   # sequences B seules
    ./vlm-par-classe.py --prefixe B_ --exclure 103746,105526 run.json

Une reponse compte comme juste quand le choix gagnant est dans l'ensemble
attendu : Griffon -> griffon, VAB -> vab, GBC -> gbc, Vehicule Civil -> toute
sous-classe civile, Person -> pas_vehicule. La colonne « famille » accepte
toute classe militaire pour un militaire. Pour les jeux COCO, la
sous-classe exacte est attendue (car -> voiture ou pickup, van -> camionnette,
truck/freight_car/Trailer -> camion...) et la colonne « famille » accepte
toute sous-classe civile.
"""
import collections
import importlib.util
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rappel_par_taille", os.path.join(HERE, "rappel-par-taille.py")
)
_rpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rpt)
BINS = _rpt.BINS

CIVILS = {"voiture", "pickup", "camionnette", "camion", "bus", "deux_roues",
          "engin"}
MILITAIRES = {"griffon", "vab", "gbc", "vt4", "masstech"}
ATTENDU = {
    "Griffon": {"griffon"},
    "VAB": {"vab"},
    "GBC": {"gbc"},
    "Véhicule Civil": CIVILS,
    "Person": {"pas_vehicule"},
    # Jeux COCO publics (DroneVehicle, AU-AIR) : sous-classes civiles.
    "car": {"voiture", "pickup"}, "Car": {"voiture", "pickup"},
    "van": {"camionnette"}, "Van": {"camionnette"},
    "truck": {"camion"}, "Truck": {"camion"},
    "freight_car": {"camion"}, "Trailer": {"camion"},
    "bus": {"bus"}, "Bus": {"bus"},
    "Motorbike": {"deux_roues"}, "Bicycle": {"deux_roues"},
    "Human": {"pas_vehicule"},
}
# Ce qui compte comme « bonne famille » : un vehicule civil range dans la
# mauvaise sous-classe reste civil, un Griffon pris pour un VAB reste militaire.
FAMILLE = {
    k: CIVILS if v <= CIVILS else MILITAIRES if v <= MILITAIRES else v
    for k, v in ATTENDU.items()
}


def tranche(side):
    for lo, hi in BINS:
        if lo <= side < hi:
            return f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"


def pct(n, d):
    return f"{100.0 * n / d:5.1f} % ({n}/{d})" if d else "      —"


def report(path, prefixe=None, exclure=()):
    run = json.load(open(path))
    keys = [c[0] for c in run["meta"]["choices"]]
    items = [s for s in run["items"] if "probs" in s
             and (prefixe is None or s["seq"].startswith(prefixe))
             and not any(e in s["seq"] for e in exclure)]
    if not items:
        print(f"\n######## {os.path.basename(path)} : aucun decoupage")
        return
    meta = run["meta"]
    a = meta["args"]
    print(f"\n######## {os.path.basename(path)}"
          + (f" (sequences {prefixe}*)" if prefixe else "")
          + (f" sans {','.join(exclure)}" if exclure else ""))
    print(f"question {a.get('question', 'complet')}, "
          f"marge {a['margin']} (min {a['margin_min']} px), cote "
          f"{a['min_side']}-{a['max_side']} px, cadre "
          f"{'non' if a['no_marker'] else 'oui'}, {a['model']} {a['dtype']}")
    ms = sorted(s["ms"] for s in items)
    print(f"{len(items)} decoupages, latence mediane {ms[len(ms) // 2]:.0f} ms"
          f", p90 {ms[int(len(ms) * 0.9)]:.0f} ms, crete "
          f"{meta.get('peak_vram_gib', '?')} Gio, masse lettres mediane "
          f"{statistics.median(s['letter_mass'] for s in items):.3f}")

    binaire = "vehicule" in keys

    def attendu(label, table):
        # En question binaire, tout vehicule annote attend « vehicule ».
        if binaire and ATTENDU[label] != {"pas_vehicule"}:
            return {"vehicule"}
        return table[label]

    for s in items:
        s["top"] = keys[max(range(len(keys)), key=lambda i: s["probs"][i])]
        s["pmax"] = max(s["probs"])
        s["ok"] = s["top"] in attendu(s["label"], ATTENDU)
        s["ok_famille"] = s["top"] in attendu(s["label"], FAMILLE)

    labels = [l for l in ATTENDU if any(s["label"] == l for s in items)]

    print("\n== Par decoupage")
    for label in labels:
        sub = [s for s in items if s["label"] == label]
        line = f"  {label:15} {pct(sum(s['ok'] for s in sub), len(sub))}"
        for mod in ("RGB", "IR"):
            m = [s for s in sub if s["modality"] == mod]
            line += f"   {mod} {pct(sum(s['ok'] for s in m), len(m))}"
        if attendu(label, FAMILLE) != attendu(label, ATTENDU):
            line += f"   famille {pct(sum(s['ok_famille'] for s in sub), len(sub))}"
        print(line)

    print("\n== Par tranche de cote min (px natifs), toutes modalites")
    header = "  " + " " * 15 + "".join(f"{tranche(lo):>18}" for lo, _ in BINS)
    print(header)
    for label in labels:
        row = f"  {label:15}"
        for lo, hi in BINS:
            m = [s for s in items
                 if s["label"] == label and lo <= s["side_min"] < hi]
            ok = sum(s["ok"] for s in m)
            row += f"{(f'{100 * ok / len(m):.0f} % /{len(m)}' if m else '—'):>18}"
        print(row)

    print("\n== Par piste (vote : somme des log-probas)")
    for label in labels:
        tracks = collections.defaultdict(list)
        for s in items:
            if s["label"] == label:
                tracks[json.dumps(s["track"])].append(s)
        line = f"  {label:15}"
        for mod in ("RGB", "IR"):
            ok = n = 0
            for crops in tracks.values():
                if crops[0]["modality"] != mod or len(crops) < 2:
                    continue
                score = [sum(math.log(max(c["probs"][i], 1e-9))
                             for c in crops) for i in range(len(keys))]
                winner = keys[max(range(len(keys)), key=score.__getitem__)]
                ok += winner in attendu(label, ATTENDU)
                n += 1
            line += f"   {mod} {pct(ok, n)}"
        print(line + "   (pistes d'au moins 2 decoupages)")

    print("\n== Reponses donnees (toutes, par classe annotee)")
    for label in labels:
        c = collections.Counter(s["top"] for s in items if s["label"] == label)
        total = sum(c.values())
        print(f"  {label:15} " + ", ".join(
            f"{k} {100 * v / total:.0f} %" for k, v in c.most_common(6)))

    print("\n== Couverture a seuil (decoupages gardes, exactitude des gardes)")
    for label in labels:
        sub = [s for s in items if s["label"] == label]
        line = f"  {label:15}"
        for t in (0.5, 0.7, 0.9):
            kept = [s for s in sub if s["pmax"] >= t]
            ok = sum(s["ok"] for s in kept)
            acc = f"{100 * ok / len(kept):.0f} %" if kept else "—"
            line += f"   >={t}: {100 * len(kept) / len(sub):3.0f} % -> {acc:>5}"
        print(line)


def main():
    args = sys.argv[1:]
    prefixe, exclure = None, ()
    while args[:1] in (["--prefixe"], ["--exclure"]):
        if args[0] == "--prefixe":
            prefixe = args[1]
        else:
            exclure = tuple(args[1].split(","))
        args = args[2:]
    if not args:
        sys.exit(__doc__)
    for path in args:
        report(path, prefixe, exclure)


if __name__ == "__main__":
    main()
