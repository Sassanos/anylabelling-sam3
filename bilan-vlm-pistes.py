#!/usr/bin/env python3
"""Page de synthèse (HTML autonome) de la vérification des pistes au VLM.

Complète rapport-vlm-pistes.py (une carte par piste) : les chiffres qui
comptent et des galeries choisies — prompts comparés sur les cas connus, taux
du pilote par strate, faux positifs rejetés et pistes gardées, parties de
véhicule vues par la géométrie et par le VLM. Les notes « relu » viennent d'une
relecture à l'œil des vignettes (2026-09-24), pas d'une vérité terrain.

    X-AnyLabeling-Server/.venv/bin/python bilan-vlm-pistes.py \\
        --out <SORTIES>/vlm-bilan.html
"""
from __future__ import annotations

import argparse
import collections
import html
import importlib.util
import json
import random
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pistes_io import SORTIES  # noqa: E402

# vignette(), pct() du rapport par piste (nom de fichier à tiret : par chemin)
_spec = importlib.util.spec_from_file_location("rapport", HERE / "rapport-vlm-pistes.py")
rapport = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rapport)

PROMPTS = [("rapide_p1", "rapide p1"), ("reflexion-low_p1", "réflexion low p1"),
           ("rapide_p2", "rapide p2"), ("rapide_p3", "rapide p3")]
STRATES = [("militaire", "Étiquetées militaires par SAM 3"),
           ("personne", "Personnes"), ("zoom", "Véhicules, zooms rapides (0000011)"),
           ("vehicule", "Véhicules")]

# Cas connus montrés en galerie, dans l'ordre du récit.
GALERIE_CONNUS = [
    ("0000012", 1085, "panneau losange jaune"), ("0000012", 1182, "panneau rond"),
    ("0000018", 102, "voiture floue de 11 px"), ("0000018", 111, "voiture floue de 10 px"),
    ("0000011", 316, "soldat"), ("0000012", 507, "voiture de parking de 25 px"),
    ("0000018", 677, "tache de 14 px"), ("0000011", 865, "piste impure"),
    ("0000012", 1215, "piste impure (saut entre deux voitures blanches)"),
    ("0000018", 162, "avant d'un camion, déjà couvert par #159"),
]
# Relecture à l'œil : (verdict, note). ok = le VLM a raison, ko = tort, ? = ambigu.
RELU = {
    ("0000018", 629): ("ok", "balise sur poteau"), ("0000011", 60): ("?", "forme blanche couchée"),
    ("0000011", 1710): ("ok", "balise jaune"), ("0000011", 98): ("ok", "balise blanche"),
    ("0000018", 1196): ("ok", "objet blanc dans l'herbe"), ("0000012", 11694): ("ok", "poteau"),
    ("0000011", 735): ("ok", "un chien"), ("0000011", 1414): ("ok", "personne sur une affiche"),
    ("0000012", 1573): ("?", "objet sombre près d'une personne"),
    ("0000018", 1229): ("ok", "sac à dos"), ("0000012", 6791): ("ok", "tache floue"),
    ("0000012", 3894): ("?", "forme blanche près d'une personne"),
    ("0000018", 1233): ("ok", "caisse ou groupe électrogène"), ("0000012", 10339): ("ok", "cuve de stockage"),
    ("0000012", 4839): ("ok", "château d'eau"), ("0000012", 3976): ("ok", "élément de bâtiment"),
    ("0000018", 1215): ("ok", "conteneur"), ("0000011", 413): ("ok", "piquet de clôture"),
    ("0000011", 1316): ("ok", "débris entre deux remorques"), ("0000018", 1251): ("ok", "caisse ou abri"),
    ("0000011", 1393): ("ok", "buse en béton"), ("0000012", 9442): ("ok", "mur"),
    ("0000011", 1553): ("ok", "armoire technique"), ("0000012", 11838): ("ok", "mur"),
    ("0000012", 10506): ("ok", "reflet sur une façade"), ("0000018", 983): ("ok", "bac ou armoire"),
    ("0000018", 1066): ("ok", "objet jaune sous un arbre"), ("0000012", 7845): ("ok", "pilier"),
    ("0000011", 449): ("ok", "flanc de camionnette, pas un mur : partie à retirer"),
    ("0000018", 935): ("?", "petite forme blanche sous un arbre"),
    ("0000011", 683): ("ko", "petite camionnette ou caravane blanche"),
    ("0000018", 565): ("?", "« bus » sombre dans le parking"), ("0000011", 1546): ("ok", "benne verte"),
    # parties : désaccords géométrie / VLM
    ("0000011", 319): ("ok", "arrière de SUV, véhicule entier sans boîte"),
    ("0000011", 486): ("ok", "toit de camionnette"), ("0000011", 1186): ("ok", "avant de voiture"),
    ("0000012", 1766): ("ok", "moitié de voiture"),
    ("0000012", 1887): ("ok", "bas de camionnette : la géométrie a raison"),
    ("0000011", 423): ("ok", "roue de secours : la géométrie a raison"),
    ("0000011", 1655): ("ko", "véhicule entier ; la boîte contenante est trop grande"),
}

CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --good:#0ca30c; --critical:#d03b3b; --warning:#b87f00; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --grid:#2c2c2a;
  --axis:#383835; --ring:rgba(255,255,255,0.10); --series-1:#3987e5; --warning:#fab219; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff;
  --ink2:#c3c2b7; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --series-1:#3987e5; --warning:#fab219; }
body { background:var(--page); color:var(--ink); font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin:0 auto; padding:0 16px 60px; max-width:1180px; }
h1 { font-size:24px; margin:28px 0 4px; } h2 { font-size:19px; margin:40px 0 6px; }
h3 { font-size:15px; margin:22px 0 6px; color:var(--ink2); }
p { max-width:880px; } .doux { color:var(--ink2); } .muet { color:var(--muted); font-size:13px; }
.tuiles { display:flex; flex-wrap:wrap; gap:12px; margin:18px 0; }
.tuile { background:var(--surface); border:1px solid var(--ring); border-radius:8px; padding:12px 16px; min-width:170px; flex:1; }
.tuile .v { font-size:30px; font-weight:600; } .tuile .l { color:var(--ink2); font-size:13px; }
table { border-collapse:collapse; background:var(--surface); margin:8px 0; }
td, th { border-bottom:1px solid var(--grid); padding:6px 10px; text-align:left; font-variant-numeric:tabular-nums; }
th { color:var(--ink2); font-weight:600; font-size:13px; }
.barre { display:inline-block; height:8px; border-radius:0 4px 4px 0; background:var(--series-1); vertical-align:middle; margin-right:6px; }
.meilleur { font-weight:700; }
.carte { background:var(--surface); border:1px solid var(--ring); border-radius:8px; padding:10px; margin:10px 0; }
.carte .tete { display:flex; flex-wrap:wrap; gap:4px 14px; align-items:baseline; }
.bande { display:flex; flex-wrap:wrap; gap:4px; margin:8px 0; align-items:flex-start; }
.bande img { display:block; border-radius:3px; } .bande .ctx { margin-left:8px; }
.puce { display:inline-block; font-size:12.5px; padding:1px 8px; border-radius:10px; margin:2px 4px 2px 0;
  border:1px solid var(--ring); color:var(--ink); }
.puce.garde::before { content:"✓ "; color:var(--good); font-weight:700; }
.puce.rejete::before { content:"✗ "; color:var(--critical); font-weight:700; }
.puce.incertain::before { content:"? "; color:var(--warning); font-weight:700; }
.puce.info::before { content:"• "; color:var(--muted); }
.relu { font-size:13px; margin-top:4px; } .relu b.ok { color:var(--good); } .relu b.ko { color:var(--critical); }
.relu b.amb { color:var(--warning); }
.grille { display:grid; grid-template-columns:repeat(auto-fill, minmax(560px, 1fr)); gap:0 12px; }
@media (max-width:600px) { .grille { grid-template-columns:1fr; } }
svg text { font-family:system-ui, -apple-system, "Segoe UI", sans-serif; }
#bulle { position:fixed; pointer-events:none; background:var(--surface); color:var(--ink);
  border:1px solid var(--ring); border-radius:6px; padding:6px 10px; font-size:13px; display:none;
  box-shadow:0 2px 8px rgba(0,0,0,0.15); max-width:320px; }
"""

JS = """
document.addEventListener('DOMContentLoaded', () => {
  const b = document.getElementById('bulle');
  document.querySelectorAll('[data-bulle]').forEach(el => {
    el.addEventListener('mousemove', e => {
      b.innerHTML = el.dataset.bulle; b.style.display = 'block';
      b.style.left = (e.clientX + 14) + 'px'; b.style.top = (e.clientY + 14) + 'px';
    });
    el.addEventListener('mouseleave', () => { b.style.display = 'none'; });
  });
});
"""


def charger(tag):
    res = {}
    for f in sorted(SORTIES.glob(f"*/vlm/*_vlm_{tag}.jsonl")):
        for ligne in open(f, encoding="utf-8"):
            r = json.loads(ligne)
            if r.get("reponse"):
                r["_zip"] = f.parent / r["vues"]
                res[(r["vol"], r["group_id"])] = r
    return res


ZIPS = {}


def bande(r, k=4, cote=100):
    """Vues réparties (k sur 8) + vue large, en base64."""
    zp = r["_zip"]
    z = ZIPS.setdefault(zp, zipfile.ZipFile(zp))
    gid = r["group_id"]
    meta = json.loads(z.read(f"{gid}/meta.json"))
    vues = meta["vues"]
    pas = max(1, round(len(vues) / k))
    out = ["<div class='bande'>"]
    for v in vues[::pas][:k]:
        out.append(f"<img src='{rapport.vignette(z, f'{gid}/{v['n']}.jpg', cote)}' width='{cote}' "
                   f"height='{cote}' alt='vue {v['n']}' title='vue {v['n']}, +{v['dt_s']:.1f} s'>")
    if meta["contexte"]:
        c = int(cote * 1.4)
        out.append(f"<img class='ctx' src='{rapport.vignette(z, f'{gid}/contexte.jpg', c)}' "
                   f"width='{c}' height='{c}' alt='vue large'>")
    out.append("</div>")
    return "".join(out)


def puce_objet(rep, nom=""):
    o = rep.get("real_object")
    cls, mot = {"yes": ("garde", "gardé"), "no": ("rejete", "rejeté"),
                "unsure": ("incertain", "incertain")}.get(o, ("info", "?"))
    return f"<span class='puce {cls}'>{nom}{mot}</span>"


def puces(r):
    rep = r["reponse"]
    out = [puce_objet(rep)]
    if rep.get("same_object") == "no":
        out.append(f"<span class='puce incertain'>impure, vues {rep.get('different_views')}</span>")
    if rep.get("box_covers") == "part":
        out.append("<span class='puce incertain'>partie (VLM)</span>")
    c = r.get("contenue") or {}
    if c.get("frac", 0) >= 0.5:
        out.append(f"<span class='puce incertain'>contenue dans #{c['par']} "
                   f"({rapport.pct(c['frac'])} des frames)</span>")
    if r["coarse"] == "vehicle" and rep.get("real_object") == "yes":
        out.append(f"<span class='puce info'>{rep.get('fine_class')} · {rep.get('affiliation')}</span>")
    return "".join(out)


def carte(r, titre_extra="", verdicts=None):
    cle = (r["vol"], r["group_id"])
    out = ["<div class='carte'><div class='tete'>",
           f"<b>{r['vol']} #{r['group_id']}</b>",
           f"<span class='doux'>SAM 3 : {r['label_sam3']}, ~{r['size_median_px']:.0f} px, "
           f"score {r['score_mean']:.2f}, {r['n_detected']} dét.</span>"]
    if titre_extra:
        out.append(f"<span>{html.escape(titre_extra)}</span>")
    out.append("</div>")
    out.append(bande(r))
    if verdicts:
        out.append("<div>" + " ".join(verdicts) + "</div>")
    else:
        out.append(f"<div>{puces(r)}</div>")
    out.append(f"<div class='doux'>« {html.escape(r['reponse']['description'])} »</div>")
    if cle in RELU:
        v, note = RELU[cle]
        cls, mot = {"ok": ("ok", "VLM juste"), "ko": ("ko", "VLM faux"),
                    "?": ("amb", "ambigu")}[v]
        out.append(f"<div class='relu'>Relu : <b class='{cls}'>{mot}</b> — {html.escape(note)}</div>")
    out.append("</div>")
    return "".join(out)


def score_connus(res_tag, attendus):
    """Comptes (justes, total) par critère, avec les attendus de p3."""
    t = {c: [0, 0] for c in ("vrai", "fp", "impur", "pur", "partie")}
    for cle, a in attendus.items():
        r = res_tag.get(cle)
        if not r:
            continue
        rep = r["reponse"]
        if a.get("partie"):
            if "box_covers" in rep:
                t["partie"][1] += 1
                t["partie"][0] += rep["box_covers"] == "part"
            continue
        if a.get("real_object") == "oui":
            t["vrai"][1] += 1
            t["vrai"][0] += rep["real_object"] == "yes"
        elif a.get("real_object") == "non":
            t["fp"][1] += 1
            t["fp"][0] += rep["real_object"] == "no"
        if a.get("pure") is False:
            t["impur"][1] += 1
            t["impur"][0] += rep["same_object"] == "no"
        elif a.get("pure") is True:
            t["pur"][1] += 1
            t["pur"][0] += rep["same_object"] == "yes"
    return t


def graphe_barres(lignes, largeur=720):
    """Barres horizontales, une série : (libellé, part, n_rejet, n, bulle)."""
    h_barre, pas, gauche, droite = 22, 40, 250, 110
    haut = 28
    H = haut + pas * len(lignes) + 26
    x = lambda v: gauche + v * (largeur - gauche - droite)
    out = [f"<svg viewBox='0 0 {largeur} {H}' width='100%' style='max-width:{largeur}px' "
           "role='img' aria-label='Part des pistes rejetées par le VLM, par strate'>"]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f"<line x1='{x(v):.1f}' x2='{x(v):.1f}' y1='{haut - 8}' y2='{H - 22}' "
                   "stroke='var(--grid)' stroke-width='1'/>")
        out.append(f"<text x='{x(v):.1f}' y='{H - 6}' font-size='12' fill='var(--muted)' "
                   f"text-anchor='middle'>{int(v * 100)} %</text>")
    for i, (lib, part, k, n, bulle) in enumerate(lignes):
        y = haut + i * pas
        w = x(part) - x(0)
        out.append(f"<text x='{gauche - 10}' y='{y + h_barre / 2 + 5}' font-size='14' "
                   f"fill='var(--ink)' text-anchor='end'>{html.escape(lib)}</text>")
        # barre : extrémité arrondie à droite seulement, ancrée sur la ligne de base
        r = min(4, w / 2)
        out.append(f"<path d='M{x(0):.1f},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h_barre - 2 * r} "
                   f"a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z' fill='var(--series-1)'/>")
        out.append(f"<text x='{x(part) + 8:.1f}' y='{y + h_barre / 2 + 5}' font-size='13' "
                   f"fill='var(--ink2)'>{part * 100:.0f} % ({k}/{n})</text>")
        # zone de survol plus large que la barre
        out.append(f"<rect x='0' y='{y - 8}' width='{largeur}' height='{pas}' fill='transparent' "
                   f"data-bulle=\"{html.escape(bulle)}\"/>")
    out.append(f"<line x1='{x(0)}' x2='{x(0)}' y1='{haut - 8}' y2='{H - 22}' "
               "stroke='var(--axis)' stroke-width='1'/>")
    out.append("</svg>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=SORTIES / "vlm-bilan.html")
    args = ap.parse_args()

    res = {tag: charger(tag) for tag, _ in PROMPTS}
    p3 = res["rapide_p3"]
    attendus = {cle: r["attendu"] for cle, r in p3.items() if r.get("attendu")}
    pilote = [r for r in p3.values()
              if {s for s, _ in STRATES} & set(r["selection"])]

    o = ["<!doctype html><html lang='fr'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width, initial-scale=1'>",
         "<title>Bilan VLM des pistes</title>",
         f"<style>{CSS}</style><script>{JS}</script></head><body><div id='bulle'></div>",
         "<h1>Vérifier les pistes SAM 3 + BoT-SORT avec Qwen3.8-27B</h1>",
         "<p class='doux'>Vols AnafiUKR 0000011, 0000012, 0000018. Qwen3.8-27B-FP8 servi par vLLM "
         "0.30 sur une A100 80 Go (Slurm), requêtes depuis le PC. Pour chaque piste : 8 vues "
         "réparties sur sa durée (boîte en rouge) et une vue large ; le VLM répond en JSON "
         "contraint, probabilités lues sur les logprobs. Pas de vérité terrain : les notes "
         "« relu » viennent d'une relecture à l'œil des vignettes.</p>"]

    # --- tuiles
    def taux(strate):
        x = [r for r in pilote if strate in r["selection"]]
        k = sum(r["reponse"]["real_object"] == "no" for r in x)
        return k, len(x)
    tuiles = [(taux("militaire"), "des pistes « militaires » de SAM 3 rejetées (12/12 relues justes)"),
              (taux("personne"), "des pistes personnes rejetées (≈ 9/12 relues justes)"),
              (taux("vehicule"), "des pistes véhicules rejetées (≈ 6/10 relues justes)")]
    o.append("<div class='tuiles'>")
    for (k, n), lib in tuiles:
        o.append(f"<div class='tuile'><div class='v'>{100 * k / n:.0f} %</div>"
                 f"<div class='l'>{lib}</div></div>")
    lat = sorted(r["latence_s"] for r in pilote)
    o.append("<div class='tuile'><div class='v'>≈ 1,3 /s</div><div class='l'>pistes vérifiées "
             "par seconde sur une A100 (≈ 3,5 h pour les 16 500 pistes des trois vols)</div></div>")
    o.append("</div>")

    # --- prompts sur les cas connus
    o.append("<h2>1. Le prompt compte plus que la réflexion</h2>")
    o.append("<p>46 pistes relues : 23 vrais objets, 7 faux positifs, 4 pistes impures, 21 pures, "
             "9 parties de véhicule. p2 et p3 font décrire l'aspect avant de nommer ; p2 ajoute "
             "« le détecteur se trompe souvent », qui fait rejeter de vrais objets. Ces cas ont "
             "servi à régler le prompt : les chiffres de p3 y sont optimistes.</p>")
    scores = {tag: score_connus(res[tag], attendus) for tag, _ in PROMPTS}
    criteres = [("vrai", "vrais objets gardés"), ("fp", "faux positifs rejetés"),
                ("impur", "pistes impures trouvées"), ("pur", "pistes pures jugées pures"),
                ("partie", "parties trouvées par le VLM")]
    o.append("<table><tr><th>critère</th>" + "".join(f"<th>{lib}</th>" for _, lib in PROMPTS) + "</tr>")
    for c, lib in criteres:
        vals = [scores[tag][c] for tag, _ in PROMPTS]
        best = max((v[0] / v[1]) for v in vals if v[1]) if any(v[1] for v in vals) else None
        o.append(f"<tr><td>{lib}</td>")
        for v in vals:
            if not v[1]:
                o.append("<td class='muet'>—</td>")
                continue
            part = v[0] / v[1]
            cls = " class='meilleur'" if best is not None and part == best else ""
            o.append(f"<td{cls}><span class='barre' style='width:{60 * part:.0f}px'></span>"
                     f"{v[0]}/{v[1]}</td>")
        o.append("</tr>")
    o.append("</table>")
    o.append("<h3>Les cas qui départagent</h3><div class='grille'>")
    for vol, gid, note in GALERIE_CONNUS:
        r = p3.get((vol, gid))
        if not r:
            continue
        verdicts = []
        for tag, lib in PROMPTS:
            x = res[tag].get((vol, gid))
            if not x:
                continue
            rep = x["reponse"]
            s = puce_objet(rep, f"{lib} : ")
            if rep.get("same_object") == "no":
                s += f"<span class='puce incertain'>{lib} : impure</span>"
            if rep.get("box_covers") == "part":
                s += f"<span class='puce incertain'>{lib} : partie</span>"
            verdicts.append(s)
        o.append(carte(r, note, verdicts))
    o.append("</div>")

    # --- pilote
    o.append("<h2>2. Pilote : 317 pistes tirées au hasard, prompt p3</h2>")
    o.append("<p>Strates tirées au hasard parmi les pistes d'au moins 10 détections : "
             "personnes, véhicules, pistes étiquetées militaires par SAM 3, et véhicules des "
             "zooms rapides de 0000011. Part des pistes que le VLM dit ne pas être l'objet "
             "annoncé (survoler une barre pour le détail par vol) :</p>")
    lignes = []
    for s, lib in STRATES:
        x = [r for r in pilote if s in r["selection"]]
        k = sum(r["reponse"]["real_object"] == "no" for r in x)
        detail = "<br>".join(
            f"{vol} : {sum(r['reponse']['real_object'] == 'no' for r in x if r['vol'] == vol)}"
            f"/{sum(r['vol'] == vol for r in x)}" for vol in ("0000011", "0000012", "0000018")
            if any(r["vol"] == vol for r in x))
        lignes.append((lib, k / len(x), k, len(x), f"<b>{lib}</b><br>{detail}"))
    lignes.sort(key=lambda t: -t[1])
    o.append(graphe_barres(lignes))
    # tableau de lecture du graphe + autres signaux
    o.append("<table><tr><th>strate</th><th>pistes</th><th>rejetées</th><th>incertaines</th>"
             "<th>impures (VLM)</th><th>parties (VLM)</th><th>contenues (géométrie)</th></tr>")
    for s, lib in STRATES:
        x = [r for r in pilote if s in r["selection"]]
        c = collections.Counter()
        for r in x:
            rep = r["reponse"]
            c["rej"] += rep["real_object"] == "no"
            c["inc"] += rep["real_object"] == "unsure"
            c["imp"] += rep["same_object"] == "no"
            c["part"] += rep.get("box_covers") == "part"
            c["geo"] += (r.get("contenue") or {}).get("frac", 0) >= 0.5
        geo = "—" if s == "personne" else c["geo"]
        o.append(f"<tr><td>{lib}</td><td>{len(x)}</td><td>{c['rej']}</td><td>{c['inc']}</td>"
                 f"<td>{c['imp']}</td><td>{c['part']}</td><td>{geo}</td></tr>")
    o.append("</table>")

    def galerie(titre, texte, strate, objet, n, relues=True, seed=2):
        """relues : les pistes relues à l'œil (RELU) ; sinon un tirage."""
        x = [r for r in sorted(pilote, key=lambda r: (r["vol"], r["group_id"]))
             if strate in r["selection"] and r["reponse"]["real_object"] == objet
             and (not relues or (r["vol"], r["group_id"]) in RELU)]
        if not relues:
            random.Random(seed).shuffle(x)
        o.append(f"<h3>{titre}</h3><p class='doux'>{texte}</p><div class='grille'>")
        o.extend(carte(r) for r in x[:n])
        o.append("</div>")

    o.append("<h2>3. Ce que le VLM rejette, et ce qu'il garde</h2>")
    galerie("Pistes « militaires » rejetées", "Le prompt « tank » de SAM 3 accroche cuves, "
            "château d'eau, conteneurs, buses, armoires : 12/12 rejets relus justes.",
            "militaire", "no", 12)
    galerie("Personnes rejetées", "Balises, panneaux, un chien, une affiche, un sac à dos : "
            "≈ 9/12 justes, 3 ambigus.", "personne", "no", 12)
    galerie("Véhicules rejetés", "≈ 6/10 justes ; une petite camionnette blanche rejetée à tort "
            "(#683).", "vehicule", "no", 10)
    galerie("Personnes gardées (non relues)", "Pour juger l'autre côté du tri.",
            "personne", "yes", 8, relues=False)
    galerie("Véhicules gardés (non relus)", "Classe fine et affiliation proposées par le VLM.",
            "vehicule", "yes", 8, relues=False)

    # --- parties
    o.append("<h2>4. Parties de véhicule : géométrie et VLM se complètent</h2>")
    o.append("<p><b>Géométrie</b> : une boîte véhicule est « contenue » sur une frame quand au "
             "moins 80 % de son aire tombe dans la boîte d'une autre piste véhicule au moins "
             "1,5 fois plus grande — le véhicule entier est déjà suivi, la piste est un doublon. "
             "<b>VLM</b> : « la boîte couvre-t-elle l'objet entier ou une partie ? ». La géométrie "
             "rate les parties dont le véhicule entier n'a pas de boîte (à revoir, pas à "
             "supprimer) ; le VLM rejette parfois une partie comme « pas un véhicule » (roue de "
             "secours, flanc) — la décision reste la même. Règle proposée : contenue → doublon à "
             "retirer ; partie selon le VLM seul → à revoir.</p>")
    veh = [r for r in pilote if r["coarse"] == "vehicle"]
    geo = lambda r: (r.get("contenue") or {}).get("frac", 0) >= 0.5
    part = lambda r: r["reponse"].get("box_covers") == "part"
    o.append("<table><tr><th>pistes véhicules du pilote</th><th>n</th></tr>"
             f"<tr><td>contenues (géométrie) et partie (VLM)</td><td>{sum(geo(r) and part(r) for r in veh)}</td></tr>"
             f"<tr><td>contenues seulement</td><td>{sum(geo(r) and not part(r) for r in veh)}</td></tr>"
             f"<tr><td>partie selon le VLM seulement</td><td>{sum(part(r) and not geo(r) for r in veh)}</td></tr>"
             f"<tr><td>ni l'un ni l'autre</td><td>{sum(not geo(r) and not part(r) for r in veh)}</td></tr></table>")
    choix = [("0000011", 319), ("0000011", 486), ("0000011", 1186), ("0000012", 1766),
             ("0000012", 1887), ("0000011", 423), ("0000011", 449), ("0000011", 1655)]
    o.append("<h3>Désaccords relus</h3><div class='grille'>")
    o.extend(carte(p3[c]) for c in choix if c in p3)
    o.append("</div>")

    # --- limites
    o.append("<h2>5. Ce qui ne marche pas encore</h2>")
    o.append("<p><b>Pureté.</b> Le VLM ne compare pas bien les 8 vues entre elles : 2 pistes "
             "impures sur 4 trouvées sur les cas connus, quel que soit le prompt ; les sauts entre "
             "deux véhicules semblables (#1215) passent. <b>Réflexion.</b> Le mode « low » "
             "coûte ~10× plus de jetons, n'aide qu'un peu, et ses probabilités tombent à 0 ou 1 : "
             "plus de seuil possible. <b>Classe fine.</b> Sans vérité terrain, on ne sait pas qui "
             "a raison entre SAM 3 et le VLM :</p>")
    m = collections.Counter((r["label_sam3"], r["reponse"].get("fine_class")) for r in pilote
                            if r["coarse"] == "vehicle" and r["reponse"]["real_object"] == "yes"
                            and {"vehicule", "militaire"} & set(r["selection"]))
    o.append("<table><tr><th>SAM 3</th><th>VLM</th><th>pistes</th></tr>")
    for (a, b), k in m.most_common(10):
        o.append(f"<tr><td>{a}</td><td>{b}</td><td>{k}</td></tr>")
    o.append("</table>")
    o.append("<p class='muet'>Page produite par bilan-vlm-pistes.py ; le détail piste par piste "
             "est dans vlm-pilote-p3.html et vlm-essai-connus.html (rapport-vlm-pistes.py).</p>")
    o.append("</body></html>")
    args.out.write_text("\n".join(o), encoding="utf-8")
    print(f"[i] {args.out} ({args.out.stat().st_size / 1e6:.1f} Mo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
