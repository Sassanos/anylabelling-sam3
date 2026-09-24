#!/usr/bin/env python3
"""Rapport HTML des vérifications de pistes au VLM (verifie-pistes-vlm.py).

Une page autonome (vignettes en base64), à ouvrir dans le navigateur : bilan
par vol et par mode, cas connus contre l'attendu, puis une carte par piste
(vues, vue large, label SAM 3, réponse du VLM), filtrable. Plusieurs modes
d'une même piste s'affichent côte à côte.

    X-AnyLabeling-Server/.venv/bin/python rapport-vlm-pistes.py \\
        <SORTIES>/0000011/vlm/0000011_vlm_rapide_p1.jsonl ... --out rapport.html
"""
from __future__ import annotations

import argparse
import base64
import collections
import html
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


def vignette(z, nom, cote):
    import cv2
    img = cv2.imdecode(np.frombuffer(z.read(nom), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (cote, cote), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


def famille(label):
    if not label:
        return None
    if label.startswith("mil_"):
        return "militaire"
    if label == "vehicle_unknown":
        return None
    return "civil"


def pct(x):
    return "—" if x is None else f"{100 * x:.0f} %"


def p_oui(r, champ):
    d = (r.get("p") or {}).get(champ)
    return None if not d else d["p"].get("yes")


def drapeaux(r):
    """Ce que le VLM signale sur cette piste (pour les filtres)."""
    rep = r.get("reponse") or {}
    f = set()
    if rep.get("real_object") == "no":
        f.add("fp")
    if rep.get("same_object") == "no":
        f.add("impure")
    if "unsure" in (rep.get("real_object"), rep.get("same_object")):
        f.add("incertain")
    if rep.get("box_covers") == "part":
        f.add("partie")
    if (r.get("contenue") or {}).get("frac", 0) >= 0.5:
        f.add("contenue")
    if r["coarse"] == "vehicle" and rep.get("fine_class"):
        if rep["fine_class"] != r["label_sam3"]:
            f.add("classe")
        fv, fs = famille(rep["fine_class"]), famille(r["label_sam3"])
        if fv and fs and fv != fs:
            f.add("famille")
    if r.get("erreur") or r.get("reponse") is None:
        f.add("erreur")
    return f


def verdict_connu(r):
    """(ok objet, ok pureté) contre l'attendu ; None si pas d'attendu."""
    a, rep = r.get("attendu"), r.get("reponse") or {}
    if not a:
        return None, None
    ok_objet = ok_pure = None
    if a.get("partie"):
        ok_pure = rep.get("box_covers") == "part"
    if a["real_object"]:
        ok_objet = {"oui": "yes", "non": "no"}[a["real_object"]] == rep.get("real_object")
    if a["pure"] is not None:
        ok_pure = ("yes" if a["pure"] else "no") == rep.get("same_object")
    return ok_objet, ok_pure


CSS = """
:root { --bg:#f6f6f4; --carte:#fff; --texte:#1c1c1c; --doux:#6b6b6b; --bord:#ddd;
  --rouge:#c0392b; --vert:#1e8449; --orange:#b9770e; --bleu:#2e5e8e; }
@media (prefers-color-scheme: dark) { :root { --bg:#161616; --carte:#222; --texte:#e8e8e8;
  --doux:#9a9a9a; --bord:#3a3a3a; --rouge:#e06c5f; --vert:#58c27d; --orange:#e0a34a; --bleu:#7fa9d6; } }
body { background:var(--bg); color:var(--texte); font:14px/1.45 system-ui, sans-serif; margin:0 16px 40px; }
h1 { font-size:20px; margin:18px 0 4px; } h2 { font-size:16px; margin:26px 0 8px; }
table { border-collapse:collapse; background:var(--carte); }
td, th { border:1px solid var(--bord); padding:4px 8px; text-align:left; font-variant-numeric:tabular-nums; }
.ok { color:var(--vert); font-weight:600; } .ko { color:var(--rouge); font-weight:600; }
.filtres { position:sticky; top:0; background:var(--bg); padding:8px 0; z-index:2; }
.filtres button { margin:0 4px 4px 0; padding:4px 10px; border:1px solid var(--bord);
  background:var(--carte); color:var(--texte); border-radius:4px; cursor:pointer; }
.filtres button.actif { border-color:var(--bleu); color:var(--bleu); font-weight:600; }
.carte { background:var(--carte); border:1px solid var(--bord); border-radius:6px; padding:10px; margin:10px 0; }
.tete { display:flex; flex-wrap:wrap; gap:4px 16px; align-items:baseline; }
.tete b { font-size:15px; } .doux { color:var(--doux); }
.vues { display:flex; flex-wrap:wrap; gap:4px; margin:8px 0; align-items:flex-start; }
.vues figure { margin:0; text-align:center; font-size:11px; color:var(--doux); }
.vues img { display:block; border:3px solid transparent; border-radius:3px; }
.vues img.ecart { border-color:var(--orange); }
.vues .ctx { margin-left:10px; }
.verdict { border-top:1px dashed var(--bord); padding-top:6px; margin-top:6px; }
.tag { display:inline-block; padding:0 6px; border-radius:3px; font-size:12px; margin-right:4px;
  border:1px solid var(--bord); }
.tag.fp, .tag.impure, .tag.famille, .tag.erreur { border-color:var(--rouge); color:var(--rouge); }
.tag.classe, .tag.incertain, .tag.partie, .tag.contenue { border-color:var(--orange); color:var(--orange); }
details { margin-top:4px; } details pre { white-space:pre-wrap; font-size:12px; max-height:320px; overflow:auto; }
"""

JS = """
function filtre(f, b) {
  document.querySelectorAll('.filtres button').forEach(x => x.classList.remove('actif'));
  b.classList.add('actif');
  document.querySelectorAll('.carte').forEach(c => {
    c.style.display = (f === 'tout' || c.dataset.f.split(' ').includes(f)) ? '' : 'none';
  });
}
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resultats", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cote", type=int, default=128, help="vignette, pixels")
    args = ap.parse_args()

    # (vol, gid) -> {mode: ligne} ; la dernière ligne sans erreur l'emporte
    pistes = collections.defaultdict(dict)
    zf = {}   # chemin du zip de vues -> ZipFile (un zip par réglage de vues)
    for chemin in args.resultats:
        for l in open(chemin, encoding="utf-8"):
            r = json.loads(l)
            cle = (r["vol"], r["group_id"])
            r["mode"] = f"{r['mode']}-{r.get('prompt', 'p1')}"
            if r.get("erreur") and r["mode"] in pistes[cle]:
                continue
            pistes[cle][r["mode"]] = r
            r["_zip"] = chemin.parent / r["vues"]
    modes = sorted({m for d in pistes.values() for m in d})

    out = ["<!doctype html><html lang='fr'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width, initial-scale=1'>",
           "<title>Vérification VLM des pistes</title>",
           f"<style>{CSS}</style><script>{JS}</script></head><body>",
           "<h1>Vérification des pistes au VLM</h1>",
           f"<p class='doux'>{len(pistes)} pistes, modes : {', '.join(modes)}. "
           "Bordure orange = vue que le VLM dit sur un autre objet. "
           "P = probabilité lue sur les logprobs.</p>"]

    # --- bilan par vol et mode
    out.append("<h2>Bilan</h2><table><tr><th>vol</th><th>mode</th><th>pistes</th>"
               "<th>faux positifs</th><th>impures</th><th>incertaines</th>"
               "<th>classe ≠ SAM 3</th><th>famille ≠ SAM 3</th><th>erreurs</th>"
               "<th>jetons sortis (moy.)</th><th>latence (moy.)</th></tr>")
    for vol in sorted({v for v, _ in pistes}):
        for m in modes:
            rs = [d[m] for (v, _), d in pistes.items() if v == vol and m in d]
            if not rs:
                continue
            c = collections.Counter(x for r in rs for x in drapeaux(r))
            sortis = [r["jetons"]["completion_tokens"] for r in rs if r.get("jetons")]
            lat = [r["latence_s"] for r in rs if r.get("latence_s")]
            vh = sum(r["coarse"] == "vehicle" for r in rs)
            out.append(
                f"<tr><td>{vol}</td><td>{m}</td><td>{len(rs)} ({vh} véh.)</td>"
                f"<td>{c['fp']}</td><td>{c['impure']}</td><td>{c['incertain']}</td>"
                f"<td>{c['classe']}</td><td>{c['famille']}</td><td>{c['erreur']}</td>"
                f"<td>{np.mean(sortis):.0f}</td><td>{np.mean(lat):.1f} s</td></tr>"
                if sortis and lat else
                f"<tr><td>{vol}</td><td>{m}</td><td>{len(rs)}</td>"
                f"<td colspan='8'>{c['erreur']} erreurs</td></tr>")
    out.append("</table>")

    # --- cas connus
    connus = [(k, d) for k, d in sorted(pistes.items())
              if any(r.get("attendu") for r in d.values())]
    if connus:
        out.append("<h2>Cas connus (relus sur planche)</h2><table><tr><th>piste</th>"
                   "<th>note</th><th>attendu objet / pure</th>"
                   + "".join(f"<th>{m} : objet (P oui)</th><th>{m} : même objet (P oui) / couvre</th>"
                             for m in modes) + "</tr>")
        bilan = collections.Counter()
        for (vol, gid), d in connus:
            # attendu du prompt le plus récent (#162 : reclassé en partie en p2)
            a = next(r["attendu"] for _m, r in sorted(
                d.items(), key=lambda x: x[0].split("-")[-1], reverse=True)
                if r.get("attendu"))
            pure = ("partie" if a.get("partie") else
                    {True: "pure", False: "impure", None: "?"}[a["pure"]])
            cells = [f"<td><a href='#{vol}-{gid}'>{vol} #{gid}</a></td>",
                     f"<td>{html.escape(a['note'])}</td>",
                     f"<td>{a['real_object'] or '?'} / {pure}</td>"]
            for m in modes:
                r = d.get(m)
                if not r or not r.get("reponse"):
                    cells.append("<td colspan='2'>—</td>")
                    continue
                ok_o, ok_p = verdict_connu(r)
                rep = r["reponse"]
                for champ, ok in (("real_object", ok_o), ("same_object", ok_p)):
                    cls = "" if ok is None else (" class='ok'" if ok else " class='ko'")
                    if ok is not None:
                        partie = (r.get("attendu") or {}).get("partie")
                        bilan[(m, "partie" if partie and champ == "same_object"
                               else champ, ok)] += 1
                    val = f"{rep.get(champ)} ({pct(p_oui(r, champ))})"
                    if champ == "same_object" and rep.get("box_covers"):
                        val += f" / {rep['box_covers']}"
                    cells.append(f"<td{cls}>{val}</td>")
            out.append("<tr>" + "".join(cells) + "</tr>")
        out.append("</table><p>")
        for m in modes:
            for champ, nom in (("real_object", "objet réel"), ("same_object", "pureté"),
                               ("partie", "partie de véhicule")):
                bon, faux = bilan[(m, champ, True)], bilan[(m, champ, False)]
                if bon + faux:
                    out.append(f"{m}, {nom} : <b>{bon}/{bon + faux}</b> conformes à l'attendu. ")
        out.append("</p>")

    # --- cartes
    out.append("<h2>Pistes</h2><div class='filtres'>")
    for f, nom in (("tout", "Toutes"), ("connu", "Cas connus"), ("fp", "Faux positifs VLM"),
                   ("impure", "Impures VLM"), ("incertain", "Incertaines"),
                   ("partie", "Parties (VLM)"), ("contenue", "Contenues (géométrie)"),
                   ("famille", "Famille ≠ SAM 3"), ("classe", "Classe ≠ SAM 3"),
                   ("erreur", "Erreurs")):
        actif = " class='actif'" if f == "tout" else ""
        out.append(f"<button{actif} onclick=\"filtre('{f}', this)\">{nom}</button>")
    out.append("</div>")
    for (vol, gid), d in sorted(pistes.items()):
        r0 = next(iter(d.values()))
        # vues du mode le plus récent (p2 > p1 : réglages de vues différents)
        zp = d[max(d, key=lambda m: m.split("-")[-1])]["_zip"]
        z = zf.setdefault(zp, zipfile.ZipFile(zp)) if zp not in zf else zf[zp]
        meta = json.loads(z.read(f"{gid}/meta.json"))
        f = set().union(*(drapeaux(r) for r in d.values()))
        if any(r.get("attendu") for r in d.values()):
            f.add("connu")
        ecarts = set()
        for r in d.values():
            ecarts |= set((r.get("reponse") or {}).get("different_views") or [])
        votes = ", ".join(f"{k} {100 * x:.0f} %" for k, x in list(r0["labels_sam3"].items())[:3])
        out.append(f"<div class='carte' id='{vol}-{gid}' data-f='{' '.join(sorted(f))}'>")
        out.append(
            f"<div class='tete'><b>{vol} #{gid}</b><span>{r0['coarse']} — SAM 3 : "
            f"<b>{r0['label_sam3']}</b> <span class='doux'>({votes})</span></span>"
            f"<span class='doux'>{r0['n_detected']} dét., {r0['duree_s']:.1f} s, "
            f"~{r0['size_median_px']:.0f} px, score {r0['score_mean']:.2f}</span>"
            f"<span class='doux'>sélection : {', '.join(r0['selection'])}</span></div>")
        if r0.get("attendu"):
            a = r0["attendu"]
            out.append(f"<div class='doux'>Attendu : {html.escape(a['note'])}</div>")
        out.append("<div class='vues'>")
        for v in meta["vues"]:
            cls = " class='ecart'" if v["n"] in ecarts else ""
            out.append(f"<figure><img{cls} src='{vignette(z, f'{gid}/{v['n']}.jpg', args.cote)}'"
                       f" width='{args.cote}' height='{args.cote}' alt='vue {v['n']}'>"
                       f"{v['n']} · +{v['dt_s']:.1f} s · {v['taille_px'][0]}×{v['taille_px'][1]}</figure>")
        if meta["contexte"]:
            c = int(args.cote * 1.5)
            out.append(f"<figure class='ctx'><img src='{vignette(z, f'{gid}/contexte.jpg', c)}'"
                       f" width='{c}' height='{c}' alt='vue large'>vue large (vue "
                       f"{meta['vue_contexte']})</figure>")
        out.append("</div>")
        for m in modes:
            r = d.get(m)
            if not r:
                continue
            out.append(f"<div class='verdict'><span class='doux'>{m}</span> ")
            out.append("".join(f"<span class='tag {x}'>{x}</span>" for x in sorted(drapeaux(r))))
            if r.get("erreur") or not r.get("reponse"):
                out.append(f"<div class='ko'>{html.escape(r.get('erreur') or r.get('brut') or '')}</div></div>")
                continue
            rep, p = r["reponse"], r.get("p") or {}
            if rep.get("appearance"):
                out.append(f"<div class='doux'>Aspect : {html.escape(rep['appearance'])}</div>")
            out.append(f"<div>« {html.escape(rep.get('description', ''))} »</div>")
            ligne = (f"même objet : <b>{rep['same_object']}</b> (P oui {pct(p_oui(r, 'same_object'))})"
                     + (f", vues à part : {rep['different_views']}" if rep.get("different_views") else "")
                     + f" · objet réel : <b>{rep['real_object']}</b> (P oui {pct(p_oui(r, 'real_object'))})")
            if rep.get("box_covers"):
                pp = ((p.get("box_covers") or {}).get("p") or {}).get("part")
                ligne += f" · boîte : <b>{rep['box_covers']}</b> (P partie {pct(pp)})"
            c = r.get("contenue")
            if c:
                ligne += (f" · contenue dans #{c['par']} sur {pct(c['frac'])} des frames"
                          if c.get("par") else " · contenue : 0 %")
            if r["coarse"] == "vehicle":
                aff = (p.get("affiliation") or {}).get("p", {})
                ligne += (f" · classe : <b>{rep.get('fine_class')}</b> (P {pct(p.get('fine_class'))})"
                          f" · {rep.get('affiliation')} (P {pct(aff.get(rep.get('affiliation')))})")
            out.append(f"<div>{ligne}</div>")
            j = r.get("jetons") or {}
            out.append(f"<div class='doux'>{j.get('prompt_tokens', '?')} jetons d'entrée, "
                       f"{j.get('completion_tokens', '?')} sortis, {r.get('latence_s', '?')} s</div>")
            if r.get("raisonnement"):
                out.append(f"<details><summary>Réflexion ({len(r['raisonnement'])} car.)</summary>"
                           f"<pre>{html.escape(r['raisonnement'])}</pre></details>")
            out.append("</div>")
        out.append("</div>")
    out.append("</body></html>")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"[i] {len(pistes)} pistes -> {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} Mo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
