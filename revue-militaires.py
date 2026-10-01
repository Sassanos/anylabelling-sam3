#!/usr/bin/env python3
"""Revue humaine rapide des pistes militaires du jeu de détection.

Pistes relues : pistes véhicule des frames exportées (par-vol/*.json) que
l'export ignore faute de famille sûre (équipement ou militaire incertain
selon m1, affiliation inconnue, objet incertain), les militaires gardés par
m1 et les parties militaires ; voir PRIORITES. Le VLM s'y trompe dans
les deux sens (groupes électrogènes pris pour des militaires par p5, Masstech
et VT4 pris pour de l'équipement par m1) : la réponse humaine remplace p5/m1
pour la famille du véhicule dans exporte-detection.py.

Une piste à la fois, vues du VLM (zip d4edf1) et vue large ; clavier :
m militaire, c civil, n pas un véhicule, i incertain, ← précédente.
Pistes triées par groupe de priorité, puis par vol et par temps, pour que
les morceaux d'un même véhicule se suivent. Réponses ajoutées à <REVUE>/annotations.jsonl (la
dernière ligne d'une piste fait foi).

    PY=X-AnyLabeling-Server/.venv/bin/python
    $PY revue-militaires.py            # http://127.0.0.1:8766
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pistes_io import chemins_vol, dernier_jeu  # noqa: E402


def _module(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, HERE / fichier)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ex = _module("exporte_detection", "exporte-detection.py")
vlm = ex.vlm
REVUE = Path("/home/cbarbier/Documents/Geolocalisation/Datasets/real/CAMPAGNE3"
             "/revue-militaires")
VERDICTS = ("militaire", "civil", "non", "incertain")
# Pistes relues, par raison de leur niveau dans l'export, dans cet ordre :
# d'abord les ignorées sans doute militaires (remarque de l'utilisateur : « beaucoup
# de pistes ignorées sont en fait militaires »), puis les militaires gardés à
# vérifier, puis les parties militaires (famille des véhicules recollés).
PRIORITES = {"equipement_m1": 1, "militaire_incertain": 1,
             "militaire_non_verifie": 1, "affiliation_inconnue": 2,
             "objet_incertain": 2, "vlm_m1": 3, "partie": 4}
NOMS_GROUPES = {1: "militaire ignoré (m1)", 2: "affiliation ou objet incertain",
                3: "militaire gardé (à vérifier)", 4: "partie militaire"}


def lire_annotations():
    faites = {}
    f = REVUE / "annotations.jsonl"
    if f.exists():
        for ligne in open(f, encoding="utf-8"):
            r = json.loads(ligne)
            faites[f"{r['vol']}/{r['group_id']}"] = r
    return faites


def selection():
    """Pistes à relire, dans l'ordre (vol, début de piste)."""
    pistes, zips = [], {}
    for vol, d in ex.lire_par_vol().items():
        presentes = {}
        for im in d["images"]:
            for b in im["boites"]:
                for g in ([b["group_id"]] if b.get("group_id") is not None
                          else b.get("morceaux") or []):
                    if g is not None:
                        presentes[g] = presentes.get(g, 0) + 1
        chemins = chemins_vol(vol)
        jeu = dernier_jeu(chemins["pistes"], vol)
        dossier = chemins["pistes"] / "vlm"
        v5 = ex.lire_verdicts(dossier / f"{vol}_vlm_rapide_p5.jsonl")
        vm1 = ex.lire_verdicts(dossier / f"{vol}_vlm_rapide_m1.jsonl")
        zips[vol] = zipfile.ZipFile(
            dossier / f"{jeu.stem}_vues-"
            f"{vlm.empreinte_vues(vlm.VUES_PROMPT['p5'])}.zip")
        debuts = {}
        for l in open(jeu, encoding="utf-8"):
            p = json.loads(l)
            if p["group_id"] in presentes:
                debuts[p["group_id"]] = p["t_min_s"]
        for g, r in v5.items():
            if g not in presentes or r["coarse"] != "vehicle":
                continue
            niveau, _cat, raison, _i = ex.verdict_piste("vehicle", r, vm1.get(g))
            groupe = PRIORITES.get(raison)
            if groupe is None:
                continue
            rep = r["reponse"]
            if raison == "partie" and rep.get("fine_class") not in vlm.MILITAIRES:
                continue
            m1 = (vm1.get(g) or {}).get("reponse") or {}
            meta = json.loads(zips[vol].read(f"{g}/meta.json"))
            pistes.append({
                "vol": vol, "group_id": g, "t": debuts.get(g, 0.0),
                "groupe": groupe, "raison": raison,
                "frames": presentes[g], "label_sam3": r["label_sam3"],
                "fine_class": rep.get("fine_class"),
                "box_covers": rep.get("box_covers"),
                "m1": f"{m1.get('kind', '—')} / militaire {m1.get('military', '—')}"
                      if m1 else "—",
                "description": rep.get("description", ""),
                "taille": r.get("size_median_px"),
                "vues": [v["n"] for v in meta["vues"]],
                "nom_groupe": NOMS_GROUPES[groupe],
                "contexte": bool(meta["contexte"])})
    pistes.sort(key=lambda p: (p["groupe"], p["vol"], p["t"], p["group_id"]))
    return pistes, zips


PAGE = """<!doctype html><meta charset=utf-8><title>Revue militaires</title>
<style>
body{font:15px system-ui,sans-serif;margin:12px;background:#111;color:#ddd}
#haut{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
#vues{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}
#vues img{width:300px;height:300px;object-fit:contain;background:#222}
#vues img.ctx{width:420px;height:420px}
.info{color:#aaa} b{color:#fff} kbd{background:#333;padding:1px 6px;border-radius:3px}
.v-militaire{color:#32cd32}.v-civil{color:#1e90ff}.v-non{color:#ff5050}.v-incertain{color:#ccc}
button{font-size:15px;margin-right:6px;padding:6px 12px;background:#333;color:#ddd;border:1px solid #555}
</style>
<div id=haut><h2 id=titre></h2><span id=avance class=info></span><span id=dejaV></span></div>
<div class=info id=vlm></div>
<div id=vues></div>
<p><button onclick="rep('militaire')"><kbd>m</kbd> militaire</button>
<button onclick="rep('civil')"><kbd>c</kbd> civil</button>
<button onclick="rep('non')"><kbd>n</kbd> pas un véhicule</button>
<button onclick="rep('incertain')"><kbd>i</kbd> incertain</button>
<button onclick="aller(i-1)"><kbd>←</kbd> précédente</button>
<button onclick="aller(i+1)"><kbd>→</kbd> suivante</button>
<button onclick="suivanteAFaire()"><kbd>espace</kbd> prochaine à faire</button></p>
<p class=info>Famille du véhicule entier, même si la boîte n'en couvre qu'une partie.
Un groupe électrogène, une remorque, un conteneur : <kbd>n</kbd>.</p>
<script>
let P=[],A={},i=0;
fetch('/api/pistes').then(r=>r.json()).then(d=>{P=d.pistes;A=d.annotations;suivanteAFaire();});
function cle(p){return p.vol+'/'+p.group_id}
function faites(){return P.filter(p=>A[cle(p)]).length}
function aller(k){if(k<0||k>=P.length)return;i=k;const p=P[i];
 document.getElementById('titre').textContent=`${p.vol} #${p.group_id}`;
 document.getElementById('avance').textContent=`${i+1}/${P.length} · faites ${faites()} · ${p.nom_groupe} · ${p.frames} boîtes exportées · ${Math.round(p.taille)} px`;
 const a=A[cle(p)];const dv=document.getElementById('dejaV');
 dv.className=a?'v-'+a.verdict:'';dv.textContent=a?'déjà : '+a.verdict:'';
 document.getElementById('vlm').innerHTML=`SAM 3 <b>${p.label_sam3}</b> · p5 <b>${p.fine_class}</b> (${p.box_covers}) · m1 <b>${p.m1}</b><br>${p.description}`;
 const v=document.getElementById('vues');v.innerHTML='';
 const n=p.vues;const choix=[n[0],n[Math.floor(n.length/3)],n[Math.floor(2*n.length/3)],n[n.length-1]];
 [...new Set(choix)].forEach(k=>{const im=new Image();im.src=`/img/${p.vol}/${p.group_id}/${k}.jpg`;v.appendChild(im)});
 if(p.contexte){const im=new Image();im.className='ctx';im.src=`/img/${p.vol}/${p.group_id}/contexte.jpg`;v.appendChild(im)}
 [i+1,i+2].forEach(k=>{if(k<P.length){const q=P[k];(new Image()).src=`/img/${q.vol}/${q.group_id}/contexte.jpg`}});}
function suivanteAFaire(){for(let k=0;k<P.length;k++){const j=(i+k+(A[cle(P[i]||{})]?1:0))%P.length;if(!A[cle(P[j])])return aller(j)}aller(i)}
function rep(v){const p=P[i];const r={vol:p.vol,group_id:p.group_id,verdict:v};
 fetch('/api/annotation',{method:'POST',body:JSON.stringify(r)}).then(()=>{A[cle(p)]=r;aller(Math.min(i+1,P.length-1))});}
document.addEventListener('keydown',e=>{const k=e.key;
 if(k==='m')rep('militaire');else if(k==='c')rep('civil');else if(k==='n')rep('non');
 else if(k==='i')rep('incertain');else if(k==='ArrowLeft')aller(i-1);
 else if(k==='ArrowRight')aller(i+1);else if(k===' '){e.preventDefault();suivanteAFaire()}});
</script>"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    pistes, zips = selection()
    REVUE.mkdir(parents=True, exist_ok=True)
    journal = REVUE / "annotations.jsonl"
    verrou = threading.Lock()

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
                return self.envoyer(200, PAGE.encode(), "text/html; charset=utf-8")
            if chemin == "/api/pistes":
                with verrou:
                    faites = lire_annotations()
                corps = json.dumps({"pistes": pistes, "annotations": faites},
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
            if r.get("verdict") not in VERDICTS:
                return self.envoyer(400, b"", "text/plain")
            r["date"] = datetime.now().isoformat(timespec="seconds")
            with verrou, open(journal, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            self.envoyer(200, b"{}", "application/json")

    serveur = ThreadingHTTPServer(("127.0.0.1", args.port), Gestion)
    print(f"[i] {len(pistes)} pistes, {len(lire_annotations())} déjà relues "
          f"-> http://127.0.0.1:{args.port}  (réponses : {journal})", flush=True)
    serveur.serve_forever()


if __name__ == "__main__":
    main()
