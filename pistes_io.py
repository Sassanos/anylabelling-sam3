"""Entrées communes à associe-pistes.py, rendu-pistes.py et planche-pistes.py.

Détections : tronçons JSONL du lot Slurm (`annots/<vol>/`), vidéo : piste 0
(DefaultVideo) du MP4 source, décodée en streaming, rien n'est écrit sur
disque. Les sorties vont en local dans `dossier_pistes(vol)`
(`Datasets/real/AnafiUKR/pistes/<vol>/` pour 0000011, 0000012 et 0000018,
`Datasets/real/CAMPAGNE3/pistes/<vol>/` pour les autres) ; les entrées ne sont
jamais modifiées.

PISTES_FLUX=visible (variable d'environnement) fait travailler tous ces
scripts sur la piste VisibleVideo — le RGB 4K à 8,57 fps des rafales
thermiques, apparié à l'IR — au lieu du DefaultVideo : détections dans
`annots-visible/<vol>/`, index samples_visible.csv, piste vidéo 1, sorties
dans `pistes-visible/<vol>/`.
Voir TRACKER-PISTES.md pour les formats.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# Entrées : détections du lot Slurm, index et vidéos, lus sans jamais être
# modifiés ni supprimés.
RACINE = Path("/media/users/cbarbier/annots-sam3")
# Sorties (pistes, cache d'indices, rendus, vues VLM), en local : <SORTIES>/<vol>/
# pour les 3 vols du lot AnafiUKR, <SORTIES_CAMPAGNE3>/<vol>/ pour les 16 autres
# vols de CAMPAGNE3 (faits le 2026-09-25) ; voir dossier_pistes().
SORTIES = Path("/home/cbarbier/Documents/Geolocalisation/Datasets/real/AnafiUKR"
               "/pistes")
SORTIES_CAMPAGNE3 = Path("/home/cbarbier/Documents/Geolocalisation/Datasets/real"
                         "/CAMPAGNE3/pistes")
VOLS_ANAFIUKR = {"0000011", "0000012", "0000018"}
# Flux annoté : nom dans l'index, dossier des détections, piste vidéo du MP4.
FLUX = os.environ.get("PISTES_FLUX", "default")
_FLUX = {"default": ("annots", 0), "visible": ("annots-visible", 1)}
if FLUX not in _FLUX:
    raise SystemExit(f"[!] PISTES_FLUX={FLUX} inconnu ({', '.join(_FLUX)})")
DOSSIER_ANNOTS, PISTE_VIDEO = _FLUX[FLUX]
# Vols RGB de jour, seuls gardés pour le VLM (consigne du 2026-09-29) : filmés
# entre 05h33 et 17h08. Exclus, sans rien supprimer : les 9 vols de 22h48 à
# 00h35 (0000229-0000236, 0000429, 0000431 : nuit, surtout IR) et 0000016
# (export de la tablette, nuit, écran filmé).
VOLS_JOUR = ("0000001", "0000002", "0000004", "0000005", "0000011", "0000012",
             "0000018", "0000019", "0000231")


@dataclass
class Frame:
    """Une frame annotée, formes réduites à ce que l'association utilise."""

    sample: int
    tick: int
    t: float
    largeur: int
    hauteur: int
    boites: np.ndarray                  # (n, 4) x1 y1 x2 y2, float32
    scores: np.ndarray                  # (n,)
    labels: List[str] = field(default_factory=list)
    grossieres: List[str] = field(default_factory=list)
    erreur: Optional[str] = None


def dossier_pistes(vol: str) -> Path:
    base = SORTIES if vol in VOLS_ANAFIUKR else SORTIES_CAMPAGNE3
    if FLUX != "default":
        base = base.with_name(f"{base.name}-{FLUX}")
    return base / vol


def chemins_vol(vol: str, racine: Path = RACINE) -> Dict[str, Path]:
    videos = sorted((racine / "campagne3_index").glob(f"*/{vol}_video.MP4"))
    if len(videos) != 1:
        raise SystemExit(f"[!] {len(videos)} vidéo(s) {vol}_video.MP4 sous "
                         f"{racine / 'campagne3_index'} (une attendue)")
    return {
        "annots": racine / DOSSIER_ANNOTS / vol,
        "video": videos[0],
        "index": (racine / "campagne3_index" / "annotation" / vol / "index"
                  / f"samples_{FLUX}.csv"),
        "pistes": dossier_pistes(vol),
    }


def troncons(dossier: Path, vol: str):
    """[(début, fin, fichier, terminé)] — .jsonl.zst préféré au .jsonl."""
    motif = re.compile(rf"^{re.escape(vol)}_chunk_(\d+)_(\d+)\.jsonl(\.zst)?$")
    trouves: Dict[Tuple[int, int], Path] = {}
    for p in dossier.iterdir():
        m = motif.match(p.name)
        if not m:
            continue
        cle = (int(m.group(1)), int(m.group(2)))
        if cle not in trouves or p.name.endswith(".zst"):
            trouves[cle] = p
    sortie = []
    for (debut, fin), p in sorted(trouves.items()):
        base = p.name[: p.name.index(".jsonl")]
        sortie.append((debut, fin, p, (dossier / f"{base}.done").exists()))
    return sortie


def lignes_jsonl(p: Path) -> Iterator[str]:
    if p.name.endswith(".zst"):
        texte = subprocess.run(["zstd", "-dc", str(p)], capture_output=True,
                               check=True).stdout.decode("utf-8")
    else:
        texte = p.read_text(encoding="utf-8")
    return iter(texte.splitlines())


def charger_frames(dossier: Path, vol: str, debut: Optional[int] = None,
                   fin: Optional[int] = None,
                   partiel: bool = False) -> Tuple[List[Frame], list]:
    """Frames annotées de [debut, fin), triées par sample_index.

    Refuse un vol dont un tronçon n'a pas son .done (en cours d'écriture),
    sauf `partiel` : les tronçons non terminés sont alors ignorés.
    """
    liste = troncons(dossier, vol)
    if not liste:
        raise SystemExit(f"[!] aucun tronçon {vol}_chunk_* dans {dossier}")
    utiles = [t for t in liste
              if (debut is None or t[1] > debut) and (fin is None or t[0] < fin)]
    en_cours = [t[2].name for t in utiles if not t[3]]
    if en_cours and not partiel:
        raise SystemExit("[!] tronçons sans .done (en cours) : "
                         + ", ".join(en_cours) + " — attendre, ou --partiel")
    frames: Dict[int, Frame] = {}
    for _d, _f, chemin, fini in utiles:
        if not fini:
            print(f"[!] ignoré (en cours) : {chemin.name}", file=sys.stderr)
            continue
        for ligne in lignes_jsonl(chemin):
            try:
                r = json.loads(ligne)
            except ValueError:
                continue
            i = r.get("sample_index")
            if i is None or (debut is not None and i < debut) or (
                    fin is not None and i >= fin):
                continue
            boites, scores, labels, grossieres = [], [], [], []
            for s in r.get("shapes") or []:
                pts = np.asarray(s["points"], float)
                boites.append([pts[:, 0].min(), pts[:, 1].min(),
                               pts[:, 0].max(), pts[:, 1].max()])
                sc = s.get("score")
                scores.append(1.0 if sc is None else float(sc))
                labels.append(s["label"])
                grossieres.append((s.get("attributes") or {}).get("coarse")
                                  or s["label"])
            frames[int(i)] = Frame(
                sample=int(i), tick=int(r["dts_ticks"]),
                t=float(r.get("dts_s") or 0.0),
                largeur=int(r.get("width") or 0),
                hauteur=int(r.get("height") or 0),
                boites=np.asarray(boites, np.float32).reshape(-1, 4),
                scores=np.asarray(scores, np.float32),
                labels=labels, grossieres=grossieres, erreur=r.get("error"))
    sources = [{"fichier": str(t[2]), "debut": t[0], "fin": t[1],
                "termine": t[3]} for t in utiles]
    return [frames[k] for k in sorted(frames)], sources


def dernier_jeu(dossier: Path, vol: str) -> Path:
    """Jeu de pistes le plus récent d'un vol (associe-pistes.py)."""
    jeux = sorted(dossier.glob(f"{vol}_pistes_*.jsonl"),
                  key=lambda p: p.stat().st_mtime)
    if not jeux:
        raise SystemExit(f"[!] aucun {vol}_pistes_*.jsonl dans {dossier} — "
                         "lancer associe-pistes.py d'abord")
    return jeux[-1]


def ticks_index(index_csv: Path) -> Dict[int, int]:
    """sample_index -> dts_ticks du flux annoté (FLUX)."""
    with open(index_csv, newline="", encoding="utf-8") as f:
        return {int(r["sample_index"]): int(r["dts_ticks"])
                for r in csv.DictReader(f) if r["stream"] == FLUX}


def timescale_index(index_csv: Path) -> int:
    """Base de temps de l'index (celle du flux) : 30000, ou 90000 pour 0000016."""
    with open(index_csv, newline="", encoding="utf-8") as f:
        return int(next(csv.DictReader(f))["timescale"])


def telemetrie_index(index_csv: Path) -> Dict[int, dict]:
    """sample_index -> ligne de l'index (hauteur, champ de vue, assiette...)."""
    with open(index_csv, newline="", encoding="utf-8") as f:
        return {int(r["sample_index"]): r for r in csv.DictReader(f)
                if r["stream"] == FLUX}


def largeur_sol_m(ligne: dict, bbox, largeur: int = 3840, hauteur: int = 2160,
                  depression_min_deg: float = 3.0) -> Optional[float]:
    """Largeur au sol (m) de la boîte, sol plat au niveau du décollage : rayon
    du bas de la boîte (contact au sol), distance = hauteur relative /
    sin(dépression). None si la télémétrie manque ou si le rayon est trop
    rasant. Grossier : relief, roulis et distorsion ignorés ; sur les 134
    pistes de la revue, personnes 0,3-1,4 m, voitures 2,4-4 m, camions 3-11 m.
    """
    import math
    try:
        h = float(ligne["rel_alt_m"])
        hfov = math.radians(float(ligne["hfov_deg"]))
        tangage = float(ligne["cam_pitch_deg"])
    except (KeyError, ValueError):
        return None
    f = (largeur / 2) / math.tan(hfov / 2)
    x1, _y1, x2, y2 = bbox
    depression = math.radians(-tangage) + math.atan((y2 - hauteur / 2) / f)
    if h < 2 or depression < math.radians(depression_min_deg):
        return None
    return (x2 - x1) * (h / math.sin(depression)) / f


def decode_frames(video: Path, ticks: List[int], hwaccel: bool = True,
                  saut_s: float = 4.0, timescale: Optional[int] = None):
    """Génère (position, image BGR ou None) pour chaque tick demandé, en ordre.

    Seek au keyframe précédent quand la prochaine frame demandée est à plus
    de `saut_s` secondes (trous IR), décodage continu sinon. Le flux a une
    base de temps 1/timescale et pas de B-frames : `frame.pts == dts_ticks`.
    timescale est celui du flux sauf s'il est imposé : 30000 pour la plupart
    des vols, 90000 pour 0000016 (l'index suit le flux).
    None signale un tick absent du flux (ne devrait pas arriver).
    """
    import av
    options = {}
    if hwaccel:
        try:
            from av.codec.hwaccel import HWAccel
            options["hwaccel"] = HWAccel(device_type="cuda",
                                         allow_software_fallback=True)
        except Exception:  # PyAV sans hwaccel : décodage logiciel
            pass
    conteneur = av.open(str(video), **options)
    flux = conteneur.streams.video[PISTE_VIDEO]
    if timescale is None:
        timescale = flux.time_base.denominator
    if flux.time_base.numerator != 1 or flux.time_base.denominator != timescale:
        raise SystemExit(f"[!] base de temps {flux.time_base}, "
                         f"1/{timescale} attendu")
    if "hwaccel" not in options:
        flux.thread_type = "AUTO"
    saut = int(saut_s * timescale)
    n, pos = len(ticks), 0
    try:
        while pos < n:
            conteneur.seek(ticks[pos], stream=flux, backward=True)
            relance = False
            for frame in conteneur.decode(flux):
                if frame.pts is None:
                    continue
                t = int(frame.pts)
                while pos < n and ticks[pos] < t:
                    yield pos, None
                    pos += 1
                if pos >= n:
                    break
                if t < ticks[pos]:
                    continue
                yield pos, frame.to_ndarray(format="bgr24")
                pos += 1
                if pos >= n:
                    break
                if ticks[pos] - t > saut:
                    relance = True
                    break
            if pos < n and not relance:  # fin de flux
                while pos < n:
                    yield pos, None
                    pos += 1
    finally:
        conteneur.close()


def en_tache_de_fond(generateur, profondeur: int = 6):
    """Fait tourner un générateur dans un fil (décodage pendant le calcul)."""
    file: queue.Queue = queue.Queue(maxsize=profondeur)
    fin = object()
    erreurs = []

    def pompe():
        try:
            for element in generateur:
                file.put(element)
        except BaseException as e:  # remonté au consommateur
            erreurs.append(e)
        finally:
            file.put(fin)

    fil = threading.Thread(target=pompe, daemon=True)
    fil.start()
    while True:
        element = file.get()
        if element is fin:
            break
        yield element
    fil.join()
    if erreurs:
        raise erreurs[0]
