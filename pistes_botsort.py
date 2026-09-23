"""BoT-SORT réécrit, sans boxmot ni ultralytics (tous deux AGPL-3.0).

D'après l'article (Aharon, Orfaig, Bobrovsky, « BoT-SORT: Robust Associations
Multi-Pedestrian Tracking », arXiv:2206.14651) et son dépôt de référence
NirAharon/BoT-SORT (licence MIT) :

* Kalman à vitesse constante sur (cx, cy, w, h), bruits proportionnels à la
  taille de la boîte ;
* compensation du mouvement caméra : la transformation affine entre deux
  frames est appliquée à l'état prédit des pistes avant association ;
* association en cascade à la ByteTrack : détections sûres d'abord (IoU et
  apparence fusionnées par un minimum, l'apparence n'étant retenue que si
  l'IoU passe la porte de proximité), puis détections faibles sur les pistes
  restées orphelines (IoU seule), puis pistes non confirmées ;
* apparence : descripteur ReID lissé par moyenne exponentielle.

Les réglages par défaut sont ceux avec lesquels BoT-SORT a été mesuré sur
UAVDT (README, « Comparaison avec des trackers établis ») : pureté 0,987,
1,03 morceau par objet.

Le module ne lit ni vidéo ni fichier : l'appelant fournit à chaque frame les
boîtes, les scores, les descripteurs et la transformation caméra. C'est ce qui
rend l'association rejouable à partir d'un cache (associe-pistes.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class ParamsBotSort:
    track_high_thresh: float = 0.5    # détection « sûre » (1re association)
    track_low_thresh: float = 0.1     # en dessous : ignorée
    new_track_thresh: float = 0.6     # score minimal pour ouvrir une piste
    track_buffer: int = 30            # frames avant d'abandonner une piste perdue
    match_thresh: float = 0.8         # coût max, 1re association
    second_match_thresh: float = 0.5  # coût max, détections faibles
    unconfirmed_match_thresh: float = 0.7
    proximity_thresh: float = 0.5     # distance IoU max pour que l'apparence compte
    appearance_thresh: float = 0.25   # distance cosinus/2 max
    fuse_score: bool = False          # IoU pondérée par le score en 1re association
    ema_alpha: float = 0.9            # lissage du descripteur
    frame_rate: float = 30.0

    @property
    def max_time_lost(self) -> int:
        return int(self.frame_rate / 30.0 * self.track_buffer)


# ---------------------------------------------------------------------------
# Kalman
# ---------------------------------------------------------------------------

POIDS_POSITION = 1.0 / 20
POIDS_VITESSE = 1.0 / 160
_F = np.eye(8)
_F[:4, 4:] = np.eye(4)
_H = np.eye(4, 8)


def kalman_initie(mesure: np.ndarray):
    w, h = mesure[2], mesure[3]
    moyenne = np.r_[mesure, np.zeros(4)]
    std = [2 * POIDS_POSITION * w, 2 * POIDS_POSITION * h,
           2 * POIDS_POSITION * w, 2 * POIDS_POSITION * h,
           10 * POIDS_VITESSE * w, 10 * POIDS_VITESSE * h,
           10 * POIDS_VITESSE * w, 10 * POIDS_VITESSE * h]
    return moyenne, np.diag(np.square(std))


def kalman_predit(moyenne, cov):
    w, h = moyenne[2], moyenne[3]
    std = [POIDS_POSITION * w, POIDS_POSITION * h,
           POIDS_POSITION * w, POIDS_POSITION * h,
           POIDS_VITESSE * w, POIDS_VITESSE * h,
           POIDS_VITESSE * w, POIDS_VITESSE * h]
    return _F @ moyenne, _F @ cov @ _F.T + np.diag(np.square(std))


def kalman_corrige(moyenne, cov, mesure):
    w, h = moyenne[2], moyenne[3]
    bruit = np.diag(np.square([POIDS_POSITION * w, POIDS_POSITION * h,
                               POIDS_POSITION * w, POIDS_POSITION * h]))
    proj_cov = _H @ cov @ _H.T + bruit
    gain = np.linalg.solve(proj_cov, _H @ cov).T
    moyenne = moyenne + gain @ (mesure - _H @ moyenne)
    cov = cov - gain @ proj_cov @ gain.T
    return moyenne, cov


def xyxy_vers_xywh(b):
    b = np.asarray(b, float)
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                     b[2] - b[0], b[3] - b[1]])


# ---------------------------------------------------------------------------
# Compensation du mouvement caméra
# ---------------------------------------------------------------------------


def gmc_flot_optique(precedente: np.ndarray, courante: np.ndarray,
                     echelle: float = 1.0) -> Optional[np.ndarray]:
    """Affine 2x3 amenant la frame précédente sur la courante.

    Flot optique clairsemé (goodFeaturesToTrack + Lucas-Kanade pyramidal) et
    similitude robuste (RANSAC), comme le `sparseOptFlow` de BoT-SORT et
    `build_gmc` de group_id_association.py, mêmes réglages. Les images sont
    des niveaux de gris déjà réduits d'un facteur `echelle` ; la translation
    est ramenée en pixels de l'image d'origine.
    """
    coins = cv2.goodFeaturesToTrack(precedente, maxCorners=1000,
                                    qualityLevel=0.01, minDistance=8,
                                    blockSize=3)
    if coins is None or len(coins) < 8:
        return None
    suivis, statut, _err = cv2.calcOpticalFlowPyrLK(
        precedente, courante, coins, None, winSize=(21, 21), maxLevel=3)
    if suivis is None or statut is None:
        return None
    ok = statut.reshape(-1) == 1
    if ok.sum() < 8:
        return None
    matrice, _inliers = cv2.estimateAffinePartial2D(
        coins[ok], suivis[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrice is None:
        return None
    matrice = matrice.astype(float)
    matrice[:, 2] /= echelle
    return matrice


# ---------------------------------------------------------------------------
# Piste
# ---------------------------------------------------------------------------

NOUVELLE, SUIVIE, PERDUE, SUPPRIMEE = range(4)


class Piste:
    """Un objet suivi : état de Kalman, descripteur lissé, détections reçues."""

    def __init__(self, boite, score, desc, ref, frame, ident, params):
        self.id = ident
        self.params = params
        self.moyenne, self.cov = kalman_initie(xyxy_vers_xywh(boite))
        self.score = score
        self.desc = None
        self._maj_desc(desc)
        self.etat = SUIVIE
        self.confirmee = False
        self.frame_debut = frame
        self.frame = frame
        # (frame, référence de la détection, boîte xyxy, score)
        self.membres: List[Tuple[int, Any, np.ndarray, float]] = [
            (frame, ref, np.asarray(boite, float), float(score))]

    def _maj_desc(self, desc):
        if desc is None:
            return
        desc = desc / max(np.linalg.norm(desc), 1e-12)
        if self.desc is None:
            self.desc = desc
        else:
            a = self.params.ema_alpha
            self.desc = a * self.desc + (1 - a) * desc
            self.desc /= max(np.linalg.norm(self.desc), 1e-12)

    @property
    def boite(self) -> np.ndarray:
        cx, cy, w, h = self.moyenne[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predit(self, pas: int = 1):
        for _ in range(max(1, pas)):
            moyenne = self.moyenne.copy()
            if self.etat != SUIVIE:
                moyenne[6:8] = 0  # une piste perdue ne change plus de taille
            self.moyenne, self.cov = kalman_predit(moyenne, self.cov)

    def compense(self, matrice: np.ndarray):
        """Transporte l'état dans le repère de la frame courante."""
        rot, trans = matrice[:, :2], matrice[:, 2]
        echelle = float(np.sqrt(abs(np.linalg.det(rot)))) or 1.0
        t = np.zeros((8, 8))
        t[0:2, 0:2] = rot
        t[2:4, 2:4] = np.eye(2) * echelle
        t[4:6, 4:6] = rot
        t[6:8, 6:8] = np.eye(2) * echelle
        self.moyenne = t @ self.moyenne
        self.moyenne[:2] += trans
        self.cov = t @ self.cov @ t.T

    def corrige(self, boite, score, desc, ref, frame):
        self.moyenne, self.cov = kalman_corrige(self.moyenne, self.cov,
                                                xyxy_vers_xywh(boite))
        self._maj_desc(desc)
        self.score = score
        self.etat = SUIVIE
        self.confirmee = True
        self.frame = frame
        self.membres.append((frame, ref, np.asarray(boite, float),
                             float(score)))


# ---------------------------------------------------------------------------
# Coûts et affectation
# ---------------------------------------------------------------------------


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aire_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    aire_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aire_a[:, None] + aire_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def affecte(cout: np.ndarray, seuil: float):
    """Affectation optimale où chaque paire retenue coûte au plus `seuil`.

    Même problème que `lap.lapjv(extend_cost=True, cost_limit=seuil)` du dépôt
    de référence : la matrice est complétée de lignes et colonnes fictives à
    seuil/2, si bien qu'apparier (i, j) n'est préféré à les laisser libres
    tous les deux que si cout[i, j] < seuil.
    """
    n, m = cout.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    etendu = np.full((n + m, m + n), seuil / 2.0)
    etendu[n:, m:] = 0.0
    etendu[:n, :m] = cout
    lignes, colonnes = linear_sum_assignment(etendu)
    paires = [(int(i), int(j)) for i, j in zip(lignes, colonnes)
              if i < n and j < m and cout[i, j] <= seuil]
    pris_i = {i for i, _ in paires}
    pris_j = {j for _, j in paires}
    return (paires, [i for i in range(n) if i not in pris_i],
            [j for j in range(m) if j not in pris_j])


class _Det:
    __slots__ = ("boite", "score", "desc", "ref")

    def __init__(self, boite, score, desc, ref):
        self.boite, self.score, self.desc, self.ref = boite, score, desc, ref


def _cout_iou(pistes, dets):
    if not pistes or not dets:
        return np.zeros((len(pistes), len(dets)))
    return 1.0 - iou_xyxy(np.array([p.boite for p in pistes]),
                          np.array([d.boite for d in dets]))


def _cout_fusionne(pistes, dets, params, fuse_score):
    """min(IoU, apparence), l'apparence ne comptant que près de la piste."""
    cout = _cout_iou(pistes, dets)
    if cout.size == 0:
        return cout
    loin = cout > params.proximity_thresh
    if fuse_score:
        cout = 1.0 - (1.0 - cout) * np.array([d.score for d in dets])[None, :]
    if all(p.desc is not None for p in pistes) and all(
            d.desc is not None for d in dets):
        a = np.array([p.desc for p in pistes])
        b = np.array([d.desc for d in dets])
        app = np.maximum(0.0, 1.0 - a @ b.T) / 2.0
        app[app > params.appearance_thresh] = 1.0
        app[loin] = 1.0
        cout = np.minimum(cout, app)
    return cout


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class BotSort:
    """Une instance par classe et par tronçon continu de frames.

    `ids` est un compteur partagé (liste d'un entier) pour que les identifiants
    internes restent uniques entre instances ; la numérotation finale des
    group_id est refaite à la sortie.
    """

    def __init__(self, params: Optional[ParamsBotSort] = None, ids=None):
        self.params = params or ParamsBotSort()
        self.ids = ids if ids is not None else [0]
        self.suivies: List[Piste] = []
        self.perdues: List[Piste] = []
        self.toutes: List[Piste] = []
        self.frame: Optional[int] = None

    def _nouvel_id(self):
        self.ids[0] += 1
        return self.ids[0]

    def update(self, frame: int, boites, scores, descs=None, matrice=None,
               refs=None):
        """Avance à `frame` (entier croissant) et associe ses détections."""
        p = self.params
        boites = np.asarray(boites, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        refs = list(range(len(boites))) if refs is None else list(refs)
        premiere = self.frame is None
        pas = 1 if premiere else max(1, frame - self.frame)
        self.frame = frame

        hautes, basses = [], []
        for k in range(len(boites)):
            if scores[k] <= p.track_low_thresh:
                continue
            d = _Det(boites[k], float(scores[k]),
                     None if descs is None else np.asarray(descs[k], float),
                     refs[k])
            if scores[k] > p.track_high_thresh:
                hautes.append(d)
            else:
                d.desc = None  # comme la référence : pas de ReID en 2e passe
                basses.append(d)

        non_confirmees = [t for t in self.suivies if not t.confirmee]
        confirmees = [t for t in self.suivies if t.confirmee]
        reserve = confirmees + self.perdues
        for t in reserve + non_confirmees:
            t.predit(pas)
            if matrice is not None:
                t.compense(matrice)

        actives, perdues_ici = [], []

        # 1. détections sûres contre pistes suivies et perdues
        cout = _cout_fusionne(reserve, hautes, p, p.fuse_score)
        paires, libres_t, libres_d = affecte(cout, p.match_thresh)
        for i, j in paires:
            d = hautes[j]
            reserve[i].corrige(d.boite, d.score, d.desc, d.ref, frame)
            actives.append(reserve[i])

        # 2. détections faibles contre pistes encore suivies (IoU seule)
        restantes = [reserve[i] for i in libres_t
                     if reserve[i].etat == SUIVIE]
        paires, libres_t2, _ = affecte(_cout_iou(restantes, basses),
                                       p.second_match_thresh)
        for i, j in paires:
            d = basses[j]
            restantes[i].corrige(d.boite, d.score, d.desc, d.ref, frame)
            actives.append(restantes[i])
        for i in libres_t2:
            restantes[i].etat = PERDUE
            perdues_ici.append(restantes[i])

        # 3. pistes non confirmées (une seule frame) contre le reste des sûres
        hautes = [hautes[j] for j in libres_d]
        cout = _cout_fusionne(non_confirmees, hautes, p, p.fuse_score)
        paires, libres_nc, libres_d = affecte(cout, p.unconfirmed_match_thresh)
        for i, j in paires:
            d = hautes[j]
            non_confirmees[i].corrige(d.boite, d.score, d.desc, d.ref, frame)
            actives.append(non_confirmees[i])
        for i in libres_nc:
            non_confirmees[i].etat = SUPPRIMEE

        # 4. nouvelles pistes
        for j in libres_d:
            d = hautes[j]
            if d.score < p.new_track_thresh:
                continue
            t = Piste(d.boite, d.score, d.desc, d.ref, frame,
                      self._nouvel_id(), p)
            # À la première frame d'un tronçon, pas de confirmation à attendre.
            t.confirmee = premiere
            self.toutes.append(t)
            actives.append(t)

        # 5. âge des pistes perdues
        perdues = [t for t in self.perdues
                   if t.etat == PERDUE and t not in actives] + perdues_ici
        vivantes = []
        for t in perdues:
            if frame - t.frame > p.max_time_lost:
                t.etat = SUPPRIMEE
            else:
                vivantes.append(t)
        self.suivies = [t for t in actives if t.etat == SUIVIE]
        self.perdues = [t for t in vivantes if t.etat == PERDUE]
        self._retire_doublons()

    def _retire_doublons(self):
        """Une piste perdue qui recouvre une suivie : garder la plus ancienne."""
        if not self.suivies or not self.perdues:
            return
        d = _cout_iou(self.suivies, self.perdues)
        jeter_s, jeter_p = set(), set()
        for i, j in zip(*np.where(d < 0.15)):
            if (self.suivies[i].frame - self.suivies[i].frame_debut
                    > self.perdues[j].frame - self.perdues[j].frame_debut):
                jeter_p.add(j)
            else:
                jeter_s.add(i)
        for i in jeter_s:
            self.suivies[i].etat = SUPPRIMEE
        for j in jeter_p:
            self.perdues[j].etat = SUPPRIMEE
        self.suivies = [t for i, t in enumerate(self.suivies)
                        if i not in jeter_s]
        self.perdues = [t for j, t in enumerate(self.perdues)
                        if j not in jeter_p]

    def pistes(self) -> List[Piste]:
        """Pistes confirmées (au moins deux détections, ou nées en 1re frame)."""
        return [t for t in self.toutes if t.confirmee]
