# Association des pistes (group_id) — notes de reprise pour la session tracker

> **Fait le 2026-09-23 (après-midi)** : `associe-pistes.py` (BoT-SORT réécrit
> sans boxmot, parité mesurée sur UAVDT) a produit les pistes des vols
> 0000011, 0000012 et 0000018, en local dans
> `Datasets/real/AnafiUKR/pistes/<vol>/`, au format proposé plus bas (plus
> `coarse`, `labels`, `shapes`). Visualisation : `rendu-pistes.py`,
> `planche-pistes.py`. Tout est décrit dans le README, section « Pistes
> BoT-SORT sur un vol entier ». Le reste de ce document est l'état du matin.

Document autonome, écrit le 2026-09-23 matin. S'adresse à une session de
programmation qui démarre à froid. À lire en entier avant de coder.

## Objectif

Les vols CAMPAGNE3 viennent d'être annotés par SAM 3 **frame par frame**
(détections, sans identité) sur le cluster Slurm. Il faut maintenant
**associer les détections d'une frame à l'autre en pistes** (`group_id`),
**hors ligne** (rejouable sans relancer l'inférence), par vol. La classe fine
par piste (VLM) et la revue humaine (client) viennent APRÈS, hors périmètre.

## Où en est le chantier (2026-09-23)

- Le lot **AnafiUKR** (3 vols : 0000011, 0000012, 0000018 — 78 742 frames RGB)
  tourne sur le cluster, fin prévue vers 14-15 h le 2026-09-23.
  **Le vol 0000011 est déjà complet** (15 954 frames, 7 tronçons `.done`) :
  c'est le terrain de jeu immédiat.
- CAMPAGNE3 complète (122 tâches, 277 074 frames) est prête à soumettre après
  validation (détails dans `slurm/README.md`).
- Les sorties du lot : **JSONL compacté**, PAS des JSON X-AnyLabeling par
  frame. C'est la principale différence avec ce qui existait avant le
  2026-09-22 — tout le code d'association existant lit des JSON par frame.

## Données d'entrée (toutes visibles depuis le PC, `/media` est monté)

- Détections : `/media/users/cbarbier/annots-sam3/annots/<vol>/`
  - `<vol>_chunk_<début>_<fin>.jsonl.zst` — compacté, tronçon terminé
    (fini, immuable, sa marqueur `.done` est à côté) ;
  - `<vol>_chunk_<début>_<fin>.jsonl` — tronçon **en cours d'écriture**
    (ne pas le lire, ou seulement ses lignes complètes : la dernière ligne
    peut être tronquée si un job vient d'être tué).
  - Une ligne JSON par frame annotée :
    ```json
    {"flight":"0000011","sample_index":21287,"dts_ticks":39122964,
     "dts_s":1304.0988,"utc_us":...,"stem":"0000011_rgb_1304098",
     "width":3840,"height":2160,
     "shapes":[{"label":"car","score":0.88,
                "points":[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                "shape_type":"rectangle",
                "attributes":{"source":"sam3","prompt":"car",
                              "coarse":"vehicle","uncertain":false}}],
     "t_inf_s":6.5}
    ```
    (les frames en échec portent `"error":"..."` à la place des shapes)
- Vidéos sources : `/media/shared/projets/pendragon/donnees/capa/inputs/NP_data_CAMPAGNE3/<date>/<vol>_video.MP4`
  — **lecture seule, ne jamais y écrire**. Le flux annoté est la **piste
  vidéo 0 (DefaultVideo)**. Les frames extraites n'existent PAS sur disque
  (quotas d'inodes) : le décodage se fait en streaming, voir
  `annote-video-sam3.py` (seek par `dts_ticks`, `frame.pts == dts_ticks`)
  et `planche-controle.py` (exemple minimal de résolution
  tick ↔ sample_index depuis l'index).
- Index par vol : `/media/users/cbarbier/annots-sam3/campagne3_index/annotation/<vol>/index/samples_default.csv`
  (une ligne par sample du flux : `sample_index`, `dts_ticks`, `timescale`
  (30000), `valid`, `mode` (RGB|IR), télémétrie) + `modes.json`.
- Taxonomie : `campagne3_index/CodeAnnotationPrep/taxonomy.yaml`
  (labels fins, `coarse`, `affiliation`).

## Code existant — ne pas le réécrire, l'adapter

- `anylabeling-views/labeling/utils/group_id_association.py` — attention,
  le vrai chemin : **`X-AnyLabeling/anylabeling/views/labeling/utils/group_id_association.py`**.
  Implémentation complète SANS Qt : IoU tamponnée en deux passes
  (BIoU 0,3 puis 1,0), bruit de Kalman proportionnel à la taille des objets,
  porte sur la distance des centroïdes normalisée par la diagonale,
  **compensation de mouvement caméra** (flot optique clairsemé
  `goodFeaturesToTrack` + `calcOpticalFlowPyrLK` + `estimateAffinePartial2D`),
  recollement des fragments a posteriori, interpolation des trous.
  **Elle charge les frames depuis des fichiers image** (`cv2.imread`) :
  c'est ce qu'il faut adapter en décodage streaming PyAV.
- `associe-group-ids.py` (racine) — façade CLI : lit un dossier de JSON
  X-AnyLabeling, appelle le module, réécrit les `group_id`. Options utiles :
  `--dry-run`, `--backup DIR`, `--max-dist`, `--min-len`, `--merge-gap`,
  `--merge-dist`, `--interpolate`, `--motion-comp`...
- Mesures de référence (dans `README.md` racine, sections « Association des
  group_id » et « Pistes pour la classe fine ») :
  - sur les annotations 1080p de Campagne_1 : 295 détections → 12 pistes
    (>= 2 frames), piste max 132 frames SANS un trou, 9,6 s de traitement ;
  - sur UAVDT avec de **vraies détections SAM 3** : tracker maison réglage
    resserré (`buffer2=0.3 max_dist=1.0 max_age=10`) : pureté 0,939,
    1,28 morceau/objet, 82 % d'un seul tenant, 2 ms/frame.
    ByteTrack : 0,985 mais 2,26 morceaux/objet. BoT-SORT (boxmot,
    `.venv-tracking/`, **AGPL-3.0**) : 0,987, 1,03 morceau — mais ReID sur
    GPU à réparer et 547 ms/frame sur CPU.
  - **Le 30 fps est ce qui rend l'association bonne** : pureté 0,955 à
    30 fps contre 0,77 à 1 fps (mesuré UAVDT) — les données sont à
    cadence native, ne pas sous-échantillonner avant association.
- Le dialogue client `Tool > Group ID Tracker` et `utils/track_review.py`
  (revue par piste) consomment des JSON X-AnyLabeling par frame : une
  exportation JSONL → JSON par frame servira plus tard à la revue ; ne pas
  en faire la sortie principale (quotas d'inodes : 78 742 frames au lot,
  ~280 000 à CAMPAGNE3).

## Livrable proposé (à discuter avec l'utilisateur avant de coder)

`pistes-<paramètres>.jsonl` **par vol**, à côté des détections (les
détections restent immuables) :

```json
{"group_id": 7, "label": "car", "n_frames": 42,
 "sample_min": 21340, "sample_max": 21581,
 "frames": [{"i": 21340, "bbox": [x1,y1,x2,y2], "score": 0.9, "interpolated": false},
            ...],
 "stats": {"iou_moyen": 0.71, "taille_mediane_px": 18}}
```

- rejouable : re-générer après retuning = relire les JSONL, quelques minutes ;
- consommable par le VLM par piste (session suivante) et par la revue
  humaine (export vers JSON par frame des seules pistes retenues) ;
- le `group_id` doit être **stable par vol** (numérotation 1..N) et cohérent
  entre relances pour un même jeu de paramètres.

## Pièges connus (tous mesurés ou constatés, les ignorer coûte des heures)

1. **Trous IR** : le DefaultVideo est un rendu d'affichage pendant les
   rafales thermiques — ces frames n'existent pas dans les JSONL (ex. vol
   0000011 : 69 % du vol seulement). Les pistes seront interrompues pendant
   des **minutes** (jusqu'à ~7 min) : le recollement a posteriori
   (`--merge-gap` en FRAMES) ne doit PAS fusionner à travers ces trous
   à moins d'un réglage explicite ; documenter ce qui est fait.
2. **Les indices de frames ne sont pas contigus** : `sample_index` saute
   les frames IR ; l'écart entre frames consécutives annotées n'est pas
   1. L'âge des pistes doit se compter en frames RÉELLEMENT vues (ou en
   temps `dts_s`), jamais en écart d'indices.
3. **Petits objets (15-25 px)** : l'IoU brute s'annule à 5 px de
   déplacement sur une boîte de 16 px — la BIoU et la porte normalisée
   du module existant sont là pour ça ; ne pas retomber sur de l'IoU nue.
4. **Véhicules denses** : le défaut du module (réglé sur des piétons
   épars) « capte le voisin » ; le réglage resserré
   `buffer2=0.3 max_dist=1.0 max_age=10` est le point de départ mesuré.
5. **Compensation de mouvement indispensable** : sous le drone, la scène
   défile — sans elle, la piste du marcheur isolé ne tient pas 132 frames.
6. **JSONL en cours d'écriture** : ne traiter que les vol dont TOUS les
   tronçons portent `.done` (le lot AnafiUKR finit vers 14-15 h ; 0000011
   est complet dès maintenant).
7. **Frames sur disque : aucune.** La compensation de mouvement exige les
   images → décodage PyAV en streaming du flux 0, seek par
   `dts_ticks` (timescale 30000, `frame.pts == dts_ticks`, pas de
   B-frames). Voir `annote-video-sam3.py` (l. `container.seek`,
   `sample_de_tick`) et `planche-controle.py`.

## Comment tester

- Vol complet : 0000011 (15 954 frames, 7 tronçons). Sortie ~90 Mo décomprimé.
- Zone de stress (campement dense, 85-98 shapes/frame) : samples
  19830-22330, la piste du test « frame 21287 » y est.
- Comparaison rapide : `planche-controle.py` peut dessiner les pistes
  (adapter `charger_annotations` pour lire `pistes-*.jsonl` et colorer par
  `group_id` au lieu de par label) — contrôle visuel immédiat.
- Les anciens outils (`associe-group-ids.py --dry-run --backup`) restent
  utilisables après export JSONL → JSON par frame ; s'en servir comme
  vérité de parité pendant le développement, pas comme sortie finale.

## Hors périmètre (sessions suivantes)

- Classe fine par piste au VLM (Qwen3.5-4B local, `mesure-vlm.py`,
  `X-AnyLabeling-Server/app/utils/crop_classifier.py`) ;
- revue humaine dans le client (`Track Review` existe) ;
- VisibleVideo (RGB des périodes IR) et flux thermique — index déjà produits
  (`samples_visible.csv`, `samples_thermal.csv`), annotation pas encore lancée ;
- l'association elle-même peut tourner sur le PC (2 ms/frame maison :
  un vol en quelques minutes une fois les frames décodées ; le décodage 4K
  logiciel domine le coût) ou sur `cpu01` du cluster — voir
  `SLURM_INSTRUCTIONS.md` pour les règles du cluster (ignoré par git,
  ne pas le committer).
