# Petits objets en haute résolution avec SAM 3 / SAM 3.1

Plan de reprise. Rédigé le 2026-09-01, pour une session ultérieure.

## Contexte

L'annotation à venir porte **à parts égales sur des images fixes et de la vidéo**, avec des
objets pouvant descendre à **32×32 px et jusqu'à moins de 8×8 px** dans l'image d'origine.
La question posée : SAM 3 / SAM 3.1 encaissent-ils la haute résolution, et risque-t-on de
perdre les petits objets ?

**Oui, et c'est structurel.** Ce document consigne les chiffres relevés dans le code, la
décision de fond qui en découle (revenir à SAM 3), et les lots de travail dans l'ordre où
ils se justifient. Rien n'a été modifié : tout ce qui suit est de la lecture de code.

## Ce qui est établi

### La chaîne de résolution

Ni le client ni le serveur ne dégradent l'image avant le modèle. Le client envoie les
**octets bruts du fichier** en base64 (`X-AnyLabeling/anylabeling/services/auto_labeling/remote_server.py:275-286`),
le serveur fait un `cv2.imdecode` sans limite de taille ni resize
(`X-AnyLabeling-Server/app/api/predict.py:39-53`). Le champ
`performance.max_image_size` de `configs/server.yaml` n'est jamais lu.

Toute la perte est **interne au modèle** :

1. **Écrasement en 1008×1008 carré, sans letterbox ni padding.**
   `app/models/sam3/model/sam3_image_processor.py:26` — `v2.Resize(size=(1008, 1008))`.
   Idem sur tous les chemins vidéo : `app/models/sam3/model/io_utils.py:57, 315, 358`.
2. **Backbone ViT (Perception Encoder), `patch_size=14`** → grille de **72×72 tokens**
   (`app/models/sam3/model_builder.py:87-96`).
3. **La détection ne voit que ce niveau 72×72.** `num_feature_levels=1`
   (`model_builder.py:156`, usage `sam3/model/sam3_image.py:126-127` :
   `vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels:]`).
   Les niveaux FPN 288 et 144 **ne servent qu'au décodeur de masques**. Ce n'est pas un
   détecteur multi-échelle au sens Deformable-DETR.
4. **Masques produits en 288×288** (`sam3_tracker_base.py:140` :
   `image_size // backbone_stride * 4`), puis `F.interpolate(..., bilinear)` vers la
   résolution native (`sam3_image_processor.py:235-243`).
5. **200 requêtes maximum par prompt et par image** (`model_builder.py:189`,
   `num_queries=200`, `num_o2m_queries=0`).

### Le tableau des tailles

| Source | Facteur x / y | 1 token de détection couvre | 1 cellule de masque |
|---|---|---|---|
| 840×712 (DroneVehicle) | ×1,20 / ×1,42 (**agrandi**) | 11,7 × 9,9 px | 2,9 × 2,5 px |
| 1920×1080 | ×0,525 / ×0,933 | 26,7 × 15 px | 6,7 × 3,8 px |
| 3840×2160 | ×0,26 / ×0,47 | 53 × 30 px | 13 × 7 px |
| 4000×3000 | ×0,25 / ×0,34 | 56 × 42 px | 14 × 10 px |

**Heuristique de travail** — **mesurée le 2026-09-02, et trop pessimiste** : elle annonce
qu'il faut ≥ 2 tokens, soit ≥ 28 px après resize. Or sur VTUAV à `image_size` 1008, des
objets à 8,4-16,8 px en espace modèle sont retrouvés à 0,817 de rappel, et ceux à
4,2-8,4 px à 0,875. SAM 3 encaisse bien plus bas. L'énoncé d'origine : pour que
le détecteur ait une chance, l'objet doit couvrir ≥ 2 tokens, soit ≥ 28 px après resize.
En pixels natifs : **≥ 53 px de large en 1920×1080**, **≥ 110 px en 4K**.

Confronté aux tailles cibles :

| Objet natif | En 840×712 | En 1920×1080 | En 4K |
|---|---|---|---|
| 32 px | 2,7 tokens — passable | **1,2 token — limite** | 0,6 token — perdu |
| 16 px | 1,4 token — limite | 0,6 token — perdu | 0,3 token — perdu |
| 8 px | **0,7 token — perdu** | 0,3 token — perdu | 0,15 token — perdu |

**Conclusion : sans tuilage, rien en dessous de ~50 px n'est fiable en 1920×1080, et le
seuil des 8 px est hors d'atteinte à toute résolution source, y compris en 840×712.**

### Aggravants propres à SAM 3.1 (le modèle actif aujourd'hui)

Tous dans `sam3-official/sam3/model_builder.py:1165-1200`, **aucun surchargé** par
`app/models/segment_anything_3_1_video.py:135-142` (qui ne passe que `checkpoint_path`,
`use_fa3`, `bpe_path`) :

| | SAM 3.1 (actif) | SAM 3 vendorisé |
|---|---|---|
| `max_num_objects` | **16** (`model_builder.py:1073`) | non passé → `-1` → 10000 (illimité) |
| `suppress_det_close_to_boundary` | **`True`**, marge 2,5 % | **`False`** (`app/models/sam3/model_builder.py:833, 859`) |
| `masklet_confirmation_enable` | **`True`**, 3 détections consécutives | `False` |
| `score_threshold_detection` | 0,4 | 0,5 |
| `new_det_thresh` | 0,65 | 0,7 |
| `image_size` | **figé à 1008**, pas d'argument dans `build_sam3_multiplex_video_predictor` | **paramètre bout-en-bout** |

Détails qui comptent pour des petits objets :

- **Le plafond de 16 objets jette par score croissant.**
  `sam3/model/sam3_video_base.py:54-80` (`get_new_det_gpu_ids`, appelé depuis
  `sam3_multiplex_base.py:1212`) puis `_drop_new_det_with_obj_limit`
  (`sam3_video_base.py:2098-2112`) : `argsort(det_scores)[::-1][:num_to_keep]`. Les petits
  objets scorent bas, ils partent en premier. Sur une vue drone à 40 véhicules, il en
  reste 16.
- **La suppression de bord élimine tout objet dont le *centre* est à moins de 2,5 % du
  bord** — 48 px en largeur sur 1920, 27 px en hauteur sur 1080
  (`app/models/sam3/model/sam3_video_base.py:306-323`). C'est **rédhibitoire pour du
  tuilage**, où les objets de bord de tuile sont précisément ce que le recouvrement doit
  rattraper.
- **Le seuil porte sur un produit de deux sigmoïdes** :
  `out_probs = sigmoid(logits) * sigmoid(presence)` (`sam3_image_processor.py:219-222`).
  Plus sévère qu'il n'y paraît.
- **`conf_threshold: 0.25` de la config ne peut rien récupérer** : il s'applique côté
  serveur (`segment_anything_3_video.py:1661-1663`) **après** les seuils durs du modèle.
  Le descendre est sans effet.

### Pertes gratuites, indépendantes du modèle

- **Double compression JPEG sur le chemin vidéo.** Le client extrait les frames en JPEG
  `-qscale:v 2` (`X-AnyLabeling/anylabeling/views/labeling/utils/video.py:269`) ou
  `cv2.imwrite` q95 par défaut (`:452-466`) ; puis le serveur les **ré-encode** en JPEG q95
  dans son dossier temporaire (`app/models/segment_anything_3_video.py:63-79`,
  `cv2.imwrite` sans `IMWRITE_JPEG_QUALITY`). Deux passes DCT + sous-échantillonnage
  chroma, ce qui coûte le plus aux objets de quelques pixels.
- **Une seule composante connexe conservée par masque.**
  `app/models/segment_anything_3.py:378-415` et `segment_anything_3_video.py:1730-1764` :
  `max(contours, key=cv2.contourArea)`. Un objet fragmenté perd ses morceaux secondaires.
  Aucun filtre d'aire minimale en revanche — un masque d'un pixel ressort.

## Décision de fond : revenir à SAM 3

Pour ce cas d'usage, **SAM 3 vendorisé est le bon choix, pas SAM 3.1** :

1. Pas de plafond de 16 objets.
2. Pas de suppression de bord — condition nécessaire au tuilage.
3. Pas de confirmation sur 3 détections consécutives, qui tue les objets intermittents.
4. `image_size` réglable bout en bout : `segment_anything_3.py:31` et
   `segment_anything_3_video.py:219` lisent `params["image_size"]` et le passent au
   builder, qui gère `image_size != 1008` en retirant les clés `freqs_cis` recalculées
   (`app/models/sam3/model_builder.py:878-895`).
5. Le chemin image de SAM 3 **découpe déjà le prompt multi-classes** sur `,` ou `.` et fait
   un forward par classe en réutilisant le backbone (`segment_anything_3.py:128-184`).
   Le patch perso à base de `reset_session` n'est nécessaire que sur le chemin vidéo.

**Mesuré le 2026-09-01, et la crainte était infondée : SAM 3 ne perd rien en capacité,
il en gagne.** Sur la vidéo de 754 frames en 1920×1080, prompt « Person », SAM 3 propage
**754/754 frames à 4,64 fps** sans déclencher le garde-fou, là où SAM 3.1 s'arrête à
**368/754 à 3,42 fps**. La note du README qui annonçait l'inverse était fausse. Détail et
protocole : section « SAM 3 ou SAM 3.1 ? » du README, harnais `./mesure-plafond-frames.py`.

Le seul avantage réel de SAM 3.1 est la **densité de détection** : sur les 133 frames que
les deux modèles identifient identiquement comme contenant des personnes, il trouve 296
instances contre 164. Sans vérité terrain on ne sait pas si c'est du rappel ou du doublon —
c'est au lot L0 de trancher, et c'est désormais la seule question qui pourrait justifier
de revenir à 3.1.

Nuance : les trois premiers aggravants de SAM 3.1 sont **annulables par override après
construction**, exactement comme `batched_grounding_batch_size` l'est déjà
(`segment_anything_3_1_video.py:152-161`). Ce n'est donc pas une raison absolue de
basculer, mais SAM 3 les donne gratuitement **et** donne `image_size` en plus.

## Deux pièges à ne pas retomber dedans

- **Ne pas « corriger » l'écrasement par un letterbox.** Tentant, mais faux. En 1920×1080,
  padder en 1920×1920 avant l'écrasement donnerait `sx = sy = 0,525` : la résolution
  horizontale ne gagne rien et **la verticale tombe de 0,933 à 0,525**. L'écrasement
  anisotrope échantillonne davantage. De plus le modèle a été **entraîné avec cet
  écrasement** : `RandomResizeAPI` avec `square: true` court-circuite
  `get_size_with_aspect_ratio` dans tous les configs d'éval
  (`sam3-official/sam3/train/transforms/basic_for_api.py:165-181`,
  `sam3/train/configs/eval_base.yaml:57-62`). Le letterbox serait hors distribution.
- **Ne pas espérer récupérer des détections en baissant `conf_threshold`.** Voir plus haut :
  les seuils durs du modèle sont en amont.

## Lots de travail

### L0 — Mesurer la courbe taille → détection sur les données réelles

Préalable à tout choix technique, et cohérent avec la méthode du reste du projet.

- Prendre un échantillon d'images déjà annotées à la main, à la résolution de production.
- Sortir l'**histogramme des tailles de boîtes** (min(largeur, hauteur) en px natifs) :
  c'est lui qui pilote toutes les décisions de tuilage.
- Faire tourner SAM 3 (chemin image) sur cet échantillon et tracer le **rappel en fonction
  de la taille de l'objet**, par tranches (0-8, 8-16, 16-32, 32-64, 64-128, >128 px).
  Objectif : trouver la taille de bascule réelle, et vérifier ou infirmer l'heuristique
  des 2 tokens.
- Rejouer la même mesure avec des **crops 1008×1008 à l'échelle 1:1** pris dans les mêmes
  images. L'écart entre les deux courbes est le gain à attendre du tuilage.

Livrable : un tableau dans le README, dans le style des tableaux VRAM existants.

**Les harnais existent depuis le 2026-09-02**, écrits pour L3 et directement réutilisables :
`mesure-rappel-coco.py` fait une passe d'inférence sur un échantillon COCO et **garde les
prédictions brutes avec leur score**, `rappel-par-taille.py` recalcule rappel et précision
par tranche hors ligne — donc une seule passe d'inférence permet de balayer les seuils et
de comparer plusieurs runs sur des lignes alignées.

```bash
./mesure-rappel-coco.py --dataset <jeu> --n 120 --category person \
    --prompt person --conf 0.15 --out run.json
./rappel-par-taille.py run-*.json --score 0.25 --iou 0.3
```

**Ce qui manque est la donnée, pas l'outil.** Les 18 jeux de `Datasets/release/` n'ont pas
la queue de distribution visée : leurs objets petits sont dans des images de faible
résolution (`rgbtdroneperson` 640×512), et leurs images haute résolution ont des objets
confortables (`nii_mapd_rgb` 4K, côté médian 52 px). Le régime réel — 8 à 32 px natifs en
1920×1080 ou plus — n'est représenté nulle part. **L0 attend les jeux de production.**

Pour comparer d'un jeu à l'autre malgré tout, raisonner en **espace modèle** :
`d × image_size / largeur_source`. C'est la seule grandeur qui transfère.

### L1 — Basculer sur SAM 3 et chiffrer ce que ça coûte — FAIT le 2026-09-01

Résultat : bascule effectuée. `configs/models.yaml` n'active plus que
**`segment_anything_3`** (chemin image) — les deux modèles vidéo sont commentés depuis
que le frame par frame a gagné, voir L3bis. Le plafond de frames de SAM 3 n'existe pas
sur cette séquence (754/754), et il est plus rapide que SAM 3.1. Chiffres dans le README.
Reste de la procédure, conservée pour mémoire :

- `X-AnyLabeling-Server/configs/models.yaml` : décommenter `segment_anything_3` et
  `segment_anything_3_video`, commenter `segment_anything_3_1_video`. Les deux modèles
  vidéo restent mutuellement exclusifs (`ModelRegistry._resolve_exclusive_models`) ;
  `segment_anything_3` (image) est un import du même paquet vendorisé, donc compatible avec
  `segment_anything_3_video`.
- Redémarrer le serveur, vérifier la ligne `Successfully loaded N/N model(s)`,
  se déconnecter/reconnecter depuis le client (il met la liste en cache).
- **Mesurer le plafond de frames de SAM 3** sur la vidéo de référence, au même protocole
  que le tableau du README (le garde-fou `vram_stop_threshold_gb` arrête proprement).
  C'est le chiffre manquant pour arbitrer.
- Vérifier que le patch perso multi-classes (boucle par classe + `reset_session` + décalage
  des group_id) fonctionne toujours : il vit dans `segment_anything_3_video.py`, hérité par
  la 3.1, donc il est déjà sur le bon fichier.

### L2 — Supprimer la double compression JPEG

Dans `app/models/segment_anything_3_video.py:63-79`, `_write_frames_to_temp_dir` :
écrire en PNG, ou en JPEG qualité 100 :

```python
cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
```

PNG est sans perte mais plus lourd sur disque et plus lent à écrire ; JPEG 100 est un bon
compromis. Mesurer l'effet sur le temps d'`init_state` (les frames sont relues par le
loader, `async_loading_frames=True`).

Effet attendu faible mais gratuit. À faire avant L0 si possible, pour ne pas mesurer une
dégradation évitable.

### L3 — Monter `image_size` sur SAM 3 — FAIT le 2026-09-02, RÉSULTAT NÉGATIF

**Mesuré, et c'est une impasse : le rappel se dégrade de façon monotone.** Les trois
valeurs se chargent sans une seule clé manquante, donc la réserve sur le chargement du
checkpoint était infondée ; celle sur le hors-distribution, elle, se vérifie.

Sur 120 images de `Datasets/release/rgb/vtuav_rgb` (1920×1080, `person`, 1158 objets),
prompt « person », seuil 0,25, serveur redémarré à chaque valeur :

| `image_size` | Rappel IoU 0,3 | Rappel IoU 0,5 | Précision | Latence | Crête VRAM |
|---|---|---|---|---|---|
| **1008** | **0,902** | **0,712** | 0,611 | **437 ms** | **4894 Mio** |
| 1400 | 0,894 | 0,647 | 0,625 | 1059 ms | 5782 Mio |
| 1680 | 0,879 | 0,585 | 0,629 | 1585 ms | 6504 Mio |

Ce n'est pas qu'une question de nombre de détections : **la localisation se dégrade**.
L'écart est de 2 points à IoU 0,3 et de 13 à IoU 0,5, et il est concentré sur les petits
objets — sous 16 px natifs, le rappel à IoU 0,5 tombe de 11/16 à 1/16.

**Cause architecturale, indépendante du jeu de données** : `window_size` est figé à
24 tokens et les poids sont entraînés à 1008, où la grille de 72×72 fait exactement
3×3 fenêtres. À 1400 elle en fait 4,17×4,17 — non entier — et à 1680, 5×5.

Contrôlé sur l'écrasement le plus violent (`nii_mapd_rgb`, 3840×2160, objet médian à
13,7 px en espace modèle) : 1680 n'y gagne rien non plus. Ce que `image_size` achète,
c'est un peu de précision, pas du rappel.

Le tableau ci-dessous reste valable pour la géométrie, pas pour l'espoir qui l'accompagnait.


Levier disponible sans écrire une ligne de code : `image_size` dans
`configs/auto_labeling/segment_anything_3.yaml` et `segment_anything_3_video.yaml`.
Valeurs à tester, **multiples de 14** :

| `image_size` | Grille | Tokens | 1 token en 1920×1080 |
|---|---|---|---|
| 1008 (défaut) | 72×72 | 5184 | 26,7 × 15 px |
| 1400 | 100×100 | 10000 (+94 %) | 19,2 × 10,8 px |
| 1680 | 120×120 | 14400 (+178 %) | 16 × 9 px |

Réserves, désormais tranchées :

- ~~Les poids sont entraînés à 1008 et `window_size=24` tokens est fixe : à 1680 la grille
  fait 120×120, soit 5×5 fenêtres au lieu de 3×3. **C'est hors distribution**, le gain
  n'est pas acquis.~~ **Vérifié le 2026-09-02 : c'est bien ce qui se passe, et le gain est
  négatif.**
- Le coût VRAM et le temps croissent en gros comme le carré du nombre de tokens dans
  l'attention. Sur 12 Go, 1680 est probablement hors budget en vidéo ; à tester d'abord
  sur le chemin image.
- `_remove_freqs_cis_keys` est bien appelé et `strict=False` quand `image_size != 1008`
  (`model_builder.py:878-895`) : le chargement du checkpoint ne cassera pas, mais
  **vérifier la liste des `missing_keys` affichée** au chargement.

### L3bis — Architecture frame par frame + association a posteriori — MESURÉ le 2026-09-01

Piste ouverte après coup et **désormais la base recommandée pour L4 et L5** : ne pas
utiliser le tracker vidéo, faire tourner `segment_anything_3` (chemin image) sur chaque
frame, puis réassocier les identités hors ligne en écrivant les `group_id`.

Elle supprime d'un coup tous les verrous listés plus haut — accumulation d'état, plafond de
16 objets, suppression de bord à 2,5 %, confirmation sur 3 frames, patch multi-classes — et
c'est **la seule qui rende le tuilage possible**. Chiffres complets dans le README, section
« Frame par frame plutôt que tracker ». En résumé : 754/754 frames, 2,34 fps (contre 4,64
pour le chemin vidéo), crête VRAM **5,86 Gio** contre 9,98, latence plate à 419 ms de
médiane sans dérive, et **1,97 détection par frame contre 1,23** pour le même SAM 3 à
travers son tracker.

**L'association est écrite depuis le 2026-09-01** : module
`anylabeling/views/labeling/utils/group_id_association.py` côté client, exposé par
**Tool > Group ID Tracker** et par `associe-group-ids.py`. Mesuré sur la vidéo de
référence : 20 pistes brutes, 12 objets retenus, 287 formes sur 295 étiquetées, aucune
collision d'identifiant, la plus longue piste couvrant 132 frames sans trou. Le
multi-classes fonctionne (une passe de backbone, +28 % pour une seconde classe).
Ce qui suit décrit le contexte d'origine.

`Shape.group_id` existe (`app/schemas/shape.py`) et n'était
pas rempli par le chemin image. Les trackers sont déjà installés dans le venv du serveur
(ultralytics 8.4.136 : BoT-SORT, ByteTrack, OC-SORT, DeepOCSORT), et BoT-SORT fait de la
compensation de mouvement global par défaut (`gmc_method: sparseOptFlow`), nécessaire sur
une caméra de drone. Point de vigilance : l'association par IoU casse sur les petits
objets — sur une boîte de 16 px, 5 px de déplacement passent sous `match_thresh: 0.8`.
Associer sur une distance de centroïde normalisée par la taille.

Cette passe est **hors ligne et rejouable** : retuner l'association ne demande pas de
relancer SAM, ce qui est impossible aujourd'hui.

### L4 — Tuilage du chemin image fixe (serveur) — FAIT le 2026-09-01

**Écrit, testé, mesuré — et le résultat ne tranche pas.** `segment_anything_3_tiled`
existe (`app/models/segment_anything_3_tiled.py`, sous-classe de `SegmentAnything3`
surchargeant le seul `_predict_with_text`), avec `app/utils/tiling.py` pour la géométrie
et la fusion, `configs/auto_labeling/segment_anything_3_tiled.yaml` pour les réglages, et
`tests/test_tiling.py` (16 cas). Le chemin image a été allégé au passage : −752 Mio de
crête VRAM à détections strictement identiques (`tests/test_mask_skipping.py`).

Sur 8 frames réelles, prompt « Person », conf 0,25 : **56 formes tuilé contre 57 en plein
cadre**. Le tuilage resserre les boîtes (côté médian 16,0 → 12,7 px) sans en trouver
davantage, et coûte 12,4× le temps. Le seuil de fusion pèse plus lourd que le tuilage :
56 formes à 0,5, 88 à 0,95. Tableau complet et discussion dans le README, section
« Le tuilage ». **Ce qui manque pour conclure est L0**, et L0 est désormais faisable :
`Datasets/release/` contient 18 jeux annotés en COCO.

Deux corrections au dimensionnement ci-dessous, venues de l'écriture :
`get_slice_bboxes` **décale les tuiles de bord vers l'intérieur** au lieu de padder, donc
le padding prévu est inutile ; et une tuile carrée écrasée en 1008² est un
redimensionnement **isotrope**, là où la frame entière subit ×0,525 en x contre ×0,933
en y — bénéfice non prévu, indépendant de la résolution.

Le plan d'origine, conservé :


C'est le vrai correctif. Dimensionnement : pour un objet de `d` px natifs, une tuile carrée
de côté `T` px donne `d × 1008 / T` px en entrée modèle. Exiger ≥ 28 px donne **`T ≤ 36 × d`**.

| Plus petit objet visé | `T` max | Tuiles pour 1920×1080, recouvrement 20 % |
|---|---|---|
| 64 px | 2304 | 1 (pas de tuilage nécessaire) |
| 32 px | 1152 | 2 |
| 16 px | 576 | 4 × 3 = 12 |
| 8 px | 288 | 9 × 5 = 45 |

Coût : **une passe backbone complète par tuile**, `set_image` relançant
`self.model.backbone.forward_image` (`sam3_image_processor.py:44-81`). Ordre de grandeur
extrapolé du 3,7 fps mesuré (≈ 0,27 s par passe détecteur) — **à mesurer, ce n'est pas un
relevé** : 6 tuiles ≈ 1,6 s/frame, 12 tuiles ≈ 3,2 s/frame, 45 tuiles ≈ 12 s/frame.

Implémentation, en suivant le patron déjà établi dans le dépôt (`segment_anything_3_1_video.py`
sous-classe `SegmentAnything3Video` et ne surcharge que ce qu'il faut) :

- Nouveau `app/models/segment_anything_3_tiled.py`, `@register_model("segment_anything_3_tiled")`,
  **sous-classe de `SegmentAnything3`**, ne surchargeant que `_predict_with_text`.
- Découpe en tuiles carrées de côté `T` (paramètre `tile_size`), recouvrement
  `tile_overlap` (défaut 0,2). ~~avec padding en bord d'image~~ — inutile :
  `get_slice_bboxes` décale les tuiles de bord vers l'intérieur, elles restent carrées.
- **Une tuile à la fois** : `set_image` → boucle `reset_all_prompts` + `set_text_prompt`
  par classe, exactement comme `_predict_with_text` actuel.
  **Ne pas essayer d'utiliser `set_image_batch`** : il écrit `original_heights` /
  `original_widths` au pluriel (`sam3_image_processor.py:92-93`) alors que
  `_forward_grounding` lit `original_height` / `original_width` au singulier
  (`:233-234`), et `find_stage` est figé à `img_ids=torch.tensor([0])` (`:34`). Le chemin
  batché est **incomplet en amont** ; le faire marcher demanderait de réécrire
  `_forward_grounding` en per-image. À garder comme optimisation ultérieure éventuelle,
  pas pour le premier jet.
- Remise en coordonnées globales : décaler les boîtes de l'offset de la tuile, recoller les
  masques dans un canevas pleine résolution.
- Fusion : NMS/NMM par classe sur les boîtes globales. La logique existe côté **client**
  dans `X-AnyLabeling/anylabeling/services/auto_labeling/utils/sahi/postprocess/`
  (NMS, NMM, GREEDYNMM) — la porter côté serveur plutôt que la réécrire.
- Nouveau `configs/auto_labeling/segment_anything_3_tiled.yaml` calqué sur
  `segment_anything_3.yaml`, plus `tile_size`, `tile_overlap`, et l'activer dans
  `configs/models.yaml`.

Effet de bord favorable : le plafond de **200 requêtes est par tuile**, donc le tuilage
multiplie aussi le budget de détections.

### L5 — Tuilage du chemin vidéo (client)

Beaucoup plus lourd : l'état du tracker est global par session, des tuiles cassent
l'identité des objets. La seule voie tractable est le **découpage spatial en sous-vidéos**,
chacune annotée comme sa propre session.

- Étendre `extract_frames_from_video`
  (`X-AnyLabeling/anylabeling/views/labeling/utils/video.py:145`) avec une option de
  découpe spatiale : écrire `<dirname>/<video>_r0c0/`, `_r0c1/`, … au lieu d'un seul
  dossier. Le dialogue `FrameExtractionDialog` (`:36-142`) n'expose aujourd'hui que
  l'intervalle et le préfixe — y ajouter `tile_size` et `overlap`.
- Annoter chaque dossier comme une vidéo à part entière (le flux existant ne change pas).
- Écrire un outil de **fusion des JSON** : décalage des coordonnées par l'offset de la
  tuile, décalage des group_id par tuile, puis déduplication des objets présents dans deux
  tuiles voisines (IoU sur les boîtes globales, frame par frame).
- **Si on reste sur SAM 3.1 pour ce chemin**, il faut impérativement désactiver la
  suppression de bord (`suppress_det_close_to_boundary`) et relever `max_num_objects`, par
  override après construction dans `segment_anything_3_1_video.py`, à côté de
  `batched_grounding_batch_size` (`:152-161`). Avec SAM 3 c'est déjà bon par défaut.
- Hypothèse à vérifier, pas à supposer : le tuilage pourrait **améliorer** le plafond de
  frames, puisque `sam2_inference_states` s'accumule par état introduisant de nouveaux
  objets et que chaque tuile en suit moins. À mesurer avant d'en tirer un argument.

### L6 — Repli pour les objets sous ~16 px : détecteur dédié + SAM en affinage

Il faut être franc : à 8×8 px, **SAM 3 n'est pas le bon outil**, tuilage ou pas. Il faudrait
des tuiles de 288 px (45 par frame en 1920×1080), et à cette taille le signal sémantique
exploitable par le grounding textuel du Perception Encoder est très faible — on demande à
un modèle vision-langage de reconnaître un concept dans 64 pixels.

L'architecture standard pour ce régime, et elle est **déjà en place des deux côtés** :

1. **Détection par un modèle petit-objet avec SAHI**, côté client, en local. Quatre modèles
   sont déjà câblés : `yolov5_sahi.py:118`, `yolov8_sahi.py:117`, `yolo11_sahi.py:117`,
   `yolo26_sahi.py:108`, tous via `get_sliced_prediction` avec `slice_height`,
   `slice_width`, `overlap_*_ratio` lus dans le YAML du modèle. C'est exactement le
   pipeline conçu pour ce problème.
2. **Affinage des masques par SAM 3**, en lui passant les boîtes obtenues comme prompts
   géométriques. Le chemin existe : le client envoie déjà des `marks`
   (`remote_server.py:301-316`), le serveur les route vers `_predict_with_boxes`
   (`segment_anything_3.py:81-82, 186+`), qui appelle `add_geometric_prompt`
   (`sam3_image_processor.py:142`, boîte en `cxcywh` normalisé).

Le travail se réduit donc à **enchaîner deux outils existants**, pas à en écrire un.
Ce lot ne se justifie que si L0 confirme une masse d'objets sous 16 px.

## Vérification

- **L0/L1/L3** : rappel par tranche de taille sur l'échantillon annoté à la main, plus les
  relevés VRAM/frames au protocole du README (`torch.cuda.memory._record_memory_history()`
  puis rejeu du flux alloc/free — c'est la méthode qui a fini par payer sur les mesures
  précédentes).
- **L2** : comparer les détections avant/après sur le même échantillon ; l'écart doit être
  nul ou favorable, et le temps d'`init_state` ne doit pas se dégrader.
- **L4** : test unitaire sur la géométrie du tuilage (une boîte connue placée à cheval sur
  deux tuiles doit ressortir une seule fois, aux bonnes coordonnées globales), dans
  `X-AnyLabeling-Server/tests/`, au format des tests existants
  (`test_video_text_prompt_state.py`, `test_vram_guard.py`, `test_propagation_frame_range.py`).
  Puis bout en bout depuis le client sur une image réelle.
- **L5** : fusionner deux tuiles voisines d'une séquence courte et vérifier à l'œil, dans le
  client, qu'un objet traversant la frontière ne produit pas deux group_id.
- Rappel de procédure : après toute modification de `configs/models.yaml`, **redémarrer le
  serveur** et se déconnecter/reconnecter depuis le client.

## Questions ouvertes

- ~~Le plafond de frames de SAM 3~~ — tranché le 2026-09-01 : pas de plafond sur 754
  frames, et 36 % plus rapide que SAM 3.1. L'arbitrage penche pour SAM 3.
- Les 296 détections de SAM 3.1 contre 164 pour SAM 3 sur les mêmes 133 frames : rappel
  supérieur ou doublons ? C'est la seule raison qui resterait de préférer 3.1, et elle
  relève du lot L0 — **qui n'est plus bloqué** : `Datasets/release/` contient 18 jeux
  annotés en COCO (282 039 images), dont `rgbtdroneperson` à 99,9 % de petits objets et
  11,2 px de côté médian, et `uavdt_rgb` à 45 % / 33 px. La même question se pose
  maintenant pour le tuilage (L4), qui resserre les boîtes sans en trouver plus.
- ~~`image_size > 1008` : gain réel ou dégradation hors distribution ?~~ Tranché le
  2026-09-02 : **dégradation**, monotone, pire sur les petits objets, pour 3,6× le temps.
  Voir L3.
- Le coût par tuile extrapolé (0,27 s) mérite d'être relevé sur le chemin image seul, où le
  tracker n'intervient pas.
- `configs/models.yaml` reste non commité des deux côtés (cf. README) : penser à consigner
  la configuration retenue dans le README plutôt que de compter sur le fichier.
