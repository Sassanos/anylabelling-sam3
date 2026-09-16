# Annotation SAM 3 / SAM 3.1 — état du chantier

Notes de reprise. Dernière mise à jour : 2026-09-16.

## Ce qu'il y a ici

| Dossier | |
|---|---|
| `X-AnyLabeling/` | client (IHM), fork de CVHub520, branche `sam3` |
| `X-AnyLabeling-Server/` | serveur d'inférence, même fork, branche `sam3` |
| `sam3-official/` | clone de `facebookresearch/sam3`, requis par SAM 3.1 |
| `run-server.sh`, `run-client.sh` | lanceurs |
| `PETITS-OBJETS-HAUTE-RESOLUTION.md` | plan de reprise : ce que la haute résolution coûte aux petits objets |
| `mesure-plafond-frames.py` | harnais de mesure : plafond de frames, crête VRAM/RSS, débit, détections |
| `mesure-frame-par-frame.py` | harnais de mesure du chemin image, une requête par frame |
| `associe-group-ids.py` | association hors ligne des `group_id` sur des annotations frame par frame |
| `mesure-rappel-coco.py` | inférence sur un échantillon COCO annoté ; garde les prédictions brutes avec leur score |
| `rappel-par-taille.py` | rappel et précision par tranche de taille, hors ligne, sur un ou plusieurs runs |
| `sam3-official-patches/` | patch local de `sam3-official` (délestage, à appliquer sur l'amont `660a5e9`) et `run_sam31.py` |

Les deux forks sont sur une branche `sam3` = `origin/main` + des patches perso
jamais poussés en amont. Tags de secours de l'état d'avant le rebase du
2026-08-31 : `sam3-backup-beta.6` (client) et `sam3-backup-v0.0.10` (serveur).

## Dépôts GitHub

Trois dépôts **privés** sur le compte `Sassanos`, poussés le 2026-09-16 :

| Local | Remote `sassanos` | Branches poussées |
|---|---|---|
| `X-AnyLabeling/` | `Sassanos/X-AnyLabeling` | `sam3` (par défaut), `main` = amont |
| `X-AnyLabeling-Server/` | `Sassanos/X-AnyLabeling-Server` | `sam3` (par défaut), `main` = amont |
| racine `Anylabelling/` | `Sassanos/anylabelling-sam3` | `main` |

- Ce sont des **miroirs, pas des forks GitHub** : un fork d'un dépôt public ne peut pas
  être privé. `origin` pointe toujours vers l'amont, pour les rebases.
- SSH via l'alias `github-sassanos` (`~/.ssh/config`, clé `~/.ssh/id_ed25519_sassanos`) :
  un autre compte GitHub sert aussi sur ce portable, chacun a sa clé et son alias.
- Identité `Sassanos <239545955+Sassanos@users.noreply.github.com>` limitée à ce dossier
  par un `includeIf` dans `~/.gitconfig` (→ `~/.gitconfig-sassanos`).
- L'auteur des commits perso a été réécrit le 2026-09-16 (contenu identique, dates
  conservées). L'état d'avant est gardé par les tags **locaux** `sam3-avant-sassanos` ;
  eux et les `sam3-backup-*` portent l'ancienne identité et ne sont pas poussés.
- `sam3-official/` n'a pas de dépôt : réappliquer le patch avec
  `git -C sam3-official apply ../sam3-official-patches/offload-device.patch`.

## Lancer

```bash
./run-server.sh     # terminal 1, attendre « Successfully loaded N/N model(s) »
./run-client.sh     # terminal 2
```

**Ne jamais utiliser `uv run` ou `uv sync` nus dans ces dépôts.** Sans `uv.lock`
ils resynchronisent sur les seules dépendances déclarées et désinstallent torch
et les extras — vérifié, 74 paquets supprimés. Utiliser `uv pip install --python
.venv/bin/python <paquet>` ou `uv run --no-sync`.

Après toute modification de `configs/models.yaml`, **redémarrer le serveur** :
la liste des modèles n'est lue qu'au démarrage. Et se déconnecter/reconnecter
depuis le client, qui met la liste en cache.

## SAM 3 ou SAM 3.1 ?

Les deux sont **mutuellement exclusifs** : ils s'importent tous deux sous le nom
`sam3` (fork vendorisé sous `app/models/sam3` contre paquet officiel en
site-packages) et le premier importé masque l'autre. Le registre refuse une
config qui active les deux et explique pourquoi.

Bascule : une ligne à inverser dans `configs/models.yaml`, puis redémarrage.

**Corrigé le 2026-09-01 : c'est SAM 3 qui va plus loin, pas SAM 3.1.** La note
précédente affirmait l'inverse (« SAM 3 échoue en OOM, SAM 3.1 annote 430 à 450
frames »). Remesuré en aller-retour le même jour, serveur fraîchement redémarré
à chaque fois, mêmes 8 modèles chargés, même vidéo de 754 frames en 1920x1080,
même prompt « Person » sur la frame 0, via `./mesure-plafond-frames.py` :

| | SAM 3 | SAM 3.1 |
|---|---|---|
| Frames propagées | **754 / 754**, aucun arrêt | 368 / 754, garde-fou VRAM |
| Débit | **4,64 fps** | 3,42 fps |
| Crête VRAM | 9,98 Gio | 10,64 Gio |
| Crête RSS serveur | 11,3 Gio | 10,9 Gio |
| Init de session (754 frames) | 27 s | **10,5 s** |
| Détections, frames 0-367 | 164 sur 133 frames | **296 sur 133 frames** |

Le 368 de SAM 3.1 est reproductible à la frame près (deux runs indépendants), et
ses 296 détections sont exactement les 296 annotations « Person » déjà sur le
disque pour cette vidéo.

**Ce qui sépare vraiment les deux, c'est la pente, pas le plafond.** SAM 3 part
haut et reste plat : 9633 → 10221 Mio sur 754 frames, soit **0,8 Mio/frame**.
SAM 3.1 part bas et grimpe : 6196 → 10893 Mio sur 368 frames, soit
**12,8 Mio/frame**, seize fois plus. Le croisement tombe vers la frame 270 ;
au-delà, SAM 3.1 est le plus gourmand des deux.

**Ce que SAM 3.1 apporte réellement, c'est la densité de détection.** Sur les
mêmes 133 frames — les deux modèles s'accordent exactement sur *lesquelles*
contiennent des personnes — il trouve 2,2 instances par frame contre 1,2 pour
SAM 3. Sans vérité terrain, on ne peut pas dire si ce surplus est du rappel ou
des doublons : c'est ce que doit trancher le lot L0 de
`PETITS-OBJETS-HAUTE-RESOLUTION.md`.

Aucun des deux ne découpe un prompt multi-classes : `"car. person"` renvoie 0.
D'où le patch perso qui boucle une classe à la fois avec un `reset_session`
entre chaque et décale les group_id de 1000. Toujours nécessaire en 3.1.

**Un prompt sans détection sur sa frame gardait la classe précédente**
(corrigé le 2026-09-01). Symptôme : prompt « Vehicle », propagation, puis
prompt « Person » sur une frame où aucune personne n'est visible — les
personnes trouvées plus loin dans la vidéo revenaient toutes étiquetées
« Vehicle », géométrie correcte. Cause : `add_prompt` retournait tôt quand
aucune classe ne rendait de masque, **avant** d'écrire `session.text_prompt`.
Le prédicteur, lui, avait bien été reset et re-prompté avec le nouveau texte,
donc la propagation détectait bien des personnes ; mais `_build_completed_results`
lit l'étiquette dans `session.text_prompt` au moment de convertir les résultats,
et y trouvait encore « Vehicle ». L'état du prompt est maintenant enregistré
dans tous les cas. Couvert par `tests/test_video_text_prompt_state.py`.

Les annotations déjà produites restent fausses sur le disque. Les group_id du
second passage sont alloués au-dessus de ceux du premier (`max en usage + 1`),
donc les personnes mal étiquetées sont isolables par plage de group_id et
réétiquetables d'un coup depuis le Group ID Manager.

## Frame par frame plutôt que tracker — mesuré le 2026-09-01

Piste : ne pas utiliser le tracker vidéo du tout. Faire tourner le **chemin image**
(`segment_anything_3`) sur chaque frame indépendamment, puis réassocier les identités
hors ligne avec un tracker classique, en écrivant les `group_id`. Le détecteur tourne
déjà sur chaque frame dans les deux chemins vidéo : le tracker est un coût *ajouté*.

Mesuré sur la même vidéo de 754 frames, prompt « Person », via
`./mesure-frame-par-frame.py` :

| | SAM 3 vidéo | SAM 3.1 vidéo | SAM 3 image, frame par frame |
|---|---|---|---|
| Couverture | 754 / 754 | 368 / 754 | 754 / 754 |
| Débit | **4,64 fps** | 3,42 fps | 2,34 fps |
| Durée | 162 s | 108 s (partiel) | 322 s |
| Crête VRAM | 9,98 Gio | 10,64 Gio | **5,86 Gio** |
| Crête RSS | 11,3 Gio | 10,9 Gio | **2,8 Gio** |

Détections sur les frames 0-367, la plage que SAM 3.1 atteint :

| | Frames avec détection | Formes | Par frame |
|---|---|---|---|
| SAM 3 vidéo | 133 | 164 | 1,23 |
| SAM 3.1 vidéo | 133 | 296 | 2,23 |
| SAM 3 image, seuil 0,50 | 132 | 260 | 1,97 |
| SAM 3 image, seuil 0,25 | 135 | 1519 | 11,25 |

**Ce que ça coûte : le débit, divisé par deux.** 5,4 min contre 2,7 pour 754 frames.
Contre-intuitif puisque le tracker a disparu, mais chaque requête refait le décodage JPEG
1920x1080, la conversion PIL, un `Sam3Processor` neuf et l'interpolation du masque en
pleine résolution, plus HTTP et base64. Le chemin vidéo amortit le chargement des frames
en asynchrone et garde tout résident.

**Ce que ça rapporte.** La VRAM tombe à 5,86 Gio et la RAM à 2,8 : c'est là que se
financent le tuilage ou un `image_size` plus grand. Surtout, **la latence est plate** —
médiane 419 ms, p90 443, max 492 sur 754 requêtes, sans la moindre dérive. La longueur de
la vidéo sort de l'équation : plus de plafond, plus de garde-fou, plus de découpage en
tronçons.

**Et la densité de détection de SAM 3.1 se récupère.** Sur la même plage, le même SAM 3
donne 1,97 instance par frame en per-frame contre 1,23 à travers son propre tracker vidéo.
Ce que perd SAM 3 vidéo, ce n'est donc pas le détecteur : **c'est son tracker qui jette des
détections**. Le dernier argument en faveur de SAM 3.1 tombe avec ça.

Les trois configurations s'accordent sur *quelles* frames contiennent des personnes
(132, 133, 133) — elles ne diffèrent que sur le nombre d'instances retenues.

**Le seuil redevient un levier.** À 0,25, inatteignable sur le chemin vidéo où
`score_threshold_detection=0.5` est en dur, on passe à 11,25 instances par frame, jusqu'à
22 sur une frame, et la première détection remonte de la frame 81 à la frame 28. Une
bonne part est sans doute du faux positif — mais c'est le régime que le lot L0 doit
explorer, et le chemin vidéo l'interdit.

**Les deux modèles SAM 3 ne cohabitent pas sur 12 Go.** Rien ne l'interdit — ils viennent
du même paquet vendorisé, le registre les charge tous les deux sans broncher (9/9) — mais
les ~3,5 Go de poids du modèle image mangent exactement la marge dont le tracker vidéo a
besoin : mesuré le 2026-09-01, la propagation tombe de **754 à 195 frames**, garde-fou à
0,95 Go libre. VRAM au repos : 3,9 Gio avec le modèle image seul, 7,5 avec les deux.
Choisir, et redémarrer.

**Ce qui reste à faire** : l'association. Le champ existe (`Shape.group_id`,
`app/schemas/shape.py`), le chemin image ne le remplit pas. Les trackers sont déjà
installés dans le venv du serveur — ultralytics 8.4.136 embarque BoT-SORT, ByteTrack,
OC-SORT, DeepOCSORT — et le défaut de BoT-SORT, `gmc_method: sparseOptFlow`, fait de la
compensation de mouvement global, indispensable sur une caméra de drone. Attention à
l'association par IoU sur de petits objets : sur une boîte de 16 px, 5 px de déplacement
passent sous le `match_thresh: 0.8` par défaut ; associer sur une distance de centroïde
normalisée par la taille.

## Association des group_id — `associe-group-ids.py`

Complète le chemin frame par frame : lit un dossier de JSON X-AnyLabeling, relie les
détections d'une frame à l'autre et réécrit `group_id`. Passe **hors ligne**, donc
rejouable : retuner l'association ne demande pas de relancer SAM.

**Pourquoi pas BoT-SORT tel quel.** Il est dans le venv (ultralytics 8.4.136, avec
ByteTrack, OC-SORT, DeepOCSORT, TrackTrack) et reste le défaut de production en 2026,
mais trois choses ne collent pas ici : il est causal, il associe par IoU, et son filtre
est réglé pour des piétons de ~100 px. Nos boîtes font **14x19 à 21x24 px**. Le script
reprend donc ses deux bonnes idées — compensation de mouvement global et association en
cascade — et change le reste :

1. **IoU tamponnée (BIoU).** Les boîtes sont dilatées d'une fraction de leur propre
   taille avant le calcul, en deux passes de tampon croissant (0,3 puis 1,0). Sur une
   boîte de 16 px, 5 px de déplacement suffisent à annuler l'IoU brute.
2. **Bruit de Kalman proportionnel à la taille de l'objet.** L'état suit le centroïde,
   la taille est lissée à part : un filtre à 8 états sur le rapport d'aspect diverge sur
   d'aussi petites boîtes.
3. **Porte sur la distance des centroïdes normalisée** par la diagonale des objets,
   pas sur un seuil en pixels.
4. **Compensation de mouvement** par flot optique clairsemé (`goodFeaturesToTrack` +
   `calcOpticalFlowPyrLK` + `estimateAffinePartial2D`), comme le `sparseOptFlow` de
   BoT-SORT. Indispensable : sous le drone, la scène défile.
5. **Recollement des fragments a posteriori** et **interpolation des trous**
   (`--interpolate`), deux choses qu'un tracker en ligne ne peut pas faire puisqu'il a
   déjà émis ses identifiants.

**Résultat mesuré** sur la sortie frame par frame de la vidéo de 754 frames (295
détections, seuil 0,5), 9,6 s de traitement, compensation de mouvement estimée sur
753/753 paires :

| | |
|---|---|
| Pistes brutes | 20 |
| Retenues (>= 2 frames) | **12** |
| Formes étiquetées | 287 sur 295 (8 singletons laissés à `null`) |
| Longueur des pistes | min 2, médiane 11, **max 132** |
| Objets simultanés | 1 à 5 |
| `group_id` dupliqué sur une frame | **aucun** |

La piste la plus longue suit un marcheur isolé sur **132 frames sans un seul trou**,
malgré le défilement de la scène — c'est la validation de la compensation de mouvement.

**Le point faible est connu et visible** : dans le groupe dense autour du campement
(frames 521-535), où le drone descend vite et où plusieurs personnes de ~10 px se tiennent
à moins de deux largeurs d'objet les unes des autres, l'identité se fragmente (trois
identifiants pour ce qui semble être une ou deux personnes). C'est le mode d'échec attendu
à cette taille. Sans vérité terrain, impossible de chiffrer les changements d'identité :
c'est au lot L0 de le faire.

Leviers dans ce cas : baisser `--max-dist` (défaut 2,5 diagonales) réduit la concurrence
entre voisins mais fragmente sur les mouvements rapides ; `--merge-gap` et `--merge-dist`
pilotent le recollement ; `--min-len` filtre les pistes courtes.

`--dry-run` affiche le bilan sans rien écrire, `--backup DIR` copie les JSON avant
réécriture. Le harnais `mesure-frame-par-frame.py --write-json DIR` produit les
annotations d'entrée et **refuse d'écrire dans le dossier des frames sources**.

### Le flux complet depuis le client

1. Ouvrir le dossier de frames (`Open Dir`).
2. Choisir le modèle **`segment_anything_3`**, saisir le prompt texte, puis **Auto Run**.
   Le mode batch texte existait déjà : `batch_processing_mode: "text_prompt"` dans la
   config du modèle, `utils/batch.py` itère dessus.
3. **Tool > Group ID Tracker** — le dialogue ajouté à côté du Group ID Manager. Il opère
   sur le dossier ouvert (ou le dossier de sortie s'il est défini), avec les réglages
   utiles exposés : labels, longueur minimale de piste, tampon de piste, distance max,
   compensation de mouvement, interpolation, sauvegarde, aperçu sans écriture.

Une seule implémentation derrière les deux points d'entrée :
`anylabeling/views/labeling/utils/group_id_association.py` (sans dépendance à Qt), le
dialogue `widgets/group_id_tracker_dialog.py` l'appelle dans un `QThread`, et
`associe-group-ids.py` n'est qu'une façade en ligne de commande par-dessus.

### Plusieurs classes : oui, et presque gratuitement

Le chemin image **découpe déjà le prompt** sur `,` ou `.` et fait un forward de grounding
par classe en ne payant **qu'une seule passe de backbone**
(`segment_anything_3.py:128-184`). Mesuré sur 220 frames avec le prompt
`"person. tent"` : 350 formes, 260 `person` et 90 `tent`, à **1,83 fps contre 2,34 pour
une seule classe** — la seconde classe coûte 28 %, pas 100 %.

C'est un net progrès sur le chemin vidéo, qui exige le patch perso (une classe à la fois,
`reset_session` entre chaque, décalage des group_id de 1000).

L'association traite **chaque classe séparément** puis numérote globalement : sur cet
essai, 6 `person` et 4 `tent`, identifiants 1 à 10, aucun partagé entre classes et aucun
doublon sur une frame.

Un défaut trouvé par les tests, et corrigé : une piste au-delà de son tampon pouvait
encore capter une détection, l'expiration étant vérifiée *après* l'association. Comme
seules les frames détectées ont un JSON, les indices sautent et une piste peut vieillir de
plusieurs dizaines de frames d'un coup — le défaut était donc matériel, pas théorique.
Couvert par `tests/test_utils/test_group_id_association.py` (12 cas).

### Deux correctifs venus de l'usage réel (2026-09-01)

**Une seule forme par objet.** `configs/auto_labeling/segment_anything_3.yaml` avait
`show_boxes: true` **et** `show_masks: true` — chaque objet ressortait donc en double, un
rectangle et un polygone (`segment_anything_3.py`, les deux branches de
`_convert_results_to_shapes` s'ajoutent au lieu de s'exclure). Le fichier vidéo, lui,
avait déjà `show_masks: false`. Aligné sur les boîtes seules, ce que veut la détection ;
inverser les deux valeurs pour annoter en masques. Le doublon faussait aussi
l'association, qui voyait deux détections par objet.

**Un prompt ne remplace plus que ses propres classes.** Inférer « Vehicle » sur une frame
effaçait les « Person » déjà annotées : le client ne gardait que les formes *verrouillées*
(`label_widget.py`, branche `replace`), et le batch écrasait `data["shapes"]` en entier
(`utils/batch.py`, `save_auto_labeling_result`). Désormais `AutoLabelingResult` transporte
la **portée du prompt** (`prompt_labels`), renseignée par `remote_server.py` avec le même
découpage que le serveur (`,` testé avant `.`), et sa méthode `supersedes_label()` décide
forme par forme. Les annotations des autres classes survivent ; relancer le même prompt
remplace bien ses propres résultats, sans doublon. Sans portée connue — un modèle qui
n'est pas piloté par un prompt texte — l'ancien comportement est conservé tel quel.

**Le comportement est réglable** : **Tool > Replace Only Prompted Classes**, une bascule
cochée par défaut. Décochée, on retrouve le remplacement complet de la frame. Le réglage
est persistant (`replace_only_prompted_classes` dans `xanylabeling_config.yaml`, fusionné
dans le `~/.xanylabelingrc` existant) et s'applique aux deux chemins, écran et batch.

À ne pas confondre avec le bouton **« preserve existing annotations »** de la barre
d'auto-annotation, qui fait autre chose : il met `replace=False`, donc **fusion sans rien
supprimer**, et les doublons s'accumulent. Trois comportements, donc :

| | Formes de la classe promptée | Formes des autres classes |
|---|---|---|
| Bascule cochée (défaut) | remplacées | **conservées** |
| Bascule décochée | remplacées | effacées |
| « preserve existing annotations » | conservées, s'ajoutent | conservées |

Couvert par `tests/test_utils/test_prompt_scoped_replace.py` (15 cas, dont la bascule dans
les deux positions et l'absence de la clé dans une config antérieure). La suite du client
passe à 466 tests, le seul échec restant étant le bug amont Qt 6.9 déjà connu.

### Le chemin image allégé — mesuré le 2026-09-01

Trois gaspillages trouvés dans le code, tous sur le chemin image, tous corrigés :

1. **Les masques étaient rapatriés sur CPU même inutilisés.**
   `segment_anything_3.py` faisait `results["masks"].cpu().float().numpy()` sans
   condition, alors que `masks` n'est lu que par la branche `show_masks` — et la config
   de production est `show_masks: false`. Un masque 1920×1080 pèse 2,07 Mo en booléen.
2. **Ils étaient surtout *produits* pour rien.** `_forward_grounding`
   (`sam3_image_processor.py`) interpole chaque masque retenu de 288² vers la résolution
   native, applique un sigmoïde et seuille — trois tenseurs pleine résolution par
   détection — et garde `masks_logits`, **que rien ne relit** dans tout `app/`.
   Corrigé par une **sous-classe** de `Sam3Processor` (`_get_processor_cls()`), le paquet
   vendorisé restant intact.
3. **Le `Sam3Processor` était reconstruit à chaque requête.** Il est désormais mis en
   cache ; seuls le seuil et le drapeau de masques varient d'une requête à l'autre.

Mesuré au même protocole, 150 frames en 1920×1080, prompt « Person », conf 0,25,
**serveur redémarré à froid des deux côtés** (VRAM au repos identique, 3968 Mio) :

| | avant | après |
|---|---|---|
| Formes détectées | 841 | **841** |
| Frames avec détection | 71 | **71** |
| Latence médiane | 438,4 ms | 426,3 ms |
| Latence p90 | 461,6 ms | 436,2 ms |
| Crête VRAM | 5646 Mio | **4894 Mio** |
| Crête RSS | 2,91 Gio | 2,75 Gio |

**Les détections sont identiques au chiffre près** — c'était le critère : ces correctifs
ne changent aucun calcul, seulement ce qui est matérialisé. Un écart aurait été un bug.
Couvert par `tests/test_mask_skipping.py` (6 cas, dont l'égalité stricte des boîtes et
des scores entre les deux chemins).

**Le gain est de la mémoire, pas du débit** : −752 Mio de crête VRAM, soit **45 % de
l'allocation transitoire** au-dessus du repos (1678 → 926 Mio). La latence ne bouge
presque pas, ce qui confirme que le coût par frame est dominé par le backbone, pas par
le post-traitement. C'est cette marge qui finance le tuilage.

### `image_size` au-dessus de 1008 — mesuré le 2026-09-02, c'est une impasse

`image_size` est réglable bout en bout sur SAM 3 (le builder retire les clés `freqs_cis`
et charge en `strict=False` quand il diffère de 1008). Les trois valeurs testées, toutes
multiples de 14, se chargent **sans une seule clé manquante**. La question était : une
grille plus fine récupère-t-elle des petits objets ?

**Non. Le rappel se dégrade de façon monotone.** Mesuré sur 120 images de
`Datasets/release/rgb/vtuav_rgb` (1920×1080, classe `person`, 1158 objets annotés),
prompt « person », seuil 0,25, serveur redémarré à chaque valeur :

| `image_size` | Grille | Rappel IoU 0,3 | Rappel IoU 0,5 | Précision | Latence médiane | Crête VRAM |
|---|---|---|---|---|---|---|
| **1008** | 72×72 | **0,902** | **0,712** | 0,611 | **437 ms** | **4894 Mio** |
| 1400 | 100×100 | 0,894 | 0,647 | 0,625 | 1059 ms | 5782 Mio |
| 1680 | 120×120 | 0,879 | 0,585 | 0,629 | 1585 ms | 6504 Mio |

Le résultat tient à tous les seuils de score de 0,15 à 0,60 et le run à 1008 se reproduit
au chiffre près sur deux passes indépendantes.

**Ce n'est pas seulement qu'il détecte moins : il localise moins bien.** À IoU 0,3 l'écart
est de 2 points ; à IoU 0,5 il est de 13. Et l'effondrement est concentré sur les petits
objets, exactement ceux qu'on espérait gagner :

| Taille native | 1008 | 1400 | 1680 |
|---|---|---|---|
| < 16 px | 11/16 | 3/16 | **1/16** |
| 16-32 px | 70/142 | 52/142 | **36/142** |
| 32-64 px | 505/696 | 455/696 | 407/696 |

*(rappel à IoU 0,5, seuil 0,25)*

**La cause est architecturale, donc elle ne dépend pas de ce jeu de données.** Les poids
sont entraînés à 1008 et `window_size` est figé à 24 tokens : à 1008 la grille de 72×72
fait exactement 3×3 fenêtres, à 1400 elle en fait 4,17×4,17 — non entier — et à 1680,
5×5. On sort de la distribution d'entraînement, et l'attention fenêtrée ne retrouve pas
ses repères. C'est la réserve annoncée par le plan, et elle se vérifie.

Ce que `image_size` achète en échange, c'est un peu de **précision** (0,611 → 0,629) :
moins de détections, mieux notées. Si un jour le problème est le faux positif et non le
rappel, c'est un levier ; pour des petits objets, non.

**Contrôle sur l'écrasement le plus violent.** Sur `nii_mapd_rgb` (3840×2160, un objet
médian de 52 px n'arrive qu'à 13,7 px en espace modèle à 1008), 1680 ne récupère rien non
plus : rappel 0,912 → 0,873 à IoU 0,3, et strictement identique à IoU 0,5, pour 1643 ms
par image. Là encore le gain est en précision (0,721 → 0,802).

**Au passage, l'heuristique des 2 tokens est trop pessimiste.** Elle annonçait que rien
sous ~28 px en espace modèle n'est fiable. Or à 1008 sur VTUAV, des objets de 16-32 px
natifs — soit **8,4 à 16,8 px en espace modèle** — sont retrouvés à 0,817 (IoU 0,3), et
ceux sous 16 px natifs (4,2 à 8,4 px modèle) à 0,875. SAM 3 encaisse bien plus bas que
prévu. À reporter sur d'autres données par la taille **en espace modèle**,
`d × image_size / largeur_source`, qui est la seule grandeur comparable d'un jeu à
l'autre.

**Limite de cette mesure, à ne pas oublier** : ces jeux publics n'ont pas la queue de
distribution des données réelles visées (objets à 8-32 px natifs en haute résolution).
Ce qui transfère ici, c'est le coût, les limites sur 12 Go, et le mécanisme de la
dégradation — pas les valeurs de rappel elles-mêmes.

### Le tuilage — écrit et mesuré le 2026-09-01, et il ne rend pas ce qu'on espérait

`segment_anything_3_tiled` (`app/models/segment_anything_3_tiled.py`, sous-classe de
`SegmentAnything3` ne surchargeant que `_predict_with_text`) découpe la frame en tuiles
carrées, prompte chacune, ramène les boîtes en coordonnées globales et déduplique.
Géométrie et fusion portées d'OBSS SAHI dans `app/utils/tiling.py` (`get_slice_bboxes`,
`greedy_nmm`, `batched_greedy_nmm`) — 160 lignes de numpy, pas la pile entière.
Réglages dans `configs/auto_labeling/segment_anything_3_tiled.yaml`, tous surchargeables
par requête puisque `params` est un dict libre. Couvert par `tests/test_tiling.py`
(16 cas : couverture de la grille, tuiles de bord décalées et non paddées, objet à cheval
sur une couture rendu une seule fois, classes qui ne se suppriment pas).

Mesuré sur 8 frames de `test_pipeline_frames`, prompt « Person », conf 0,25 :

| Configuration | Formes | Côté médian | < 8 px | 8-16 | 16-32 | Latence |
|---|---|---|---|---|---|---|
| plein cadre seul | 57 | 16,0 px | 0 | 29 | 27 | 438 ms |
| tuile 576 + plein cadre | 56 | 12,7 px | 3 | 41 | 12 | 5430 ms |
| tuile 576 seule | 48 | 12,1 px | 3 | 37 | 8 | 4961 ms |
| tuile 576, fusion 0,8 | 66 | 12,7 px | 3 | 48 | 15 | 5396 ms |
| tuile 576, fusion 0,95 | 88 | 12,8 px | 5 | 60 | 23 | 5509 ms |
| tuile 864 + plein cadre | 53 | 12,6 px | 0 | 41 | 12 | 2974 ms |

**Le tuilage ne trouve pas plus d'objets** : 56 contre 57 au réglage par défaut. Ce qu'il
change, ce sont les **boîtes** — le côté médian tombe de 16,0 à 12,7 px, la tranche 16-32
s'effondre de 27 à 12 pendant que 8-16 gonfle de 29 à 41. Cela peut être un
resserrement légitime des boîtes à résolution effective plus haute, ou de la
fragmentation ; **sans vérité terrain on ne peut pas trancher**, et c'est exactement ce
que le lot L0 doit faire.

**Et le seuil de fusion pèse plus lourd que le tuilage lui-même** : 56 formes à 0,5,
66 à 0,8, 88 à 0,95. Si la fusion ne supprimait que de vrais doublons, la relâcher ne
ferait que les réintroduire. Qu'elle fasse varier le compte de 57 % dit qu'un grand
nombre de paires sont limites — ce qui est attendu quand des objets de 12 px se tiennent
à moins de deux largeurs les uns des autres. Sur 20 frames, 633 détections brutes
tombent à 196 après fusion (facteur 3,2 pour au plus 5 recouvrements possibles).

**Le coût est de 12,4×** (438 → 5430 ms à la tuile 576, 13 passes de backbone), 6,8× à
la tuile 864. Une passe complète sur 754 frames prendrait 68 min.

**Conclusion honnête : L4 est écrit, testé et mécaniquement correct, mais rien ne montre
qu'il récupère sur ces données des objets que le plein cadre manque.** Il reste le seul
chemin possible pour un prompt texte ouvert sur de très petits objets, et le seul moyen
de dépasser le plafond de 200 requêtes par prompt. Mais la question « rappel ou
doublons ? » est la même que celle qui oppose SAM 3 à SAM 3.1, et elle demande la même
chose : **le lot L0, désormais faisable** — `Datasets/release/` contient 18 jeux annotés
en COCO, dont `rgbtdroneperson` (99,9 % d'objets < 32², côté médian 11,2 px).

## Chiffres mesurés (RTX 4000 Ada, 12 Go)

Crête VRAM de propagation SAM 3.1 ~= `4,2 + 0,65 x detector_batch_size` Go.
Le défaut amont de 16 ne passe pas. Détections **identiques** à 1, 2 et 4 : le
lotissement ne fait que découper une boucle sur des frames indépendantes.

| `detector_batch_size` | Crête | Frames propagées | Débit |
|---|---|---|---|
| 16 (défaut amont) | 10,8 Go | 8 | — |
| 4 | 8,7 Go | 240 | 3,3 fps |
| 2 | 6,7 Go | ~320 | 3,2 fps |
| 1 (notre défaut) | 5,7 Go | 448 | 3,7 fps |

Le lot de 1 gagne sur les trois axes à la fois : c'est lui le défaut depuis le
2026-09-01 (il était à 2, par prudence, sans que la mesure le justifie).

**Deux réserves sur ce tableau.** Les crêtes sont relevées sur **60 frames** :
la crête grandit aussi avec la longueur de la séquence, et le même lot de 1
monte à 7,41 Go sur 200 frames. Et le plafond de 448 frames a été remesuré à
**432** le 2026-09-01 dans un processus neuf, puis à **368** deux fois le même
jour avec le prompt « Person » sur la vidéo de 754 frames. Le plafond dépend donc
autant du **prompt** — c'est-à-dire du nombre d'objets introduits — que de ce qui
occupe la carte. Ne pas citer un chiffre unique : mesurer sur le prompt visé.

Second plafond, non levé : `sam2_inference_states` accumule **9,3 Mo par frame**
introduisant de nouveaux objets. C'est lui qui borne la propagation.
Vu depuis la RAM, une fois l'état délesté, la même accumulation se relève à
11,7 Mo/frame — mêmes tenseurs, plus le surcoût de l'allocateur.
Le garde-fou `vram_stop_threshold_gb` (1 Go) arrête la propagation avant l'OOM
et conserve les annotations ; le client affiche où reprendre.

**Le garde-fou se bloquait après son premier déclenchement** (corrigé le
2026-09-01). Symptôme : une fois la limite atteinte, plus rien ne s'annote au
delà de la frame de prompt, à chaque tentative, jusqu'au redémarrage du serveur.
Cause : il lit `torch.cuda.mem_get_info()`, la mémoire libre vue du *pilote*,
qui compte comme occupé tout ce que l'allocateur cache de PyTorch retient — or
celui-ci ne rend jamais rien au pilote sans `empty_cache()`, qui n'était appelé
nulle part. Après un arrêt, `reset_session` libère bien l'état de tracking, mais
côté pilote la carte reste pleine, donc le test repassait vrai dès la première
frame de la propagation suivante.

Correctif : `_release_vram_cache()` appelle `empty_cache()` **une fois au début
de chaque propagation** et une fois à l'arrêt. La lecture du garde-fou reste
celle du pilote — tentant mais faux d'y ajouter le cache récupérable
(`memory_reserved - memory_allocated`) : ce cache est fragmenté en blocs aux
tailles déjà en usage, donc le compter en marge fait déclencher trop tard, tout
près de l'OOM qu'on veut éviter. Se tromper tôt coûte des frames, se tromper
tard coûte le run. Couvert par `tests/test_vram_guard.py`.

Flux de travail sur vidéo longue **en SAM 3.1** : prompt frame 0 → quelques
centaines de frames → se placer sur la dernière annotée → reprompt → suite.
Recoller les group_id à la jonction avec le Group ID Manager. **En SAM 3 ce
découpage n'est pas nécessaire** sur une séquence de cette longueur : les 754
frames passent d'une traite.

**Restreindre la plage de frames faisait planter la propagation** (corrigé le
2026-09-01). Le serveur passait la borne au prédicteur via
`max_frame_num_to_track` ; l'amont en dérive `valid_frame_end`
(`sam3_multiplex_detector.py:729`) puis calcule `chunk_start = (frame_idx //
batch_size) * batch_size` **sans le borner** (l. 741). Dès que le tracker
atteint la borne, `chunk_end` tombe à ≤ `chunk_start`, `chunk_find_inputs` est
vide et `[0]` lève un `IndexError` — reproduit à 176/200 et 16/30 frames. La
borne est désormais appliquée côté serveur, en cessant de consommer le
générateur (`_reached_frame_limit`), ce qui ne coûte rien : les frames au-delà
ne sont jamais calculées. Couvert par `tests/test_propagation_frame_range.py`.

## Petits objets et haute résolution

Toute image est **écrasée en 1008×1008 carré** avant le backbone (patch 14 → grille
72×72), et **la détection ne voit que ce niveau** : `num_feature_levels=1`, les niveaux
FPN 288 et 144 ne servent qu'au décodeur de masques. En 1920×1080 un token de détection
couvre 26,7 × 15 px natifs ; rien sous ~50 px de large n'est fiable.

SAM 3.1 ajoute trois plafonds absents de SAM 3, tous non surchargés par le serveur :
`max_num_objects=16` (qui jette par score croissant, donc les petits d'abord),
`suppress_det_close_to_boundary=True` à 2,5 % du bord, et une confirmation exigeant
3 détections consécutives. Et son builder n'expose pas `image_size`, que SAM 3 accepte.

Chiffres complets, arbitrage SAM 3 / SAM 3.1 et lots de travail :
**`PETITS-OBJETS-HAUTE-RESOLUTION.md`**.

## Délestage de l'état de tracking — mesuré, et insuffisant

`offload_state_to_cpu` est exposé dans `segment_anything_3_1_video.yaml`,
**défaut `false`**. Mesuré le 2026-09-01, même vidéo, `detector_batch_size: 1` :

| 200 frames, charge égale | détections | fps | Crête VRAM |
|---|---|---|---|
| défaut (état en VRAM) | 188 | 3,28 | 7,41 Go |
| `offload_state_to_cpu: true` | **188** | 3,15 | **5,72 Go** |

Les détections sont **identiques** : le cast bf16 des `maskmem_features` est
appliqué sur les deux chemins, donc le délestage n'est que du déplacement de
données. Le coût en vitesse est de **4 %**, bien moins que les 11-12 %
annoncés en commentaire par l'amont. Et il économise 1,7 Go de VRAM.

**Mais le plafond ne bouge pratiquement pas, il déménage.** Le coût RAM est de
**11,7 Mo/frame**, stable sur cinq relevés. Sur cette machine (30 Go, ~11 Go
disponibles au lancement) :

| | Frames atteintes | Mur |
|---|---|---|
| défaut | 432 | OOM VRAM à 10,60 Go |
| délesté | 450 | RAM disponible à 1,0 Go, RSS 15,46 Go |

Le délestage ne divise pas non plus la croissance VRAM par zéro : elle passe
de ~14 à ~4,4 Mo/frame, elle ne disparaît pas.

**L'espoir des « 754 frames d'une traite » est donc invalidé sur cette
machine.** Ce qui bloque désormais, ce sont les ~10 Go de coût fixe en RAM :
5,3 Go de modèle et 4,7 Go de frames décodées retenues par
`offload_video_to_cpu`. Piste suivante s'il faut y revenir : streamer les
frames depuis le disque au lieu de les garder en RAM libérerait ~4,7 Go, soit
~400 frames de marge. C'est là qu'est le levier, pas dans le délestage.

**Deux patches locaux dans `sam3-official/`** (non commités, `git diff` propre)
ont été nécessaires pour que le délestage fonctionne du tout, dans
`video_tracking_multiplex.py` : `_merge` (l. 2838) et `_append` (l. 2819)
ne convertissent que le dtype, pas le device, et cassent dès que l'état est en
RAM. Ils sont inertes tant que `offload_state_to_cpu: false` (le `.to(device)`
est un no-op quand tout est déjà sur GPU). Le mécanisme d'offload existe donc
dans la classe mais n'a manifestement jamais été exercé sur le chemin multiplex
dynamique.

Au passage : le builder instancie bien `Sam3VideoTrackingMultiplexDemo`
(`model_builder.py:985`), pas `VideoTrackingDynamicMultiplex` — c'est la
docstring du builder, qui annonce le type de base, qui avait égaré la note
précédente.

## Pistes mortes — ne pas refaire

- **`offload_state_to_cpu` contre le *premier* plafond** : l'état retenu ne fait
  que 10 Mo/frame (0,07 Go à 6 frames), pas 650. Il ne joue que sur le second.
  Implémenté et mesuré depuis — voir « Délestage de l'état de tracking », qui
  conclut qu'il ne lève pas non plus le second sur cette machine.
- **`hotstart_delay`** : 15, 8, 4 ou 0 donnent la même crête à 10,98 Go.
- **Attention non optimisée** : `use_fa3=False` retombe sur
  `F.scaled_dot_product_attention`, déjà économe sur sm_89.
- **`transformers`** : sa `Sam3VideoModel` implémente SAM 3, pas le multiplex
  (0 clé `interactive` contre 174 dans le checkpoint).
- **Baisser `multiplex_count`** : figé à 16 par les poids eux-mêmes.
- **Baisser `image_size`** : non exposé par le builder multiplex (SAM 3 le
  permet, pas 3.1).
- **Le *monter* au-dessus de 1008** : mesuré le 2026-09-02, le rappel se dégrade de
  façon monotone (0,902 → 0,879 à IoU 0,3 ; 0,712 → 0,585 à IoU 0,5) pour 3,6× le
  temps. Les poids sont entraînés à 1008 avec `window_size=24` figé : au-delà, la
  grille ne tombe plus sur un nombre entier de fenêtres. Détail dans
  « `image_size` au-dessus de 1008 ».
- **Attendre de la vitesse de SAM 3.1.** Non seulement il n'en apporte pas, mais
  il est **plus lent** : 3,42 fps contre 4,64 pour SAM 3, mesuré en aller-retour
  le 2026-09-01 (la note précédente disait « même débit », c'était faux). C'est
  structurel : le coût par frame est dominé par le détecteur, que
  `run_backbone_and_detection` appelle sur **chaque** frame dans les deux
  versions, avec le même backbone à `image_size=1008`. Le multiplex ne change que
  le *tracker* (16 objets par passe partagée au lieu d'une passe par objet), donc
  le gain suit le nombre d'objets suivis — et sur cette vidéo il y en a trop peu
  pour l'amortir. S'ajoute que FlashAttention 3, une partie du gain annoncé en
  amont, exige sm_90 : sur Ada (sm_89) on retombe sur
  `scaled_dot_product_attention`. Ce qu'apporte 3.1 ici, c'est la **densité de
  détection**, pas la capacité ni le débit.

Méthode qui a fini par payer : `torch.cuda.memory._record_memory_history()` puis
rejeu du flux alloc/free pour identifier ce qui est vivant au pic. Les quatre
premières hypothèses ci-dessus venaient d'un raisonnement par analogie architecturale.

## Pièges d'installation

- L'extra `sam3` du serveur oublie **`regex`** → `No module named 'regex'`.
- Le paquet officiel `sam3` oublie **`einops`, `psutil`, `pycocotools`**.
- Installer le paquet officiel avec **`--no-deps`** : son pin `numpy<2` casserait
  `opencv-python-headless` qui exige `numpy>=2`.
- `configs/models.yaml` est versionné et l'amont y change les défauts : après un
  pull, des modèles activés localement peuvent être recommentés en silence.
  Vérifier la ligne `Successfully loaded N/N`.
- FlashAttention 3 exige sm_90 ; sur Ada (sm_89) `use_fa3` est auto-détecté.

## En suspens

- **`configs/models.yaml` reste non commité**, côté serveur uniquement (le client
  n'en a plus de modification locale). Son état au 2026-09-01 :
  `segment_anything_3` seul, les deux modèles vidéo commentés — voir
  « Frame par frame plutôt que tracker » pour pourquoi, et « Association des
  group_id » pour la contrainte de cohabitation sur 12 Go. `segment_anything_3_tiled`
  y figure aussi, commenté : il partage le paquet vendorisé, donc aucun conflit
  d'import, mais chacun charge ses ~3,5 Go de poids — n'en activer qu'un.
- Le test client `test_first_tool_button_stays_inside_vertical_toolbar` échoue —
  bug amont avec Qt 6.9, reproduit sur `origin/main` vanilla. Seul échec sur les
  467 tests du client.
- Deux écarts `black` pré-existants dans `segment_anything_3_video.py`, dans les
  commits perso. Volontairement non reformatés. **`black` n'est pas installé
  dans le venv du serveur**, donc le code serveur ajouté le 2026-09-01 n'a pas pu
  être vérifié — il a été écrit à la main dans le style du fichier. Côté client
  en revanche `black` est présent : les fichiers ajoutés sont conformes
  (`line-length = 79`). Restent non conformes `tests/test_utils/test_file_search.py`
  et `test_image.py`, tous deux amont et intouchés.

## Historique des commits perso

Branche `sam3` = `origin/main` + patches jamais poussés en amont.

| | Client | Serveur |
|---|---|---|
| 2026-06-11 | 6 commits | 2 commits |
| 2026-09-01 | 4 (arrêt anticipé signalé, association des group_id, portée du prompt, formatage) | 8 (mémoire de propagation, SAM 3.1, prompt conservé, garde-fou VRAM, `detector_batch_size`, `offload_state_to_cpu`, plage de frames, une forme par objet) |
| 2026-09-16 | — | 2 (processeur réutilisé et masques sautés sur le chemin image, tuilage) |
| **Total** | **10** | **12** |

Seul `configs/models.yaml` reste non commité (serveur). Tous les commits perso ont
l'auteur `Sassanos` depuis le 2026-09-16 — voir « Dépôts GitHub ».
