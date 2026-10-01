# Annotation SAM 3 / SAM 3.1 — état du chantier

Notes de reprise. Dernière mise à jour : 2026-09-29.

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
| `mesure-vlm.py` | classe au VLM des objets annotés (Campagne_1 ou jeu COCO) ; garde les probabilités de chaque choix |
| `vlm-par-classe.py` | exactitude par classe, modalité, taille, piste et seuil, hors ligne, sur un ou plusieurs runs |
| `annote-video-sam3.py` | annotation à la cadence native d'un flux vidéo via le serveur : décodage streaming (aucune frame sur disque), JSONL par tronçon, reprise à la frame près |
| `planche-controle.py` | planches visuelles : frames décodées en streaming + boîtes des JSONL (couleur par label) |
| `slurm/` | déploiement cluster : jobs array, génération de tâches, bilan de lot — voir `slurm/README.md` |
| `TRACKER-PISTES.md` | notes de reprise pour l'association des pistes (group_id) sur les JSONL du cluster |
| `associe-pistes.py` | pistes (group_id) d'un vol entier, BoT-SORT hors ligne sur les JSONL du cluster → `Datasets/real/AnafiUKR/pistes/<vol>/` (0000011, 0000012, 0000018) ou `Datasets/real/CAMPAGNE3/pistes/<vol>/` (les 16 autres), en local |
| `pistes_botsort.py`, `reid_osnet.py`, `pistes_io.py` | BoT-SORT réécrit (sans boxmot), descripteurs OSNet, lecture des tronçons et décodage NVDEC en streaming |
| `rendu-pistes.py` | vidéo MP4 de contrôle des pistes (couleur et numéro par piste, recadrage auto) |
| `planche-pistes.py` | planche par piste : une ligne de vignettes réparties sur sa durée (contrôle de pureté) |
| `verifie-pistes-vlm.py` | vérification des pistes au VLM (Qwen3.8 servi par vLLM sur Slurm) : faux positifs, pureté, parties de véhicule, classe fine ; + contenance géométrique |
| `revue-pistes.py`, `revue-pistes.html` | revue humaine de pistes tirées au hasard (strates), page locale d'annotation : vérité terrain pour les prompts |
| `evalue-prompts-vlm.py` | un prompt du VLM contre la revue humaine : part réglage / part contrôle, désaccords |
| `rapport-vlm-pistes.py`, `bilan-vlm-pistes.py` | pages HTML locales : une carte par piste (filtrable) / synthèse avec graphiques et galeries relues |
| `poids/` | `osnet_x0_25_msmt17.pt` (zoo torchreid, 3 Mo) — **hors git** (`*.pt`) |
| `SLURM_INSTRUCTIONS.md` | règles du cluster pour l'assistant — **HORS GIT** (ignoré, ne pas le committer) |

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
- **Le port 22 est bloqué sur ce réseau**, seul le proxy HTTP `troie.ia` sort. L'alias
  passe donc par `ssh.github.com:443` avec
  `ProxyCommand nc -X connect -x proxy-cache.utils.dmz.troie.ia:3128 %h %p`. Symptôme
  sans ça : `ssh: connect to host github.com port 22: Connection timed out`.
- `gh` 2.101.0 est installé dans `~/.local/bin`, mais pas authentifié.
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

## VLM après SAM 3 — lot V0, mesuré le 2026-09-16

But : donner une classe fine aux détections SAM 3 (Griffon, VAB, GBC, VT4, Masstech T4 ;
voiture, pick-up, camionnette, camion, bus, deux-roues, engin pour les civils) et filtrer
ses faux positifs. Modèle : **Qwen3.5-4B** en bf16, local.

**Mécanique** (`X-AnyLabeling-Server/app/utils/crop_classifier.py`, tests
`tests/test_crop_classifier.py`) : découpage carré centré (marge 0,5 × côté, 16 px min),
gris hors image, agrandi à 448-896 px — un token Qwen3.5 couvre 32 px (patch 16, fusion
2×2). Choix fermés en lettres, lus sur les logits du prochain token, thinking coupé
(`enable_thinking=False`) : une passe, pas de parsing. Backend `transformers` ou
OpenAI/vLLM (`logprobs`, non encore essayé). **9,1 Gio de crête, 130-200 ms par
découpage**, masse des lettres 0,98-0,999 : le modèle répond bien par une lettre.

**Vérité terrain : `Datasets/real/Campagne_1`**, lue dans les zips. Labels à normaliser
(`person`/`Person`, `Véhicule civil`/`Véhicule Civil`). Échantillon : 627 découpages,
au plus 6 frames par piste. **Peu de véhicules physiques** : les scans tournent autour
d'un même Griffon ou VAB, donc 131 découpages Griffon ≠ 131 Griffon. Les chiffres par
piste sont les plus honnêtes, et ils restent sur un ou deux engins par type. GBC : 10
découpages, tous sous 16 px — rien de mesurable.

| Question | Griffon | VAB | Civil | Personnes rejetées |
|---|---|---|---|---|
| complète (sous-classe + « pas un véhicule » + « incertain »), cadre | 37 % | 27 % | 49 % | 100 % |
| sous-classe seule, cadre | 35 % (famille 95) | **71 %** | 84 % | — |
| sous-classe seule, sans cadre | **51 %** (famille 96) | 56 % | 84 % | — |
| binaire véhicule / non, cadre, argmax | 79 % | 100 % | 63 % | 78 % |
| binaire, sans cadre, argmax | 89 % | 100 % | 86 % | 58 % |

(« famille » = militaire bien rangé chez les militaires ; civil = toute sous-classe civile.)

Ce qu'on en tire :
1. **Ne jamais mettre le rejet dans la liste des sous-classes.** L'option « pas un
   véhicule » y aspire 45 % des Griffon et 49 % des civils, y compris des voitures nettes
   à 0,6-0,9. Le 100 % de personnes rejetées ne vaut donc rien. SAM 3 a déjà dit
   « véhicule » : on ne demande que le type, et le rejet se fait par une question à part.
2. **Militaire / civil : fiable. Type exact : non.** Famille 95-100 % ; mais Griffon, VAB,
   VT4 et Masstech se confondent (35-71 % selon le cadre). Par piste, sans cadre :
   Griffon RGB 9/13, civil RGB 24/25.
3. **Filtre de faux positifs : seuiller P(véhicule), pas l'argmax.** Question binaire avec
   cadre, objets ≥ 16 px : seuil 0,1 → 99 % des véhicules gardés, 75 % des personnes
   rejetées ; seuil 0,2 → 96 % / 85 % ; 0,3 → 90 % / 93 %. **Les personnes ne sont qu'un
   substitut** : les vrais faux positifs de SAM 3 (ombres, buissons, bâches) restent à
   mesurer.
4. **Le cadre rouge aide le binaire et le VAB, gêne le Griffon.** Pas de réglage unique ;
   à retrancher sur plus de véhicules.
5. Échecs lisibles sur planche (`runs/vlm/planche-*.jpg`) : Griffon **sous filet ou bâche**,
   frames quasi noires, blindés **vus du dessus** sur piste, et un VAB peu connu du modèle.

### Séquences B, bâches exclues

Les vidéos `B_*` sont la cible prioritaire. **Les quatre séquences B avec un Griffon le
montrent bâché sous un filet** (103746, 105526, 145902 `00h00m00s` et `00h02m30s`,
vérifié sur toutes les frames) ; ce cas est jugé trop difficile et **mis de côté**. Restent
`145902 00h05m00s` et `00h07m30s` : 163 découpages de véhicules civils (21 pistes),
195 de personnes, 9 GBC sous 16 px et flous. Filtre :
`./vlm-par-classe.py --prefixe B_ --exclure 103746,105526,00h00m00s,00h02m30s run.json`.

- **Civils, sous-classe seule, sans cadre, descriptions v1 : famille 87 % par découpage,
  19/21 pistes.** Au-dessus de 0,5 de confiance : 74 % des découpages gardés, **100 %
  justes**. Relu sur planche (`runs/vlm/planche-B-civils-sousclasses.jpg`, sans vérité
  terrain fine) : camionnettes à 0,96-0,99, voitures à 0,9-0,99 ; les erreurs sont toutes
  sous 0,5 (véhicule à moitié caché, utilitaire de 37 px).
- **Filtre binaire, avec cadre, objets ≥ 16 px** : seuil 0,1 → 98 % des véhicules gardés,
  80 % des personnes rejetées ; 0,2 → 96 % / 91 % ; 0,3 → 93 % / 96 %.
- **Descriptions v2 (`--descriptions v2`) : à ne pas garder.** Écrites pour le Griffon
  bâché (essieux, cabine séparée, « souvent sous filet »), elles le font passer de 7 % à
  53-62 % sur les séquences bâchées, mais envoient 29 % des civils vers `gbc` et font
  tomber les civils de 87 % à 57 %. Le « 100 % GBC » qu'elles affichent vient du même
  attracteur, pas d'une reconnaissance.
- GBC : rien de mesurable (9 découpages de 8-16 px).

Descriptions données au modèle : `MILITAIRES` et `CIVILS` en tête de `mesure-vlm.py`
(sources defense.gouv.fr et Wikipédia pour VT4 et Masstech T4). Sorties brutes dans
`runs/vlm/` (hors git).

## Pistes pour la classe fine — le tracker mesuré sur UAVDT, 2026-09-16

Flux visé : frames à 30 fps, SAM 3 frame par frame en classes grossières
(`vehicle`, `person`), association des `group_id`, puis **une classe fine choisie par
piste** (proposée par le VLM, validée par l'humain). Tout repose sur la **pureté** des
pistes : une piste coupée en deux coûte un clic, une piste qui mélange deux objets donne
une classe fausse à une partie de ses frames sans que rien ne le signale.

`mesure-tracker.py` donne à `group_id_association.py` les boîtes de vérité terrain
d'UAVDT **sans leur identité** (`Datasets/opensource/raw/UAVDT`, format Supervisely,
tag `target id`, véhicules vus de drone, 30 fps, 1024×540, ~19 px médian), à la cadence
voulue, éventuellement dégradées (`--drop` détections retirées au hasard, `--jitter`
bruit en fraction de la taille). Réglage sur 10 séquences `train`, résultats sur 10
séquences `test`. Pureté = part des détections d'une piste appartenant à son objet
majoritaire.

| test, 30 fps | pureté | morceaux / objet | objets d'un seul tenant |
|---|---|---|---|
| boîtes parfaites, réglage par défaut | 0,955 | 1,04 | 96 % |
| 20 % de détections perdues, défaut | **0,684** | 2,62 | 41 % |
| 20 % perdues, `buffer2=0.3 max_dist=1.0 max_age=10` | 0,801 | 1,60 | 70 % |
| 20 % perdues + bruit 10 %, resserré | 0,703 | 1,95 | 55 % |

1. **Le 30 fps est justifié.** Boîtes parfaites : pureté 0,955 à 30 fps, 0,91 à 10 fps,
   0,88 à 5 fps, **0,77 à 1 fps** — la cadence des annotations actuelles de Campagne_1.
2. **Le tracker s'effondre dès que des détections manquent.** Mécanisme vu sur M0208 :
   quand un objet n'est plus détecté, sa piste continue sur sa lancée et **capte le
   voisin** (une piste : 55 détections de l'objet 9 puis 33 de l'objet 3). La passe 2 est
   trop permissive pour des véhicules denses : tampon 1,0, IoU > 0, centres jusqu'à
   3,75 diagonales (~70 px pour 19 px d'objet). Elle a été réglée sur des piétons épars.
3. **Resserrer la passe 2 aide sans rien casser** (0,684 → 0,801, 2,62 → 1,60 morceaux)
   mais ne suffit pas : sans apparence, deux véhicules voisins restent interchangeables.
   Le recollement des fragments n'y est pour rien (`--no-merge` : mêmes chiffres).
4. **Réserve** : les pertes simulées sont indépendantes d'une frame à l'autre ; celles de
   SAM 3 sont corrélées (un petit objet manqué plusieurs frames d'affilée). La mesure qui
   tranche passe par de vraies détections SAM 3 sur UAVDT — faite, ci-dessous.

### Comparaison avec des trackers établis (boxmot) — vraies détections SAM 3

`boxmot` 25.0.0 est installé dans un venv à part, **`.venv-tracking/`** (hors git) : il
tirait torch cu130, que le pilote (CUDA 12.9) refuse — torch cu128 réinstallé par-dessus
avec `--no-deps`. La ReID de BoT-SORT échoue sur GPU (`CUDNN_STATUS_NOT_INITIALIZED`,
cuDNN mal apparié dans ce venv) : elle tourne sur CPU. `mesure-tracker.py` charge
`group_id_association.py` par chemin, donc tourne dans les deux venvs.
**Licence boxmot : AGPL-3.0** — à regarder avant tout déploiement en service.

**Vérité terrain dégradée, 10 séquences test, 30 fps, 20 % de pertes** :

| tracker | pureté | morceaux / objet | couverture |
|---|---|---|---|
| maison | 0,684 | 2,62 | 1,000 |
| ByteTrack | **0,987** | 1,22 | 0,993 |
| OC-SORT | 0,992 | 1,23 | 0,633 (jette un tiers des détections) |

**Vraies détections SAM 3** (`detecte-uavdt.py`, prompt `vehicle`, conf 0,15 gardée, seuil
rejoué hors ligne ; M0801 basse altitude, M0601 haute altitude, M0403 dense ; 1 184 frames,
~410 ms/frame). Moyenne pondérée, seuil de score 0,5 :

| tracker | pureté | morceaux / objet | objets d'un seul tenant | coût (CPU) |
|---|---|---|---|---|
| maison, défaut | 0,885 | 1,52 | 75 % | 2 ms/frame |
| maison, `buffer2=0.3 max_dist=1.0 max_age=10` | 0,939 | 1,28 | 82 % | 2 ms/frame |
| ByteTrack | 0,985 | 2,26 | 71 % | 10 ms/frame |
| OC-SORT | 0,992 | 2,20 | 69 % | 13 ms/frame |
| **BoT-SORT** (ReID OSNet + compensation caméra) | **0,987** | **1,03** | **97 %** | 547 ms/frame |

1. **BoT-SORT est le seul à être à la fois pur et d'un seul tenant.** ByteTrack et OC-SORT
   sont purs mais coupent chaque objet en ~2 morceaux (4 sur M0601, haute altitude) : autant
   de décisions en plus à la revue. Le tracker maison fragmente moins qu'eux mais mélange.
2. **L'apparence fait la différence** : sur M0601, 145 pistes pour 51 objets avec BoT-SORT,
   723 avec ByteTrack, pour la même pureté.
3. **Le seuil SAM 3 compte peu pour la pureté** (0,985-0,987 de 0,15 à 0,5 pour BoT-SORT) ;
   à 0,5 la couverture est la meilleure.
4. **Coût** : 547 ms/frame sur CPU, soit 2,7 h pour 10 min de vidéo à 30 fps. Il faut la
   ReID sur GPU (réparer cuDNN dans `.venv-tracking`, ou l'installer dans un venv sain).
5. **« FP en piste » (~35 %) surestime les faux positifs de SAM 3** : UAVDT a des zones
   ignorées où des véhicules visibles ne sont pas annotés. La pureté, calculée sur les seules
   détections appariées, n'en dépend pas.
6. **Portée** : 3 séquences, véhicules seuls, ReID entraînée sur des piétons (MSMT17).

**Les annotations existantes ont le même défaut.** Dans `Track Review` sur la séquence B
`145902 00h07m30s` (1 fps), la piste 20 alterne voiture bleue / voiture blanche d'une frame
à l'autre : ce n'est pas un seul changement d'objet, une découpe ne suffit pas.

### Revue des pistes dans le client — `Tool > Track Review`

`widgets/track_review_dialog.py` sur `utils/track_review.py` (sans Qt), tests
`tests/test_utils/test_track_review.py` et `tests/test_widgets/test_track_review_dialog.py`.
Une ligne par `group_id` : vignette de la boîte la plus grande (hors interpolées), durée,
classe actuelle, proposition VLM si les attributs `vlm_class`/`vlm_prob` existent, menu de
classe. La classe choisie est écrite sur toutes les formes de la piste, la classe grossière
d'origine gardée dans l'attribut `coarse_label`. La piste sélectionnée montre 12 frames
réparties sur sa durée ; **« Split from selected frame »** donne un nouveau `group_id` à
cette frame et aux suivantes. « Accept suggestions above threshold » adopte les propositions
VLM au-dessus du seuil. Seuls les fichiers touchés sont réécrits (écriture atomique).
Manque : déplacer une frame isolée vers une autre piste (cas des alternances), supprimer une
piste de faux positifs.

## Pistes BoT-SORT sur un vol entier — `associe-pistes.py`, 2026-09-23

Suite de `TRACKER-PISTES.md` : associer en pistes les détections SAM 3 frame par frame
du lot Slurm, **hors ligne**, par vol. Fait sur le vol **0000011** (15 954 frames,
296 170 détections).

### BoT-SORT réécrit, sans boxmot

BoT-SORT était le meilleur sur UAVDT (section précédente), mais boxmot et ultralytics sont
sous **AGPL-3.0**. Réécrit d'après l'article (arXiv:2206.14651) et le dépôt de référence
`NirAharon/BoT-SORT` (MIT), sans une ligne de boxmot :

- `pistes_botsort.py` — Kalman (cx, cy, w, h) à bruit proportionnel à la taille,
  compensation caméra appliquée aux états prédits, cascade ByteTrack (sûres : IoU et
  apparence fusionnées par un minimum ; faibles : IoU seule ; non confirmées), affectation
  équivalente à `lap.lapjv(cost_limit)` via scipy. Ne lit ni vidéo ni fichier : l'appelant
  donne boîtes, scores, descripteurs et affine caméra, d'où le rejeu depuis un cache.
- `reid_osnet.py` — OSNet x0.25 réécrit avec les noms de couches de torchreid (MIT) pour
  charger son poids `osnet_x0_25_msmt17.pt` tel quel (daté d'avril 2021, copié depuis
  `.venv-tracking` dans `poids/`, hors git). **Descripteurs identiques à ceux de boxmot** :
  cosinus ≥ 0,9999999 sur les mêmes boîtes.
- Tourne dans **`X-AnyLabeling-Server/.venv`** (torch cu128, cuDNN sain, PyAV 18 avec
  NVDEC), rien à installer. `.venv-tracking` ne sert plus qu'à boxmot.

**Parité mesurée** (`mesure-tracker.py --tracker botsort-maison`, mêmes 3 séquences UAVDT,
vraies détections SAM 3, score ≥ 0,5) :

| | pureté | morceaux / objet | d'un seul tenant | M0601 | M0403 | coût |
|---|---|---|---|---|---|---|
| boxmot BoT-SORT | 0,987 | 1,03 | 97 % | 0,971 / 1,08 | 0,997 / 1,00 | 547 ms/frame (CPU) |
| **botsort-maison** | **0,987** | **1,03** | **96,8 %** | 0,971 / 1,08 | 0,997 / 1,00 | ~40 ms/frame (GPU) |

`fuse_score` (IoU pondérée par le score, défaut du dépôt de référence) : 1,04 morceau,
pas mieux, laissé à `False` comme boxmot.

### Seuils : SAM 3 n'est pas YOLOX

Les seuils de BoT-SORT (0,5 pour une détection sûre, 0,6 pour ouvrir une piste) supposent
des scores de type YOLOX. SAM 3 note les **personnes entre 0,25 et 0,59** : aucune piste
personne ne naissait. Sur UAVDT (score ≥ 0,25) :

| `track_high` / `new_track` | pureté | morceaux / objet | couverture |
|---|---|---|---|
| 0,5 / 0,6 (référence) | 0,986 | 1,03 | 0,84 |
| 0,4 / 0,4 | 0,986 | 1,06 | 0,92 |
| **0,3 / 0,3** (défaut de `associe-pistes.py`) | **0,986** | 1,06 | **0,96** |

Descendre les seuils ne coûte rien en pureté. `ParamsBotSort` garde les défauts de
référence ; `associe-pistes.py` applique `SEUILS_SAM3`.

### Le pipeline

**Où sont les sorties.** En local, dans `SORTIES` (`pistes_io.py`) =
`~/Documents/Geolocalisation/Datasets/real/AnafiUKR/pistes/<vol>/`. Les entrées restent
lues dans `/media/users/cbarbier/annots-sam3/` (détections, index, miroir des vidéos), où
**rien n'est jamais supprimé ni modifié**. 0000011 et 0000012 ont d'abord été écrits sous
`annots-sam3/pistes/`, puis copiés en local ; `--sortie` choisit un autre dossier.

1. **Indices** (une fois par vol, mis en cache) : décodage NVDEC du flux 0 en streaming,
   affine caméra entre frames successives (flot optique clairsemé, mêmes réglages que
   `group_id_association.py`, en 960 px de large), descripteur OSNet de **chaque**
   détection → `<vol>/indices/<vol>_indices.npz` (304 Mo pour 0000011). Le cache
   est reconnu par une empreinte du contenu des détections, pas des fichiers : recompresser
   un tronçon ne l'invalide pas.
2. **Association** (16 s pour le vol), rejouable à volonté :
   - **doublons inter-prompts fusionnés** par classe grossière (NMS à IoU 0,6) : SAM 3
     répond une fois par prompt, et sur 0000011 **132 000 paires de boîtes véhicule à
     IoU > 0,7 portent deux labels fins différents** sur la même frame (car/van 51 000,
     truck/van 28 000, car/truck 25 000). Sans fusion, chaque véhicule aurait deux pistes.
     La boîte au meilleur score représente la grappe, tous ses labels votent ;
   - BoT-SORT **par classe grossière** (`vehicle`, `person`) et **par tronçon continu** ;
   - **trous IR** : un écart de plus de `--coupure` (5) frames arrête toutes les pistes.
     Rien n'est recollé à travers un trou IR (0000011 : 8 tronçons continus) ;
   - classe fine de la piste = **vote des scores** sur toutes ses frames, distribution
     gardée (`labels`) ;
   - trous internes **interpolés** (`interpolated: true`), pistes de moins de `--min-len`
     (5) détections écartées.
3. Sortie : `<vol>/<vol>_pistes_<tag>.jsonl` (une piste par ligne) et le bilan
   `.json` (réglages, sources, tronçons continus, chiffres). `tag` =
   `botsort-<empreinte des réglages>` : mêmes réglages → même fichier, mêmes `group_id`
   (1..N, triés par apparition).

```json
{"group_id": 7, "coarse": "vehicle", "label": "car", "labels": {"car": 0.62, "van": 0.38},
 "n_frames": 42, "n_detected": 40, "sample_min": 21340, "sample_max": 21381,
 "t_min_s": 1307.4, "t_max_s": 1308.8, "score_mean": 0.81, "size_median_px": 41.0,
 "frames": [{"i": 21340, "bbox": [x1, y1, x2, y2], "score": 0.9, "label": "car",
             "shapes": [3, 17]},
            {"i": 21341, "bbox": [...], "interpolated": true}, ...]}
```

`shapes` = indices des formes de la frame dans le JSONL source (la grappe, tête d'abord) :
de quoi revenir aux détections d'origine, pour l'export vers le client ou le VLM.

```bash
PY=X-AnyLabeling-Server/.venv/bin/python
$PY associe-pistes.py --vol 0000011                           # vol entier
$PY associe-pistes.py --vol 0000011 --set track_buffer=60     # rejeu, cache réutilisé
$PY associe-pistes.py --vol 0000011 --debut 19830 --fin 22330 --tag essai
```

Refuse un vol dont un tronçon n'a pas son `.done` (`--partiel` pour passer outre).

### Résultat sur 0000011

| | |
|---|---|
| Indices (décodage 4K NVDEC + GMC + ReID) | 6,9 min, 38 frames/s, 0 frame introuvable |
| Association | 16 s |
| Détections → après fusion des doublons | véhicules 255 650 → 151 399 ; personnes 40 520 (un seul prompt) |
| Pistes ≥ 5 détections | **1 968** : 1 580 véhicules, 388 personnes ; 1 389 de plus d'une seconde |
| Couverture (grappes dans une piste) | véhicules 0,89, personnes 0,85 |
| Longueur (détections) | médiane 44, p90 204, max 917 (une personne suivie 30,8 s) |
| Classes fines votées | car 1 006, van 270, truck 185, mil_tank 59, mil_truck 41, motorcycle 15 |

**Contrôle visuel (planches, pas de vérité terrain) :**

- **Personnes : propres**, y compris à 8 px (#933) et sur 26-30 s pour les soldats
  devant le bâtiment (#89, #314-#317). Les pistes fausses sont des faux positifs à
  score bas (#443 une clôture à 0,39, #531 à 0,33).
- **Véhicules en scène stable** (parking, zone dense 19830-22330) : une piste par
  véhicule, tenue à travers le dézoom de la caméra (échelle 0,88 par frame à 21136).
- **Les impuretés se concentrent dans les zooms rapides et le flou** (tronçon
  10000-12489, boîtes de 20 à 800 px) : #276 et #865 changent d'objet, #386 est ambigu.
  Autour des toilettes cyan, étiquetées `car`, des pistes mélangent faux positif et
  véhicule voisin.
- **Le label fin de SAM 3 n'est pas fiable par frame** : sur une piste véhicule, le
  label majoritaire ne pèse que 75 % des votes en médiane (car/van surtout). Ce qui
  confirme qu'il faut décider la classe fine par piste (VLM, revue).
- Essayé et **abandonné** : un score de « rupture d'apparence » par piste (cosinus
  entre descripteurs OSNet avant/après la meilleure coupure). Les pistes pures qui
  traversent un zoom (#483 : 0,35) sortent plus « rompues » que les vraies impures
  (#276 : 0,31). Le zoom change l'apparence autant qu'un changement d'objet.

### Résultat sur 0000012

Vol quatre fois plus dense, vu de dessus, véhicules plus petits (23 px médians contre 60) :

| | |
|---|---|
| Frames, détections | 40 066 frames (6 tronçons continus), 1 170 966 détections |
| Indices / association | 20,6 min (32 frames/s) / 125 s ; cache 1,2 Go, pistes 92 Mo |
| Après fusion des doublons | véhicules 1 049 916 → 795 395 (−24 %, contre −41 % sur 0000011 : presque tout sort en `car`) |
| Pistes ≥ 5 détections | **13 230** : 11 719 véhicules, 1 511 personnes ; 8 069 de plus d'une seconde |
| Couverture | véhicules 0,84, personnes 0,75 |
| Longueur (détections) | médiane 28, p90 134, max 1 063 |
| Pic | ~630 détections de ~22 px par frame autour du sample 27540 (grand parc de véhicules), 536 pistes actives dont 414 détectées sur la frame |

- **Véhicules** : les pistes longues sont nettes (#507 à #514, voitures d'un parking
  suivies 29 s chacune à travers un zoom). Des sauts d'objet restent possibles quand la
  caméra bouge vite (#1215, #1661).
- **Personnes : beaucoup de faux positifs suivis**, dont un panneau jaune (#1085), un
  panneau rond (#1182) et des taches floues. Leurs scores (0,41-0,47) sont ceux des vraies
  personnes (#135, #1400) : le score ne les sépare pas, c'est au tri par piste (VLM,
  revue) de le faire.

### Résultat sur 0000018

Premier vol écrit directement en local. Vol clairsemé, beaucoup de rafales IR :

| | |
|---|---|
| Frames, détections | 22 722 frames (**15 tronçons continus**), 162 635 détections |
| Indices / association | 9,5 min (40 frames/s) / 11 s |
| Après fusion des doublons | véhicules 149 120 → 104 763 (−30 %) ; personnes 13 515 |
| Pistes ≥ 5 détections | **1 253** : 1 064 véhicules, 189 personnes ; 843 de plus d'une seconde |
| Couverture | véhicules 0,88, personnes 0,79 |
| Longueur (détections) | médiane 39, p90 185, max 1 230 (une voiture suivie 41 s pendant que la vue tourne) |

- **Véhicules propres**, y compris des voitures floues de 10-12 px au score 0,39-0,47
  (#83, #102, #111). #162, d'abord lu comme « un camion à demi caché par un arbre », est
  en fait **le bas de l'avant** d'un camion que #159 couvre en entier (contenu dans #159
  sur 87 % de ses frames, relu le 2026-09-24) : une partie, pas un véhicule.
- **Personnes** : les soldats sont bien suivis (#368, #369, #373). Faux positifs et cas
  ambigus : une ombre de 668 px (#41), des boîtes contre une portière (#370, #383), des
  taches de 14 px (#677, #680).

### Visualiser

```bash
# vidéo 1080p, une couleur et un numéro par piste, recadrée sur les pistes,
# détections restées hors piste en gris (≈1 min pour 2 500 frames)
$PY rendu-pistes.py --vol 0000011 --debut 19830 --fin 22330 --zone auto --detections
# survol du vol entier en accéléré x2 (≈4 min)
$PY rendu-pistes.py --vol 0000011 --pas 2 --zone auto
# verdicts du VLM : classe fine du VLM, rejetées en gris (ou --masquer-rejets),
# suffixes partie / impure / ? (incertain) / mil (personne militaire)
$PY rendu-pistes.py --vol 0000004 --debut 9000 --fin 10000 --zone auto --vlm p5
# planches : pistes les plus longues, tirées au hasard, une classe, ou des group_id précis
$PY planche-pistes.py --vol 0000011 --n 40 --tri longueur --out planche.jpg
$PY planche-pistes.py --vol 0000011 --tri aleatoire --classes person --out p.jpg
$PY planche-pistes.py --vol 0000011 --gid 276 865 --k 12 --out douteuses.jpg
```

Sorties dans `<vol>/rendus/`. Dans mpv ou VLC, `.` avance d'une frame :
suffisant pour voir une piste sauter d'un objet à l'autre. Boîte pleine = détection,
pointillée = interpolée ; traîne = centres des 20 dernières frames (repère image, non
compensé). La vidéo saute les rafales IR ; le bandeau donne le sample_index.

Quand la caméra balaie (0000012 autour de 27540 : 530 pistes sur toute l'image), les
traînes deviennent de longs traits horizontaux, qui montrent le mouvement de la caméra et
non celui des véhicules. `--zone auto` ne peut pas zoomer quand les pistes couvrent
toute l'image. Préférer `--traine 0` et une zone fixe en pixels 4K
(`--zone 1000 300 2600 1200`).

### Suite

- Lot AnafiUKR complet (0000011, 0000012, 0000018). CAMPAGNE3 : même commande par vol,
  compter ~40 frames/s d'indices plus 10 s à 2 min d'association selon la densité.
- Export des pistes retenues vers des JSON X-AnyLabeling par frame pour `Track Review`
  (les `shapes` y mènent), puis classe fine par piste au VLM.
- Zooms rapides : pistes impures, c'est là que la revue doit regarder d'abord. Piste
  possible : couper une piste quand la taille de la boîte diverge de l'échelle estimée
  par la GMC.
- Licence du poids OSNet : code torchreid MIT, mais entraîné sur MSMT17 (jeu de
  recherche) — à regarder avant un déploiement hors recherche.

## Vérifier les pistes au VLM — Qwen3.8-27B sur Slurm, 2026-09-24

But : voir ce qu'un VLM plus gros que Qwen3.5-4B (lot V0) apporte aux pistes SAM 3 +
BoT-SORT : rejeter les faux positifs, repérer les pistes impures et les parties de
véhicule, proposer une classe fine. Modèle : `/media/shared/models/qwen/Qwen3.8-27B-FP8`
(architecture Qwen3.5, FP8 par blocs, ~29 Go).

**Visuels** (locaux, hors git, images base64) dans `Datasets/real/AnafiUKR/pistes/` :
`vlm-bilan.html` (synthèse : chiffres, graphique, galeries relues), `vlm-essai-connus.html`
(46 cas connus, quatre prompts côte à côte), `vlm-pilote-p3.html` (317 pistes, filtrable).

### Mise en place

- **Serveur** : `slurm/vllm-qwen38.sbatch`, vLLM 0.30.0 (torch 2.13 cu130), **une A100
  80 Go** suffit (Marlin FP8 sur Ampere ; driver 580 des nœuds GPU : CUDA 13 OK). Chargement
  ~8 min depuis ceph. Projet cluster : `/media/users/cbarbier/vlm-pistes/`.
- **Environnement** : `slurm/setup-vllm-env.sbatch`, lancé une fois sur la partition `cpu`
  (le `/tmp` du nœud de connexion fait 6 Go, un venv vLLM 7,7 Go). Construit dans `/dev/shm`
  au chemin où le serveur le détarre, archivé en un seul fichier
  (`vlm-pistes/env/vllm-qwen38-v1.tar.zst`, 3,3 Go) : aucun venv sur ceph. Tous les caches
  (uv, pip, virtualenv, vLLM, Triton, Inductor, FlashInfer, CUDA) vont dans `/dev/shm`,
  statistiques d'usage de vLLM coupées. Proxy du site passé par `https_proxy`.
- **Réseau** : un pare-feu bloque les ports des nœuds de calcul, **même depuis le nœud de
  connexion**. Ce qui marche : SSH jusqu'au nœud où tourne le job, en saut par le login,
  `ssh -N -J master.slurm.troie.ia -L <port>:127.0.0.1:<port> gpu02.slurm.vm.troie.ia`
  (port et nœud dans le journal du job).
- **Client** (`verifie-pistes-vlm.py`, venv du serveur X-AnyLabeling) : 8 vues réparties sur
  les frames détectées de la piste (boîte en rouge, un peu plus grande que la boîte) + une
  vue large, décodées sur le PC (NVDEC, streaming) et rangées dans un zip par vol ; requêtes
  parallèles ; réponse en **JSON contraint** (`response_format` json_schema), probabilités
  lues sur les logprobs bruts. Reprise piste par piste.

```bash
PY=X-AnyLabeling-Server/.venv/bin/python
$PY verifie-pistes-vlm.py --vol 0000018 --selection connus --serveur http://localhost:13610/v1
$PY verifie-pistes-vlm.py --vol 0000018 --selection pilote --serveur ... --prompt p3
$PY rapport-vlm-pistes.py <SORTIES>/0000018/vlm/0000018_vlm_rapide_p3.jsonl --out r.html
$PY bilan-vlm-pistes.py                 # -> <SORTIES>/vlm-bilan.html
```

Questions posées, séparées (leçon de V0 : le rejet mis parmi les sous-classes aspire les
vrais véhicules) : `same_object` + `different_views` (pureté), `real_object` (faux positif),
`box_covers` (entier ou partie), et pour les véhicules `fine_class` (les 12 labels véhicule de
`taxonomy.yaml`, décrits) + `affiliation`. Le VLM ne voit pas le label fin de SAM 3.

### Le prompt compte plus que la réflexion — 46 cas relus

Cas du README relus sur planche + 9 parties de véhicule vérifiées (dont #162 de 0000018) :

| | rapide p1 | réflexion low p1 | rapide p2 | **rapide p3** |
|---|---|---|---|---|
| vrais objets gardés (23) | 23 | 22 | 19 | **22** |
| faux positifs rejetés (7) | 2 | 4 | 7 | **6** |
| pistes impures trouvées (4) | 1 | 2 | 2 | 2 |
| pistes pures jugées pures (21) | 21 | 21 | 21 | 21 |
| parties trouvées par le VLM (9) | — | — | 5 | **6** |
| jetons sortis par piste (médiane) | 59 | 629 | 128 | 127 |

1. **p1 hallucine des personnes sur les panneaux** (#1085 losange jaune « personne en gilet
   haute visibilité », #1182 panneau rond), à P = 0,99. La réflexion ne corrige pas : elle
   raisonne sur la vue large, où il y a d'autres personnes, et sur l'intro « le tracker a
   suivi une personne ».
2. **Décrire l'aspect avant de nommer (p2, p3) règle les panneaux.** p2 ajoute « le détecteur
   se trompe souvent » : 4 vrais objets rejetés à P ≈ 0 (voitures de 10-11 px, un soldat),
   le même effet d'attraction que le rejet dans V0. p3 retire cette phrase : meilleur
   compromis. Ces 46 cas ont servi à régler le prompt : chiffres optimistes.
3. **Réflexion `low`** : ~10× plus de jetons, ~0,4 piste/s au lieu de ~1,3, gain faible, et
   **les probabilités tombent à 0/1** — plus de seuil possible. Pas retenue.
4. Restent en p3 : #507 (voiture de 25 px vue de très haut, « boîtier ») rejetée, #677
   (tache de 14 px) gardée comme personne.

### Pilote — 317 pistes tirées au hasard, p3

Strates tirées au hasard parmi les pistes d'au moins 10 détections, ~5 min de VLM :

| strate | pistes | rejetées par le VLM | relu à l'œil |
|---|---|---|---|
| étiquetées militaires par SAM 3 | 44 | **70 %** | 12/12 rejets justes |
| personnes | 98 | **43 %** | ~9/12 justes, 3 ambigus |
| véhicules, zooms rapides (0000011) | 15 | 27 % | — |
| véhicules | 118 | **17 %** | ~6/10 justes, 1 faux (#683 de 0000011, petite camionnette) |

1. **Le prompt « tank » de SAM 3 accroche des cuves, un château d'eau, des conteneurs, des
   buses en béton, des armoires** : 70 % des pistes `mil_*` ne sont pas des véhicules.
2. **Personnes** : balises, panneaux, un chien, une personne sur une affiche, un sac à dos —
   le tri que le score SAM 3 ne faisait pas sur 0000012.
3. **Pureté : faible.** Le VLM compare mal 8 vues : les sauts entre deux véhicules semblables
   (#1215) passent. 10 % des véhicules dits impurs, 20 % dans les zooms rapides.
4. **Classe fine** : sur 79 `car` SAM 3 gardés, 44 confirmés, 19 → `van`, 8 → `truck` (vues
   de haut), 7 → militaire ; pas de vérité terrain pour trancher.

### Parties de véhicule : géométrie d'abord, VLM en second

**Contenance géométrique** (sans VLM, `contenance()`) : une boîte véhicule est contenue sur
une frame quand >= 80 % de son aire tombe dans la boîte d'une autre piste véhicule >= 1,5
fois plus grande. Pistes contenues sur >= 80 % de leurs frames : **223/1 580 (14 %) sur
0000011**, 262/11 719 (2 %) sur 0000012, 44/1 064 (4 %) sur 0000018. Les 9 relues sont
toutes des parties (flanc ou bas de camionnette, arrière ou roues de SUV, avant de camion).

Sur le pilote, les deux signaux se complètent (désaccords relus dans `vlm-bilan.html`) :
- la géométrie rate les parties **dont le véhicule entier n'a pas de boîte** (arrière de SUV
  #319, toit de camionnette #486, moitié de voiture #1766) — à revoir, pas à supprimer ;
- le VLM rejette parfois une partie comme « pas un véhicule » (roue de secours #423, flanc
  #449) — même décision au bout ;
- fausse alerte géométrique : #1655 de 0000011, véhicule entier pris dans une boîte trop
  grande.

Règle proposée : **contenue → doublon à retirer ; partie selon le VLM seul → à revoir.**

### Coût et suite

~1 800-2 000 jetons d'entrée par piste, **~1,3 piste/s** sur une A100 à 36 requêtes
parallèles : ~3,5 h pour les ~16 500 pistes des trois vols, plus 20-40 min de vignettes par
vol (décodage via le réseau depuis le PC ; faisable sur le cluster). Suite possible : lot
complet en p3 ; attributs `vlm_*` écrits pour `Track Review` ; la pureté demande autre chose
qu'un VLM qui regarde 8 vues (comparer vue à vue, ou couper aux zooms).

## Revue humaine et prompt p5 — 2026-09-29

### Pistes des 16 autres vols

Les pistes des 16 vols CAMPAGNE3 hors lot AnafiUKR ont été faites le 2026-09-25
(mêmes réglages, `botsort-54e109`) dans `Datasets/real/CAMPAGNE3/pistes/<vol>/` ;
`pistes_io.dossier_pistes()` choisit le dossier selon le vol. 0000004 s'arrête avant
le mur de corruption (`--fin 24067` ; `associe-pistes.py` ne redemande plus les frames
en erreur au décodeur). **0000016 est le seul vol en base de temps 1/90000** (les autres
1/30000) : `decode_frames` prend désormais celle du flux et `timescale_index()` celle de
l'index ; 457 pistes.

**Vols gardés pour le VLM : les 9 vols RGB de jour** (`VOLS_JOUR`, filmés de 05h33 à
17h08). Exclus sans rien supprimer : les 9 vols de 22h48 à 00h35 (0000229-0000236,
0000429, 0000431 : nuit, 70-85 % d'IR, vues sombres) et 0000016 (export de la tablette :
écran filmé, nuit).

### Revue humaine — `revue-pistes.py`

200 pistes des 19 vols, tirées par strates (7 parties de véhicule par contenance
géométrique, 5 zooms, puis par label SAM 3, rares sur-représentés : on cherche des
échecs, pas des taux), diversifiées en vol, taille, longueur et score ; pistes déjà
passées au VLM exclues. Page locale (`serveur`, http://127.0.0.1:8765) avec les vues
mêmes du VLM (réglage p3), questions du VLM + nature des faux positifs et « jugeable
sur ces vues ». Réponses : `Datasets/real/CAMPAGNE3/revue-vlm/annotations.jsonl`.

Sur les 134 de jour, ce que la revue apprend :
- faux positifs : **groupes électrogènes** (5, pris pour des `mil_*`), cuves, blocs de
  béton, bidon, mallettes, chaises, sacs, poubelle, statue ;
- **VT4** (4x4 militaire, base Ford Ranger) → `mil_other`, étiquetée car, van,
  mil_truck ou mil_apc_ifv par SAM 3 ; moteur de GBC → `mil_truck` ;
- **31 boîtes sur une partie** de véhicule (portière, coffre, moteur, roue).

### Prompts — `evalue-prompts-vlm.py`

Part réglage (85 pistes) pour lire les erreurs, part contrôle (49, rang multiple de 3)
lue à la fin. Totaux sur les 134 :

| | p3 | p4 | **p5** | p6 | p7 | SAM 3 |
|---|---|---|---|---|---|---|
| vrais véhicules gardés (71) | 65 | 57 | **67** | 64 | 64 | — |
| faux positifs véhicule rejetés (29) | 19 | 23 | **21** | 22 | 22 | — |
| parties trouvées (30) | 24 | 18 | **24** | 27 | 26 | — |
| classe fine exacte (71) | 47 | 39 | **50** | 45 | 48 | 30 |
| famille civil/militaire (71) | 63 | 65 | **64** | 64 | 66 | 60 |
| vraies personnes gardées (24) | 21 | 21 | **23** | 20 | 23 | — |
| pistes impures trouvées (7) | 2 | 2 | **2** | 2 | 2 | — |
| affiliation des personnes (22) | — | 8 | **9** | 10 | 9 | — |

1. **p4** (longue liste de leurres dont « electric generator or other equipment on
   wheels ») : les morceaux de véhicule y tombent (portière, roue de secours, arrière de
   VT4 → « un générateur ») et sont rejetés ; l'exemple Berlingo pousse les voitures
   blanches en `van`. Nommer les groupes électrogènes n'en fait rejeter aucun.
2. **p5, retenu** = p3 + `box_covers` demandé **avant** `real_object` et « une boîte sur
   une partie reste yes », leurres concrets sans « équipement », VT4/GBC et « la peinture
   militaire rend militaire », affiliation des personnes. Meilleur sur la part contrôle.
3. **p6/p7 : largeur au sol** (`largeur_sol_m` : rayon du bas de la boîte, sol plat,
   hauteur relative, tangage, champ de vue). Ordres de grandeur justes (personnes
   0,3-1,4 m, voitures 2,4-4 m, camions 3-11 m, bidon 0,5 m, statue 2,9 m), mais pas de
   gain net : un faux positif de plus, trois vrais véhicules de moins — tout texte ajouté
   attire le rejet (véhicules pris pour « un arbre », « un oiseau »). Occlusions et
   personnes vues à la verticale faussent la lecture de la taille.
4. Restent faibles : pureté, affiliation des personnes (« unknown »), groupes
   électrogènes remorqués, VT4 vue comme un SUV civil sombre.

### Lot complet

p5 sur toutes les pistes (≥ 5 détections) des 9 vols de jour : **41 544 pistes**.
Vues d'abord (PC, ~26 frames/s décodées, 2-3 h), vol par vol, le VLM suit. `--serveur`
accepte plusieurs URL (un serveur vLLM par GPU, pistes réparties par `group_id`) ;
~1,1 piste/s par A100 à 32 requêtes.

```bash
PY=X-AnyLabeling-Server/.venv/bin/python
$PY verifie-pistes-vlm.py --vol 0000019 --selection toutes --prompt p5 --vues-seules
$PY verifie-pistes-vlm.py --vol 0000019 --selection toutes --prompt p5 \
    --serveur http://localhost:13863/v1 http://localhost:13864/v1 --paralleles 32
```

## Jeu de détection 3 classes — `exporte-detection.py`, 2026-09-30

But : fine-tuner un YOLO Ultralytics (déjà entraîné sur les jeux publics de
`Datasets/opensource`, cf. `ENTRAINEMENT.md`) avec une **tête à 3 classes** :
`person`, `civilian_vehicle`, `military_vehicle`. Sortie :
`Datasets/real/CAMPAGNE3/detection-p5/` (hors git).

```
frames/<vol>/<vol>_rgb_<sample:07d>.jpg   frames 4K intactes (JPEG q95)
par-vol/<vol>.json                        toutes les boîtes, niveaux et raisons
annotations/{train,val,test}.json         COCO pivot, coordonnées 4K
yolo/{images,labels}/<split>/, data.yaml  tuiles 1024, zones ignorées grisées
dataset.json  bilan.json  controle.html
```

**Trois niveaux par boîte**, verdicts du VLM par piste propagés à chaque frame :

| Cas | Niveau |
|---|---|
| p5 `real_object` no (P ≥ 0,5) | retirer |
| p5 `real_object` incertain, piste sans verdict | ignorer |
| personne | garder `person` |
| véhicule, `box_covers` part, contenu (≥ 80 % de l'aire) dans un véhicule ≥ 1,5× sur **cette** frame | retirer |
| partie non contenue ; véhicule « entier » contenu dans un autre | ignorer |
| `fine_class` civile | garder `civilian_vehicle` |
| `vehicle_unknown` | ignorer |
| `fine_class` militaire : m1 dit véhicule et pas civil | garder `military_vehicle` |
| militaire : m1 dit équipement, ou civil, ou incertain | ignorer |
| frame interpolée d'une piste gardée | ignorer |
| shape SAM 3 hors piste (score ≥ 0,3, IoU < 0,5 avec les pistes) | ignorer |

*Garder* : `iscrowd=0`. *Ignorer* : `iscrowd=1` en COCO (une annotation par catégorie
neutralisée — pycocotools n'ignore une détection que dans la même catégorie — liées par
`attributes.groupe_ignore`) et **zone grisée (114)** dans les tuiles YOLO, jamais sur une
boîte gardée. *Retirer* : absent du COCO, compté dans `bilan.json`. `same_object` n'est
pas utilisé : sur la revue, il écarterait 7 pistes pures pour 2 impures, et une piste
impure garde des boîtes justes frame par frame.

**m1, 2e passe sur les militaires** (`verifie-pistes-vlm.py --selection militaires
--prompt m1`, mêmes vues que p5) : sur la revue, p5 ne garde juste que 4 « militaires »
sur 10 — les 5 autres sont des **groupes électrogènes** remorqués. Question fermée
« véhicule / équipement remorqué / autre » + « peinture militaire ». Revue : 5/5
groupes électrogènes reconnus, mais le robot UGV et un VT4 vu de dessus pris pour des
équipements, d'où *ignorer* et non *retirer*. Sur les 2 241 candidats (2 A100,
15 min) : 180 militaires confirmés (8/12 justes à l'œil ; erreurs : cabines ou
portières de camion, une voiture beige), 1 686 « équipement » (à l'œil, environ la
moitié sont de vrais 4x4 militaires sous les arbres : m1 surappelle l'équipement),
207 « civil ».

**Résultat** (9 vols de jour, décodage ~5 frames/s écrites, 36 Go de frames) :

| Vol | Split | Frames | person | civilian | military |
|---|---|---|---|---|---|
| 0000001 | train | 918 | 3 441 | 7 561 | 114 |
| 0000002 | train | 1 294 | 10 774 | 5 524 | 179 |
| 0000004 | train | 1 049 | 1 252 | 15 | 60 |
| 0000011 | train | 1 115 | 2 735 | 6 026 | 240 |
| 0000012 | train | 2 435 | 6 118 | 34 152 | 232 |
| 0000019 | train | 2 600 | 4 948 | 77 751 | 1 143 |
| 0000231 | train | 1 962 | 3 280 | 8 239 | 593 |
| 0000005 | val | 1 327 | 2 327 | 787 | 247 |
| 0000018 | test | 1 095 | 840 | 3 975 | 118 |

Boîtes : 182 671 gardées ; ignorées : 26 227 interpolées, 17 054 « équipement » m1,
11 814 hors piste, 10 835 parties, 7 165 entières contenues, 3 547 militaires
incertaines, 3 519 affiliation inconnue, 2 263 objets incertains ; retirées : 62 411
rejets du VLM, 7 714 parties contenues. Côté médian des boîtes gardées : 20 à 65 px pour
les personnes, 19 à 145 px pour les civils selon le vol (0000012 et 0000019, filmés
haut, sont les plus petits).

**Tuiles YOLO** (1024 px, recouvrement 20 %, échelle ×1, 22 Go ; une boîte coupée est
gardée si ≥ 40 % de son aire reste dans la tuile, sinon grisée comme un moignon ; tuile
sans objet gardée avec une probabilité de 0,1). Validées par les chargeurs
d'Ultralytics 8.4 (0 corrompue) :

| Split | Tuiles | dont fond | person | civilian | military |
|---|---|---|---|---|---|
| train | 62 106 | 12 234 | 66 904 | 281 274 | 5 440 |
| val | 5 023 | 1 627 | 5 033 | 1 681 | 572 |
| test | 3 173 | 1 434 | 1 505 | 5 913 | 194 |

**Véhicules gros et proches coupés : les coutures de SAM 3 tuilé.** Retour
utilisateur sur `controle.html` (0000001, samples 1135 et 3035) : camionnettes et
véhicule militaire ignorés. Cause : SAM 3 tuilé (tuiles 1008, recouvrement 0,2,
coutures x = 807/1008, 1614/1815, 2421/2622, 2832/3429, y = 807/1008, 1152/1815) sort un
véhicule plus gros qu'une tuile en morceaux coupés net aux coutures ; la passe pleine
image le voit entier, mais la fusion SAHI (NMM en IOS, meilleur score) garde souvent le
morceau. Sur 0000001 : 1 715 des 2 135 parties contenues et 13 % des boîtes gardées
étaient coupées à une couture. **Recollage à l'export** (`recoller()`) : deux boîtes
véhicule, l'une finissant sur la fin de la tuile i, l'autre commençant au début de la
tuile i+1, même étendue sur l'autre axe (IoU ≥ 0,5), sont unies (union-find : coins,
plusieurs coutures) ; famille votée à l'aire sur les verdicts des morceaux (p5 donne la
classe du véhicule entier même sur une partie). 487 véhicules recollés gardés sur les 9
vols. Limite : rien à recoller quand la fusion de SAM 3 n'a gardé qu'un morceau.

**Revue humaine des militaires** — `revue-militaires.py` (http://127.0.0.1:8766) :
Masstech pris pour un équipement par m1, VT4 vus civils, et « beaucoup de pistes
ignorées sont en fait militaires ». Une piste à la fois (vues du VLM et vue large),
clavier `m` militaire, `c` civil, `n` pas un véhicule, `i` incertain. 3 341 pistes des
frames exportées, par priorité : militaires ignorés par m1 (1 908), affiliation ou objet
incertain (739), militaires gardés par m1 (188), parties militaires (506). Réponses dans
`Datasets/real/CAMPAGNE3/revue-militaires/annotations.jsonl` (la dernière fait foi) ;
`exporte-detection.py` les lit et elles remplacent p5/m1 pour la famille (`non` →
retirée). Au 2026-09-30 au soir : 564 relues (161 militaires, 204 civils, 188 pas un
véhicule, 11 incertaines).

**Limites connues** : les VT4 (voitures militaires sombres) sont presque toujours
`car` pour le VLM, donc gardés en `civilian_vehicle` ; la classe militaire est
petite ; les véhicules tronqués par le bord de l'image passent souvent pour des parties
(ignorés). Aucune vérité terrain humaine : le test (0000018, vol test du jeu AnafiUKR)
est en pseudo-labels, à relire.

```bash
PY=X-AnyLabeling-Server/.venv/bin/python
$PY exporte-detection.py seuils              # règles contre la revue humaine (134 pistes)
$PY exporte-detection.py vol 0000001 ...     # frames + par-vol/<vol>.json (reprise)
$PY exporte-detection.py controle 0000001    # controle.html
$PY exporte-detection.py assemble            # COCO, dataset.json, bilan.json
$PY exporte-detection.py yolo                # tuiles 1024, recouvrement 20 %
```

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
