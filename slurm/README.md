# Annotation SAM 3 à grande échelle sur le cluster Slurm

Flux visé (cf. `../README.md`) : frames à cadence native, SAM 3 image tuilé
frame par frame, classes du prompt mappées sur la taxonomie fine. L'association
des `group_id` puis la classe fine par piste (VLM) se font **après**, hors ligne.

## Architecture

```
vidéo MP4 (/media/shared, LECTURE SEULE, jamais écrite)
   │  décodage streaming (PyAV) — AUCUNE frame sur disque (quotas d'inodes)
   ▼
client annote-video-sam3.py ──HTTP──► serveur X-AnyLabeling-Server (1 GPU H100,
   │                                  modèle segment_anything_3_tiled seul)
   ▼
annots/<vol>/<vol>_chunk_<début>_<fin>.jsonl   (+ marqueur .done)
```

- **1 tâche de job array = 1 tronçon d'un vol** (2500 frames valides). Le job
  lance son propre serveur sur `127.0.0.1:12000+SLURM_ARRAY_TASK_ID` (jamais
  du numéro de job : toutes les tâches d'un array le partagent).
- Idempotent : le `.done` fait qu'une resoumission ne refait que ce qui manque.
- Les frames `mode=IR` du DefaultVideo (rendu d'affichage gelé pendant les
  rafales thermiques) sont exclues par l'index (`build_index.py`).

## Arborescence sur le cluster (`/media/users/$USER/annots-sam3/`)

| Dossier | Contenu |
|---|---|
| `X-AnyLabeling-Server/` | déploiement du serveur (copie du fork, sam3.pt 3,4 Go), `configs/models.yaml` réduit à `segment_anything_3_tiled` |
| `campagne3_index/` | miroir : symlinks vers les MP4 de `/media/shared/.../NP_data_CAMPAGNE3` + pipeline d'index (copie) + `annotation/<vol>/index/` |
| `scripts/` | `annote-video-sam3.py` (client), copie de la racine du repo |
| `slurm/` | `setup_env.sh`, `annots-sam3.sbatch`, `genere-taches.py`, `tasks.json`, `requirements-freeze.txt` |
| `annots/<vol>/` | sorties : 1 JSONL + 1 `.done` par tronçon |
| `slurm_logs/` | journaux Slurm (2 par tâche) |

Environnement : `/media/users/$USER/envs/annots-sam3` (créer avec
`slurm/setup_env.sh`, une fois, depuis le nœud de connexion). Dépôt PyPI public
direct (pas d'Artifactory) ; `decord` et `zai-sdk` omis (inutiles au chemin
image, cf. pièges d'installation du README principal).

## Mise en route (résumé)

```bash
# 1. index des vols (PC ou cluster, pur python, lecture du moov seul) :
cd campagne3_index/CodeAnnotationPrep && python3 build_index.py --all

# 2. tâches (une par tronçon de 2500 frames) :
python3 slurm/genere-taches.py --flights-root campagne3_index \
    --out slurm/tasks.json            # [--flights 0000011 0000012 0000018]

# 3. environnement (une fois, nœud de connexion) :
bash slurm/setup_env.sh               # --force pour reconstruire

# 4. soumission (créer slurm_logs/ d'abord) :
mkdir -p slurm_logs
sbatch --array=0-$(($(python3 -c 'import json;print(len(json.load(open("slurm/tasks.json"))))')-1)) \
    slurm/annots-sam3.sbatch
```

## Sortie JSONL

Une ligne par frame annotée (format X-AnyLabeling par shape, identique à
`preannotate_sam3.py` du pipeline AnafiUKR) :

```json
{"flight": "0000011", "sample_index": 21287, "dts_s": 1304.099,
 "stem": "0000011_rgb_1304098", "width": 3840, "height": 2160,
 "shapes": [{"label": "car", "score": 0.88, "points": [...],
             "attributes": {"source": "sam3", "prompt": "car",
                            "coarse": "vehicle", "uncertain": false}}],
 "t_inf_s": 3.2}
```

Les frames en échec après retries portent `"error": "..."` ; un relancement
avec `--retry-errors` les reprend.

## Choix mesurés

- **Tuilé 1008 partout** (décision utilisateur) : le plein cadre 4K écrase en
  1008² et ne voit rien sous ~50 px ; testé sur 0000011 frame 21287 : plein
  cadre = 0 shape, tuilé 1008 = 84 shapes/frame.
- Prompt : `person. car. van. truck. bus. motorcycle. tank. armored vehicle.
  military truck` (9 classes = 9 forwards de grounding par tuile, backbone
  partagé).
- ~19,7 s/frame du tuilé 1008 sur RTX 4000 Ada (PC) ; à mesurer sur H100.

## Limites connues

- `build_index.py` : des queues de flux sans télémétrie (ex. 0000004, 3
  dernières frames) sont tolérées en avertissement (valid=0 → non annotées)
  dans la copie du miroir, l'assert amont ferait échouer `--all`.
- Les vols 100 % IR (DefaultVideo = rendu gelé) produisent zéro tâche RGB —
  ils attendent le flux VisibleVideo/thermique (périmètre suivant).
