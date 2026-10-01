#!/bin/bash
#SBATCH --job-name=annots-sam3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --output=/media/users/%u/annots-sam3/slurm_logs/%x-%A_%a.out
#SBATCH --error=/media/users/%u/annots-sam3/slurm_logs/%x-%A_%a.err
#
# Annotation SAM 3 à la cadence native : une tâche d'array = un tronçon d'un vol.
# Soumettre avec l'étendue d'array correspondant à tasks.json :
#   sbatch --array=0-33 slurm/annots-sam3.sbatch
# Les tronçons sont idempotents (marqueur .done) : resoumettre reprend ce qui
# manque. Le port du serveur est un port libre demandé au système : un port
# fixe (12000 + tâche) tombait sur les serveurs vLLM d'autres utilisateurs du
# même nœud, dont le /health répond aussi — le client leur envoyait les frames.
#
# Environnement : pas de venv sur ceph (quota d'inodes ~25-30k, un venv torch
# en fait 40-60k) — une archive du venv (1 fichier) est détarrée dans /dev/shm
# (tmpfs RAM, 93 Go), partagée entre les jobs d'un même nœud, purgée au reboot.
# Une tâche = 1 GPU H100, 8 cœurs, 56 Go, 6 h max (à recaler après le job test).

set -eo pipefail

PROJECT_DIR="/media/users/$USER/annots-sam3"
SERVER_DIR="$PROJECT_DIR/X-AnyLabeling-Server"
SCRIPT="$PROJECT_DIR/scripts/annote-video-sam3.py"
# tasks.json par défaut ; surchargeable pour un test : sbatch --export=ANNOTS_TASKS_FILE=...
TASKS_FILE="${ANNOTS_TASKS_FILE:-$PROJECT_DIR/slurm/tasks.json}"
OUT_ROOT="${ANNOTS_OUT_ROOT:-$PROJECT_DIR/annots}"
ENV_TAR="$PROJECT_DIR/env/annots-sam3-env-v1.tar.zst"
SHM_ENV="/dev/shm/annots-sam3-env-v1"        # nom versionné avec l'archive
PY="$SHM_ENV/.venv/bin/python"
export HF_HOME="/media/users/$USER/.cache/huggingface"
export PYTHONDONTWRITEBYTECODE=1

# --- Environnement Python : détarrer dans /dev/shm si absent (le premier job
# d'un nœud le fait, les suivants réutilisent ; mv atomique contre les courses).
if [[ ! -x "$PY" ]]; then
    echo "Détarrage de l'env dans $SHM_ENV (~2 min, une fois par nœud)..."
    mkdir -p "/dev/shm/.annots-sam3-env.tmp.$$"
    tar -C "/dev/shm/.annots-sam3-env.tmp.$$" --zstd -xf "$ENV_TAR"
    if mv "/dev/shm/.annots-sam3-env.tmp.$$" "$SHM_ENV" 2>/dev/null; then
        echo "Env détarré."
    else
        rm -rf "/dev/shm/.annots-sam3-env.tmp.$$"   # un autre job l'a posé
    fi
fi
[[ -x "$PY" ]] || { echo "Env indisponible après détarrage" >&2; exit 3; }

echo "Job $SLURM_JOB_ID (tâche $SLURM_ARRAY_TASK_ID) sur $SLURM_NODELIST"
echo "GPU visibles : ${CUDA_VISIBLE_DEVICES:-non défini}"
"$PY" --version
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"

# --- État du tronçon SANS charger le serveur (économie de GPU si déjà fini)
STATE=$("$PY" "$SCRIPT" --tasks-file "$TASKS_FILE" --task-id "$SLURM_ARRAY_TASK_ID" \
    --out-dir "$OUT_ROOT" --print-only) || exit 3
read -r ST FLIGHT START END <<< "$STATE"
if [[ "$ST" != "todo" ]]; then
    echo "Tronçon [$START,$END) du vol $FLIGHT : $ST, rien à faire."
    exit 0
fi
echo "Tâche : vol $FLIGHT, tronçon [$START, $END)"

# --- Serveur d'inférence (chargement de sam3.pt, ~3,4 Go ; attendre « Successfully loaded »)
PORT=$("$PY" -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
echo "Port du serveur : $PORT"
cd "$SERVER_DIR"
"$PY" -c "from app.cli import main; main()" --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
cd - > /dev/null
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

"$PY" - "$PORT" "$SERVER_PID" <<'PY'
import os, sys, time
import requests
url = f"http://127.0.0.1:{sys.argv[1]}/health"
for _ in range(150):
    try:
        os.kill(int(sys.argv[2]), 0)
    except OSError:  # port pris entre-temps, poids illisibles...
        print("Le serveur s'est arrêté au démarrage", file=sys.stderr)
        sys.exit(3)
    try:
        if requests.get(url, timeout=5).ok:
            print("Serveur prêt.")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(4)
print("Serveur injoignable après 600 s", file=sys.stderr)
sys.exit(3)
PY

# --- Annotation du tronçon
set +e
"$PY" "$SCRIPT" \
    --tasks-file "$TASKS_FILE" --task-id "$SLURM_ARRAY_TASK_ID" \
    --out-dir "$OUT_ROOT" \
    --server "http://127.0.0.1:$PORT" \
    --model segment_anything_3_tiled \
    --tile-size 1008 --conf 0.25
RC=$?
set -e

# --- Compaction : tronçon terminé -> JSONL compressé (quota de volume de
# /media/users ; ~10x sur du JSON). Le .done suffit à la reprise/idempotence.
# ATTENTION : le marqueur du client est <chunk>.done (avec .jsonl remplacé),
# pas <chunk>.jsonl.done — confusion qui a fait échouer silencieusement toutes
# les compactations précédentes.
if [[ $RC -eq 0 ]]; then
    CHUNK=$(printf "%s/%s/%s_chunk_%07d_%07d.jsonl" "$OUT_ROOT" "$FLIGHT" "$FLIGHT" "$START" "$END")
    DONE="${CHUNK%.jsonl}.done"
    if [[ -f "$DONE" && -f "$CHUNK" ]]; then
        zstd -T4 -12 -q "$CHUNK" && rm -f "$CHUNK" && echo "JSONL compacté : ${CHUNK}.zst"
    fi
fi

if [[ $RC -ne 0 ]]; then
    echo "Client terminé avec le code $RC"
fi
exit $RC
