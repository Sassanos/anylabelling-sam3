#!/bin/bash
#SBATCH --job-name=vllm-qwen38
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/media/users/%u/vlm-pistes/slurm_logs/%x-%j.out
#SBATCH --error=/media/users/%u/vlm-pistes/slurm_logs/%x-%j.err
#
# Serveur vLLM de Qwen3.8-27B-FP8 (API OpenAI) pour verifie-pistes-vlm.py.
# Un GPU suffit : ~31 Go de poids FP8 (Marlin sur Ampere). Autre GPU :
#   sbatch --gres=gpu:a6k:1 slurm/vllm-qwen38.sbatch
# Le serveur tourne jusqu'à scancel ou la limite de temps. L'adresse est
# écrite dans le journal (« Adresse : http://<nœud>:<port>/v1 »), le client
# y passe par un tunnel SSH depuis le PC.
#
# Environnement : archive de slurm/setup-vllm-env.sbatch, détarrée dans
# /dev/shm (même mécanique que annots-sam3.sbatch), partagée entre les jobs
# d'un même nœud. Caches de compilation (vLLM, Triton, Inductor) dans
# /dev/shm, propres au job, effacés à la fin : rien sur ceph.

set -eo pipefail

PROJECT_DIR="/media/users/$USER/vlm-pistes"
MODEL_PATH="/media/shared/models/qwen/Qwen3.8-27B-FP8"
SERVED_NAME="qwen3.8-27b"
ENV_TAR="$PROJECT_DIR/env/vllm-qwen38-v1.tar.zst"
SHM_ENV="/dev/shm/vllm-qwen38-v1"            # nom versionné avec l'archive
VENV="$SHM_ENV/.venv"
PORT=$((13000 + SLURM_JOB_ID % 1000))        # 12000+ : ports de annots-sam3
MAX_MODEL_LEN=16384                          # ~2k d'entrée + réflexion
CACHE="/dev/shm/vllm-qwen38-cache-$SLURM_JOB_ID"
export HF_HOME="/media/users/$USER/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$CACHE/vllm" TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" XDG_CACHE_HOME="$CACHE/xdg"
export VLLM_CONFIG_ROOT="$CACHE/vllm-config" FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer"
export CUDA_CACHE_PATH="$CACHE/nv"
# Pas de statistiques d'usage envoyées ni écrites dans ~/.config/vllm.
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PYTHONDONTWRITEBYTECODE=1
# Contournement : nvcc 12.0 sur les nœuds, paquets en CUDA 13 (SLURM_INSTRUCTIONS §5).
export VLLM_USE_FLASHINFER_SAMPLER=0

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Erreur : modèle introuvable : $MODEL_PATH" >&2
    exit 1
fi

# --- Environnement : détarrer dans /dev/shm si absent (mv atomique contre
# les courses entre jobs d'un même nœud).
if [[ ! -x "$VENV/bin/vllm" ]]; then
    echo "Détarrage de l'env dans $SHM_ENV (une fois par nœud)..."
    mkdir -p "/dev/shm/.vllm-qwen38-env.tmp.$$"
    tar -C "/dev/shm/.vllm-qwen38-env.tmp.$$" --zstd -xf "$ENV_TAR"
    if mv "/dev/shm/.vllm-qwen38-env.tmp.$$" "$SHM_ENV" 2>/dev/null; then
        echo "Env détarré."
    else
        rm -rf "/dev/shm/.vllm-qwen38-env.tmp.$$"   # un autre job l'a posé
    fi
fi
[[ -x "$VENV/bin/vllm" ]] || { echo "Env indisponible après détarrage" >&2; exit 3; }
mkdir -p "$CACHE"
trap 'rm -rf "$CACHE"' EXIT

source "$VENV/bin/activate"
set -u

echo "Job : $SLURM_JOB_ID"
echo "Nœud : $SLURM_NODELIST"
echo "GPU visibles : ${CUDA_VISIBLE_DEVICES:-non défini}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "Port : $PORT"
echo "Adresse : http://$(hostname):$PORT/v1"

vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 12, "video": 0}' \
    --reasoning-parser qwen3 \
    --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'

echo "Serveur vLLM arrêté."
