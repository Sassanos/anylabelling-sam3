#!/bin/bash
#SBATCH --job-name=setup-vllm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/media/users/%u/vlm-pistes/slurm_logs/%x-%j.out
#SBATCH --error=/media/users/%u/vlm-pistes/slurm_logs/%x-%j.err
#
# Construit UNE FOIS l'environnement vLLM du serveur Qwen3.8 et l'archive en
# un seul fichier sur ceph. À relancer seulement pour changer de version.
#
# Pourquoi un job et pas le nœud de connexion : un venv vLLM fait ~12 Go et
# le /tmp du nœud de connexion 6 Go ; et pas de venv sur ceph (quota
# d'inodes ~25-30k, un venv torch en fait 40-60k). Le venv est donc construit
# dans /dev/shm du nœud, AU CHEMIN où les jobs serveur le détarrent
# (/dev/shm/vllm-qwen38-v1/.venv : les shebangs y pointent), archivé, puis
# effacé du nœud. Rien n'est écrit sur ceph hors de l'archive et du freeze.

set -eo pipefail

PROJECT_DIR="/media/users/$USER/vlm-pistes"
VERSION="v1"
VLLM_VERSION="0.30.0"
SHM_ENV="/dev/shm/vllm-qwen38-$VERSION"
ENV_TAR="$PROJECT_DIR/env/vllm-qwen38-$VERSION.tar.zst"
FREEZE="$PROJECT_DIR/slurm/requirements-vllm-$VERSION-freeze.txt"
# Proxy du site (même valeur que sur le nœud de connexion, pas de secret).
export https_proxy="http://proxy-cache.utils.dmz.troie.ia:3128"
export http_proxy="$https_proxy" HTTPS_PROXY="$https_proxy" HTTP_PROXY="$https_proxy"
export no_proxy=".troie.ia,127.0.0.1,localhost,10.0.0.0/8,172.16.0.0/12"
export NO_PROXY="$no_proxy"
# Caches hors ceph (inodes), effacés avec le reste à la fin.
export UV_CACHE_DIR="$SHM_ENV.cache/uv" PIP_CACHE_DIR="$SHM_ENV.cache/pip"
export VIRTUALENV_OVERRIDE_APP_DATA="$SHM_ENV.cache/virtualenv"   # sinon ~/.local/share
export XDG_CACHE_HOME="$SHM_ENV.cache/xdg"
export PYTHONDONTWRITEBYTECODE=1

echo "Job $SLURM_JOB_ID sur $SLURM_NODELIST, $(nproc) cœurs"
df -h /dev/shm | tail -1

if [[ -f "$ENV_TAR" ]]; then
    echo "Erreur : $ENV_TAR existe déjà (changer VERSION pour en faire un autre)." >&2
    exit 1
fi
if [[ -e "$SHM_ENV" ]]; then
    echo "Erreur : $SHM_ENV existe déjà sur ce nœud." >&2
    exit 1
fi
trap 'rm -rf "$SHM_ENV" "$SHM_ENV.cache"' EXIT

curl -sSf -m 30 -o /dev/null https://pypi.org/simple/pip/ \
    || { echo "Erreur : PyPI injoignable depuis $(hostname)." >&2; exit 2; }

# Même amorçage que slurm/setup_env.sh : le virtualenv Debian ne sème pas pip.
mkdir -p "$SHM_ENV.cache"
virtualenv -p /usr/bin/python3 "$SHM_ENV/.venv"
PY="$SHM_ENV/.venv/bin/python"
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$SHM_ENV.cache/get-pip.py"
"$PY" "$SHM_ENV.cache/get-pip.py" --quiet
"$PY" -m pip install --quiet uv
# vLLM tire torch, transformers et flashinfer aux versions qu'il épingle.
"$PY" -m uv pip install --python "$PY" "vllm==$VLLM_VERSION"

"$PY" -m uv pip freeze --python "$PY" > "$FREEZE.tmp" && mv "$FREEZE.tmp" "$FREEZE"
"$PY" - <<'PY'
import torch, transformers, vllm
from vllm.model_executor.models.registry import ModelRegistry
print("vllm", vllm.__version__, "| torch", torch.__version__, "cuda", torch.version.cuda,
      "| transformers", transformers.__version__)
arch = "Qwen3_5ForConditionalGeneration"
print(arch, "supporté :", arch in ModelRegistry.get_supported_archs())
PY
du -sh "$SHM_ENV/.venv"

# Archive : .venv à la racine, détarrée dans $SHM_ENV par le job serveur.
tar -C "$SHM_ENV" -cf - .venv | zstd -T"$(nproc)" -6 -q > "$ENV_TAR.tmp"
mv "$ENV_TAR.tmp" "$ENV_TAR"
ls -lh "$ENV_TAR"
echo "Environnement archivé : $ENV_TAR"
