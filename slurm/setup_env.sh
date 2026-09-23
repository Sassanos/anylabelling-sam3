#!/bin/bash
# setup_env.sh : crée l'environnement Python du projet d'annotation SAM 3.
#
# À lancer UNE FOIS, À LA MAIN, depuis le nœud de connexion du cluster, puis
# à relancer uniquement si la liste des paquets change. Les jobs Slurm ne font
# qu'activer l'environnement existant — jamais le créer ni le modifier.
#
# Dépôt : PyPI public (le proxy du site sort directement), PAS Artifactory.
# decord est volontairement omis : pas de wheel py3.12 et seul le chemin vidéo
# (inutilisé ici) en a besoin.
#
# Après une installation réussie, la version exacte des paquets est figée dans
# requirements-freeze.txt pour pouvoir reconstruire à l'identique.
set -eo pipefail

PROJECT_DIR="/media/users/$USER/annots-sam3"
ENV_DIR="/media/users/$USER/envs/annots-sam3"
SERVER_DIR="$PROJECT_DIR/X-AnyLabeling-Server"
export PIP_CACHE_DIR="/media/users/$USER/.cache/pip"
export PYTHONDONTWRITEBYTECODE=1

if [[ ! -f "$SERVER_DIR/pyproject.toml" ]]; then
    echo "Erreur : $SERVER_DIR/pyproject.toml introuvable." >&2
    exit 1
fi
if [[ -d "$ENV_DIR" && "$1" != "--force" ]]; then
    echo "Erreur : $ENV_DIR existe déjà. Pour le reconstruire (interdit pendant" >&2
    echo "que des jobs l'utilisent) : $0 --force" >&2
    exit 1
fi
mkdir -p "$PIP_CACHE_DIR"

virtualenv -p python3 "$ENV_DIR"
source "$ENV_DIR/bin/activate"

# Le virtualenv Debian (seeder FromAppData) ne copie PAS pip dans
# $ENV_DIR/bin : sans ça, "pip" résout vers /usr/bin/pip et refuse
# (PEP 668, externally-managed). On sème pip via get-pip.py, standard.
# Toute la suite passe par "python -m pip" pour ne plus dépendre du PATH.
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$PIP_CACHE_DIR/get-pip.py"
python "$PIP_CACHE_DIR/get-pip.py"

# Le paquet serveur lui-même, sans ses deps (l'extra sam3 tire decord).
python -m pip install --no-deps -e "$SERVER_DIR"

# Dépendances core du serveur (pyproject.toml, hors extras). zai-sdk (API
# cloud Z.ai pour le seul modèle glm_4_6v_grounding_api, import paresseux)
# est omis : inutile pour SAM 3 et jamais importé avec ce models.yaml.
python -m pip install "fastapi[standard]>=0.115.0" "pydantic>=2.12.0" "openai>=1.99.1" \
    "requests>=2.26.0" "packaging>=23.2" "opencv-python-headless>=4.11.0" \
    loguru numpy pillow

# Extra sam3 du pyproject SANS decord, + PyAV pour le décodage streaming côté
# client. torch vient de PyPI (wheel CUDA 12.x) — compatible H100 (sm_90).
python -m pip install einops "ftfy==6.1.1" huggingface_hub "hydra-core>=1.3.2" \
    "iopath>=0.1.10" "numpy>=1.26" pandas pycocotools scikit-image scikit-learn \
    "timm>=1.0.17" typing_extensions "torch>=2.7.0" av

# Figer les versions exactes pour reconstruire à l'identique si besoin.
python -m pip freeze --exclude-editable > "$PROJECT_DIR/slurm/requirements-freeze.txt"

# Test minimal : le serveur doit savoir s'importer et torch voir CUDA (sur
# le nœud de connexion sans GPU, on vérifie juste l'import).
python -c "import torch, fastapi, av, cv2, einops, ftfy; print('imports OK, torch', torch.__version__)"

echo "Environnement prêt : $ENV_DIR"
