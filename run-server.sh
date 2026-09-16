#!/usr/bin/env bash
# Lance le serveur d'inférence X-AnyLabeling (SAM3 vidéo, YOLO11, YOLOE).
# Les arguments sont transmis tels quels : ./run-server.sh --port 8001
#
# NE PAS utiliser "uv run" nu ici : sans uv.lock il resynchroniserait l'env
# sur les seules deps déclarées et désinstallerait torch et les extras sam3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/X-AnyLabeling-Server"
EXE="$REPO/.venv/bin/x-anylabeling-server"

if [[ ! -x "$EXE" ]]; then
    cat >&2 <<EOF
Environnement introuvable : $REPO/.venv

Pour le recréer :
  cd "$REPO"
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python \\
      --index-url https://download.pytorch.org/whl/cu128 torch torchvision
  uv pip install --python .venv/bin/python -e ".[sam3,ultralytics]"
  uv pip install --python .venv/bin/python regex   # oubli de l'upstream
EOF
    exit 1
fi

# Les poids (sam3.pt, bpe_simple_vocab_16e6.txt.gz, yolo*.pt) et configs/
# sont référencés en relatif dans models.yaml : il faut partir du repo.
cd "$REPO"

echo "Modèles activés dans configs/models.yaml :"
sed -n '/^enabled_models:/,/^[^ #-]/p' configs/models.yaml \
    | grep -E '^\s+- ' | sed 's/^/  /'
echo
echo "Chargement en cours (~90 s, SAM3 pèse 3,4 Go)."
echo "Attendre la ligne « Successfully loaded N/N model(s) »."
echo

exec "$EXE" "$@"
