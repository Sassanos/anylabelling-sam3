#!/usr/bin/env bash
# Lance l'IHM X-AnyLabeling. Les arguments sont transmis tels quels :
# ./run-client.sh /chemin/vers/mes/images
#
# NE PAS utiliser "uv run" nu ici (voir run-server.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/X-AnyLabeling"
EXE="$REPO/.venv/bin/xanylabeling"
SERVER_URL="${XANYLABELING_SERVER_URL:-http://127.0.0.1:8000}"

if [[ ! -x "$EXE" ]]; then
    cat >&2 <<EOF
Environnement introuvable : $REPO/.venv

Pour le recréer :
  cd "$REPO"
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -e ".[gpu,dev]"
EOF
    exit 1
fi

# Le client marche sans serveur (annotation manuelle), on avertit sans bloquer.
if curl -sf -m 2 "$SERVER_URL/health" >/dev/null 2>&1; then
    echo "Serveur d'inférence joignable sur $SERVER_URL"
else
    echo "/!\\ Serveur d'inférence injoignable sur $SERVER_URL"
    echo "    L'auto-labeling SAM3 ne sera pas disponible."
    echo "    Lance ./run-server.sh dans un autre terminal."
fi
echo

cd "$REPO"
exec "$EXE" "$@"
