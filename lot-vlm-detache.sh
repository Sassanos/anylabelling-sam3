#!/bin/bash
# Lot VLM détaché de la session : vues puis vérification au VLM, vol par vol,
# sur les serveurs vLLM de jobs Slurm déjà lancés (slurm/vllm-qwen38.sbatch),
# puis annulation de ces jobs pour ne pas laisser les GPU tourner à vide.
#
# - un tunnel SSH par job (saut par le nœud de connexion), rouvert s'il tombe,
#   tant que le job tourne ;
# - vues (verifie-pistes-vlm.py --vues-seules) en parallèle du VLM, qui prend
#   chaque vol dès que ses vues sont faites ;
# - requêtes réparties sur les serveurs qui répondent au moment du vol ;
# - deux passes : la seconde reprend les pistes en erreur (reprise piste par
#   piste, rien n'est refait) ;
# - à la fin (ou si plus aucun serveur ne répond) : scancel des jobs.
#
# Lancer détaché (survit à la fermeture du terminal ou de la session) :
#   setsid nohup ./lot-vlm-detache.sh 12863:13863 12864:13864 > /dev/null 2>&1 < /dev/null &
# Suivre : tail -f $JOURNAUX/lot.log ; arrêter : kill -- -<pgid> (voir lot.pid)
set -u

ICI="$(cd "$(dirname "$0")" && pwd)"
PY="$ICI/X-AnyLabeling-Server/.venv/bin/python"
LOGIN="master.slurm.troie.ia"
PROMPT="${PROMPT:-p5}"
MIN_LEN="${MIN_LEN:-5}"
PARALLELES="${PARALLELES:-32}"
# petits vols d'abord : le VLM démarre pendant que les gros préparent leurs vues
VOLS="${VOLS:-0000004 0000005 0000018 0000231 0000001 0000011 0000002 0000012 0000019}"
JOURNAUX="${JOURNAUX:-/home/cbarbier/Documents/Geolocalisation/Datasets/real/CAMPAGNE3/lot-vlm-$PROMPT}"
[[ $# -ge 1 ]] || { echo "usage : $0 <job>:<port> [<job>:<port> ...]" >&2; exit 2; }

mkdir -p "$JOURNAUX"
LOG="$JOURNAUX/lot.log"
FAITES="$JOURNAUX/vues-faites.txt"
: > "$FAITES"
echo "$$ $(ps -o pgid= $$ | tr -d ' ')" > "$JOURNAUX/lot.pid"
journal() { echo "$(date '+%F %T') $*" >> "$LOG"; }

job_actif() { ssh -o BatchMode=yes "$LOGIN" "squeue -h -j $1 -o %T" 2>/dev/null | grep -q RUNNING; }

tunnel() {  # job port : tunnel vers le nœud du job, rouvert tant que le job tourne
    local job=$1 port=$2 noeud
    while job_actif "$job"; do
        noeud=$(ssh -o BatchMode=yes "$LOGIN" "squeue -h -j $job -o %N")
        ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
            -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -J "$LOGIN" -L "$port:127.0.0.1:$port" "$noeud.slurm.vm.troie.ia" \
            2>> "$JOURNAUX/tunnels.log"
        journal "tunnel $job:$port fermé, réouverture dans 10 s si le job tourne"
        sleep 10
    done
    journal "job $job terminé : plus de tunnel"
}

serveurs_vivants() {
    local jp
    for jp in "$@"; do
        curl -s -m 10 "http://localhost:${jp#*:}/v1/models" > /dev/null \
            && echo "http://localhost:${jp#*:}/v1"
    done
}

fin() {
    journal "annulation des jobs : ${JOBS[*]}"
    ssh -o BatchMode=yes "$LOGIN" "scancel ${JOBS[*]}" >> "$LOG" 2>&1
    # boucles d'arrière-plan et leurs enfants (ssh des tunnels, python des vues)
    for p in "${ENFANTS[@]}"; do
        pkill -P "$p" 2>/dev/null
        kill "$p" 2>/dev/null
    done
    journal "fin du lot"
}

JOBS=()
ENFANTS=()
for jp in "$@"; do
    JOBS+=("${jp%%:*}")
    tunnel "${jp%%:*}" "${jp#*:}" &
    ENFANTS+=($!)
done
trap fin EXIT
trap 'journal "signal reçu : arrêt"; exit 130' INT TERM HUP
journal "lot $PROMPT, pistes >= $MIN_LEN détections, vols : $VOLS, jobs : $*"
sleep 15

# Vues, en parallèle du VLM
(
    cd "$ICI" || exit 1
    for v in $VOLS; do
        "$PY" verifie-pistes-vlm.py --vol "$v" --selection toutes --min-len "$MIN_LEN" \
            --prompt "$PROMPT" --vues-seules >> "$JOURNAUX/vues.log" 2>&1
        echo "$v $?" >> "$FAITES"
    done
) &
VUES=$!
ENFANTS+=("$VUES")

# Serveurs lancés en même temps que le script : ~12 min de chargement.
# Attendre qu'ils répondent tous (30 min au plus ; au moins un pour continuer).
attente=0
until [[ $(serveurs_vivants "$@" | wc -l) -eq $# || $attente -ge 1800 ]]; do
    sleep 30
    attente=$((attente + 30))
done
journal "$(serveurs_vivants "$@" | wc -l)/$# serveur(s) prêt(s) après $attente s"

cd "$ICI" || exit 1
for passe in 1 2; do
    for v in $VOLS; do
        until grep -q "^$v " "$FAITES"; do
            kill -0 "$VUES" 2>/dev/null || break
            sleep 30
        done
        mapfile -t serveurs < <(serveurs_vivants "$@")
        if [[ ${#serveurs[@]} -eq 0 ]]; then
            journal "aucun serveur ne répond : arrêt"
            exit 1
        fi
        journal "passe $passe, vol $v : ${#serveurs[@]} serveur(s)"
        "$PY" verifie-pistes-vlm.py --vol "$v" --selection toutes --min-len "$MIN_LEN" \
            --prompt "$PROMPT" --serveur "${serveurs[@]}" --paralleles "$PARALLELES" \
            >> "$JOURNAUX/vlm.log" 2>&1
        journal "passe $passe, vol $v : code $?"
    done
done
wait "$VUES"
journal "lot terminé"
