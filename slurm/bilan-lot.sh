#!/bin/bash
# bilan-lot.sh — où en est l'annotation SAM 3 à grande échelle ?
#
#   ./slurm/bilan-lot.sh              # état complet (Slurm + fichiers + frames)
#   ./slurm/bilan-lot.sh 12437        # idem, en filtrant sur ce job Slurm
#
# Peut être lancé depuis le PC (lecture du ceph monté) ou depuis le cluster
# (la partie Slurm devient locale, la partie ssh est ignorée si indisponible).
set -uo pipefail
ROOT="/media/users/$USER/annots-sam3"
JOB="${1:-}"

echo "=== Tâches Slurm (cbarbier) ==="
if [[ -n "$JOB" ]]; then
    SSH_MASTER="master.slurm.troie.ia"
    if ssh -o BatchMode=yes -o ConnectTimeout=5 master.slurm.troie.ia true 2>/dev/null; then
        ssh -o BatchMode=yes master.slurm.troie.ia \
            "squeue -u $USER -h -o '%.10i %.2t %.9M %N' ; \
             echo '--- historique ---'; \
             sacct -j $JOB -X -o JobID,State,Elapsed,End | tail -n +2" 2>/dev/null \
            || echo "(Slurm injoignable depuis ici)"
    else
        squeue -u $USER -h -o '%.10i %.2t %.9M %N' 2>/dev/null
        sacct -j $JOB -X -o JobID,State,Elapsed,End 2>/dev/null | tail -n +2
    fi
else
    echo "(préciser un numéro de job pour l'historique, ex. $0 12437)"
    squeue -u $USER 2>/dev/null || echo "(lancé hors cluster : partie Slurm indicative)"
fi

echo
echo "=== Tronçons d'annotation ==="
DONE=$(ls "$ROOT"/annots/*/*.done 2>/dev/null | wc -l)
ENCOURS=$(ls "$ROOT"/annots/*/*.jsonl 2>/dev/null | wc -l)
echo "terminés (.done) : $DONE   |   en cours d'écriture : $ENCOURS"

echo
echo "=== Frames annotées par tronçon ==="
TOT=0
for f in "$ROOT"/annots/*/*.jsonl "$ROOT"/annots/*/*.jsonl.zst; do
    [[ -f "$f" ]] || continue
    case "$f" in
        *.zst) n=$(zstd -dc "$f" 2>/dev/null | wc -l) ;;
        *)     n=$(wc -l < "$f") ;;
    esac
    TOT=$((TOT + n))
    printf '  %-24s %6d lignes\n' "$(basename "$(dirname "$f")")/$(basename "$f")" "$n"
done
echo "  TOTAL : $TOT frames annotées (lot AnafiUKR : 78 742 attendues)"

echo
echo "=== Vols CAMPAGNE3 indexés ==="
ls "$ROOT"/campagne3_index/annotation/*/index/modes.json 2>/dev/null \
    | sed "s|.*/annotation/||; s|/index/modes.json||" | tr '\n' ' '
echo
