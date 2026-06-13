#!/usr/bin/env bash
# Deploy trained Venus model artifacts from local machine to VPS.
#
# Usage:
#   ./scripts/deploy_models.sh ubuntu@1.2.3.4
#   ./scripts/deploy_models.sh ubuntu@1.2.3.4 EURUSD          # single symbol
#   ./scripts/deploy_models.sh ubuntu@1.2.3.4 "" ~/.ssh/id_ed25519  # custom key
#
# After upload the bot process on the VPS must be restarted so Venus
# reloads the new model.pt from disk (it loads once at startup via inference.py).

set -euo pipefail

VPS="${1:?Usage: $0 USER@HOST [SYMBOL] [SSH_KEY]}"
SYMBOL="${2:-}"           # if blank, deploy all symbols found under models/artifacts/
SSH_KEY="${3:-}"

LOCAL_BASE="models/artifacts"
REMOTE_BASE="~/aethel/models/artifacts"

if [[ ! -d "$LOCAL_BASE" ]]; then
  echo "ERROR: $LOCAL_BASE not found — run retrain first" >&2
  exit 1
fi

SSH_OPT=()
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPT=(-e "ssh -i $SSH_KEY")
fi

if [[ -n "$SYMBOL" ]]; then
  DIRS=("$LOCAL_BASE/$SYMBOL")
else
  # all subdirectories that contain a model.pt
  mapfile -t DIRS < <(find "$LOCAL_BASE" -name "model.pt" -printf "%h\n" | sort -u)
fi

if [[ ${#DIRS[@]} -eq 0 ]]; then
  echo "No trained models found under $LOCAL_BASE"
  exit 1
fi

for dir in "${DIRS[@]}"; do
  sym=$(basename "$dir")
  remote="${VPS}:${REMOTE_BASE}/${sym}/"
  echo "→ deploying $sym  ($dir/ → $remote)"
  rsync -avz --mkpath "${SSH_OPT[@]}" "${dir}/" "$remote"
done

echo ""
echo "✓ done — restart the bot on $VPS to reload models:"
echo "    ssh $VPS 'cd aethel && docker compose restart bot'"
echo "  or, if running systemd:"
echo "    ssh $VPS 'sudo systemctl restart aethel'"
