#!/usr/bin/env bash
# Mirror data/ to or from the Backblaze B2 bucket configured in .env.
#
# Two explicit one-way operations, never both at once:
#   push  (DEFAULT)  local data/  ->  B2      "subir lo que produje"
#   pull             B2           ->  local data/
#
# This replaced an `rclone bisync` setup. Bisync needs a baseline built by
# --resync, and it aborts the WHOLE run on any fatal error demanding a fresh
# --resync to recover — which is exactly what happened when B2 returned
# 403 download_cap_exceeded mid-pull: 4298 of 13133 files had landed and the
# baseline was gone. One-way copy just resumes where it left off.
#
# Both directions are ADDITIVE by default (rclone copy): nothing on the
# destination is ever deleted or overwritten-to-nothing, so a push cannot
# destroy remote data and a pull cannot destroy local data. Pass --delete to
# mirror instead (rclone sync); that path shows a dry run first and asks
# before touching anything, and refuses outright when not on a terminal.
#
# In-flight files are skipped: *.partial is how scripts/common/ai_prefilter.py
# stages a parquet part before renaming it into place, so uploading one would
# publish a truncated part.
#
# Usage:
#   scripts/common/sync_data_b2.sh                 # push (default)
#   scripts/common/sync_data_b2.sh push --dry-run
#   scripts/common/sync_data_b2.sh pull
#   scripts/common/sync_data_b2.sh pull --delete   # mirror, asks first

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_DIR="$REPO_ROOT/data"
ENV_FILE="$REPO_ROOT/.env"

DIRECTION="push"
DRY_RUN=0
DELETE=0
for arg in "$@"; do
  case "$arg" in
    push|pull) DIRECTION="$arg" ;;
    --dry-run) DRY_RUN=1 ;;
    --delete)  DELETE=1 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $arg (expected push|pull [--dry-run] [--delete])" >&2; exit 1 ;;
  esac
done

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone no está instalado. Instalalo con: brew install rclone" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo ".env no encontrado en $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${B2_KEY_ID:?Falta B2_KEY_ID en .env}"
: "${B2_APPLICATION_KEY:?Falta B2_APPLICATION_KEY en .env}"
: "${B2_BUCKET:?Falta B2_BUCKET en .env (nombre del bucket de Backblaze)}"
: "${RCLONE_REMOTE_NAME:?Falta RCLONE_REMOTE_NAME en .env}"
B2_PREFIX="${B2_PREFIX:-}"

mkdir -p "$LOCAL_DIR"

REMOTE_ENV_NAME="$(echo "$RCLONE_REMOTE_NAME" | tr '[:lower:]-' '[:upper:]_')"
export "RCLONE_CONFIG_${REMOTE_ENV_NAME}_TYPE"=b2
export "RCLONE_CONFIG_${REMOTE_ENV_NAME}_ACCOUNT"="$B2_KEY_ID"
export "RCLONE_CONFIG_${REMOTE_ENV_NAME}_KEY"="$B2_APPLICATION_KEY"

REMOTE_PATH="${RCLONE_REMOTE_NAME}:${B2_BUCKET}"
if [[ -n "$B2_PREFIX" ]]; then
  REMOTE_PATH="${REMOTE_PATH}/${B2_PREFIX}"
fi

if [[ "$DIRECTION" == "push" ]]; then
  SOURCE="$LOCAL_DIR"; DESTINATION="$REMOTE_PATH"
else
  SOURCE="$REMOTE_PATH"; DESTINATION="$LOCAL_DIR"
fi

OPERATION="copy"
[[ $DELETE -eq 1 ]] && OPERATION="sync"

RCLONE_ARGS=(
  "$OPERATION" "$SOURCE" "$DESTINATION"
  --exclude "*.partial"
  --exclude ".DS_Store"
  --transfers 8
  --checkers 16
)

echo "Operación: $DIRECTION ($OPERATION)"
echo "  desde: $SOURCE"
echo "  hacia: $DESTINATION"
echo

if [[ $DRY_RUN -eq 1 ]]; then
  rclone "${RCLONE_ARGS[@]}" --dry-run -v
  echo
  echo "(--dry-run: no se aplicó nada)"
  exit 0
fi

# Mirroring can delete. Show exactly what would go and get a yes first — and
# never guess an answer when nothing is there to answer (cron, CI).
if [[ $DELETE -eq 1 ]]; then
  echo "Revisando qué borraría (dry-run)..."
  DRY_LOG="$(mktemp)"
  trap 'rm -f "$DRY_LOG"' EXIT
  if ! rclone "${RCLONE_ARGS[@]}" --dry-run -v > "$DRY_LOG" 2>&1; then
    echo "El dry-run falló:" >&2
    cat "$DRY_LOG" >&2
    exit 1
  fi
  DELETIONS="$(grep -Ei 'delet' "$DRY_LOG" || true)"
  if [[ -n "$DELETIONS" ]]; then
    echo "$DELETIONS"
    echo
    if [[ ! -t 0 ]]; then
      echo "Hay borrados pendientes y no hay terminal para confirmar. Cancelado." >&2
      exit 1
    fi
    read -r -p "¿Aplicar estos borrados en $DESTINATION? [y/N] " CONFIRM
    case "$CONFIRM" in
      y|Y|yes|si|sí) ;;
      *) echo "Cancelado, no se aplicó nada."; exit 0 ;;
    esac
  else
    echo "Sin borrados pendientes."
  fi
fi

rclone "${RCLONE_ARGS[@]}" -v --stats 30s --stats-one-line
echo "$DIRECTION completo."
