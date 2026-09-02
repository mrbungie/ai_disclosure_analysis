#!/usr/bin/env bash
# Mirror data/ to/from the Backblaze B2 bucket configured in .env.
#
# Behavior:
#   - Bidirectional sync (rclone bisync): new local files go up, new remote
#     files come down.
#   - On a real conflict (same relpath changed on both sides), the REMOTE
#     copy wins (--conflict-resolve=path2) — "arriba manda".
#   - Deletions are never applied silently: a dry run is inspected first,
#     and if it would delete or overwrite-via-conflict anything, the diff is
#     shown and you're asked to confirm before the real sync runs.
#   - First run against a given local/remote pair needs --resync (rclone
#     bisync requirement) to build its baseline; the script won't do this
#     on its own.
#
# Usage:
#   scripts/common/sync_data_b2.sh              # normal sync
#   scripts/common/sync_data_b2.sh --resync      # first run / baseline reset
#   scripts/common/sync_data_b2.sh --dry-run     # show what would happen, do nothing

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_DIR="$REPO_ROOT/data"
ENV_FILE="$REPO_ROOT/.env"

RESYNC=0
DRY_RUN_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --resync) RESYNC=1 ;;
    --dry-run) DRY_RUN_ONLY=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
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

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "No existe $LOCAL_DIR — nada que sincronizar." >&2
  exit 1
fi

REMOTE_ENV_NAME="$(echo "$RCLONE_REMOTE_NAME" | tr '[:lower:]-' '[:upper:]_')"
export "RCLONE_CONFIG_${REMOTE_ENV_NAME}_TYPE"=b2
export "RCLONE_CONFIG_${REMOTE_ENV_NAME}_ACCOUNT"="$B2_KEY_ID"
export "RCLONE_CONFIG_${REMOTE_ENV_NAME}_KEY"="$B2_APPLICATION_KEY"

REMOTE_PATH="${RCLONE_REMOTE_NAME}:${B2_BUCKET}"
if [[ -n "$B2_PREFIX" ]]; then
  REMOTE_PATH="${REMOTE_PATH}/${B2_PREFIX}"
fi

STATE_DIR="$REPO_ROOT/.rclone-bisync-state"
mkdir -p "$STATE_DIR"

BISYNC_ARGS=(
  bisync "$LOCAL_DIR" "$REMOTE_PATH"
  --workdir "$STATE_DIR"
  --conflict-resolve path2
  --conflict-loser pathname
  --recover
)
if [[ $RESYNC -eq 1 ]]; then
  BISYNC_ARGS+=(--resync)
fi

echo "Local:  $LOCAL_DIR"
echo "Remoto: $REMOTE_PATH"
echo

echo "Revisando cambios (dry-run)..."
DRY_LOG="$(mktemp)"
trap 'rm -f "$DRY_LOG"' EXIT
if ! rclone "${BISYNC_ARGS[@]}" --dry-run -v > "$DRY_LOG" 2>&1; then
  echo "El dry-run de rclone bisync falló:" >&2
  cat "$DRY_LOG" >&2
  exit 1
fi

if [[ $DRY_RUN_ONLY -eq 1 ]]; then
  cat "$DRY_LOG"
  echo
  echo "(--dry-run: no se aplicó nada)"
  exit 0
fi

# Anything that looks like a deletion or a conflict getting resolved is
# treated as "raro": show it and ask before touching real data.
RISKY_LINES="$(grep -Ei 'delet|conflict|error|won|renam' "$DRY_LOG" || true)"

if [[ -n "$RISKY_LINES" ]]; then
  echo "El dry-run detectó cambios que ameritan revisión (borrados/conflictos):"
  echo
  echo "$RISKY_LINES"
  echo
  read -r -p "¿Continuar con el sync real? [y/N] " CONFIRM
  case "$CONFIRM" in
    y|Y|yes|si|sí) ;;
    *) echo "Cancelado, no se aplicó nada."; exit 0 ;;
  esac
else
  echo "Sin borrados ni conflictos detectados, aplicando sync..."
fi

rclone "${BISYNC_ARGS[@]}" -v
echo "Sync completo."
