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
# PARTIAL SYNCS — no hace falta bajar todo el bucket:
#   --path <subruta>   limita la operación a ese subárbol. Se aplica a AMBOS
#                      lados, así que B2 sólo lista/transfiere ese prefijo
#                      (esto es lo que evita gastar el download cap bajando
#                      data/ entera). Repetible. La subruta es relativa a
#                      data/, y un "data/" al principio se ignora, así que
#                      `--path interim/manifests` y `--path data/interim/
#                      manifests` son lo mismo.
#   --include <glob>   sólo transfiere lo que matchee (repetible). Al usar
#                      --include, todo lo demás queda excluido.
#   --exclude <glob>   nunca transfiere lo que matchee (repetible).
# --path recorta ANTES de listar; --include/--exclude filtran lo listado.
# Para un archivo suelto conviene combinarlos: --path interim/manifests
# --include 'filing_manifest.parquet'.
#
# Con --delete el borrado queda acotado al mismo subárbol/filtro: lo que
# --path o --include dejan fuera no se mira y por lo tanto no se borra.
#
# In-flight files are skipped: *.partial is how scripts/enrichment/ai_prefilter.py
# stages a parquet part before renaming it into place, so uploading one would
# publish a truncated part.
#
# Usage:
#   scripts/common/sync_data_b2.sh                 # push (default)
#   scripts/common/sync_data_b2.sh push --dry-run
#   scripts/common/sync_data_b2.sh pull
#   scripts/common/sync_data_b2.sh pull --delete   # mirror, asks first
#   scripts/common/sync_data_b2.sh push --delete --yes  # mirror without asking
#   scripts/common/sync_data_b2.sh pull --path interim/manifests
#   scripts/common/sync_data_b2.sh pull --path raw/filings_html --include '*2023*'
#   scripts/common/sync_data_b2.sh push --path interim/sections --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_DIR="$REPO_ROOT/data"
ENV_FILE="$REPO_ROOT/.env"

usage() {
  awk 'NR > 1 { if (substr($0, 1, 1) != "#") exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

DIRECTION="push"
DRY_RUN=0
ASSUME_YES=0
DELETE=0
PATHS=()
INCLUDES=()
EXCLUDES=()

# Subpaths are relative to data/; tolerate the leading "data/" a user naturally
# types after tab-completing, and refuse ".." so a filter can't escape the tree.
norm_subpath() {
  local p="$1"
  p="${p#./}"
  p="${p#/}"
  p="${p#data/}"
  [[ "$p" == "data" ]] && p=""
  p="${p%/}"
  if [[ "$p" == ".." || "$p" == ../* || "$p" == */../* || "$p" == */.. ]]; then
    echo "Subruta inválida (no se permite '..'): $1" >&2
    exit 1
  fi
  printf '%s' "$p"
}

# Flags that take a value accept both `--flag value` and `--flag=value`.
while [[ $# -gt 0 ]]; do
  case "$1" in
    push|pull) DIRECTION="$1" ;;
    --dry-run) DRY_RUN=1 ;;
    --delete)  DELETE=1 ;;
    -y|--yes)  ASSUME_YES=1 ;;
    --path)    [[ $# -ge 2 ]] || { echo "--path necesita un valor" >&2; exit 1; }
               PATHS+=("$(norm_subpath "$2")"); shift ;;
    --path=*)  PATHS+=("$(norm_subpath "${1#*=}")") ;;
    --include) [[ $# -ge 2 ]] || { echo "--include necesita un valor" >&2; exit 1; }
               INCLUDES+=("$2"); shift ;;
    --include=*) INCLUDES+=("${1#*=}") ;;
    --exclude) [[ $# -ge 2 ]] || { echo "--exclude necesita un valor" >&2; exit 1; }
               EXCLUDES+=("$2"); shift ;;
    --exclude=*) EXCLUDES+=("${1#*=}") ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1 (expected push|pull [--dry-run] [--delete] [--yes] [--path P] [--include G] [--exclude G])" >&2; exit 1 ;;
  esac
  shift
done

# No --path means the whole tree: one pass rooted at data/.
[[ ${#PATHS[@]} -eq 0 ]] && PATHS=("")

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

OPERATION="copy"
[[ $DELETE -eq 1 ]] && OPERATION="sync"

# rclone applies filter rules in command-line order, and an implicit
# "exclude everything else" once any --include exists. So: built-in excludes
# first (a *.partial must never win an --include), then the user's excludes,
# then the user's includes.
FILTER_ARGS=(--exclude "*.partial" --exclude ".DS_Store")
for pat in ${EXCLUDES[@]+"${EXCLUDES[@]}"}; do
  FILTER_ARGS+=(--exclude "$pat")
done
for pat in ${INCLUDES[@]+"${INCLUDES[@]}"}; do
  FILTER_ARGS+=(--include "$pat")
done

# Sets SOURCE/DESTINATION/RCLONE_ARGS for the subpath in $1 ("" = whole tree).
prepare_pass() {
  local sub="$1" local_side="$LOCAL_DIR" remote_side="$REMOTE_PATH"
  if [[ -n "$sub" ]]; then
    local_side="$LOCAL_DIR/$sub"
    remote_side="$REMOTE_PATH/$sub"
  fi
  if [[ "$DIRECTION" == "push" ]]; then
    SOURCE="$local_side"; DESTINATION="$remote_side"
  else
    SOURCE="$remote_side"; DESTINATION="$local_side"
  fi
  RCLONE_ARGS=(
    "$OPERATION" "$SOURCE" "$DESTINATION"
    "${FILTER_ARGS[@]}"
    --transfers 8
    --checkers 16
  )
}

# A --path that doesn't exist locally makes rclone fail with a bare
# "directory not found"; say which subpath and on which side.
if [[ "$DIRECTION" == "push" ]]; then
  for sub in "${PATHS[@]}"; do
    if [[ -n "$sub" && ! -d "$LOCAL_DIR/$sub" ]]; then
      echo "No existe localmente: $LOCAL_DIR/$sub (nada que subir desde --path $sub)" >&2
      exit 1
    fi
  done
fi

echo "Operación: $DIRECTION ($OPERATION)"
for sub in "${PATHS[@]}"; do
  prepare_pass "$sub"
  echo "  desde: $SOURCE"
  echo "  hacia: $DESTINATION"
done
for pat in ${EXCLUDES[@]+"${EXCLUDES[@]}"}; do echo "  excluye: $pat"; done
for pat in ${INCLUDES[@]+"${INCLUDES[@]}"}; do echo "  incluye: $pat"; done
echo

if [[ $DRY_RUN -eq 1 ]]; then
  for sub in "${PATHS[@]}"; do
    prepare_pass "$sub"
    rclone "${RCLONE_ARGS[@]}" --dry-run -v
  done
  echo
  echo "(--dry-run: no se aplicó nada)"
  exit 0
fi

# Mirroring can delete. Show exactly what would go — across every --path pass,
# so one confirmation covers the whole run — and get a yes first; never guess
# an answer when nothing is there to answer (cron, CI).
if [[ $DELETE -eq 1 ]]; then
  echo "Revisando qué borraría (dry-run)..."
  DRY_LOG="$(mktemp)"
  trap 'rm -f "$DRY_LOG"' EXIT
  for sub in "${PATHS[@]}"; do
    prepare_pass "$sub"
    if ! rclone "${RCLONE_ARGS[@]}" --dry-run -v >> "$DRY_LOG" 2>&1; then
      echo "El dry-run falló:" >&2
      cat "$DRY_LOG" >&2
      exit 1
    fi
  done
  DELETIONS="$(grep -Ei 'delet' "$DRY_LOG" || true)"
  if [[ -n "$DELETIONS" ]]; then
    echo "$DELETIONS"
    echo
    if [[ $ASSUME_YES -eq 1 ]]; then
      echo "--yes: se aplican los borrados listados."
    else
      if [[ ! -t 0 ]]; then
        echo "Hay borrados pendientes y no hay terminal para confirmar (usa --yes). Cancelado." >&2
        exit 1
      fi
      read -r -p "¿Aplicar estos borrados? [y/N] " CONFIRM
      case "$CONFIRM" in
        y|Y|yes|si|sí) ;;
        *) echo "Cancelado, no se aplicó nada."; exit 0 ;;
      esac
    fi
  else
    echo "Sin borrados pendientes."
  fi
fi

for sub in "${PATHS[@]}"; do
  prepare_pass "$sub"
  [[ -n "$sub" ]] && echo "-> $sub"
  rclone "${RCLONE_ARGS[@]}" -v --stats 30s --stats-one-line
done
echo "$DIRECTION completo."
