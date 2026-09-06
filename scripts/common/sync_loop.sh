#!/usr/bin/env bash
# Empuja data/ a B2 y los commits a git, cada N segundos (default 600).
#
# Sólo EMPUJA lo ya comprometido. No hace `git add` ni `git commit`:
# el árbol de trabajo puede tener cambios a medio escribir de otra sesión
# (pasó hoy: scripts/analytics/ tenía trabajo en curso ajeno), y
# autocommitearlo los mete bajo un mensaje inventado y a medias. Si querés
# que también commitee, se agrega, pero es una decisión aparte.
#
# El push a B2 es `rclone copy` vía sync_data_b2.sh: aditivo, nunca borra.
#
# Uso:  scripts/common/sync_loop.sh [segundos]
#       kill $(pgrep -f sync_loop.sh)   # para pararlo
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTERVAL="${1:-600}"
cd "$REPO_ROOT"

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"

  pendientes="$(git log --oneline @{u}..HEAD 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${pendientes:-0}" -gt 0 ]]; then
    if git push origin HEAD >/dev/null 2>&1; then
      echo "[$ts] git: $pendientes commit(s) pusheados"
    else
      echo "[$ts] git: FALLÓ el push de $pendientes commit(s)"
    fi
  else
    echo "[$ts] git: al día"
  fi

  sucio="$(git status --porcelain | wc -l | tr -d ' ')"
  [[ "${sucio:-0}" -gt 0 ]] && echo "[$ts] git: $sucio archivo(s) sin commitear (no los toco)"

  if out="$(scripts/common/sync_data_b2.sh push 2>&1 | tail -3)"; then
    echo "[$ts] b2: $(echo "$out" | grep -oE '[0-9]+ / [0-9]+, [0-9]+%' | tail -1 || echo ok)"
  else
    echo "[$ts] b2: FALLÓ"
    echo "$out" | sed 's/^/       /'
  fi

  sleep "$INTERVAL"
done
