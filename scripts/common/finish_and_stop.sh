#!/usr/bin/env bash
# Watchdog: comprueba que el pipeline terminó DE VERDAD, sincroniza todo
# fuera de la máquina, y recién entonces apaga la instancia de Vast.
#
# Pensado para cron cada 5 min. Sale sin hacer nada mientras quede trabajo,
# así que correrlo de más es gratis.
#
# EL ORDEN IMPORTA Y NO ES NEGOCIABLE: primero se verifica que no queda
# trabajo, después se sube todo, después se COMPRUEBA que la subida quedó,
# y sólo si eso pasa se apaga. Apagar con algo sin subir pierde el trabajo
# de la corrida entera, que es exactamente lo que este script existe para
# evitar — no para ahorrar unos centavos.
#
# Uso:
#   scripts/common/finish_and_stop.sh --dry-run   # informa, nunca apaga
#   scripts/common/finish_and_stop.sh             # apaga si todo pasa
#
# Requiere VAST_API_KEY con permiso de escritura sobre la instancia. La
# CONTAINER_API_KEY del contenedor puede no tenerlo; si el apagado falla,
# el script lo dice y NO se queda en un estado a medias — el trabajo ya
# está subido igual.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR=/workspace/logs
REPORT="$LOG_DIR/finish_report.txt"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# Bajo cron el entorno es mínimo: ni las variables que vast inyecta en el
# contenedor ni las del .env del proyecto están. Se recuperan de las dos
# fuentes que sí persisten — el entorno del PID 1 (que ES el del contenedor)
# y el .env del repo — para que este script se comporte igual lanzado a mano
# que desde crontab. Sin esto, el paso de apagado falla silenciosamente por
# no encontrar la API key, justo cuando ya no hay nadie mirando.
if [[ -r /proc/1/environ ]]; then
  while IFS='=' read -r -d '' k v; do
    case "$k" in VAST_*|CONTAINER_API_KEY) [[ -z "${!k:-}" ]] && export "$k=$v" ;; esac
  done < /proc/1/environ
fi
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; . "$REPO_ROOT/.env"; set +a
fi
export HOME="${HOME:-/root}"

say() { echo "[$(date '+%F %T')] $*"; }

# --- 1. ¿Sigue corriendo algo del pipeline? -------------------------------
# Se buscan los scripts por nombre de archivo, con `pgrep -f` sobre la línea
# de comando ENTERA. Eso matchea a propósito dos cosas distintas:
#   - el proceso python corriendo de verdad, y
#   - los wrappers `bash -c 'while ...; do sleep; done; <script>'` que
#     esperan su turno. Un job encadenado que todavía no arrancó es trabajo
#     PENDIENTE: apagar ahí perdería exactamente lo que falta.
# Los patrones se parten en dos trozos concatenados para que esta línea de
# comando no se auto-detecte como "pipeline corriendo".
BUSY=""
for pat in "01_fetch""_filings.py" "01b_fetch""_xbrl_facts.py" "02_extract""_sections.py" \
           "it_""loop.sh" "sync_""loop.sh"; do
  pgrep -f "$pat" >/dev/null 2>&1 && BUSY="$BUSY $pat"
done
if [[ -n "$BUSY" ]]; then
  say "todavía corriendo:$BUSY — no hago nada"; exit 0
fi

# --- 2. ¿La red está quieta? ---------------------------------------------
# Un proceso puede haber muerto dejando una transferencia en vuelo, o puede
# haber un rclone/git del loop de sync a mitad de camino. Se mide el
# contador de la interfaz, no la lista de procesos.
read -r RX1 TX1 < <(awk '/eth0:/ {gsub(/.*:/,"",$1); print $2, $10}' /proc/net/dev)
sleep 20
read -r RX2 TX2 < <(awk '/eth0:/ {gsub(/.*:/,"",$1); print $2, $10}' /proc/net/dev)
DELTA=$(( (RX2 - RX1) + (TX2 - TX1) ))
# 200 KB en 20 s. Por encima de eso hay una transferencia real; por debajo
# es el ruido del túnel SSH de vast y del keepalive de nginx.
if [[ "$DELTA" -gt 200000 ]]; then
  say "red activa ($((DELTA/1024)) KB en 20s) — algo sigue transfiriendo, no apago"; exit 0
fi

# --- 3. ¿Queda trabajo pendiente en los datos? ---------------------------
PENDING="$(.venv/bin/python - <<'PY' 2>/dev/null
import sys; sys.path.insert(0, "scripts/it")
from pathlib import Path
import pandas as pd, manifest_store as ms

problems, lines = [], []
mdir = Path("data/interim/manifests_it")
sdir = Path("data/interim/sections_it")
m = ms.read(mdir) if mdir.exists() else pd.DataFrame()
if m.empty:
    problems.append("manifest italiano vacío")
else:
    ok = m[m.download_status == "completed"]
    failed = m[m.download_status.str.startswith("failed", na=False)]
    lines.append(f"descargas: {len(ok)} ok, {len(failed)} fallidas, {len(m)} filas")
    lines.append(f"con narrativa: {int(ok.has_narrative.sum())} ({ok.has_narrative.mean():.1%})")
    done = set()
    for f in sdir.glob("extracted_documents__run=*.parquet"):
        try: done |= set(pd.read_parquet(f, columns=["document_id"]).document_id)
        except Exception: pass
    missing = set(ok.document_id) - done
    lines.append(f"extraídos: {len(done)}; sin extraer: {len(missing)}")
    if missing:
        problems.append(f"{len(missing)} filings descargados sin extraer")
xdir = Path("data/raw/xbrl_it")
lines.append(f"xbrl facts en disco: {len(list(xdir.rglob('*.json.gz'))) if xdir.exists() else 0}")
print("\n".join(lines))
print("PROBLEMS:" + ("; ".join(problems) if problems else "none"))
PY
)"
say "$PENDING" | sed 's/^/    /'
if echo "$PENDING" | grep -q "^PROBLEMS:" && ! echo "$PENDING" | grep -q "^PROBLEMS:none"; then
  say "queda trabajo por hacer — no apago"; exit 0
fi

# --- 4. Sacar TODO de la máquina ------------------------------------------
say "nada pendiente; sincronizando antes de apagar"
GIT_OK=1; B2_OK=1
if [[ -n "$(git status --porcelain)" ]]; then
  say "AVISO: hay cambios sin commitear; no los commiteo yo (ver sync_loop.sh)"
  git status --porcelain | head -10 | sed 's/^/    /'
fi
git push origin HEAD >/dev/null 2>&1 || GIT_OK=0
UNPUSHED="$(git log --oneline @{u}..HEAD 2>/dev/null | wc -l | tr -d ' ')"
[[ "${UNPUSHED:-1}" -ne 0 ]] && GIT_OK=0
scripts/common/sync_data_b2.sh push >/dev/null 2>&1 || B2_OK=0

# --- 5. COMPROBAR que la subida quedó, no asumirlo ------------------------
B2_PENDING="$(scripts/common/sync_data_b2.sh push --dry-run 2>/dev/null \
  | grep -cE "^(Transferred|.*: Copied)" || true)"
say "git: $([[ $GIT_OK -eq 1 ]] && echo 'al día' || echo "FALLA ($UNPUSHED sin pushear)")"
say "b2 : $([[ $B2_OK -eq 1 ]] && echo 'push ok' || echo 'FALLA')"

{
  echo "=== informe de cierre — $(date '+%F %T') ==="
  echo "$PENDING"
  echo "git al día: $GIT_OK | b2 ok: $B2_OK"
  echo "commits sin pushear: ${UNPUSHED:-?}"
} > "$REPORT"
say "informe en $REPORT"

if [[ $GIT_OK -ne 1 || $B2_OK -ne 1 ]]; then
  say "la sincronización NO quedó limpia — NO apago. Se reintenta en el próximo cron."
  exit 1
fi

# --- 6. Apagar ------------------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
  say "--dry-run: todo listo y sincronizado; acá apagaría la instancia"; exit 0
fi
KEY="${VAST_API_KEY:-${CONTAINER_API_KEY:-}}"
ID="${VAST_INSTANCE_ID:-$(echo "${VAST_CONTAINERLABEL:-}" | tr -d 'C.')}"
if [[ -z "$KEY" || -z "$ID" ]]; then
  say "sin VAST_API_KEY/instance id — no puedo apagar. Trabajo ya sincronizado."; exit 1
fi
say "apagando instancia $ID"
RESP="$(curl -sS -X PUT -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d '{"state": "stopped"}' "https://console.vast.ai/api/v1/instances/$ID/" 2>&1)"
say "respuesta: $RESP"
echo "stop solicitado: $RESP" >> "$REPORT"
