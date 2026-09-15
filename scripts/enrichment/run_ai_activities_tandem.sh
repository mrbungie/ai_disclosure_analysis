#!/usr/bin/env bash
# Corre ai_activities_from_frames.py en loop mientras pass-1 (ai_classify.py)
# sigue escribiendo partes nuevas -- cada invocación llama fetch_pending()
# desde cero, así que agarra cualquier frame disparador nuevo que haya
# aparecido desde la vuelta anterior. Sin --limit: procesa todo lo pendiente
# en cada vuelta, y si no hay nada nuevo simplemente vuelve a chequear tras
# la pausa (fetch_pending vacío no cuesta llamadas a LLM).
set -uo pipefail
cd "$(dirname "$0")/../.."

SLEEP_SECONDS="${1:-60}"
CHILD_PID=""
STOP=0

# Reenvía la señal al proceso Python en curso (bash no lo hace solo) y frena
# el loop -- misma garantía que ya probamos en ai_classify.py: lo que esté en
# el buffer del hijo se escribe antes de salir, nada se pierde.
_forward() {
    STOP=1
    [ -n "$CHILD_PID" ] && kill -TERM "$CHILD_PID" 2>/dev/null
}
trap _forward SIGINT SIGTERM

while [ "$STOP" -eq 0 ]; do
    .venv/bin/python3 scripts/enrichment/ai_activities_from_frames.py --concurrency 6 --part-rows 250 --progress-every 50 &
    CHILD_PID=$!
    wait "$CHILD_PID"
    CHILD_PID=""
    [ "$STOP" -eq 1 ] && break
    echo "--- vuelta terminada, durmiendo ${SLEEP_SECONDS}s ---"
    sleep "$SLEEP_SECONDS" &
    wait $!
done
echo "run_ai_activities_tandem.sh: detenido"
