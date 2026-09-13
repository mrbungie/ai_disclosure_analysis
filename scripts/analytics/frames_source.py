"""Lee los frames sin depender del archivo DuckDB compartido.

`duckdb/thesis.duckdb` es un solo archivo y DuckDB no permite que dos procesos
lo usen si uno lo tiene abierto para escritura — ni siquiera al segundo en
read-only:

    IO Error: Could not set lock on file "duckdb/thesis.duckdb":
    Conflicting lock is held in /usr/bin/python3.12 (PID 73956)

Con varias corridas en paralelo (fetch de earnings calls, pipeline de Italia,
rebuild de vistas) eso convierte cualquier análisis en una lotería: o falla, o
queda esperando. Y no es un problema del análisis — es del archivo.

Todo lo que un análisis a nivel empresa necesita ya vive en parquet:

    frames        data/interim/ai_classify/ai_frames__session=*.parquet
    población     data/interim/prefilter_predictions_unique/prefilter_predictions__run=*.parquet
    manifiestos   data/interim/manifests/filing_manifest*.parquet

Este módulo arma la misma tabla que la vista `gold_ai_frames` —dedup por LLAMADA
del juez, acotada a la población del despliegue VIGENTE, unida a los
manifiestos— pero contra los parquet, en una conexión `:memory:`. Sin lock, sin
esperar a nadie.

Diferencia deliberada con la vista: acá NO se expande a una fila por instancia
de párrafo. Un análisis a nivel empresa quiere frames únicos (ver
`08_definiciones_de_washing.md`, unidad de análisis); la expansión por instancia sólo
sirve para contar apariciones en el corpus.

Uso:
    from frames_source import load_frames_parquet
    frames = load_frames_parquet()                    # 10-K, DEF 14A, 8-K
    frames = load_frames_parquet(include_10q=True)    # + 10-Q
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMES_GLOB = "data/interim/ai_classify/ai_frames__session=*.parquet"
PREDICTIONS_GLOB = "data/interim/prefilter_predictions_unique/prefilter_predictions__run=*.parquet"
MANIFEST = "data/interim/manifests/filing_manifest.parquet"
MANIFEST_10Q = "data/interim/manifests/filing_manifest_10q.parquet"


def _base_sql(manifest: str, form_expression: str) -> str:
    return f"""
        SELECT fm.ticker, fm.cik, {form_expression} AS form_type, fm.filing_date,
               extract(year from fm.filing_date)::INT AS year,
               f.text_hash, f.frame_id, f.subject, f.ai_type, f.temporal,
               f.concepts, f.specificity, f.rhetoric, f.valence
        FROM latest_frames f
        JOIN current_population p ON p.text_hash = f.text_hash
        JOIN read_parquet('{manifest}') fm ON fm.accession_number = f.accession_number
        -- El manifiesto en parquet no trae `country_code` (la vista de DuckDB lo
        -- agrega); `accession_number` es un identificador de la SEC, así que
        -- alcanza para EE.UU. El filtro de país queda del lado de los frames.
        WHERE f.country_code = 'us' AND f.has_frame AND fm.ticker IS NOT NULL
    """


def load_frames_parquet(include_10q: bool = False,
                        repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        # `_base_sql` is called once per form (10-K, optionally 10-Q) and its
        # results UNIONed -- a view referenced from both branches gets
        # RE-EVALUATED once per branch. Harmless for `all_frames`/`latest_frames`
        # (tens of thousands of rows), but `current_population`'s QUALIFY
        # window runs over `prefilter_predictions_unique`'s ~79M rows: as a
        # VIEW that ran the full window function twice, driving memory
        # footprint to several GB and stalling on this machine (observed
        # 2026-09-13: 15+ minutes, near-zero CPU, workers idle -- classic
        # memory-pressure/swap, not a deadlock). TABLE materializes it once.
        con.execute(f"""
            CREATE TEMP TABLE all_frames AS
            SELECT * FROM read_parquet('{repo_root / FRAMES_GLOB}', union_by_name=True)
            WHERE error IS NULL;

            -- Dedup por LLAMADA del juez, no por (texto, frame): un mismo texto
            -- se clasifica varias veces y mezclar el frame 0 de una llamada con
            -- el 1 de otra junta dos lecturas distintas del mismo párrafo.
            CREATE TEMP TABLE latest_call AS
            SELECT text_hash, session_id, classified_at FROM (
                SELECT DISTINCT text_hash, session_id, classified_at FROM all_frames)
            QUALIFY row_number() OVER (
                PARTITION BY text_hash ORDER BY session_id DESC, classified_at DESC) = 1;

            CREATE TEMP TABLE latest_frames AS
            SELECT f.* FROM all_frames f JOIN latest_call c
              ON c.text_hash = f.text_hash AND c.session_id = f.session_id
             AND c.classified_at = f.classified_at;

            -- Población del despliegue vigente, no la unión histórica.
            CREATE TEMP TABLE current_population AS
            SELECT text_hash FROM (
                SELECT text_hash, is_ai_prefiltered FROM read_parquet(
                    '{repo_root / PREDICTIONS_GLOB}', union_by_name=True)
                QUALIFY row_number() OVER (
                    PARTITION BY text_hash ORDER BY model_version DESC) = 1
            ) WHERE is_ai_prefiltered;
        """)
        query = _base_sql(str(repo_root / MANIFEST), "fm.form_type")
        if include_10q:
            query += " UNION ALL " + _base_sql(str(repo_root / MANIFEST_10Q), "'10-Q'")
        frames = con.execute(query).fetchdf()
    finally:
        con.close()
    return frames.drop_duplicates(["ticker", "form_type", "filing_date",
                                   "text_hash", "frame_id"])


if __name__ == "__main__":
    for flag in (False, True):
        data = load_frames_parquet(include_10q=flag)
        print(f"include_10q={flag}: {len(data):,} frames | "
              f"{data['ticker'].nunique():,} empresas | "
              f"formularios {sorted(data['form_type'].unique())}")
