# Analytics — índice

Diseño final. Un documento por bloque, sobre el corpus congelado
(`docs/FREEZE.md`) y en modo extensivo: todas las empresas con filings, cero
cuando no hablan de IA, intensidad por 1.000 párrafos (`10_pipeline.md`).

| # | documento | pregunta | script |
|---|---|---|---|
| 00 | `00_funnel_del_corpus.md` | de documentos a frames: qué hay en el corpus | consulta SQL en el doc |
| 01 | `01_evolucion_2021_2025.md` | qué pasó con la divulgación de IA 2021-2025 | `evolution_figures.py` |
| 02 | `02_segmentacion.md` | qué tipos de divulgación existen (RQ1) | `build_segments.py` |
| 03 | `03_voz_conducta.md` | ¿desacople entre cómo habla y lo que describe? (RQ2) | `build_voice_behavior_grid.py` |
| 04 | `04_perfiles_economicos.md` | qué empresas son los segmentos, dentro de sector × año (RQ3) | `economic_profiles.py` |
| 05 | `05_senal_incremental.md` | **¿el contenido aporta señal más allá de volumen y fundamentals?** (RQ4) | `incremental_signal.py` |
| 06 | `06_brecha_entre_canales.md` | la misma empresa en la call y en el filing | `channel_gap_analysis.py` |
| 07 | `07_shocks_sec_deepseek.md` | SEC 2024 y DeepSeek: no-identificación (RQ5, RQ6) | `shock_analysis.py`, `shock_did_simple.py` |
| 08 | `08_definiciones_de_washing.md` | tres definiciones de AI-washing y por qué no coinciden | `washing_score.py`, `validate_washing_score.py` |
| 09 | `09_actividades_ia.md` | qué dicen las empresas que hacen con IA: acciones, funciones, etapa, proveedores; por segmento y por empresa | `ai_activities_from_frames.py`, `activity_profiles.py`, `activity_grounding.py` |
| 10 | `10_pipeline.md` | cómo se produce todo y cómo se regenera | `make analytics` |

`apendice/`: material secundario que no entra al cuerpo — descriptivos SQL
del corpus, correlaciones simples y FDR (superadas por `05`), factores
voz-conducta (sustento técnico de `03`).

Lo que no está acá no forma parte del diseño final: arquetipos A/B/C/D,
clusters de comportamiento, ROIC−WACC por grupos de washing, controles
sintéticos. Está en git.
