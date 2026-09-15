# Plan de expansión de tipos de documento

El pipeline actual solo cubre 10-K/10-Q. Esos son, por diseño, el peor
lugar para encontrar AI-washing real: legal los revisa específicamente
para minimizar riesgo de litigio, así que tienden a lenguaje cauteloso y
genérico (ver `docs/analytics/01_ai_disclosure_analytics.md`, sección "Evolución
temporal"). El AI-washing que le preocupa a la SEC vive en otros
documentos — algunos ya están en EDGAR con la misma infraestructura que
`scripts/us/` ya usa, otros requieren una fuente nueva.

Ordenado de más fácil a más difícil de agregar. Cada fase asume que las
anteriores ya están hechas.

---

## Fase 1 — 8-K (Item 7.01 / 2.02 / 8.01)

**Por qué**: comunicados de prensa y anuncios de producto/partnership de
IA se adjuntan como exhibit a un 8-K (`EX-99.1` típicamente). Es el
"anuncio" sin el filtro legal completo de un 10-K — mucho más cerca de
donde realmente se exagera.

**Por qué es lo más fácil**: mismo EDGAR, mismas 517 empresas del
universo, mismo mecanismo de descarga (`scripts/us/01_fetch_filings.py`
ya sabe pedir un `form` específico a EDGAR — solo hay que agregar `8-K`
a la lista de formularios, no escribir un fetcher nuevo). El texto de un
8-K es HTML igual que un 10-K, así que `clean_html_to_lines` /
`_paragraph_select_sql` (`build_duckdb.py`) deberían funcionar sin
cambios, extendiendo el `UNION ALL BY NAME` existente con un tercer
branch `_paragraph_select_sql('8-K', 'filing_sections_8k')`.

**Trabajo real**:
1. `scripts/us/01_fetch_filings.py`: agregar `8-K` a los formularios
   descargados, filtrando a los Items 7.01/2.02/8.01 (los que suelen
   traer comunicados/resultados) para no bajar los ~30 items de 8-K que
   son puramente administrativos (cambio de directorio, etc. — ruido).
2. `scripts/raw_processing/us/section_segmenter.py` (o un segmentador nuevo si el 8-K
   no tiene la estructura de Items 1/1A/7 del 10-K): un 8-K es corto y
   casi todo es el cuerpo del Item + el exhibit — probablemente NO
   necesita segmentación por sección, solo extraer el texto completo del
   documento y del exhibit adjunto.
3. `build_duckdb.py`: nuevo branch en el `UNION ALL BY NAME` de
   `paragraphs`, análogo al de 10-K/10-Q.
4. Re-correr `ai_prefilter.py --source-relation unique_paragraphs` (el
   nuevo volumen de 8-K entra a `unique_paragraphs` automáticamente en
   cuanto está en `paragraphs`, sin tocar el script — mismo diseño
   país-agnóstico de `docs/prefilter_evaluation.md` §8.9/§8.12).

**Riesgo a vigilar**: un 8-K puede repetir CASI el mismo texto de un
comunicado de prensa trimestral tras trimestre — el dedup por
`text_hash` (§8.7/§8.8) ya maneja eso, pero vale la pena revisar
`duplicate_count` específicamente en 8-K una vez cargado.

---

## Fase 2 — Cartas de comentario de la SEC (UPLOAD / CORRESP)

**Por qué**: si la SEC específicamente cuestionó las afirmaciones de IA
de una empresa, queda en el expediente público de EDGAR como
`UPLOAD` (carta de la SEC) y `CORRESP` (respuesta de la empresa). Es la
única señal de este plan que no es "más texto para clasificar" sino
**ground truth de un tercero independiente**: un caso donde ya se sabe,
sin depender del propio clasificador, que hubo un problema real de
divulgación de IA. Sirve para VALIDAR si los arquetipos de
`docs/analytics/01_ai_disclosure_analytics.md` realmente separan washing de
divulgación creíble, no solo para describir más variación.

**Por qué es fácil pese a ser una fuente nueva**: EDGAR expone estas
cartas por el mismo mecanismo de full-text search ya usado en
`scripts/raw_ingestion/us/00_build_firm_universe.py` (ver
`scripts/raw_ingestion/us/docs/sp500_2021_universe_provenance.md`, que ya usa
`efts.sec.gov/LATEST/search-index` para resolver CIKs). La misma técnica,
apuntada a `forms=UPLOAD,CORRESP` con la query `"artificial intelligence"`
o `"generative AI"`, da la lista de cartas relevantes sin necesidad de
scraping nuevo.

**Trabajo real**:
1. Script nuevo `scripts/us/0X_fetch_sec_comment_letters.py`: full-text
   search de EDGAR (`forms=UPLOAD,CORRESP`, query de términos de IA),
   sobre el universo de 517 empresas o sobre TODO EDGAR (esto es
   independiente del universo — la SEC pudo haberle escrito a una
   empresa fuera del universo actual, y ESE caso también sirve).
2. Extraer: empresa, fecha, si es carta de la SEC o respuesta de la
   empresa, y el texto completo (suelen ser PDFs o HTML cortos, no hay
   Items que segmentar).
3. No necesita el pipeline de prefiltro/clasificación semántica — el
   volumen es chico (probablemente decenas a bajos cientos de cartas en
   todo EDGAR sobre IA) y el valor está en LEER cada una a mano o con
   un prompt específico ("¿la SEC está cuestionando una afirmación de
   IA, y cuál?"), no en pasarla por el pipeline de frames diseñado para
   párrafos de 10-K.
4. Cruzar: de las empresas con carta, ¿cuál es su arquetipo (A/B/C/D) en
   `docs/analytics/01_ai_disclosure_analytics.md`? Si las cartas caen
   desproporcionadamente en el cluster D (líderes vocales), es evidencia
   fuerte de que el arquetipo captura señal real.

---

## Fase 3 — DEF 14A (proxy statement)

**Por qué**: gobernanza real — supervisión del directorio sobre IA,
comités de riesgo tecnológico, compensación ejecutiva ligada a
"transformación con IA". El esquema de frames ya tiene 6 conceptos
`gov_*` (`docs/classification_model.md`) que en el 10-K probablemente
están casi vacíos (`docs/analytics/01_ai_disclosure_analytics.md` #6 no midió
gobernanza directamente, pero el 10-K rara vez detalla esto) — el proxy
es donde ese eje del esquema de clasificación finalmente tendría con qué
trabajar.

**Trabajo real**:
1. `scripts/us/01_fetch_filings.py`: agregar `DEF 14A` a los formularios.
2. Segmentación: un proxy no tiene Items 1/1A/7 — tiene secciones como
   "Board Committees", "Risk Oversight", "Compensation Discussion and
   Analysis" (CD&A). Necesita su propio segmentador (heurística por
   encabezados, no reusa `section_segmenter.py` de 10-K directamente).
3. Mismo `UNION ALL BY NAME` en `build_duckdb.py`.
4. Esperable: volumen bajo de párrafos relevantes por proxy (la mayoría
   del documento es compensación no relacionada con IA) — el prefiltro
   léxico/semántico ya construido debería filtrar bien esto sin ajustes,
   dado que es agnóstico al tipo de documento fuente.

---

## Fase 4 — Transcripciones de earnings calls

**Por qué**: donde el CEO realmente "vende" la historia de IA sin el
filtro legal completo del 10-K — la fuente más directamente comparable
al 10-K/10-Q para contrastar tono (call = menos filtrado, filing = más
filtrado), la comparación en sí misma sería un resultado de tesis.

**Por qué es más difícil**: NO está en EDGAR — hay que decidir fuente
(sitio de relaciones con inversionistas de cada empresa, un proveedor de
datos pago tipo Seeking Alpha/AlphaSense, o scraping de transcripciones
públicas). Sin una fuente estructurada y consistente para las 517
empresas, esto se vuelve 517 scrapers distintos en la práctica. Requiere
decidir la fuente ANTES de estimar el esfuerzo real — no es una
extensión trivial de `scripts/us/01_fetch_filings.py`.

**No se detalla más hasta decidir la fuente** — es la primera fase que
necesita una decisión de producto/alcance antes de una decisión técnica.

---

## Fase 5 — S-1/S-3 (prospectos de IPO/oferta) y presentaciones a inversionistas

**Por qué**: los IPOs 2023-2024 fueron notorios por pitchear
"AI-powered" agresivamente — casos extremos útiles para calibrar el
extremo alto de la escala de especificidad/promoción. Las presentaciones
a inversionistas (a veces furnished como exhibit de 8-K, a veces solo en
el sitio de IR) son marketing dirigido a analistas, sin filtro.

**Por qué va al final**: la población relevante (empresas que salieron a
bolsa recientemente) es distinta del universo actual de 517 empresas
S&P 500 establecidas — construir esto bien implica antes decidir si se
expande el universo de firmas o se trata como una muestra aparte
("empresas IPO 2023-2024 con narrativa de IA"), una decisión de diseño
de investigación, no solo de scraping. Las presentaciones a
inversionistas comparten el problema de fuente no estructurada de la
Fase 4.

---

## Resumen de prioridad

| Fase | Documento | Esfuerzo técnico | Valor para la tesis |
|---|---|---|---|
| 1 | 8-K | Bajo (mismo EDGAR/pipeline) | Alto — versión menos filtrada del mismo tipo de anuncio |
| 2 | Cartas SEC (UPLOAD/CORRESP) | Bajo (mismo mecanismo de full-text search ya usado) | Muy alto — único ground truth externo del plan |
| 3 | DEF 14A | Medio (segmentador nuevo) | Medio-alto — activa el eje de gobernanza del esquema |
| 4 | Earnings calls | Alto (fuente no resuelta) | Alto, pero bloqueado en decisión de fuente |
| 5 | S-1/S-3, investor decks | Alto (población distinta) | Medio — casos extremos útiles, no central |

No se empieza ninguna fase hasta que se confirme explícitamente cuál —
igual que la integración de Chile (`docs/prefilter_evaluation.md` §8.9),
esto queda documentado como plan, no como trabajo en curso.
