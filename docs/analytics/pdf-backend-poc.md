# POC: qué motor usar para parsear los PDFs chilenos

Comparación medida de tres modelos de document-parsing contra el
extractor que el pipeline chileno venía usando (PyMuPDF + heurísticas),
sobre un PDF real. Motivó el refactor de `scripts/cl/cmf_pdf_paragraphs.py`
hacia `scripts/common/pdf/`, que es country-agnostic.

**Resultado corto:** PaddleOCR-VL 1.6 gana. Empata a MinerU2.5-Pro en
calidad de corpus y recuperación de tablas, le gana en velocidad (~2x) y
en recuperación de infografías; dots.mocr queda tercero por una razón
concreta y no cosmética (ver "Por qué dots.mocr pierde").

## Setup

- **Documento:** Banco de Chile, Memoria Anual 2024 (`97004000`), 466
  páginas, born-digital (tiene capa de texto). Es el caso difícil del
  corpus: Memoria Integrada glossy, dos y tres columnas, infografías,
  y ~250 páginas de notas a los EEFF con tablas densas.
- **6 páginas de test**, elegidas por tipo de dificultad, no al azar:

  | pág. | por qué está |
  |---|---|
  | 35 | **la página que importa**: narrativa de IA a dos columnas ("el programa de exploración del uso de herramientas de inteligencia artificial…"), con marcadores CMF/GRI en el margen izquierdo |
  | 17 | infografía de KPIs: donuts, cifras sueltas, sin flujo de lectura |
  | 24 | tabla limpia (veinte mayores accionistas) |
  | 40 | tabla de directorio: 4 columnas, celdas con párrafos largos |
  | 120 | prosa a varias columnas + lista de bullets |
  | 440 | 4 tablas de descalce de liquidez, 52 celdas numéricas |

- **Hardware:** RTX 5090 (Blackwell, sm_120, 32 GB). Los tres modelos
  corren en GPU. Nota de instalación, porque cuesta encontrarla: en
  sm_120 hay que usar `paddlepaddle-gpu` **cu129** (la ficha del modelo
  sugiere cu126, que no compila para esta arquitectura), y `flash-attn`
  no tiene wheel para sm_120 — dots.mocr lo importa sin condicional, así
  que se sustituye por un shim exacto sobre SDPA
  (`vlm_dots.install_flash_attn_shim`).
- **Render:** 200 dpi. Los tres leen píxeles; el baseline lee la capa de
  texto del PDF.
- Los tres corrieron por `transformers`/pipeline local, sin vLLM, para
  que la comparación de latencia sea entre iguales.

## Resultados

| | baseline PyMuPDF | **PaddleOCR-VL 1.6** | MinerU2.5-Pro | dots.mocr |
|---|---|---|---|---|
| bloques emitidos al corpus (6 págs) | 90 | 57 | 55 | 57 |
| **fugas de chrome al corpus** | 3 | **0** | **0** | 7 |
| fragmentos cortados a media frase | 11 | 24 | 13 | 23 |
| **celdas de tabla recuperadas (pág. 440)** | 42/52 (80,8 %) | **52/52** | **52/52** | **52/52** |
| KPIs de infografía (pág. 17) | 10/11 | 7/11 | 7/11 | 0/11 |
| fidelidad de transcripción vs. capa de texto | 1,0 por construcción | 0,9995 | 0,9997 | 0,9997 |
| latencia media (s/página, 6 págs difíciles) | 0,076 | **6,5** | 11,9 | 11,1 |
| régimen permanente (doc. completo de 30 págs) | 0,042 (23,6 p/s) | **5,95** (0,17 p/s) | — | — |

Scripts: `/workspace/ocr_poc/score.py` (métricas de corpus/tablas) y
`ocr_accuracy.py` (fidelidad de caracteres). El ground truth de la pág.
440 son las 52 celdas numéricas leídas a ojo de la imagen renderizada.

### Lo que el baseline pierde, concretamente

No son diferencias de estilo. Son tres pérdidas de datos verificadas:

1. **Columnas de tabla que desaparecen sin error.** Pág. 440, tabla de
   descalce: el ground truth tiene 4 columnas (`De 0 a 7 / 15 / 30 / 90
   días`). El baseline devuelve tres números por fila y pierde las
   etiquetas de fila:
   ```
   18.893.992 | 21.207.713 | 24.766.475
   ```
   Los tres VLMs devuelven las 4 columnas con su etiqueta y con
   `rowspan`/`colspan` correctos.

2. **Marcadores de margen incrustados dentro de una frase real.** En la
   página 35 — la página con la narrativa de IA — los marcadores CMF
   viven en el canalón izquierdo, a la misma altura vertical que un
   párrafo. El orden de lectura del baseline los intercala:
   ```
   3.1.v 3.6.vii Este enfoque ha permitido altos niveles de
   participación en los 3.6.ix procesos de adaptación competitiva…
   ```
   PaddleOCR-VL y MinerU los clasifican como `aside_text` y los sacan
   del corpus. dots.mocr los devuelve como bloques `Text` normales.

3. **Orden de lectura roto en tablas sin líneas de grilla.** Pág. 40:
   el baseline atribuye "Nacionalidad: Chileno / Género: Masculino /
   Edad: 66 años" al director equivocado, porque `find_tables()` no
   detecta la tabla y los bloques se ordenan por posición. Los tres VLMs
   la devuelven como tabla de 4 columnas con cada director en su fila.

### Por qué dots.mocr pierde

Empata en tablas y su markup es de hecho el más rico (`<thead>`, `<br>`
dentro de celdas, `<strong>` en la fila de subtotal). Pierde por dos
cosas medibles:

- **No tiene categoría para notas de margen.** Sus 7 fugas de chrome son
  exactamente los marcadores CMF/GRI de las páginas 35 y 24, emitidos
  como `Text`. En este corpus eso no es un caso raro: están en todas las
  páginas de las Memorias que siguen la norma CMF.
- **Descarta infografías enteras.** La página 17 vuelve como un solo
  bloque `Picture` sin texto: 0 de 11 KPIs. PaddleOCR-VL y MinerU
  recuperan 7.

### Sobre el 7/11 y el 10/11 de la infografía

El baseline "gana" esa fila y conviene ser explícito sobre por qué: lee
la capa de texto, así que los glifos están ahí por construcción — pero
salen desordenados y sin asociación ("Colocaciones totales $52.095 Banca
Minorista 65% Banca Mayorista" y luego "35%" como bloque aparte). Los
VLMs releen píxeles y pierden lo que está dibujado dentro del donut. Es
un trade-off real, no un empate: **si esas cifras importaran, el
baseline tampoco sirve, porque no dice a qué corresponde cada una.**

La fidelidad de transcripción (0,9995–0,9997) dice lo mismo desde el
otro lado: releer píxeles cuesta ~3 caracteres cada 6.000. Es barato,
pero no es cero, y el baseline no lo paga.

## Costo a escala del corpus

Esto es lo que decide la arquitectura, no la calidad:

- Corpus chileno: **2.112 PDFs** (576 Memorias + 1.536 Análisis
  Razonados), 6,8 GB comprimidos ≈ **~300.000 páginas** (calibrado con
  páginas/MB sobre los dos PDFs realmente abiertos; es orden de
  magnitud, no cifra exacta).
- A 5,95 s/página en un solo stream (medido en régimen permanente sobre
  un documento completo, no sobre las 6 páginas difíciles): **~496
  horas**. No es viable.
- Por eso `scripts/common/pdf/pipeline.py` tiene **triage por página**:
  sólo va al VLM la página que lo necesita (sin capa de texto, con
  tablas, o multi-columna). Medido sobre los dos documentos de prueba:

  | documento | páginas | al VLM | al backend CPU |
  |---|---|---|---|
  | Memoria Banco de Chile 2024 | 466 | 354 (76 %) | 112 |
  | Análisis Razonado SALMOCAM 2024-Q4 | 30 | 17 (57 %) | 13 |

  El triage ahorra ~25-45 %, no un orden de magnitud: las Memorias
  glossy son difíciles casi enteras. Con el triage puesto, el corpus
  completo baja de ~496 h a **~350 h en un solo stream**. Sigue sin ser
  viable. **La conclusión honesta es que el triage solo no alcanza: para
  correr el corpus completo hay que servir el modelo con vLLM**
  (`pdf.runtime: vllm` en `configs/cl/config.yaml`), que es donde el
  batching hace la diferencia — el modelo son 0,9B parámetros y una 5090
  de 32 GB está muy lejos de saturarse con un stream (el POC usó ~12 GB).
  Medir el throughput con vLLM es el siguiente paso pendiente; este POC
  no lo midió.

El triage cuesta ~50 ms/página (usa `layout_signals()` del backend CPU,
que es el mismo análisis que ese backend haría igual), así que es gratis
frente a la llamada al VLM que evita.

## Decisión

`pdf.backend: paddleocr_vl` con `triage.enabled: true` y fallback a
`pymupdf`, en `configs/cl/config.yaml`. Los cuatro backends quedan
registrados en `scripts/common/pdf/backends/` y son intercambiables por
config o por `--backend`, así que revisar esta decisión no requiere
tocar código.

**Lo que NO se decidió acá:** si conviene re-extraer todo el corpus
chileno con el VLM. Eso invalida los embeddings y las etiquetas del
prefilter ya calculadas sobre el corpus actual, y es una decisión de
costo, no técnica. Lo que este POC establece es que el corpus actual
tiene pérdidas medibles en tablas y contaminación de chrome, y cuánto
costaría arreglarlo.

## Reproducir

```bash
# backend CPU (por defecto, sin GPU)
uv run python scripts/cl/02_extract_text.py --backend pymupdf --limit 2

# un VLM, sin triage (todas las páginas al modelo)
uv run python scripts/cl/02_extract_text.py --backend paddleocr_vl --no-triage --limit 2

# tests
uv run pytest scripts/common/tests/test_pdf_blocks.py
CL_SAMPLE_PDF=<un.pdf> uv run pytest scripts/common/tests/test_pdf_pymupdf_backend.py
```
