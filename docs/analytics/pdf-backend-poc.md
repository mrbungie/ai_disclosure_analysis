# Qué tan malo es el extractor de PDFs chileno, y qué motor conviene

Dos preguntas distintas, medidas por separado:

1. **¿El baseline (PyMuPDF + heurísticas) funciona?** Auditoría sobre 125
   documentos / **12.381 páginas** reales, 92 empresas.
2. **¿Qué VLM conviene si hay que reemplazarlo?** POC sobre PaddleOCR-VL
   1.6, MinerU2.5-Pro y dots.mocr.

**Resumen ejecutivo, y va en contra de la primera versión de este
documento:** el baseline no pierde texto narrativo de forma masiva. Sobre
páginas difíciles, al baseline le falta el **2,8 %** de las cifras que lee
un VLM, y de hecho captura *más* tokens en total. Lo que rompe es la
**estructura**: qué número pertenece a qué columna, qué párrafo sigue a
cuál, qué dato es de qué director. Y hay un **2,5 % de páginas que
devuelve completamente vacías**, de las cuales ~12 % tienen contenido real.

---

## Parte 1 — Auditoría del baseline (12.381 páginas)

`scripts/verif/pdf_audit/baseline_audit.py`. Muestra estratificada: 45
Memorias + 80 Análisis Razonados, una por empresa, 92 empresas.

### Páginas que el baseline no puede ver

| | páginas | % |
|---|---|---|
| sin capa de texto pero con tinta | 265 | **2,1 %** |
| genuinamente en blanco | 2 | 0,0 % |
| que no producen ningún párrafo | 306 | **2,5 %** |

No hay error, no hay párrafo vacío: la página simplemente no existe aguas
abajo. **47 de 125 documentos (37,6 %) tienen al menos una.**

Corrí PaddleOCR-VL sobre 60 de esas páginas invisibles, de 34 empresas,
para ver qué hay realmente en ellas:

| | páginas | % |
|---|---|---|
| decorativa (foto, portada) — 0-4 palabras | 43 | 71,7 % |
| poco texto — 5-29 palabras | 10 | 16,7 % |
| **contenido real — 30+ palabras** | **7** | **11,7 %** |

O sea: **~0,25 % de las páginas del corpus pierden texto sustantivo en
silencio**, y ~0,6 % pierde algo. Es poco, pero no es cero, y lo que se
pierde no es decorativo:

- `90227000/memoria_2025` pág. 214 — 336 palabras, dictamen de auditoría
  ("*Concluimos sobre lo adecuado de la utilización, por la
  Administración, de la base contable de empresa en marcha…*")
- `93007000/memoria_2025` pág. 216 — **DECLARACIÓN DE RESPONSABILIDAD** de
  SQM, escaneada. Documento regulatorio obligatorio, invisible.
- `96542300/memoria_2022` pág. 56 — **SUSCRIPCIÓN DE LA MEMORIA** bajo NCG
  N° 30, escaneada.

### Daño estructural en el corpus emitido

| | valor | denominador |
|---|---|---|
| páginas multi-columna | 7.038 | **56,8 %** de páginas |
| páginas con tabla | 5.332 | 43,1 % |
| **páginas expuestas (una u otra)** | 8.969 | **72,4 %** |
| **tablas con celdas perdidas** | 3.410 | **9,9 %** de las tablas |
| filas de tabla incompletas | 28.189 | 34,4 % de las filas |
| chrome emitido como párrafo | 140 | 0,2 % de la prosa |
| marcador incrustado en una frase | 74 | 0,1 % de la prosa |

Por documento: **84 % tiene al menos una tabla con celdas perdidas**,
44,8 % al menos una página que no produce nada, 20,8 % al menos un
marcador incrustado dentro de una frase.

Memorias vs Análisis Razonados: expuestas 76,9 % vs 48,5 %; invisibles
2,4 % vs 0,8 %. Las Memorias glossy son el problema.

### Head-to-head sobre 50 páginas "en riesgo" (45 empresas)

Baseline vs MinerU2.5-Pro, mismas páginas, muestreadas al azar entre las
multi-columna o con tabla:

| | valor |
|---|---|
| páginas donde el VLM lee cifras que el baseline no tiene | 10 / 50 (**20 %**) |
| ... con 3 o más cifras faltantes | 3 / 50 (6 %) |
| cifras sólo-VLM / sólo-baseline / total | 39 / **58** / 1.384 |
| **al baseline le falta** | **2,8 %** de las cifras |
| palabras totales | baseline 19.943, VLM 13.957 |

**Este es el resultado que corrige la versión anterior de este
documento.** El baseline captura *más* tokens que el VLM (lee la capa de
texto, no se le escapa un glifo; el VLM además descarta chrome a
propósito). Su problema no es perder texto — es que el texto que entrega
está mal ordenado y mal asociado.

Los tres ejemplos concretos, verificados a mano sobre la Memoria 2024 de
Banco de Chile, siguen siendo válidos y son de *estructura*, no de
cobertura:

1. **Columnas de tabla mal asignadas** (pág. 440). Los 52 números están
   en el output; la fila queda como `18.893.992 | 21.207.713 |
   24.766.475` — tres valores donde hay cuatro columnas, sin etiqueta de
   fila. Los dígitos están; saber a qué corresponden, no.
2. **Marcador de margen dentro de una frase** (pág. 36, la página con la
   narrativa de IA): "*…altos niveles de participación en los 3.6.ix
   procesos de adaptación competitiva…*".
3. **Atributos asignados al director equivocado** (pág. 40): "Nacionalidad:
   Chileno / Edad: 66 años" termina pegado a otra persona.

### Qué significa para esta tesis

El objeto de estudio son **oraciones de disclosure narrativo**, no tablas.
Sobre prosa el baseline es mediocre pero no catastrófico: 0,2 % de chrome
y 0,1 % de marcadores incrustados. El daño se concentra en tablas
(9,9 % con celdas perdidas), que este trabajo casi no usa.

**Por lo tanto: re-extraer el corpus entero con un VLM no está
justificado por esta evidencia.** Lo que sí está justificado, y es barato:

- recuperar las ~0,25-0,6 % de páginas escaneadas con contenido real
  (declaraciones de responsabilidad, dictámenes) — son ~1.000-2.000
  páginas del corpus, no 120.000;
- correr el VLM sólo sobre tablas si en algún momento se usan.

---

## Parte 2 — Cuál VLM

### Calidad (6 páginas de la Memoria 2024 de Banco de Chile)

| | baseline | PaddleOCR-VL 1.6 | MinerU2.5-Pro | dots.mocr |
|---|---|---|---|---|
| fugas de chrome | 3 | **0** | **0** | 7 |
| celdas de tabla (pág. 440) | 42/52 | **52/52** | **52/52** | **52/52** |
| KPIs de infografía (pág. 17) | 10/11 | 7/11 | 7/11 | 0/11 |
| fidelidad de transcripción | 1,0 por construcción | 0,9995 | 0,9997 | 0,9997 |
| s/página (régimen permanente) | 0,042 | **5,95** | ~10,1 | ~11 |

**Estas 6 páginas las elegí yo, y sobreestiman dos fallas.** Las fugas de
chrome fueron 3 en 6 páginas acá, pero 0,2 % de la prosa en el corpus. No
generalizar desde esta tabla; para eso está la Parte 1.

### Confiabilidad — y por qué el ganador de la tabla no es el default

PaddleOCR-VL corre su VLM en un worker interno. Ese worker **se cuelga**:
sin error, sin timeout propio, GPU al 0 %, indefinidamente. Y no se
recupera — el mensaje es `VLM worker did not terminate in time` y a partir
de ahí el objeto pipeline está muerto, así que **un cuelgue cuesta el
resto de la corrida, no una página.**

| | páginas intentadas | cuelgues |
|---|---|---|
| PaddleOCR-VL 1.6 | ~125 | **2 (~1,6 %)** |
| MinerU2.5-Pro | 50 | **0** |

Una de las páginas donde se colgó es un estado de resultados
absolutamente común de Coca-Cola Embonor
(`93281000/analisis_razonado_202412` pág. 0) — no un caso patológico.

A 1,6 % sobre 120.000 páginas son ~2.000 cuelgues, cada uno matando la
corrida. **Por eso el default es `mineru`**, aunque sea ~2x más lento y
haya perdido la tabla de calidad por poco. `paddleocr_vl` sigue
disponible por config y es la mejor opción para una corrida corta que
estés mirando.

De ahí también sale `scripts/common/pdf/watchdog.py`: deadline por página,
para que un cuelgue cueste una página y una línea de log.

### Costo a escala (corregido)

La primera versión de este documento dijo "~350 h" a partir de un
estimado de 300.000 páginas extrapolado de **dos** PDFs. Con 12.381
páginas medidas:

- Memorias 232 págs/doc (n=45), AR 24,3 págs/doc (n=80)
- Corpus = 576 × 232 + 1.536 × 24,3 = **~171.000 páginas** (sobreestimé 1,8x)
- El triage manda **~120.000 (71 %)** al VLM

| escenario | horas (con triage) |
|---|---|
| single-stream medido, paddleocr_vl 5,95 s/pág | 200 h |
| single-stream medido, mineru 10,1 s/pág | 339 h |
| si vLLM da 3x | 67-113 h |
| si vLLM da 6x | 33-57 h |

**El single-stream es el peor caso y nadie lo correría así**: es batch=1
sobre un modelo de 0,9-1,2B en una 5090 que usa 12 de 32 GB. Cuánto da
vLLM realmente **no está medido** — es el siguiente paso si alguna vez se
decide re-extraer.

---

## Reproducir

```bash
# auditoría del baseline (CPU, sin GPU)
uv run python scripts/verif/pdf_audit/baseline_audit.py \
    --pdf-dir data/raw/filings_pdf_cl --out audit.json

# extracción con un backend concreto
uv run python scripts/cl/02_extract_text.py --backend pymupdf --limit 2
uv run python scripts/cl/02_extract_text.py --backend mineru --no-triage --limit 2

uv run pytest scripts/common/tests/test_pdf_blocks.py
CL_SAMPLE_PDF=<un.pdf> uv run pytest scripts/common/tests/test_pdf_pymupdf_backend.py
```

## Limitaciones de esta medición

- Muestra de 125 de 2.112 documentos (6 %). Las tasas tienen error de
  muestreo; ninguna conclusión acá depende de la tercera cifra
  significativa.
- `chrome_paragraphs` y `spliced_paragraphs` usan regexes ajustadas a las
  plantillas CMF/GRI. Son **cotas inferiores**: un emisor con otra
  plantilla de chrome no se cuenta.
- `ragged_tables` detecta filas más cortas que la más ancha de su tabla.
  Una tabla con celdas combinadas legítimas cuenta como rota, así que
  el 9,9 % también es aproximado — inspeccionadas a mano, las que miré
  eran daño real.
- El head-to-head compara contra el VLM, no contra ground truth humano.
  Mide desacuerdo, no verdad.
