# Selección de modelo/proveedor para Pass-1 (pipeline v2)

Decisión de infraestructura, no de metodología: qué modelo LLM, con qué
proveedor de OpenRouter y qué nivel de razonamiento, ejecuta
`scripts/common/ai_classify.py` (extracción de frames) a escala del corpus
completo (~42.000 párrafos únicos). El schema y el prompt están fijados en
`docs/classification_model.md` / el spec v2; este documento solo registra la
comparación empírica de motores de inferencia y la elección resultante.

## Método

Mismos 6 párrafos reales (10-K de AMD, `0000002488-23-000047`, distintos
`paragraph_index`), corridos en paralelo (concurrencia 6) contra cada
candidato, con el `SYSTEM_PROMPT`/schema de `ai_classify.py` vigente al
momento de cada prueba. Métricas: latencia por llamada, tokens de output,
tasa de `valence=model_inconsistent` (ver más abajo), fallas, y una revisión
cualitativa rápida de si el modelo encuentra el mismo contenido de IA que los
demás candidatos en el mismo párrafo (recall) y si fragmenta una sola
proposición en frames duplicados.

n=6 por candidato: indicativo, no estadísticamente definitivo. Suficiente para
descartar candidatos claramente peores (fallas, latencia de minutos, recall
pobre), no para certificar el ganador con precisión.

## Dos defectos de diseño encontrados y corregidos en el camino

1. **`valence` opcional permitía "null" como respuesta gratis.** El campo
   debía llevar un valor solo si el frame tenía un concepto `*_outcome`
   (regla condicional cruzada entre dos campos). Un modelo sin razonamiento
   (`reasoning: none`/`low`) tiende a no verificar esa condición y default
   a omitirlo. Fix: `valence` pasó a ser obligatorio con un cuarto valor
   explícito `not_applicable` -- el modelo siempre elige uno de 4 valores,
   nunca "nada". La inconsistencia real (dice `positive` sin concepto
   outcome, o `not_applicable` con uno presente) se corrige en
   post-procesamiento a un sentinel `model_inconsistent`, que **nunca es
   parte del `Literal` que ve el modelo** (verificado contra
   `AIFrame.model_json_schema()`) -- no puede elegirlo como salida perezosa.
   Antes, la misma inconsistencia lanzaba una excepción de Pydantic y podía
   tirar el frame completo (subject/temporal/concepts correctos incluidos)
   por un solo campo secundario mal puesto.
2. **Sobre-fragmentación: un modelo puede partir una proposición en varios
   frames casi idénticos**, violando la regla del prompt ("mismo subject y
   temporal → un frame"). Fix determinístico post-hoc en
   `merge_duplicate_frames()`: fusiona frames que comparten `subject` +
   `temporal` Y al menos una oración de evidencia (`sentence_ids` se
   solapan) -- criterio estricto a propósito, para no fusionar dos
   proposiciones genuinamente distintas que coincidan en subject/temporal
   por azar. También se deduplican valores repetidos dentro de una misma
   lista (`concepts`/`specificity`/`rhetoric`) vía `_dedupe_lists()`.

## Resultados

| Candidato | Proveedor | Reasoning | Latencia/llamada | `valence` inconsistente | Fallas / recall |
|---|---|---|---|---|---|
| `google/gemini-3.8-flash` | (default) | (default) | ~22s | -- | output descontrolado (hasta 7.473 tokens/llamada por razonamiento oculto) |
| `qwen/qwen3.8-flash` | `makora` | `none` | ~2s | ~25-33% | 1 falla transitoria (no reprodujo aislada) |
| `openai/gpt-5.4-nano` | `openai/flex` | (default) | ~3s | ~9% | fragmentación (arreglada por `merge_duplicate_frames`) |
| `minimax/minimax-m3` | `coreweave/fp4` | (default) | 30-183s | -- | 4/6 fallaron |
| `deepseek/deepseek-v4.1-flash` | `siliconflow/fp8` | `low` | 11-67s | -- | recall pobre (missing `*_outcome` que otros sí encontraban; 2 párrafos con 0 frames) |
| **`openai/gpt-5.6-luna`** | **`amazon-bedrock/us-east-1`** | **`none`** | **~3s** | **0%** | **sin fallas, sin fragmentación patológica** |
| `z-ai/glm-5.3-flash` | `baseten/fp8` | `low` | 8-250s | -- | recall pobre + latencia peor de toda la comparación |

## Decisión

**`openai/gpt-5.6-luna` vía `amazon-bedrock/us-east-1`, `reasoning: {"effort": "none"}`**
queda como `DEFAULT_JUDGE_MODEL` de `scripts/common/ai_classify.py`.

Config exacta (`OpenRouterModelSettings`):
```python
OpenRouterModelSettings(
    openrouter_provider={"order": ["amazon-bedrock/us-east-1"], "allow_fallbacks": True},
    extra_body={"reasoning": {"effort": "none"}},
)
```
`allow_fallbacks: True` a propósito (vs. el `False` usado en pruebas
puntuales) -- para la corrida del corpus completo se prefiere degradar a otro
proveedor de OpenRouter antes que frenar toda la corrida si `amazon-bedrock`
tiene un problema puntual; la fila queda igual sujeta al reintento
estructural (`retries=2`) y al reintento semántico propio, y si falla del
todo se registra como fila de error y se reintenta en la corrida siguiente
(nunca bloquea el resto del batch).

## Resultado de la corrida real (2026-09-12)

Ambas pasadas corrieron sobre el corpus completo, en tándem (pass-2 consume
el output de pass-1 incrementalmente, sin esperar a que termine):

- **Pass-1**: 42.359 textos únicos AI-positivos (población final, tras el
  fix de segmentación de `sentences` -- ver commit `f893a71` y el hallazgo
  de +292/-65 en `is_ai_prefiltered` que ese fix destapó). 87.962 frames.
  1 error permanente (no relacionado al bug de segmentación).
- **Pass-2**: 25.960 textos con frames disparadores. ~59.850 actividades.
  2 errores permanentes.
- **Reloj de pared, ambas fases en tándem**: ~2h20min.

`gpt-5.6-luna` vía `amazon-bedrock/us-east-1` sí se comportó bien en pass-2
(schema y prompt distintos a pass-1) -- no hubo degradación de calidad
visible ni tasa de error anormal en esa tarea. El precio real vía
`amazon-bedrock` específicamente sigue sin confirmarse contra factura, pero
la corrida completa. Ver `thesis_document/thesis.qmd`, Apéndice A, para el
párrafo con estos números.
