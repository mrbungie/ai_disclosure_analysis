# Preguntas extendidas: ¿cambió la divulgación de IA tras los shocks de 2024-2025?

Responde las preguntas extendidas de `docs/thesis_proposal.md`:

> ¿Cambiaron las empresas su divulgación después del escrutinio de la SEC sobre
> AI-washing (marzo 2024)? ¿Afectó DeepSeek (enero 2025) cómo enmarcan sus
> capacidades?

Producido por `scripts/analytics/shock_analysis.py`. Determinístico, sin LLM.

## El diseño

Event study empresa-trimestre:

```
y[i,t] = a[i] + t[q] + SUM_k b[k] · 1{tiempo_evento = k} · exposición[i]
         + mezcla_documental[i,t] + log(frames)[i,t] + e[i,t]
```

con efectos fijos de empresa y de trimestre, panel balanceado (≥2 trimestres a
cada lado), errores estándar clusterizados por empresa, y `t−1` como trimestre de
referencia. Los coeficientes previos al evento **son** el test del supuesto, y se
evalúan con un test conjunto F, no uno por uno.

**El tratamiento es la exposición previa a IA** —volumen de frames antes de
2023—, no la vaguedad. La versión original definía los grupos con
`z(promocional) − z(especificidad)` pre-evento y después medía esas mismas
métricas: la convergencia resultante es mecánica, no un efecto, y las tendencias
previas no pueden ser paralelas (F=5,11, p=0,001). Con exposición el tratamiento
sale de cuánto habla la empresa del tema, no de cómo.

**Qué afirma este diseño y qué no.** La propuesta pide, para estas preguntas, un
análisis de **cambio diferencial** ("quasi-causal techniques like DiD *may* be
used"), no la identificación causal del enforcement. El diseño sostiene lo
primero. No sostiene lo segundo: el evento es común en tiempo calendario, así que
cualquier coeficiente post recoge todo lo que le pasó a las empresas expuestas a
IA en esa fecha — el boom de IA generativa incluido. Para separarlos haría falta
variación de canal o de jurisdicción (`docs/pregunta_identificacion_sec.md`).

## Resultado 1: la SEC no produjo un cambio diferencial medible

Ventana ±5 trimestres, 388 empresa-trimestre, 54 empresas:

| outcome | tendencias previas | cambio post |
|---|---|---:|
| `promotional_rate` | **pasa** (F=1,26, p=0,298) | +0,025 (p=0,336) |
| `hypothetical_share` | **pasa** (F=1,73, p=0,156) | −0,007 (p=0,355) |
| `specificity_index` | límite (F=2,18, p=0,084) | +0,015 (p=0,132) |
| `risk_share` | falla (F=3,31, p=0,017) | — |

**Donde el diseño es utilizable, el cambio es cero.** Las empresas más expuestas
a IA no se volvieron ni más ni menos promocionales, ni más ni menos hipotéticas,
que las menos expuestas después de marzo de 2024.

### Y la sensibilidad, que es la parte importante

Una versión anterior de este análisis reportó **+8,4 p.p. (p<0,001)** para
`promotional_rate`. No sobrevive. Barriendo el ancho de ventana:

| ventana | empresas | tendencias previas | cambio post |
|---:|---:|---|---:|
| ±3 | 24 | **falla** (p=0,047) | −0,020 (p=0,60) |
| ±4 | 34 | pasa (p=0,733) | +0,013 (p=0,79) |
| ±5 | 54 | pasa (p=0,298) | +0,025 (p=0,34) |
| ±6 | 62 | pasa (p=0,279) | +0,049 (p=0,24) |
| ±7 | 90 | **falla** (p=0,011) | **+0,057 (p=0,03)** |

**El efecto sólo aparece con la ventana en la que el test de tendencias previas
empieza a fallar.** Eso es la firma de contaminación por tendencia, no de un
evento: al abrir la ventana entran trimestres de 2022 y de 2026 donde los grupos
ya venían separándose, y esa separación se cuela en el coeficiente post. En las
tres ventanas donde el supuesto se sostiene, el efecto no es distinguible de
cero.

Lección metodológica, y vale para toda la tesis: **un resultado que aparece
exactamente cuando el test de identificación falla no es un resultado.**

## Resultado 2: sobre DeepSeek, este diseño no puede decir nada

| ventana | empresas | tendencias previas |
|---:|---:|---|
| ±3 | 35 | falla (p=0,021) |
| ±4 | 50 | falla (p=0,001) |
| ±5 | 72 | falla (p<0,001) |
| ±6 | 73 | falla (p<0,001) |
| ±7 | 77 | falla (p<0,001) |

Las tendencias previas fallan en **todas** las ventanas para
`promotional_rate`, `specificity_index` e `hypothetical_share`. Los grupos ya
venían divergiendo antes de enero de 2025 — algo esperable, porque el corte cae
en medio de la difusión de la IA generativa, cuando las empresas expuestas se
estaban separando del resto por su cuenta. El único outcome que pasa
(`risk_share`, p=0,786) da +0,029 con p=0,153: cero.

**No es que DeepSeek no haya tenido efecto: es que este diseño no puede
distinguirlo de la tendencia.** Reportarlo como efecto sería exactamente el
error de la ventana ±7 de arriba.

## Resultado 3: los segmentos no responden distinto

Cambio post en `promotional_rate` por segmento de divulgación
(`11_segmentacion.md`), evento SEC:

| segmento | cambio post | p | n (empresa-trimestre) |
|---|---:|---:|---:|
| adoptantes con gobernanza | +0,014 | 0,740 | 72 |
| desplegadores de producto | −0,003 | 0,910 | 307 |
| listadores de riesgo | −0,111 | 0,116 | 9 |

Ninguno significativo. El −11 p.p. de los listadores de riesgo tiene 9
observaciones detrás: es una anécdota, no un hallazgo.

## Qué queda para la tesis

1. **La respuesta a la pregunta extendida es un nulo bien medido**: no hay
   evidencia de cambio diferencial en la divulgación de IA alrededor del
   escrutinio de la SEC, en las especificaciones donde el diseño se sostiene.
   Un nulo con el supuesto testeado y reportado vale más que un efecto que
   depende de la ventana.
2. **DeepSeek queda fuera de alcance** con este diseño, y se dice.
3. Lo que sí sería identificable —y no está hecho— es el **contraste por canal**:
   la misma empresa, el mismo trimestre, filings (con responsabilidad legal)
   contra earnings calls (sin ella). Los 483.214 párrafos de transcripciones ya
   están extraídos pero sin procesar. Ver `docs/pregunta_identificacion_sec.md`.

## Limitaciones

- 54 empresas en la ventana utilizable del evento SEC: la potencia para detectar
  un efecto chico es baja, así que "no hay evidencia de cambio" no es "no hubo
  cambio".
- El outcome es una etiqueta de LLM sin validación humana
  (`docs/problemas_academicos.md` #1).
- El panel exige ≥3 frames por empresa-trimestre, así que la entrada al panel es
  endógena a cuánto habla la empresa de IA ese trimestre (#12).
