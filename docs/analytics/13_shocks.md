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
2023—, no la vaguedad. Definir los grupos con `z(promocional) −
z(especificidad)` pre-evento y después medir esas mismas métricas produce una
convergencia mecánica, no un efecto, y las tendencias previas no pueden ser
paralelas (F=5,11, p=0,001 con ese diseño). Con exposición el tratamiento
sale de cuánto habla la empresa del tema, no de cómo.

**Qué afirma este diseño y qué no.** La propuesta pide, para estas preguntas, un
análisis de **cambio diferencial** ("quasi-causal techniques like DiD *may* be
used"), no la identificación causal del enforcement. El diseño sostiene lo
primero. No sostiene lo segundo: el evento es común en tiempo calendario, así que
cualquier coeficiente post recoge todo lo que le pasó a las empresas expuestas a
IA en esa fecha — el boom de IA generativa incluido. Para separarlos haría falta
variación de canal o de jurisdicción (`docs/pregunta_identificacion_sec.md`).

## Resultado 1: la SEC no produjo un cambio diferencial medible

Ventana ±5 trimestres, 393 empresa-trimestre, 55 empresas:

| outcome | tendencias previas | cambio post |
|---|---|---:|
| `promotional_rate` | **pasa** (F=1,06, p=0,385) | +0,027 (p=0,251) |
| `hypothetical_share` | **pasa** (F=1,63, p=0,180) | −0,007 (p=0,296) |
| `specificity_index` | límite (F=2,18, p=0,083) | +0,015 (p=0,109) |
| `risk_share` | falla (F=3,55, p=0,012) | — |

**Donde el diseño es utilizable, el cambio es cero.** Las empresas más expuestas
a IA no se volvieron ni más ni menos promocionales, ni más ni menos hipotéticas,
que las menos expuestas después de marzo de 2024. `specificity_index` da
+0,015 con p=0,109: tampoco.

### Y la sensibilidad, que es la parte importante

Con una ventana ancha aparece un efecto de **+6,0 p.p. (p=0,03)** para
`promotional_rate`. Barriendo el ancho de ventana:

| ventana | empresas | tendencias previas | cambio post |
|---:|---:|---|---:|
| ±3 | 24 | **falla** (p=0,034) | −0,020 (p=0,59) |
| ±4 | 34 | pasa (p=0,682) | +0,014 (p=0,79) |
| ±5 | 55 | pasa (p=0,385) | +0,027 (p=0,25) |
| ±6 | 63 | pasa (p=0,293) | +0,052 (p=0,19) |
| ±7 | 91 | **falla** (p=0,021) | **+0,060 (p=0,03)** |

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
| ±3 | 35 | falla (p=0,020) |
| ±4 | 51 | falla (p<0,001) |
| ±5 | 73 | falla (p<0,001) |
| ±6 | 74 | falla (p<0,001) |
| ±7 | 79 | falla (p<0,001) |

Las tendencias previas fallan en **todas** las ventanas para
`promotional_rate`, `specificity_index` e `hypothetical_share`. Los grupos ya
venían divergiendo antes de enero de 2025 — algo esperable, porque el corte cae
en medio de la difusión de la IA generativa, cuando las empresas expuestas se
estaban separando del resto por su cuenta. El único outcome que pasa
(`risk_share`, p=0,717) da +0,019 con p=0,173: cero.

**No es que DeepSeek no haya tenido efecto: es que este diseño no puede
distinguirlo de la tendencia.** Reportarlo como efecto sería exactamente el
error de la ventana ±7 de arriba.

## Resultado 3: los segmentos no responden distinto

Cambio post en `promotional_rate` por segmento de divulgación
(`11_segmentacion.md`), evento SEC:

| segmento | cambio post | p | n (empresa-trimestre) |
|---|---:|---:|---:|
| adoptantes con gobernanza | +0,010 | 0,821 | 72 |
| desplegadores de producto | −0,004 | 0,883 | 312 |
| listadores de riesgo | −0,112 | 0,107 | 9 |

Ninguno significativo. El −11 p.p. de los listadores de riesgo tiene 9
observaciones detrás: es una anécdota, no un hallazgo.

## Resultado 4: margen extensivo — con las empresas que no hablan de IA adentro, el diseño no identifica

Todo lo anterior condiciona a ≥3 frames por empresa-trimestre: la empresa
que deja de hablar de IA sale del panel en vez de contar como cero.
`shock_analysis.py --margin extensive` y `shock_did_simple.py --margin
extensive` corren los mismos diseños sobre **todos los filings** (10-K,
10-Q, DEF 14A, 8-K) por empresa-trimestre, con intensidad por 1.000
párrafos y cero cuando el documento no habla de IA (`ai_intensity.py`):
11.302 empresa-trimestre, 510 empresas; 5.355 y 488 en la ventana del
evento SEC. Exposición = frames de IA por 1.000 párrafos antes de 2023.

| outcome por 1.000 párrafos | tendencias previas | cambio post (SEC) |
|---|---|---:|
| promocionales | **falla** (p=0,024) | +0,40 |
| especificidad | pasa (p=0,20) | +0,70 |
| riesgo | **falla** (p=0,003) | +0,64 |
| hipotético | **falla** (p<0,001) | +0,44 |
| frames de IA | **falla** (p=0,002) | +3,6 |

DiD simple (grupo = mitad más promocional por párrafo antes de 2024; 108
tratadas, 381 controles, 5.364 empresa-trimestre):

| outcome por 1.000 párrafos | DiD | p | tendencias previas |
|---|---:|---:|---|
| promocionales *(dimensión del grupo)* | +0,32 | 0,040 | **falla** (p=0,003) |
| cuantificados | +0,29 | 0,140 | límite (p=0,058) |
| gobernanza | +0,30 | 0,037 | **falla** (p=0,017) |
| especificidad | +0,60 | 0,007 | **falla** (p<0,001) |

**En el margen extensivo las tendencias previas fallan casi en todo.** Las
empresas más expuestas venían aumentando su intensidad de IA relativa desde
antes de 2024, en todas las dimensiones, porque el margen extensivo es
donde vive el boom de 2023: el 36% de las empresas-año de 2021 hablaba de
IA y el 92% de 2025. Un antes/después de marzo de 2024 sobre esa curva
captura la curva. El único outcome con tendencias previas planas
(especificidad por párrafo, +0,70) tiene su salto en t+3, que es 2025Q1: la
temporada de 10-K, no el evento. El diseño de arriba, condicionado a hablar
de IA, es el que puede decir algo sobre el regulador precisamente porque
normaliza por cuánto se habla; el extensivo describe el boom y no puede
separarlo del escrutinio.

## Qué queda para la tesis

1. **La respuesta a la pregunta extendida es un nulo bien medido**: no hay
   evidencia de cambio diferencial en la divulgación de IA alrededor del
   escrutinio de la SEC, en las especificaciones donde el diseño se sostiene.
   Un nulo con el supuesto testeado y reportado vale más que un efecto que
   depende de la ventana.
2. **DeepSeek queda fuera de alcance** con este diseño, y se dice.
3. Lo que sí sería identificable —y no está hecho— es el **contraste por canal**:
   la misma empresa, el mismo trimestre, filings (con responsabilidad legal)
   contra earnings calls (sin ella). Las transcripciones ya están clasificadas
   (16.267 frames de 403 empresas, `earnings_calls_analysis.py`) y el diseño
   empresa × trimestre × canal está en `14_brecha_entre_canales.md`: la brecha
   promocional entre call y filing es de +10 p.p. y el escrutinio de la SEC
   no la cerró (b=+0,015, p=0,36); lo que cambió después de 2024 es que el
   filing agregó gobernanza de IA, no que la call bajara el tono.

## Limitaciones

- 55 empresas en la ventana utilizable del evento SEC: la potencia para detectar
  un efecto chico es baja, así que "no hay evidencia de cambio" no es "no hubo
  cambio".
- El outcome es una etiqueta de LLM sin validación humana
  (`docs/problemas_academicos.md` #1).
- El panel exige ≥3 frames por empresa-trimestre, así que la entrada al panel es
  endógena a cuánto habla la empresa de IA ese trimestre (#12).

---

# Versión simple: dos grupos, pre/post, con figura

`scripts/analytics/shock_did_simple.py`. Es la especificación que se lleva a la
tesis, porque se explica en dos líneas y es la que la propuesta pide como
extensión:

```
Y[i,t] = a[i] + lambda[t] + beta * (AltoRiesgo[i] x Post[t]) + e[i,t]
```

`AltoRiesgo` = la mitad de empresas cuya divulgación de IA **pre-2024** era más
promocional que la mediana. `Post` = desde 2024Q2. Efectos fijos de empresa y de
trimestre, errores clusterizados por empresa, ventana ±5 trimestres. **40
empresas de alto riesgo, 26 de bajo, 462 empresa-trimestre.**

**La decisión que hace que el diseño funcione**: el grupo se define con una
dimensión (retórica promocional) y los outcomes se miden en OTRAS —
cuantificación, gobernanza, especificidad. Definir el grupo por el nivel
pre-evento de la misma variable que después se mide produce convergencia
mecánica: el grupo alto sólo puede bajar, haya pasado algo o no.
`promotional_rate` se reporta igual, pero **como control interno**: es donde el
artefacto debe aparecer.

| outcome | DiD | SE | p | tendencias previas (p) |
|---|---:|---:|---:|---:|
| `promotional_rate` *(misma dimensión que el grupo)* | **−9,67 p.p.** | 2,33 | **0,000** | 0,120 |
| `quantified_rate` | +3,37 p.p. | 4,32 | 0,435 | 0,580 |
| `gov_share` | +2,74 p.p. | 2,15 | 0,202 | 0,117 |
| `specificity_index` | −0,35 p.p. | 1,63 | 0,829 | 0,980 |

## Cómo se lee

**En la dimensión que define los grupos hay convergencia grande (−9,7 p.p.) y en
las demás no pasa nada.** Y la trayectoria por trimestre muestra por qué hay que
desconfiar de la primera: el diferencial ya venía bajando **antes** del evento
(+8,8\* → +8,1 → +4,9 → −4,8 en los cuatro trimestres previos), con el test
conjunto de tendencias previas en p=0,120 — no falla al 5%, pero tampoco es
plano. Es el perfil de una reversión a la media que empezó antes del corte.

En cambio `quantified_rate` (p=0,580) y `specificity_index` (p=0,980) tienen
tendencias previas planas y limpias, y ahí el efecto es **cero**.

**La afirmación defendible para la tesis:**

> Después del escrutinio de la SEC, las empresas que ex ante se veían más
> expuestas a AI-washing **no se volvieron más concretas, ni más cuantificadas,
> ni más orientadas a gobernanza** que las demás. Su tono promocional converge
> hacia el del resto, pero esa convergencia ya estaba en marcha antes del evento.

Es un resultado, no un no-resultado: la hipótesis natural —"el escrutinio empuja
a las vagas hacia divulgación más sustantiva"— **no se verifica en las
dimensiones donde el diseño se sostiene**.

## Las figuras

`data/processed/clusters/shock_did_<outcome>.png`: coeficiente estimado por
trimestre con intervalo de 95%, normalizado a t−1. Se grafica **lo que el modelo
estima** (la diferencia entre grupos ajustada por efectos fijos), no medias
crudas — con 66 empresas, las medias crudas son puro ruido y no muestran el
supuesto. En la figura se lee de una si las barras previas cruzan el cero y si
las posteriores se despegan.

## Detalle técnico que hay que dejar escrito

La primera versión del event study usaba `C(ev, Treatment(reference=-1))` sobre
un categórico de enteros y **patsy ignoraba la referencia**: estimaba todos los
períodos, el coeficiente de t−1 salía +9,7 p.p. en vez de cero, y el test de
tendencias previas heredaba ese salto de nivel (daba p=0,000 cuando en realidad
es 0,120). Corregido con dummies explícitas y el período omitido a mano.
