# Señal incremental: ¿el contenido aporta información más allá del volumen y de los fundamentals?

Es el análisis central de la tesis (RQ4). Producido por
`scripts/analytics/shock/incremental_signal.py` sobre todas las empresas-año con
filings, ejercicios 2021-2025 (`firm_year_master_v2.parquet`; 2.475 filas,
508 empresas, 59% con algún frame de IA). Financieros winsorizados 1/99.

## Diseño

Para cada outcome Y, cuatro modelos anidados con efectos fijos sector × año:

| modelo | qué sabe el modelo |
|---|---|
| M0 | fundamentals: log market cap, margen bruto, margen operativo, rotación de activos |
| M1 | M0 + **volumen**: log(1 + frames de IA por 1.000 párrafos) |
| M2 | M1 + **contenido**: log(1 + x por 1.000 párrafos) para realizado, despliegue, capacidad, riesgo, gobernanza, promocional, especificidad |
| M3 | M1 + dummies de segmento (`02_segmentacion.md`) |

Lo que se reporta es **ΔR² = R²(M2) − R²(M1)** —cuánto agregan las palabras
una vez observados sus fundamentals, sector, año y volumen de divulgación de
IA— y el R² parcial del bloque semántico, (R²M2 − R²M1)/(1 − R²M1), con
intervalo bootstrap por empresa (300 réplicas). Tres controles sobre el
mismo ΔR², porque siete regresores más nunca bajan el R² crudo:

- **R² ajustado** de M0–M3, contando las celdas sector × año absorbidas como
  parámetros. Si el bloque semántico no aportara, ΔR² ajustado sería ≤ 0.
- **Test conjunto de Wald** (F con SE cluster por empresa) de H0: los siete
  coeficientes semánticos son cero en M2.
- **Test de permutación**: 200 réplicas permutando las siete features entre
  empresas-año DENTRO de cada celda sector × año. Conserva sector, año,
  fundamentals y volumen; destruye sólo la asociación entre contenido y
  outcome. p = fracción de ΔR² nulos ≥ el observado.

Cinco outcomes, ninguno más: beta, **volatilidad idiosincrática** (desviación
estándar de los residuos del modelo de mercado sobre los 252 días previos al
filing, anualizada; `build_market_factors.py`), P/S, R&D/ventas, crecimiento
de ingresos t+1. La volatilidad total pre-filing queda como robustez.

## Resultado

| outcome | n | R² M0 | R² M1 | R² M2 | ΔR² volumen | **ΔR² contenido** | IC 95% bootstrap | ΔR² aj. contenido | R² parcial | Wald F (7) | p Wald | p perm. | ΔR² segmentos |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| beta | 1.037 | 0,039 | 0,078 | 0,110 | +0,039 | **+0,032** | [+0,019, +0,085] | +0,029 | 3,5% | 3,08 | 0,004 | 0,005 | +0,010 |
| volatilidad idiosincrática | 1.037 | 0,182 | 0,198 | 0,223 | +0,016 | **+0,025** | [+0,015, +0,059] | +0,022 | 3,1% | 3,76 | 0,001 | 0,005 | +0,007 |
| P/S | 1.039 | 0,233 | 0,234 | 0,261 | +0,000 | **+0,027** | [+0,015, +0,079] | +0,024 | 3,5% | 2,98 | 0,005 | 0,005 | +0,001 |
| R&D / ventas | 649 | 0,453 | 0,497 | 0,519 | +0,044 | **+0,021** | [+0,011, +0,070] | +0,017 | 4,2% | 2,51 | 0,019 | 0,005 | +0,005 |
| crecimiento ingresos t+1 | 1.003 | 0,021 | 0,043 | 0,066 | +0,022 | **+0,023** | [+0,007, +0,088] | +0,018 | 2,4% | 1,89 | 0,072 | 0,015 | +0,012 |

R² ajustado (M0 / M1 / M2 / M3): beta −0,098 / −0,054 / −0,026 / −0,047;
volatilidad idiosincrática 0,065 / 0,083 / 0,105 / 0,088; P/S 0,125 / 0,124 /
0,149 / 0,123; R&D 0,409 / 0,455 / 0,472 / 0,458; crecimiento −0,122 /
−0,099 / −0,081 / −0,089. (Negativos en beta y crecimiento porque las 200-y-
tantas celdas sector × año cuestan más de lo que explican; lo que importa es
que M2 sube el ajustado en los cinco.) El ΔR² nulo por permutación tiene
media +0,005 a +0,009 y percentil 95 entre +0,010 y +0,017: el observado lo
supera en los cinco outcomes.

**El contenido semántico agrega entre 2 y 4 puntos de R² parcial sobre
fundamentals y volumen, en los cinco outcomes.** Los intervalos bootstrap
excluyen el cero, el ΔR² ajustado es positivo en todos, el Wald conjunto
rechaza a 5% en cuatro (crecimiento a 7%) y la permutación dentro de sector ×
año rechaza en los cinco. Es del mismo orden que lo que agrega el volumen
(0-4 puntos) y más que lo que agregan los segmentos como dummies (0-1
puntos). Modesto, no cero, y consistente.

Qué dimensión lo carga (coeficientes estandarizados de M2, SE cluster por
empresa):

| outcome | dimensión con más peso | β estandarizado | p |
|---|---|---:|---:|
| beta | especificidad | +0,37 | <0,01 |
| volatilidad idiosincrática | especificidad | +0,32 | <0,01 |
| P/S | especificidad | +0,45 | <0,01 |
| R&D / ventas | despliegue (−), volumen (+) | −0,24 / +0,25 | 0,03 / 0,09 |
| crecimiento ingresos t+1 | especificidad | +0,37 | 0,01 |

**La señal es la especificidad**: cuánto de lo que la empresa dice de IA
viene con procesos, productos, proveedores, cifras o fechas. Dado el volumen,
la empresa que habla de IA en concreto tiene más riesgo idiosincrático, es
más cara y crece más que la que habla de IA en abstracto. La promoción no
aporta (β ≈ −0,1 en P/S, p=0,07; cero en el resto); la gobernanza resta beta
(−0,12, p=0,04).

## Robustez

Mismo estimador, ΔR² del bloque semántico (entre paréntesis el p del Wald
conjunto):

| outcome | principal | composición (shares) | sólo 10-K | sin IT ni comunicaciones (SIC 35/36/48/73) | efectos fijos de empresa |
|---|---:|---:|---:|---:|---:|
| beta | +0,032 (0,004) | +0,031 (0,014) | +0,031 (0,001) | +0,056 (0,002) | +0,022 (0,005) |
| volatilidad idiosincrática | +0,025 (0,001) | +0,019 (0,091) | +0,026 (0,001) | +0,063 (0,022) | +0,010 (0,197) |
| P/S | +0,027 (0,005) | +0,014 (0,111) | +0,015 (0,078) | +0,012 (0,516) | +0,006 (0,188) |
| R&D / ventas | +0,021 (0,019) | +0,023 (0,029) | +0,021 (0,042) | +0,028 (0,013) | +0,008 (0,281) |
| crecimiento ingresos t+1 | +0,023 (0,072) | +0,012 (0,259) | +0,034 (0,007) | +0,007 (0,407) | +0,012 (0,016) |
| volatilidad total pre-filing (60 d) | +0,021 (0,068) | — | +0,021 (0,055) | — | — |

- **Composición (shares)**: el bloque semántico entra como frames_k / frames
  de IA, encogido hacia la media del corpus con el prior empírico-Bayes de
  `02_segmentacion.md` (sin frames, el share queda en el prior), con el
  volumen aparte. Separa "cómo se reparte" de "cuánto". Sobrevive en beta y
  R&D; en volatilidad, P/S y crecimiento el ΔR² baja a la mitad y el Wald
  deja de rechazar. Parte de la señal de contenido, por tanto, es
  intensidad de cada dimensión y no sólo su mezcla; la mezcla sola sigue
  aportando en riesgo sistemático e I+D.
- **Sólo 10-K**: igual. No es el proxy ni el 8-K.
- **Sin IT ni comunicaciones**: igual o mayor en riesgo (beta, volatilidad);
  menor en valuación y crecimiento. La señal no es "software vs. el resto".
- **Efectos fijos de empresa**: la mitad o menos, y el Wald sólo rechaza en
  beta y crecimiento. Como en todo el proyecto, la mayor parte de la
  información del texto es transversal —separa empresas— y una parte menor,
  pero no nula, es within-firm.
- **Volatilidad total** (60 días pre-filing, sin descontar el mercado):
  misma magnitud que la idiosincrática, algo menos precisa.
- Winsorización 1/99 en todo; sector × año en todo salvo la variante de
  efectos fijos de empresa.

## ¿Estilo semántico o actividad identificable?

El bloque semántico mide cómo se documenta la IA (realizado, específico,
promocional…). Las actividades divulgadas (`09_actividades_ia.md`) miden
qué se dice hacer. Sobre la misma muestra principal, seis familias de
actividad por 1.000 párrafos de los filings del año, con ceros (despliegue
a clientes, despliegue interno, IA propia, proveedor nombrado,
infraestructura, resultado cuantificado; log1p), entran como bloque
alternativo (M1 + actividades) y como bloque adicional (M1 + semántica +
actividades). ΔR² sobre M1; p del Wald conjunto con SE cluster por empresa:

| outcome | semántica | actividades | ambos | actividades dado semántica | semántica dado actividades | p act. | p act. dado sem. | p sem. dado act. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| beta | +0,032 | +0,028 | **+0,051** | +0,019 | +0,023 | 0,012 | 0,055 | 0,021 |
| volatilidad idiosincrática | +0,025 | +0,030 | **+0,039** | +0,014 | +0,009 | <0,001 | 0,131 | 0,504 |
| P/S | +0,027 | +0,023 | **+0,044** | +0,017 | +0,022 | 0,034 | 0,029 | 0,008 |
| R&D / ventas | +0,021 | +0,020 | **+0,037** | +0,015 | +0,017 | 0,002 | 0,008 | 0,025 |
| crecimiento ingresos t+1 | +0,023 | +0,052 | **+0,063** | +0,040 | +0,011 | 0,042 | 0,035 | 0,235 |

Los dos bloques aportan, y aportan cosas distintas: juntos suben el ΔR² a
4-6 puntos, y cada uno conserva la mitad o más de su incremento cuando el
otro ya está, con dos excepciones simétricas. En volatilidad idiosincrática
la actividad identificable absorbe la mayor parte del estilo (semántica dado
actividades +0,009, p=0,50): el riesgo idiosincrático responde a qué hace la
empresa, no a cómo lo cuenta. En crecimiento de ingresos las actividades
aportan el doble que el estilo (+0,052 contra +0,023) y el estilo casi
desaparece dado la actividad (p=0,24): lo que anticipa el crecimiento es la
actividad concreta que la empresa describe. En beta y P/S los dos se
sostienen. La lectura para H1: la especificidad importa además de la
actividad, no en lugar de ella; y donde el outcome es "hacer" (riesgo
propio, crecimiento), manda la actividad.

## Lectura para la tesis

> El parseo semántico produce una taxonomía de divulgación coherente
> (`02_segmentacion.md`) y, además, señal económica incremental: 2-4 puntos de
> R² parcial sobre fundamentals y volumen en riesgo idiosincrático,
> valuación, I+D y crecimiento, concentrada en la especificidad de las
> afirmaciones; sobrevive al R² ajustado, a un Wald conjunto con SE cluster y
> a una permutación dentro de sector × año. Es una señal de tipo de empresa
> más que de cambio dentro de la empresa. Descompuesta, la señal viene
> tanto de la actividad identificable como del estilo con que se documenta;
> en riesgo idiosincrático y crecimiento, de la actividad.

El resultado sostiene la tesis con cualquier signo: si ΔR² fuera cero, la
conclusión sería "taxonomía sí, señal económica no". Salió positivo y chico,
y eso es lo que hay que escribir.

## Limitaciones

- R² bajos en beta, volatilidad y crecimiento: los outcomes son ruidosos y el
  ΔR² se lee en relación a ellos (por eso el R² parcial).
- La volatilidad idiosincrática sale de un modelo de mercado de un factor
  (exceso sobre Mkt−RF); no descuenta tamaño, valor ni momentum.
- Con shares en vez de intensidades la señal se debilita en tres de cinco
  outcomes: el contenido no es sólo mezcla.
- R&D/ventas tiene 44% de cobertura XBRL; su fila tiene la mitad de la muestra.
- Etiquetas de un LLM sin validación humana (`docs/problemas_academicos.md`
  #1); la sensibilidad a colapsar campos con acuerdo mediocre queda pendiente
  de la anotación (`ui-validator/`).
