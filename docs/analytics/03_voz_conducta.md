# Grilla voz × comportamiento: dos ejes, nueve celdas

Producido por `scripts/analytics/build_voice_behavior_grid.py`. Es la forma
directa de la pregunta de la tesis —¿habla más de lo que hace?— con los dos ejes
explícitos en vez de escondidos dentro de un clustering. **Modo de análisis
final: las 510 empresas con filings entran**; las 17 sin ningún frame de IA
tienen una celda propia, **sin IA**, porque no hay voz ni conducta que ubicar.
En el panel empresa-año esa celda es el 66% de 2021 y el 8% de 2026.

## Los dos ejes

| eje | definición | media del corpus |
|---|---|---|
| **VOZ** | % de las afirmaciones de IA de la empresa en registro promocional o estratégico | 14,4% |
| **CONDUCTA DIVULGADA** | % que describe conducta concreta: etapa de uso, resultado o capacidad | 44,9% |

**El eje de conducta mide conducta divulgada, no conducta operativa observada.**
Si la empresa dice "desplegamos IA en X", eso cuenta como conducta; nadie
verificó el despliegue. Lo que sí es externo al texto son los perfiles
financieros de `04` y los outcomes de `05`: ahí "hablar contra hacer" se
mide contra I+D, beta y valuación, no contra otra frase del mismo filing.

Los dos son **porcentajes sobre los frames de la misma empresa**, así que ninguno
premia a la que más habla. La confianza que merece cada tasa sí depende del
volumen, y de eso se encarga el encogimiento empírico-Bayes: una empresa con 9
frames se corre hacia el promedio del corpus; una con 0 no tiene tasa y va a
"sin IA".

**Correlación entre ejes: 0,286.** No son el mismo eje — que es la condición
para que la grilla tenga contenido.

Cada eje se corta en terciles → 9 celdas.

## La grilla

|  | conducta baja | conducta media | conducta alta |
|---|---:|---:|---:|
| **voz alta** | **desacople absoluto: 36** | 44 | **vocales sustantivos: 90** |
| **voz media** | 60 | 68 | 42 |
| **voz baja** | **silenciosos: 79** | 53 | **sustancia callada: 38** |

Más la celda **sin IA**: 17 empresas (y 1.088 empresas-año en el panel).

Las cuatro esquinas son las categorías que la tesis necesita nombrar:

| esquina | empresas | % voz | % conducta | frames (mediana) | ejemplos (por volumen) |
|---|---:|---:|---:|---:|---|
| **Desacople absoluto voz-conducta** (voz alta, conducta baja; región consistente con washing) | 36 | 33,1 | 22,8 | 17 | UNH, CI, AAPL, PYPL, ELV, DISCA, AMT, AAL, MPWR, SO |
| **Sustancia callada** (voz baja, conducta alta) | 38 | 4,3 | 70,5 | 30 | FTNT, OKTA, DDOG, STX, FFIV, NET, EXPE, ZBH, ZTS, HOLX |
| **Vocales sustantivos** | 90 | 28,7 | 72,6 | 77 | MSFT, NVDA, GOOGL, ADBE, INTC, CRM, HPE, SNOW, WDAY, AMD |
| **Silenciosos** | 79 | 1,8 | 19,0 | 14 | BAC, MCHP, ISRG, BAX, WLTW, WELL, UAL, FOX, LNC, PGR |

## Qué tan estable es (leer antes de usarla)

Remuestreando los frames de cada empresa, 25 réplicas:

| medida | 3×3 |
|---|---:|
| cae en la MISMA celda | 61,2% |
| cae en la misma o una ADYACENTE | **98,8%** |

**Cuando una empresa se mueve, se mueve un paso; nunca cruza la grilla.** El
azar con 9 celdas sería 11%.

Cada empresa trae su propia **`confianza_celda`**: en qué fracción de los
remuestreos cae en la celda que se le asignó. 149 de 510 superan 0,80. La
confianza mediana por celda va de 0,40 (el centro de la grilla, al borde de
dos cortes) a 0,80 (silenciosos). **Para análisis río abajo: filtrar por
confianza, no usar las 510 por igual.** Con confianza ≥0,70 las esquinas
quedan en 16 / 15 / 49 / 52 empresas (washing / callada / vocales /
silenciosos). "Sin IA" tiene confianza 1 por regla: no hay frames que
remuestrear.

Persistencia año a año en el panel (2.964 empresas-año): 48,5%. Es baja y no
hay que disimularla: con pocos frames el porcentaje salta solo. **Para series
de tiempo conviene usar los ejes continuos, no la celda.**

## Las esquinas separan cosas que no entraron a construirlas

Medianas por empresa (`firm_year_master_v2`):

| | Desacople absoluto | Sustancia callada | Vocales sustantivos | Silenciosos |
|---|---:|---:|---:|---:|
| frames de IA por 1.000 párrafos | 1,3 | 2,8 | **7,7** | 1,1 |
| I+D / ingresos | **1,5%** | 6,9% | **11,7%** | 2,2% |
| Margen bruto | 35,5% | 43,8% | **56,2%** | 45,5% |
| Beta | **0,76** | 0,97 | **1,05** | 0,78 |
| P/E | 21,1 | 25,6 | **26,8** | 22,2 |
| Crecimiento ingresos t+1 | 8,0% | 7,3% | 6,9% | 6,5% |

La esquina de desacople absoluto es **bajo I+D, bajo beta, no-tech, y habla poco de IA en
volumen**: empresas que dedican poco filing a IA pero lo poco que dicen es
estratégico, sin conducta. La de sustancia callada se parece a los vocales
sustantivos en beta y valuación con la mitad de la intensidad.

## Qué hay detrás del eje de conducta

El eje de conducta es una proporción de afirmaciones; las actividades
divulgadas (`09_actividades_ia.md`) dicen qué acción hay detrás y cuán
concreta es. Por empresa, la **concreción conductual** es la media de cinco
proporciones de sus actividades: con función declarada, desplegada o
escalada, con producto o proceso nombrado, con resultado cuantificado, con
proveedor nombrado (`activity_profiles.py`). Y una empresa está
**respaldada** si tiene al menos una actividad desplegada o escalada con
producto o proceso nombrado (`activity_grounding.py`).

| esquina | empresas | actividades (mediana) | concreción (media) | % respaldadas | % sin ninguna actividad |
|---|---:|---:|---:|---:|---:|
| Desacople absoluto (voz alta, conducta baja) | 36 | 6 | 0,31 | **53** | 8 |
| Sustancia callada | 38 | 34,5 | 0,41 | 92 | 0 |
| Vocales sustantivos | 90 | 77,5 | 0,39 | **94** | 0 |
| Silenciosos | 79 | 4 | 0,33 | 48 | 18 |
| resto de la grilla | 250 | 18,5 | 0,34 | 77 | 4 |

Spearman entre el eje de conducta y la concreción: +0,33 (n=471). Miden
cosas relacionadas pero distintas: el eje dice cuánto de lo que se afirma
es conducta; la concreción dice cuánto de esa conducta tiene nombre, etapa,
cifra o proveedor.

**Entre las 170 empresas de voz alta hay dos poblaciones.** 142 describen al
menos una actividad desplegada con producto nombrado (mediana 52
actividades; I+D 10% de ventas, beta 0,93, 2,4 frames por 1.000 párrafos:
NVDA, MSFT, ADBE, GOOGL, IBM, INTC, CRM, PANW). 28 no describen ninguna
(mediana 3 actividades; I+D 1,7%, beta 0,77, 0,28 frames por 1.000: CL,
AWK, BMY, AAL, PVH, FANG, MHK). Esta segunda es la que corresponde a la
noción intuitiva de washing: afirmaciones estratégicas sin una actividad
identificable detrás. En la esquina de desacople absoluto, 19 de las 36
tienen alguna actividad desplegada con nombre (AAPL, UNH, ELV, PYPL, MPWR,
CI, AMT, HUM, DFS, USB…) y 17 no la tienen (TXT, ROL, OKE, CSX, GIS, HLT,
COP, FE, KMB, MDLZ, KEY, AAL, FANG, MHK…). La esquina no es homogénea, y
la concreción es la que la parte en dos.

## Cómo se relaciona con el score de `08_definiciones_de_washing.md`

Son dos definiciones distintas y ambas hacen falta:

| | grilla (esta) | score exacto (09) |
|---|---|---|
| pregunta | ¿habla mucho y describe poco, **en absoluto**? | ¿habla más de lo que **su propia** conducta predice? |
| unidad | esquina de una grilla | test binomial con FDR, y residuo de intensidad para las 510 |
| resultado | 36 empresas | 8 empresas |
| dónde caen las 8 del score | **7 en "vocales sustantivos"**, 1 (YUM) en voz alta / conducta media | — |

O sea: el score marca empresas que describen MUCHA conducta y aun así hablan más
de lo que eso justifica (GOOGL, PANW, CRWD); la grilla marca empresas que hablan
sin describir conducta (UNH, CI, AAPL). **No se contradicen: son washing
relativo y washing absoluto.** La brecha entre canales (`06_brecha_entre_canales.md`) es una
tercera definición, ortogonal a las dos.

## Limitaciones

- "Sin IA" es una regla (cero frames), no una posición en los ejes.
- La celda del centro de la grilla es la menos confiable: son empresas que
  están al borde de los dos cortes.
- Todo descansa en etiquetas de un LLM sin validación humana
  (`docs/problemas_academicos.md` #1).
