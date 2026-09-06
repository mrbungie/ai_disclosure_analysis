# Grilla voz × comportamiento: dos ejes, nueve celdas

Producido por `scripts/analytics/build_voice_behavior_grid.py`. Es la forma
directa de la pregunta de la tesis —¿habla más de lo que hace?— con los dos ejes
explícitos en vez de escondidos dentro de un clustering.

## Los dos ejes

| eje | definición | media del corpus |
|---|---|---|
| **VOZ** | % de las afirmaciones de IA de la empresa en registro promocional o estratégico | 14,9% |
| **COMPORTAMIENTO** | % que describe conducta concreta: etapa de uso, resultado o capacidad | 48,5% |

Los dos son **porcentajes sobre los frames de la misma empresa**, así que ninguno
premia a la que más habla. La confianza que merece cada tasa sí depende del
volumen, y de eso se encarga el encogimiento empírico-Bayes (una empresa con 9
frames se corre hacia el promedio del corpus).

**Correlación entre ejes: 0,297.** No son el mismo eje — que es la condición
para que la grilla tenga contenido. (Con la medición del cruce de clusters de `06_...md`, tasas crudas de
15 conceptos, los bloques compartían 82% de la varianza y la matriz no decía
nada; ver `06_...md`.)

Cada eje se corta en terciles → 9 celdas.

## La grilla

|  | conducta baja | conducta media | conducta alta |
|---|---:|---:|---:|
| **voz alta** | **washing: 26** | 40 | **vocales sustantivos: 74** |
| **voz media** | 48 | 54 | 36 |
| **voz baja** | **silenciosos: 66** | 45 | **sustancia callada: 30** |

Las cuatro esquinas son las categorías que la tesis necesita nombrar:

| esquina | empresas | % voz | % conducta | ejemplos (por volumen) |
|---|---:|---:|---:|---|
| **Washing** (voz alta, conducta baja) | 26 | 28,6 | 26,2 | UNH, CI, AAPL, PYPL, ELV, DISCA, AMT, AAL |
| **Sustancia callada** (voz baja, conducta alta) | 30 | 4,7 | 71,3 | FTNT, OKTA, STX, DDOG, FFIV, NET, EXPE, ZTS |
| **Vocales sustantivos** | 74 | 27,3 | 72,5 | MSFT, NVDA, GOOGL, ADBE, INTC, CRM, SNOW, HPE |
| **Silenciosos** | 66 | 2,9 | 23,0 | BAC, MCHP, ISRG, BAX, WLTW, DVA, WELL, UAL |

## Qué tan estable es (leer antes de usarla)

Remuestreando los frames de cada empresa, 20 réplicas:

| medida | 3×3 | 2×2 |
|---|---:|---:|
| cae en la MISMA celda | 60,4% | 76,2% |
| cae en la misma o una ADYACENTE | **99,1%** | 100% |
| se mantiene en la esquina "washing" | 61,5% | 69,5% |
| se mantiene en "vocales sustantivos" | 74,7% | 81,8% |

**Cuando una empresa se mueve, se mueve un paso; nunca cruza la grilla.** El
azar con 9 celdas sería 11%.

Cada empresa trae su propia **`confianza_celda`**: en qué fracción de los
remuestreos cae en la celda que se le asignó. 86 de 419 superan 0,80. La
confianza mediana por celda va de 0,44 (el centro de la grilla, donde todo está
al borde de un corte) a 0,78 (vocales sustantivos). **Para análisis río abajo:
filtrar por confianza, no usar las 419 por igual.** Con confianza ≥0,70 las
esquinas quedan en 10 / 10 / 43 / 37 empresas (washing / callada / vocales /
silenciosos).

Persistencia año a año en el panel: 36,4%. Es baja y no hay que disimularla: una
empresa-año necesita apenas 4 frames para entrar, y con 4 frames el porcentaje
salta solo. **Para series de tiempo conviene usar los ejes continuos, no la
celda.**

## Las esquinas separan cosas que no entraron a construirlas

Medianas sobre empresa-año:

| | Washing | Sustancia callada | Vocales sustantivos | Silenciosos |
|---|---:|---:|---:|---:|
| I+D / ingresos | **1,8%** | 8,3% | **12,6%** | 3,6% |
| Margen bruto | 38,7% | 50,5% | **56,5%** | 41,3% |
| Beta | **0,60** | 0,96 | **1,06** | 0,77 |
| P/E | 21,9 | **30,6** | 26,9 | 20,7 |
| Crecimiento ingresos t+1 | 6,4% | **7,5%** | 7,0% | 4,9% |

La esquina de washing es **bajo I+D, bajo beta, no-tech**: empresas que hablan de
IA en registro estratégico sin describir casi ninguna conducta. La de sustancia
callada es lo contrario y se parece financieramente a los vocales sustantivos —
mismo perfil tech, distinto volumen de discurso.

## Cómo se relaciona con el score de `09_washing_score.md`

Son dos definiciones distintas y ambas hacen falta:

| | grilla (esta) | score exacto (09) |
|---|---|---|
| pregunta | ¿habla mucho y describe poco, **en absoluto**? | ¿habla más de lo que **su propia** conducta predice? |
| unidad | esquina de una grilla | test binomial con FDR |
| resultado | 26 empresas | 8 empresas |
| dónde caen las 8 del score | **7 en "vocales sustantivos"**, 1 (YUM) en voz alta / conducta media | — |

O sea: el score marca empresas que describen MUCHA conducta y aun así hablan más
de lo que eso justifica (GOOGL, PANW, CRWD); la grilla marca empresas que hablan
sin describir conducta (UNH, CI, AAPL). **No se contradicen: son washing
relativo y washing absoluto.** Para la tesis, la grilla ordena la muestra y el
score dice dónde hay evidencia estadística.

## Limitaciones

- 419 de 493 empresas (≥8 frames). Las que quedan afuera son las que menos
  divulgan.
- La celda del centro de la grilla es la menos confiable (0,44): son empresas
  que están al borde de los dos cortes.
- Todo descansa en etiquetas de un LLM sin validación humana
  (`docs/problemas_academicos.md` #1).
