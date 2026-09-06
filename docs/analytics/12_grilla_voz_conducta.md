# Grilla voz × comportamiento: dos ejes, nueve celdas

Producido por `scripts/analytics/build_voice_behavior_grid.py`. Es la forma
directa de la pregunta de la tesis —¿habla más de lo que hace?— con los dos ejes
explícitos en vez de escondidos dentro de un clustering.

## Los dos ejes

| eje | definición | media del corpus |
|---|---|---|
| **VOZ** | % de las afirmaciones de IA de la empresa en registro promocional o estratégico | 13,8% |
| **COMPORTAMIENTO** | % que describe conducta concreta: etapa de uso, resultado o capacidad | 47,6% |

Los dos son **porcentajes sobre los frames de la misma empresa**, así que ninguno
premia a la que más habla. La confianza que merece cada tasa sí depende del
volumen, y de eso se encarga el encogimiento empírico-Bayes (una empresa con 9
frames se corre hacia el promedio del corpus).

**Correlación entre ejes: 0,294.** No son el mismo eje — que es la condición
para que la grilla tenga contenido. (Con la medición anterior, tasas crudas de
15 conceptos, los bloques compartían 82% de la varianza y la matriz no decía
nada; ver `06_...md`.)

Cada eje se corta en terciles → 9 celdas.

## La grilla

|  | conducta baja | conducta media | conducta alta |
|---|---:|---:|---:|
| **voz alta** | **washing: 27** | 41 | **vocales sustantivos: 72** |
| **voz media** | 47 | 54 | 39 |
| **voz baja** | **silenciosos: 66** | 45 | **sustancia callada: 29** |

Las cuatro esquinas son las categorías que la tesis necesita nombrar:

| esquina | empresas | % voz | % conducta | ejemplos (por volumen) |
|---|---:|---:|---:|---|
| **Washing** (voz alta, conducta baja) | 27 | 28,0 | 26,8 | UNH, CI, AAPL, LMT, PYPL, ELV, AMT, AAL |
| **Sustancia callada** (voz baja, conducta alta) | 29 | 4,6 | 71,1 | ETSY, FTNT, OKTA, DDOG, STX, FFIV, NET, ZTS |
| **Vocales sustantivos** | 72 | 27,5 | 72,5 | MSFT, NVDA, ADBE, INTC, CRM, SNOW, HPE, AMD |
| **Silenciosos** | 66 | 2,8 | 23,0 | MCHP, BAC, ISRG, RTX, BAX, TYL, WELL, UAL |

## Qué tan estable es (leer antes de usarla)

Remuestreando los frames de cada empresa, 20 réplicas:

| medida | 3×3 | 2×2 |
|---|---:|---:|
| cae en la MISMA celda | 60,4% | 76,2% |
| cae en la misma o una ADYACENTE | **99,3%** | 100% |
| se mantiene en la esquina "washing" | 61,5% | 69,5% |
| se mantiene en "vocales sustantivos" | 74,7% | 81,8% |

**Cuando una empresa se mueve, se mueve un paso; nunca cruza la grilla.** El
azar con 9 celdas sería 11%.

Cada empresa trae su propia **`confianza_celda`**: en qué fracción de los
remuestreos cae en la celda que se le asignó. 109 de 420 superan 0,80. La
confianza mediana por celda va de 0,45 (el centro de la grilla, donde todo está
al borde de un corte) a 0,78 (vocales sustantivos). **Para análisis río abajo:
filtrar por confianza, no usar las 420 por igual.** Con confianza ≥0,70 las
esquinas quedan en 10 / 13 / 43 / 44 empresas.

Persistencia año a año en el panel: 37,8%. Es baja y no hay que disimularla: una
empresa-año necesita apenas 4 frames para entrar, y con 4 frames el porcentaje
salta solo. **Para series de tiempo conviene usar los ejes continuos, no la
celda.**

## Las esquinas separan cosas que no entraron a construirlas

Medianas sobre empresa-año:

| | Washing | Sustancia callada | Vocales sustantivos | Silenciosos |
|---|---:|---:|---:|---:|
| I+D / ingresos | **2,2%** | 12,7% | **14,2%** | 4,2% |
| Margen bruto | 36,0% | **62,4%** | 57,5% | 42,9% |
| Beta | **0,53** | 1,03 | **1,09** | 0,70 |
| P/E | 20,6 | **31,7** | 30,3 | 20,4 |
| Crecimiento ingresos t+1 | 5,6% | **9,4%** | 8,7% | 5,1% |

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
| resultado | 27 empresas | 8 empresas |
| dónde caen las 8 del score | **6 en "vocales sustantivos"**, 2 fuera de esquina | — |

O sea: el score marca empresas que describen MUCHA conducta y aun así hablan más
de lo que eso justifica (GOOGL, PANW, CRWD); la grilla marca empresas que hablan
sin describir conducta (UNH, CI, AAPL). **No se contradicen: son washing
relativo y washing absoluto.** Para la tesis, la grilla ordena la muestra y el
score dice dónde hay evidencia estadística.

## Limitaciones

- 420 de 494 empresas (≥8 frames). Las que quedan afuera son las que menos
  divulgan.
- La celda del centro de la grilla es la menos confiable (0,45): son empresas
  que están al borde de los dos cortes.
- Todo descansa en etiquetas de un LLM sin validación humana
  (`docs/problemas_academicos.md` #1).
