# Perfil financiero de los segmentos voz × comportamiento

> **Recalculado 2026-09-06 sobre el prefiltro v2 y con builders versionados.**
> Las tablas de este documento salen de la corrida actual de `make analytics`
> (`report_crosscheck_stats.py` reproduce las numéricas): panel de 1.426
> empresas-año y 460 empresas, frames de 10-K, DEF 14A y 8-K, población
> marcada por el prefiltro v2 (árboles, umbral 0,17 — `prefilter_evaluation.md`
> §8.16). Lado contable/mercado producido por `build_firm_financials.py`,
> `build_market_factors.py` y `build_roic_wacc.py` (ERP geométrico 6,48%).
> Las corridas anteriores y sus deltas están en `10_builders_y_recalculo.md`.

> Los clusters de comportamiento se re-ajustaron sobre la población del
> prefiltro v2 (tamaños 20 / 226 / 138 / 67; antes 25 / 212 / 132 / 74). El
> grupo de washing (D + mínimo) quedó en 12 empresas (20 en la corrida
> anterior, 6 en la original y en el pase intermedio con umbral 0,30) —
> leer la sección 2.


Cruza los clusters de comportamiento y los grupos de washing/sustancia
callada de `06_voice_vs_behavior_clustering.md` con los ratios,
valuación, beta y volatilidad de `04_ratios_factors_and_volatility.md`
(mediana por empresa de la mediana anual, sobre
`data/processed/clusters/segment_financials.parquet`). Objetivo: ¿los
segmentos que la voz y el comportamiento definen por separado se ven
distintos también en lo financiero — no solo en el texto?

```python
firm_fin = master.groupby('ticker')[FIN_COLS].median().reset_index()
bf = beh[['ticker','behavior_cluster']].merge(firm_fin, on='ticker')
```

## 1. Perfil financiero por cluster de comportamiento

Medianas por empresa (promediando sus años), n = 451 empresas con
etiqueta de cluster, de las cuales 272 tienen ratios contables:

| Cluster de comportamiento | n | Margen bruto | Margen operativo | R&D/revenue | Beta | Vol. pre-filing | P/E | P/S | Market cap |
|---|---|---|---|---|---|---|---|---|---|
| 0 — Intermedio interno | 20 | 42,0% | 14,2% | 4,9% | 0,77 | 25,4% | 22,0 | 1,98 | $26,9B |
| 1 — Comportamiento mínimo | 226 | 41,7% | 16,7% | 3,4% | 0,74 | 26,0% | 22,8 | 2,97 | $37,2B |
| 2 — Desplegadores de producto | 138 | **59,5%** | **16,9%** | **12,3%** | **1,05** | **30,3%** | **32,5** | **4,29** | **$44,3B** |
| 3 — Inversores en infraestructura | 67 | 50,3% | 14,6% | 4,8% | 0,90 | 26,8% | 27,9 | 3,07 | $41,8B |

El perfil del cluster 2 es el más nítido y se mantiene entre las tres
corridas en lo que importa: margen bruto más alto, R&D/revenue 3,6x el
del cluster mínimo, los múltiplos más caros y el mayor beta. Es un perfil
de software/semiconductores, y esa es también la advertencia — el cluster
puede estar capturando sector más que conducta de disclosure.

Los clusters 3 (infraestructura) y 0 tienen los peores márgenes
operativos (14,6% y 14,2%): invertir en infraestructura de IA es caro y
todavía no aparece como rentabilidad.

**Con esta partición el mayor riesgo está donde uno lo esperaría a
priori: en el cluster 2.** Vale dejar registrado que no es estable: en el
pase intermedio con umbral 0,30, el cluster chico (0, 34 empresas
entonces, 20 ahora) tenía beta 1,32, volatilidad 36,6% y el P/E más alto,
el mismo patrón que la versión original atribuía a "narradores de
revenue". Ese cluster cambia de identidad en cada re-ajuste y es el que
menos se puede interpretar; lo estable es el contraste entre el 2 y el 1.

## 2. Candidatos a washing (voz D + comportamiento mínimo)

> **La definición del grupo está superada** por `09_washing_score.md`, cuya
> versión validada deja 8 empresas en la cola de washing (ninguna de ellas
> en este grupo) y muestra que buena parte del grupo de 20 de la corrida
> anterior salía de la mezcla documental. Los perfiles financieros de abajo
> describen un grupo cuya construcción resultó ser en buena parte un
> artefacto de volumen de texto.


El grupo quedó en **12 empresas** (`06_...md`): ALL, CL, CMS, ELV, GE,
GILD, GL, GPC, HUM, LDOS, PFE, UDR. Nueve estaban entre las 20 de la
corrida anterior (entran GE, GL y PFE; salen AAPL, ADI, BMY, CHD, ECL, FE,
JPM, RL, SO, SOLS y TFC). De ellas, 6 tienen ratios y 5 tienen R&D.

| | Washing (n=12) | Resto de D (n=91) |
|---|---|---|
| Margen bruto | 63,9% | 59,6% |
| Margen operativo | **19,3%** | 15,8% |
| **R&D/revenue** | **3,2%** (n=5) | **12,4%** (n=56) |
| Beta | **0,44** | **1,02** |
| Deuda/equity | 0,82 | 0,50 |
| Current ratio | 1,18 | 1,35 |
| P/E | 28,1 | 29,1 |
| Market cap | $36,2B | $47,0B |

**El hallazgo titular sobrevive en dirección; el respaldo va y viene.**
La versión original reportaba R&D/revenue de 0,6% en el grupo washing
contra 12,5% en el resto de D sobre 2 empresas con dato; la corrida
anterior, 2,4% contra 12,2% sobre 8; hoy es **3,2% contra 12,4%** sobre 5
empresas con dato (y el pase intermedio con umbral 0,30 lo dejó en 2
empresas). La dirección y el orden de magnitud (4-5x) se mantienen en
todas las corridas, con el resto de D prácticamente inmóvil; el grupo de
washing, en cambio, cambia de miembros y de tamaño en cada re-ajuste.

El beta refuerza el mismo cuadro: 0,44 contra 1,02. **Las empresas que
hablan como líderes de IA sin comportamiento observable son empresas de
bajo riesgo sistemático y bajo gasto en I+D** — seguros de salud,
utilities, farma, consumo básico, un conglomerado industrial. Su margen
operativo es incluso mejor que el del resto de D (19,3% vs. 15,8%), o
sea que no son empresas en problemas: son empresas de otro negocio,
hablando el idioma del sector tech.

**La interpretación se debilita por persistencia, no por magnitud.**
El grupo se reconstruye distinto en cada corrida: 6 empresas en la
original, 20 en la anterior (de las cuales sólo AAPL y CL sostenían la
etiqueta D en todos sus años), 12 hoy, con AAPL fuera. El contraste
financiero es real cada vez que se mide; lo que no está establecido es
que corresponda a una identidad de disclosure sostenida en vez de a un
promedio de años heterogéneos.

Lectura defendible: **"empresas de sectores no-tech con bajo I+D adoptan
transitoriamente un registro vocal sobre IA"**, que es una afirmación más
débil y más precisa que "estas 12 empresas hacen AI-washing".

## 3. Sustancia callada

### Grupo A (voz cautelosa + comportamiento de despliegue): 11 empresas

AVB, CCI, CDW, CERN, EXC, GNRC, KLAC, MTCH, QRVO, RHI, T. En la corrida
anterior era CDW sola; el re-ajuste movió a A varias empresas que antes
estaban en B (AVB, CCI, CERN, KLAC, QRVO, RHI, T). Sigue siendo chico para
perfilar.

### Grupo B (voz genérica + comportamiento de despliegue): 40 empresas

AIZ, ALGN, AON, CDAY, CPRT, DDOG, DE, EMR, ETSY, EXPE, FAST, FIS, FLT,
HAL, ICE, IRM, JKHY, LUV, MA, MSCI, MSI, NET, NWS, OKTA, OTIS, PLTR, RCL,
RIVN, RMD, ROP, SEDG, SQ, STX, SYK, TFC, TROW, ULTA, V, WYNN, ZTS.
Veintitrés de las 39 de la corrida anterior siguen; entran, entre otras,
MA, PLTR, DE, DDOG, ETSY y SQ.

| | Callados B (n=40) | Resto de B (n=148) |
|---|---|---|
| Margen bruto | 51,8% | 44,7% |
| Margen operativo | 15,8% | 16,8% |
| **R&D/revenue** | **8,6%** (n=22) | **4,9%** (n=57) |
| Beta | 0,97 | 0,76 |
| P/E | **36,5** | 23,2 |
| P/S | **4,64** | 3,07 |
| Deuda/equity | 0,68 | 0,76 |

El contraste es consistente y en la dirección esperada: **invierten 1,8x
más en I+D que sus pares de voz equivalente, con mejor margen bruto y
algo menos de deuda, y el mercado les paga una prima considerable** (P/E
36,5 vs. 23,2; P/S 4,64 vs. 3,07). Lo único que se dio vuelta es el
margen operativo, que ahora es 1,0 p.p. menor (antes empataban).

Es el cuadrante con más respaldo muestral de los cuatro (40 empresas, 23
con ratios) y el único cuyo contraste no depende de un puñado de
observaciones. Si algo de esta matriz 2D merece seguirse, es esto: **hay
un grupo de empresas que despliega IA con perfil financiero de tech,
habla de eso en registro llano, y cotiza con prima igual.** El mercado
parece estar viendo lo que el lenguaje del filing no dice.

## Lectura conjunta

Con datos financieros de por medio, la matriz voz × comportamiento dejó
de ser solo un artefacto de texto: **los 6 candidatos a washing tienen
un perfil financiero (R&D casi nulo, beta bajo, múltiplos bajos) que el
propio mercado ya trata como "no-tech"**, mientras que los grupos de
"sustancia callada" tienen un perfil financiero (R&D alto, beta alto,
en el caso de A también mejor margen y mayor tamaño) más parecido al de
un líder vocal genuino, pese a no sonar como uno. La divergencia
voz-comportamiento identificada solo con texto en `06_...md` se
confirma independientemente con datos contables y de mercado — no es
ruido de la clusterización de texto.

## Limitaciones

- **El grupo de washing tiene 12 empresas y sólo 5 con dato de R&D**,
  que es la columna que sostiene el hallazgo. La corrida anterior tenía 8
  con dato; el pase intermedio con umbral 0,30, 2. Sigue siendo poco.
- **La persistencia, no el tamaño, es la limitación seria**: 2 de 20
  sostienen su etiqueta año a año (`06_...md`). El contraste financiero
  está bien medido sobre un grupo cuya definición es inestable.
- El grupo de sustancia callada A tiene n=1. Ignorarlo.
- La lección general de haber corrido esto dos veces sobre poblaciones
  distintas: **un hallazgo construido sobre un cluster de K-means de menos
  de ~30 empresas no debería titular nada.** Entre versiones desaparecieron
  los "narradores de revenue" y se rehízo por completo la composición del
  grupo de washing, aunque el contraste de R&D sobreviviera.
- Igual que `04_...md`: sin efectos fijos de empresa, ratios con outliers
  winsorizados mecánicamente (2-98%), SIC-2 no usado aquí como control
  adicional. El perfil del cluster 2 (margen bruto y R&D altos, múltiplos
  caros) es indistinguible a priori del perfil "software/semis", así que
  atribuirlo a la conducta de disclosure y no al sector requiere un
  control que este documento no tiene.
- Las medianas por empresa promedian años con cobertura financiera
  desigual: una empresa con 1 año de datos pesa igual que una con 6.
