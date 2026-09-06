# Perfil financiero de los segmentos voz × comportamiento

Los grupos de este documento —clusters de comportamiento, "washing = voz D ×
comportamiento mínimo", "sustancia callada"— son particiones sobre CÓMO habla
de IA quien habla de IA (panel condicionado a ≥3 frames por año, 451
empresas). No existen para la empresa que no habla; el perfil financiero por
nivel de intensidad de IA sobre todas las empresas-año está en `04_...md`.
Financieros por empresa: mediana de sus años (`build_firm_panels.py`,
`segment_financials.parquet`).

Cruza los clusters de comportamiento y los grupos de washing/sustancia
callada de `06_voice_vs_behavior_clustering.md` con los ratios,
valuación, beta y volatilidad de `04_ratios_factors_and_volatility.md`
(mediana por empresa de sus años, sobre
`data/processed/clusters/segment_financials.parquet` —
`build_firm_panels.py` agrega con `.median()`). Objetivo: ¿los
segmentos que la voz y el comportamiento definen por separado se ven
distintos también en lo financiero — no solo en el texto?

```python
firm_fin = master.groupby('ticker')[SEGMENT_COLUMNS].median().reset_index()
bf = beh[['ticker','behavior_cluster']].merge(firm_fin, on='ticker')
```

## 1. Perfil financiero por cluster de comportamiento

Medianas por empresa (mediana de sus años), n = 451 empresas con
etiqueta de cluster, de las cuales 272 tienen ratios contables:

| Cluster de comportamiento | n | Margen bruto | Margen operativo | R&D/revenue | Beta | Vol. pre-filing | P/E | P/S | Market cap |
|---|---|---|---|---|---|---|---|---|---|
| 0 — Intermedio interno | 20 | 42,0% | 15,2% | 4,9% | 0,68 | 24,6% | 21,4 | 1,98 | $27,2B |
| 1 — Comportamiento mínimo | 226 | 41,3% | 16,8% | 3,4% | 0,73 | 25,3% | 21,6 | 2,85 | $36,8B |
| 2 — Desplegadores de producto | 138 | **59,5%** | **17,0%** | **12,2%** | **1,05** | **30,3%** | **30,2** | **3,99** | $39,9B |
| 3 — Inversores en infraestructura | 67 | 49,9% | 14,6% | 4,6% | 0,88 | 26,4% | 26,5 | 3,07 | **$42,0B** |

El perfil del cluster 2 es el más nítido: margen bruto más alto,
R&D/revenue 3,6x el del cluster mínimo, los múltiplos más caros y el
mayor beta. Es un perfil de software/semiconductores, y esa es también la
advertencia — el cluster puede estar capturando sector más que conducta
de disclosure.

El cluster 3 (infraestructura) tiene el peor margen operativo (14,6%):
invertir en infraestructura de IA es caro y todavía no aparece como
rentabilidad.

**El mayor riesgo está donde uno lo esperaría a priori: en el cluster
2.** El cluster chico (0, 20 empresas) es el que menos se puede
interpretar: K-means lo reconstruye distinto ante cualquier cambio de
población. Lo estable es el contraste entre el 2 y el 1.

## 2. Candidatos a washing (voz D + comportamiento mínimo)

> **La definición del grupo está superada** por `09_washing_score.md`, cuya
> versión validada deja 8 empresas en la cola de washing (ninguna de ellas
> en este grupo) y muestra que buena parte de este cruce sale de la mezcla
> documental. Los perfiles financieros de abajo describen un grupo cuya
> construcción es en buena parte un artefacto de volumen de texto.


El grupo son **12 empresas**: ALL, CL, CMS, ELV, GE, GILD, GL, GPC, HUM,
LDOS, PFE, UDR. De ellas, 6 tienen ratios y 5 tienen R&D.

| | Washing (n=12) | Resto de D (n=91) |
|---|---|---|
| Margen bruto | 66,0% | 59,7% |
| Margen operativo | **19,8%** | 16,4% |
| **R&D/revenue** | **3,3%** (n=5) | **12,1%** (n=56) |
| Beta | **0,46** | **1,02** |
| Deuda/equity | 0,81 | 0,48 |
| Current ratio | 1,18 | 1,34 |
| P/E | 20,3 | 28,0 |
| Market cap | $36,5B | $45,8B |

**El hallazgo titular: R&D/revenue de 3,3% en el grupo washing contra
12,1% en el resto de D**, un contraste de 4x sobre 5 empresas con dato.
El resto de D es un grupo estable; el de washing cambia de miembros y de
tamaño ante cualquier re-ajuste de K-means, así que el contraste está bien
medido pero el grupo no.

El beta refuerza el mismo cuadro: 0,46 contra 1,02. **Las empresas que
hablan como líderes de IA sin comportamiento observable son empresas de
bajo riesgo sistemático y bajo gasto en I+D** — seguros de salud,
utilities, farma, consumo básico, un conglomerado industrial. Su margen
operativo es incluso mejor que el del resto de D (19,8% vs. 16,4%), o
sea que no son empresas en problemas: son empresas de otro negocio,
hablando el idioma del sector tech.

**La interpretación se debilita por persistencia, no por magnitud.**
Pocas de estas empresas sostienen la etiqueta D en todos sus años; la
mayoría migra dentro y fuera de D o tiene 1-2 años de panel. El contraste
financiero es real; lo que no está establecido es que corresponda a una
identidad de disclosure sostenida en vez de a un promedio de años
heterogéneos.

Lectura defendible: **"empresas de sectores no-tech con bajo I+D adoptan
transitoriamente un registro vocal sobre IA"**, que es una afirmación más
débil y más precisa que "estas 12 empresas hacen AI-washing".

## 3. Sustancia callada

### Grupo A (voz cautelosa + comportamiento de despliegue): 11 empresas

AVB, CCI, CDW, CERN, EXC, GNRC, KLAC, MTCH, QRVO, RHI, T. Chico para
perfilar.

### Grupo B (voz genérica + comportamiento de despliegue): 40 empresas

AIZ, ALGN, AON, CDAY, CPRT, DDOG, DE, EMR, ETSY, EXPE, FAST, FIS, FLT,
HAL, ICE, IRM, JKHY, LUV, MA, MSCI, MSI, NET, NWS, OKTA, OTIS, PLTR, RCL,
RIVN, RMD, ROP, SEDG, SQ, STX, SYK, TFC, TROW, ULTA, V, WYNN, ZTS.

| | Callados B (n=40) | Resto de B (n=148) |
|---|---|---|
| Margen bruto | 51,8% | 44,6% |
| Margen operativo | 15,9% | 17,1% |
| **R&D/revenue** | **8,4%** (n=22) | **4,9%** (n=57) |
| Beta | 0,97 | 0,75 |
| P/E | **34,6** | 22,5 |
| P/S | **4,66** | 3,06 |
| Deuda/equity | 0,65 | 0,76 |

El contraste es consistente y en la dirección esperada: **invierten 1,7x
más en I+D que sus pares de voz equivalente, con mejor margen bruto y
algo menos de deuda, y el mercado les paga una prima considerable** (P/E
34,6 vs. 22,5; P/S 4,66 vs. 3,06). El margen operativo es 1,2 p.p. menor.

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
  que es la columna que sostiene el hallazgo. Es poco.
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
