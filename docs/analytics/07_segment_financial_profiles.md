# Perfil financiero de los segmentos voz × comportamiento

> **Recalculado con builders versionados (ver `10_builders_y_recalculo.md`).**
> El lado contable/mercado ya no viene de un script perdido: lo producen
> `build_firm_financials.py`, `build_market_factors.py` y `build_roic_wacc.py`
> (`make analytics`). Con mejor cobertura XBRL y el ERP corregido, varias
> cifras de este documento se movieron — las tablas de abajo son las de la
> corrida anterior; los deltas están listados en `10_...md`.

> **Recalculado 2026-09-05 con DEF 14A y 8-K.** Los insumos financieros
> no cambiaron; las etiquetas sí (K-means re-ajustado, ver `01_...md` y
> `06_...md`). En el camino se corrigió un bug de reproducibilidad en
> `gold_ai_frames` — ver `01_...md`. El resultado central **sobrevive en
> magnitud pero se debilita en interpretación** — leer la sección 2.


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

Medianas por empresa (promediando sus años), n = 443 empresas con
etiquetas y datos financieros:

| Cluster de comportamiento | n | Margen bruto | Margen operativo | R&D/revenue | Beta | Vol. pre-filing | P/E | P/S | Market cap |
|---|---|---|---|---|---|---|---|---|---|
| 0 — Intermedio interno | 25 | 45,8% | **18,2%** | 5,0% | 0,62 | 25,0% | 21,8 | 3,39 | $37,6B |
| 1 — Comportamiento mínimo | 212 | 43,2% | 16,8% | 2,9% | 0,75 | 26,1% | 20,9 | 3,03 | $35,4B |
| 2 — Desplegadores de producto | 132 | **59,0%** | 17,3% | **12,5%** | **1,06** | **29,1%** | **28,0** | **4,27** | **$41,3B** |
| 3 — Inversores en infraestructura | 74 | 46,4% | 12,7% | 6,0% | 0,91 | 26,6% | 23,4 | 2,52 | $39,8B |

El perfil del cluster 2 es el más nítido y se mantiene intacto entre
versiones: margen bruto más alto, R&D/revenue 4,3x el del cluster mínimo,
los múltiplos más caros y el mayor beta. Es un perfil de
software/semiconductores, y esa es también la advertencia — el cluster
puede estar capturando sector más que conducta de disclosure.

El cluster 3 (infraestructura) tiene el peor margen operativo (12,7%):
invertir en infraestructura de IA es caro y todavía no aparece como
rentabilidad.

**Se cae el hallazgo de "narradores de revenue".** La versión anterior
destacaba que el cluster de mayor beta (1,21) y volatilidad (37,5%) no
era el de despliegue sino el de narradores de revenue. Ese cluster no
existe en la nueva partición (`06_...md`). Con la partición actual el
mayor riesgo está donde uno lo esperaría a priori: en el cluster 2.

## 2. Candidatos a washing (voz D + comportamiento mínimo)

> **La definición del grupo está superada** por `09_washing_score.md`, cuya
> versión validada deja 8 empresas en la cola de washing (ninguna de ellas
> AAPL/CL) y muestra que 9 de las 20 de acá salían de la mezcla documental.
> Los perfiles financieros de abajo describen un grupo cuya construcción
> resultó ser en buena parte un artefacto de volumen de texto.


El grupo pasó de 6 a **20 empresas** (`06_...md`): AAPL, ADI, ALL, BMY,
CHD, CL, CMS, ECL, ELV, FE, GILD, GPC, HUM, JPM, LDOS, RL, SO, SOLS, TFC,
UDR. De ellas, 14 tienen ratios y 8 tienen R&D.

| | Washing (n=20) | Resto de D (n=99) |
|---|---|---|
| Margen bruto | 52,7% | 62,1% |
| Margen operativo | **18,4%** | 15,8% |
| **R&D/revenue** | **2,4%** (n=8) | **12,2%** (n=55) |
| Beta | **0,46** | **1,00** |
| Deuda/equity | 0,82 | 0,58 |
| Current ratio | 1,21 | 1,29 |
| P/E | 28,4 | 26,4 |
| Market cap | $43,3B | $43,3B |

**El hallazgo titular de la versión anterior sobrevive.** Reportaba
R&D/revenue de 0,6% en el grupo washing contra 12,5% en el resto de D, un
contraste de 20x sobre 2 empresas con dato. Hoy es **2,4% contra 12,2%**
— un contraste de 5x, sobre 8 empresas con dato y con el resto de D
prácticamente idéntico al de entonces. La magnitud bajó, la dirección y
el orden de magnitud se mantienen.

El beta refuerza el mismo cuadro: 0,46 contra 1,00. **Las empresas que
hablan como líderes de IA sin comportamiento observable son empresas de
bajo riesgo sistemático y bajo gasto en I+D** — utilities, consumo
básico, seguros de salud, banca. Su margen operativo es incluso mejor que
el del resto de D (18,4% vs. 15,8%), o sea que no son empresas en
problemas: son empresas de otro negocio, hablando el idioma del sector
tech.

**Pero la interpretación se debilita por persistencia, no por magnitud.**
Como documenta `06_...md`, sólo 2 de las 20 (AAPL y CL) sostienen la
etiqueta D en todos sus años con ≥3 años de panel. El resto migra dentro
y fuera de D o tiene 1-2 años de datos. El contraste financiero es real y
está bien medido; lo que no está establecido es que corresponda a una
identidad de disclosure sostenida en vez de a un promedio de años
heterogéneos.

Lectura defendible: **"empresas de sectores no-tech con bajo I+D adoptan
transitoriamente un registro vocal sobre IA"**, que es una afirmación más
débil y más precisa que "estas 20 empresas hacen AI-washing".

## 3. Sustancia callada

### Grupo A (voz cautelosa + comportamiento de despliegue): CDW

Una sola empresa. No es un grupo.

### Grupo B (voz genérica + comportamiento de despliegue): 39 empresas

ADSK, AON, AVB, CCI, CDAY, CERN, CPRT, DAL, EMR, EQIX, EXPE, FAST, FLT,
ICE, IRM, J, JKHY, KHC, KLAC, LH, LUV, MSCI, NET, NWS, OKTA, OTIS, QRVO,
RHI, RMD, ROP, SEDG, STX, T, TEL, TER, ULTA, V, WAB, ZTS.

| | Callados B (n=39) | Resto de B (n=142) |
|---|---|---|
| Margen bruto | 50,5% | 45,4% |
| Margen operativo | 16,8% | 16,8% |
| **R&D/revenue** | **7,9%** (n=16) | **4,9%** (n=44) |
| Beta | 0,86 | 0,81 |
| P/E | **31,1** | 21,3 |
| P/S | **4,03** | 3,05 |
| Deuda/equity | 0,49 | 0,67 |

El contraste es consistente y en la dirección esperada: **invierten 1,6x
más en I+D que sus pares de voz equivalente, con mejor margen bruto y
menos deuda, y el mercado les paga una prima considerable** (P/E 31,1 vs.
21,3; P/S 4,03 vs. 3,05).

Es el cuadrante con más respaldo muestral de los cuatro (39 empresas, 32
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

- **El grupo de washing tiene 20 empresas pero sólo 8 con dato de R&D**,
  que es la columna que sostiene el hallazgo. Es mejor que las 2 de la
  versión anterior, y sigue siendo poco.
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
