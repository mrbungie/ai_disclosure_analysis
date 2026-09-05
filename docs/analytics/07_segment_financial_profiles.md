# Perfil financiero de los segmentos voz × comportamiento

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

| Cluster de comportamiento | n | Margen bruto | Margen operativo | R&D/revenue | Beta | Vol. pre-filing | P/E | P/S | Market cap |
|---|---|---|---|---|---|---|---|---|---|
| 0 — Narradores de revenue | 26 | 45,2% | **22,2%** | 7,4% | **1,21** | **37,5%** | 20,6 | 3,16 | $25,7B |
| 1 — Comportamiento mínimo | 233 | 43,8% | 16,8% | 2,9% | 0,71 | 24,5% | 20,9 | 2,99 | $32,5B |
| 2 — Desplegadores de producto | 114 | **62,9%** | 15,7% | **12,8%** | 1,03 | 29,5% | **27,1** | **4,07** | $38,8B |
| 3 — Inversores en infraestructura | 48 | 46,1% | 13,8% | 6,1% | 0,91 | 27,6% | 23,6 | 2,36 | **$42,4B** |

**Hallazgo no anticipado**: el cluster de mayor riesgo sistemático
(beta 1,21) y mayor volatilidad (37,5% anualizada) **no es el de
"desplegadores de producto" (cluster 2, beta 1,03) — es el de
"narradores de revenue" (cluster 0)**, el grupo más chico (26
empresas). Tiene sentido narrativo: enmarcar la IA como motor de
ingresos (no de eficiencia interna) es la apuesta más "growth story",
y el mercado las trata como tal — más beta, más volatilidad, más
momentum (16,6% acumulado 12-1 meses, el más alto de los 4). El
cluster 2 (despliegue real) es donde está la sustancia de margen bruto
y R&D, pero el cluster 0 es donde está el riesgo/especulación de
mercado.

## 2. Candidatos a washing (voz D + comportamiento mínimo) — la evidencia más clara de todo el proyecto

CCL, FE, GPC, HII, IQV, NEM (§`06_...md`) vs. el resto de D:

| | Washing (n=6) | Resto de D (n=79) |
|---|---|---|
| Margen bruto | 36,3% | 61,2% |
| **R&D/revenue** | **0,6%** | **12,5%** |
| Beta | 0,63 | 1,06 |
| Deuda/equity | 0,81 | 0,52 |
| Current ratio | 0,90 | 1,39 |
| P/E | 17,9 | 26,8 |
| P/S | 1,58 | 3,88 |
| Market cap | $29,8B | $46,9B |

**Esta es la evidencia financiera más limpia de todo el proyecto.**
Los 6 candidatos a washing no solo tienen comportamiento textual
mínimo (por construcción) — tienen una R&D/revenue de **0,6% vs. 12,5%
del resto de D, veinte veces menos**, cotizan a múltiplos de valor
(P/E, P/S) claramente más bajos, tienen MENOS beta (0,63 vs. 1,06,
es decir, el mercado los trata como acciones normales, no como
apuestas de crecimiento tech), y peor liquidez/apalancamiento. **El
mercado no les está creyendo el discurso de IA** — la valuación y el
riesgo sistemático de estas 6 empresas se parecen al de sus industrias
reales (cruceros, utilities, autopartes, defensa, servicios de salud,
minería), no al de un "líder vocal de IA".

Detalle por empresa — casi ninguna reporta R&D como línea material, y
**la mayoría tiene muy pocos años/frames respaldando la etiqueta**
(ver corrección de persistencia en `06_...md`):

| Ticker | Negocio real | R&D/revenue | Beta | P/E | Años en panel | n_frames (esos años) |
|---|---|---|---|---|---|---|
| CCL | Cruceros | — (sin dato) | 0,63 | 14,0 | 1 | 4 |
| FE | Utility eléctrica | — (sin dato) | 0,10 | 27,5 | 1 | 3 |
| GPC | Distribución autopartes | — (sin dato) | 0,62 | 18,1 | 3 | 8, 5, 6 |
| HII | Astilleros/defensa | 0,3% | 0,64 | 14,4 | 4 (migra D→D→B→A) | 5, 5, 16, 14 |
| IQV | Servicios/datos de salud | — (sin dato) | 1,09 | 23,9 | 2 | 29, 32 |
| NEM | Minería de oro | 0,9% | 0,59 | 17,8 | 2 | 3, 3 |

> **Importante**: solo GPC tiene algo parecido a presencia estable en
> D año a año (3/3); CCL, FE y NEM descansan en 1-2 años de muy pocos
> frames, y HII directamente migra fuera de D en 2025-2026. El perfil
> financiero (R&D≈0, beta bajo) es un dato real e independiente del
> texto, pero la etiqueta "voz de líder vocal" detrás de estos 6 casos
> es más frágil de lo que sugiere la tabla agregada — ver el detalle
> completo en `06_voice_vs_behavior_clustering.md`.

Ninguna de estas 6 es remotamente una empresa de tecnología — son
negocios tradicionales de sectores muy distintos entre sí (cruceros,
utility, autopartes, defensa, salud, minería) cuyo único denominador
común es que su VOZ sobre IA (especificidad + tono) se pareció lo
suficiente a la de NVDA/MSFT/GOOGL como para caer en el mismo cluster
de voz — sin nada del perfil de R&D, margen o riesgo de mercado que
acompaña a esas empresas.

## 3. Sustancia callada — el espejo del hallazgo anterior

### Grupo A (voz cautelosa/riesgo + comportamiento de despliegue): CERN, DXCM, EMR, FISV, GM, HAL, LUV, MCO, ROP, ULTA

| | Sustancia callada (n=10) | Resto de A (n=140) |
|---|---|---|
| Margen bruto | **66,6%** | 42,1% |
| **R&D/revenue** | **10,6%** | **2,9%** |
| Beta | 0,98 | 0,74 |
| P/E | 36,8 | 22,1 |
| P/S | 4,30 | 3,53 |
| Market cap | **$57,7B** | $33,2B |

El inverso exacto del hallazgo de washing: 10 empresas con voz de
"riesgo cauteloso" (boilerplate genérico, poco promocional) tienen
**margen bruto y R&D/revenue varias veces más altos** que el resto de
su propio arquetipo de voz, más beta, más múltiplo, y casi el doble de
market cap mediano. DXCM (Dexcom, dispositivos médicos con IA
embebida), MCO (Moody's, analítica de datos), ROP (Roper Technologies)
son negocios genuinamente intensivos en tecnología que simplemente
NO lo cuentan con lenguaje promocional — el patrón opuesto al
"AI-washing": sustancia real, subvendida.

### Grupo B (voz genérica + comportamiento de despliegue): 40 empresas incl. ADSK, WDAY, TEAM, OKTA, PLTR, V, WMT

| | Sustancia callada (n=40) | Resto de B (n=135) |
|---|---|---|
| **R&D/revenue** | **9,0%** | 5,8% |
| Beta | **1,03** | 0,80 |
| Vol. pre-filing | 29,6% | 25,4% |
| Momentum 12-1m | 4,2% | 10,0% |

Mismo patrón que el grupo A pero más moderado: más R&D-intensivas y de
mayor beta que el resto de B, pese a una voz igual de "plana". Nota:
su momentum de 12 meses es MENOR que el resto de B (4,2% vs. 10,0%) —
a diferencia del grupo A de sustancia callada, este grupo no viene
acompañado de mejor desempeño bursátil reciente, solo de mayor
sustancia/riesgo estructural.

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

- n=6 para el grupo de washing — cualquier estadístico ahí es
  descriptivo, no inferencial; no se puede correr un test de hipótesis
  serio con esa muestra.
- Igual que `04_...md`: sin efectos fijos de empresa, ratios con
  outliers winsorizados mecánicamente (2-98%), SIC-2 no usado aquí como
  control adicional (aunque los 6 candidatos de washing ya son
  visiblemente heterogéneos en sector, lo que sugiere que el patrón no
  es un artefacto de un solo sector).
- `rd_intensity` con varios NaN en la tabla de washing (CCL, FE, GPC,
  IQV no reportan R&D como línea XBRL separada) — consistente con "no
  son empresas de tecnología" pero también reduce la muestra efectiva
  para esa columna específica a 2 de 6 empresas.
