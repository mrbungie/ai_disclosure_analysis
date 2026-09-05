# Cruce con mercado y contabilidad (EE.UU.)

Primer cruce de los arquetipos/comportamientos de divulgación de IA
(`01_ai_disclosure_analytics.md`) con datos externos al texto: XBRL
(¿la sustancia declarada se refleja en los números?) y precios
(¿el mercado reacciona al contenido del filing, no solo a su
existencia?). Responde al hueco identificado explícitamente en la
revisión de literatura de `docs/thesis_proposal.md` (Eisfeldt et al.,
Basnet et al. — reacción de mercado a narrativas de IA), que hasta
ahora no tenía ningún análisis conectándolo con el trabajo de NLP.

**Estado: primer pase exploratorio, no resultado de tesis.** Los
números de abajo son reales y las correlaciones están bien calculadas,
pero el diseño es simple (sin ajuste por mercado/beta en retornos, sin
efectos fijos de empresa/sector en las regresiones de "talk vs. walk")
— ver limitaciones al final antes de citar cualquier cifra.

> **Corrección aplicada (revisión de alineamiento temporal)**: la
> primera versión de esta sección calculaba "año siguiente" con un
> shift fijo de −1 año calendario sobre `period_end`, asumiendo cierre
> fiscal en diciembre para todas las empresas. Es falso para una
> fracción grande del panel — NVDA, por ejemplo, cierra en enero, así
> que su 10-K filed en `2025-02-26` reporta el FY que termina
> `2025-01-26` (mismo año calendario, sin desfase), mientras que KO
> (cierre diciembre) filed en `2024-02-20` reporta el FY que termina
> `2023-12-31` (desfasado un año). El shift fijo no era leakage (nunca
> se usaron datos que no existieran aún al momento del filing — el
> error apuntaba a datos DEMASIADO futuros, dos años fiscales adelante
> en vez de uno, para las empresas con cierre no-enero), pero mezclaba
> horizontes distintos entre empresas y estaba mal etiquetado. Se
> corrigió usando el calendario fiscal real de cada ticker (ver
> "Datos y construcción" abajo) y se recalcularon todos los números de
> esta sección — las conclusiones cualitativas se mantienen, con
> cobertura mejor (787 vs. 548 observaciones para el cruce de
> `revenue_outcome`, porque ahora el alineamiento funciona para
> cualquier mes de cierre fiscal, no solo por coincidencia calendario).
> Auditoría completa del bug, la verificación del fix sobre las 2.325
> filas del panel (99,7% con gap correcto de 340-380 días) y un
> checklist para futuros cruces con datos fechados: ver
> `leakage-checking.md`.

> **Actualización posterior (`04_ratios_factors_and_volatility.md`)**:
> dos conclusiones de esta sección se debilitan al agregar control
> sectorial y ajuste por riesgo — la correlación `revenue_outcome` →
> crecimiento real cae de r≈0,09 a r≈0,03 dentro de sector-año, y la
> mayor intensidad de R&D de D resultó ser casi enteramente composición
> sectorial (dentro de su sector, D gasta lo esperado, no más). Leer
> `04_...md` antes de citar cualquier número de esta sección como
> evidencia de "sustancia real".

## Datos y construcción

Tres tablas nuevas, todas en `data/processed/clusters/`, todas
derivadas de datos ya existentes (sin nueva extracción LLM):

| Archivo | Contenido | Fuente |
|---|---|---|
| `firm_year_financials.parquet` | revenue, R&D, capex, SG&A del FY disclosed en cada 10-K + YoY del FY SIGUIENTE, alineado por fecha real de filing, por (ticker, año) | `data/raw/xbrl_facts/us/*.parquet` + `filing_manifest` |
| `firm_year_filing_returns.parquet` | retorno de ventana corta [-1, +5 días hábiles] alrededor de la fecha del 10-K, por (ticker, año) | `data/raw/market/prices/*.parquet` (yfinance) |
| `firm_year_full_crosscheck.parquet` | merge de las dos anteriores con `firm_year_archetype_behaviors.parquet` (`01_...md`, sección "Panel empresa-año") | — |

**Financials — alineamiento por fecha real, no por año calendario.**
Para cada `(ticker, filing_date)` de un 10-K:
1. Se busca, entre los hechos XBRL de duración anual (340-380 días) de
   ESE ticker, el `period_end` más reciente que sea `<= filing_date`
   (con 10 días de margen) → es el FY efectivamente **disclosed en ese
   documento** (dato ya conocido al momento del filing, no un outcome
   futuro).
2. Se busca el **siguiente** `period_end` de ese mismo ticker (el FY
   que empieza justo después del disclosed) → es el FY que todavía NO
   había terminado cuando se filed el documento, genuinamente futuro.
   `next_<metric>_yoy` = crecimiento de esa métrica entre el FY
   disclosed y ese FY siguiente.
3. XBRL repite cada cifra anual en 2-3 filings distintos (comparativos)
   — se dedupea a un valor por `(ticker, concept, period_end)` con la
   mediana de las repeticiones antes del paso 1-2.

Cobertura: 429/429 tickers del panel tienen algo de XBRL, pero `capex`
tiene huecos grandes (el tag `PaymentsToAcquirePropertyPlantAndEquipment`
no todos lo reportan de forma consistente).

**Retornos**: `adj_close` del día hábil anterior al `filing_date` del
10-K hasta 5 días hábiles después — 2.746 observaciones. **Sin ajuste
por mercado** (no se resta un benchmark ni se calcula beta) — es
retorno crudo de la acción, no retorno anormal. Esta parte no tenía el
bug de alineamiento (usa `filing_date` real de principio a fin, sin
pasar por año calendario). 411/429 tickers del panel tienen cobertura
de precios.

```python
# financials: dedup + filtro de duración anual + alineamiento por fecha real
CONCEPTS = {'us-gaap:Revenues': 'revenue',
            'us-gaap:ResearchAndDevelopmentExpense': 'rd_expense',
            'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment': 'capex',
            'us-gaap:SellingGeneralAndAdministrativeExpense': 'sga_expense'}
df['duration_days'] = (df['period_end'] - df['period_start']).dt.days
df = df[(df['duration_days'] >= 340) & (df['duration_days'] <= 380)]
dd = df.groupby(['metric','period_end'])['value'].median()  # dedup re-reportings

# por (ticker, filing_date): último period_end <= filing_date = FY disclosed;
# el period_end INMEDIATAMENTE siguiente de ESE ticker = FY futuro real
idx_disclosed = np.searchsorted(pe, filing_date + pd.Timedelta(days=10), side='right') - 1
disclosed = fg.iloc[idx_disclosed]
next_fy   = fg.iloc[idx_disclosed + 1]
next_revenue_yoy = next_fy.revenue / disclosed.revenue - 1

# retornos: ventana [-1, +5] días hábiles alrededor del filing_date
idx = px['date'].searchsorted(filing_date)
p0 = px['adj_close'].iloc[idx - 1]
p1 = px['adj_close'].iloc[idx + 5]
ret_5d = p1 / p0 - 1
```

## Resultados: "talk vs. walk"

### 1. `revenue_outcome` (lo que dicen) vs. crecimiento de revenue real al año fiscal siguiente

Correlación entre `behavior_share_revenue_outcome` del año *t* y el
crecimiento real de revenue del FY siguiente (`next_revenue_yoy`):
**r = 0,093** (n=786, recortando outliers >300% YoY). Débil pero
positiva y en la dirección esperada — hay algo de sustancia real
detrás de la narrativa de revenue, no es puro ruido.

Por arquetipo — crecimiento de revenue del FY **siguiente**:

| Arquetipo | Mediana next FY revenue YoY | Media | n |
|---|---|---|---|
| A cauteloso | 4,7% | 6,1% | 165 |
| B genérico | 7,0% | 8,8% | 310 |
| C cuantificador | 9,3% | 10,5% | 35 |
| D vocal | **9,3%** | **12,6%** | 276 |

Con el alineamiento corregido, C y D quedan prácticamente empatados en
mediana (ambos ~9,3%), con D todavía arriba en media. A sigue siendo,
con claridad, el arquetipo con menor crecimiento subsecuente — y sigue
siendo el que menos habla de resultados (§ Comportamiento por
arquetipo, `01_...md`). El ranking ordinal (A < B < C≈D) se mantiene
respecto a la versión anterior del análisis, aunque C se movió de
"medio" a "empatado con D".

### 2. `ai_investment` / `ai_infrastructure` vs. capex y R&D reales

Correlación con crecimiento real de capex/R&D del FY siguiente:

| Comportamiento declarado (t) | vs. crecimiento real (FY t+1) | r | n |
|---|---|---|---|
| `ai_investment` | `capex_yoy` | 0,034 | 563 |
| `ai_investment` | `rd_expense_yoy` | 0,009 | 418 |
| `ai_infrastructure` | `capex_yoy` | 0,113 | 563 |
| `ai_infrastructure` | `rd_expense_yoy` | 0,084 | 418 |

**Corrección (`05_circularity_and_robustness_checks.md`)**: `ai_infrastructure`
vs. `capex_yoy` (r=0,113) SÍ es estadísticamente significativo — de
hecho es la correlación más significativa de las 11 testeadas en todo
el cruce contable/mercado, incluso tras corrección por comparaciones
múltiples (FDR). "Sin señal" abajo se refiere a magnitud económica
chica, no a que la relación sea estadísticamente inexistente — los
otros 3 pares de esta tabla sí quedan sin señal tras esa corrección.

Magnitud económica chica en los cuatro casos. Hablar de "estamos
invirtiendo en IA" o "infraestructura de IA" predice muy poco del capex
o R&D real del año siguiente (aunque el par `ai_infrastructure`/`capex_yoy`
es estadísticamente real, ver corrección arriba). Esto contrasta con el
resultado de revenue abajo — la narrativa de resultado (revenue) tiene
algo de sustancia, la narrativa de insumo
(inversión) no, al menos a este nivel de agregación anual.

### 3. `cost_outcome` vs. SG&A real — la señal más cercana a "washing" encontrada

Correlación: **r = −0,027** (n=419) — sigue siendo negativa (dirección
contraria a lo esperado), aunque más débil que en la versión anterior
del análisis. Partiendo la muestra en dos por intensidad de
`cost_outcome`:

| Grupo | SG&A YoY (FY t+1) mediana | n |
|---|---|---|
| Menos `cost_outcome` (mitad baja) | 4,6% | 210 |
| Más `cost_outcome` (mitad alta) | 5,5% | 209 |

Sigue siendo la señal más parecida a "washing" del pase: las empresas
que más enmarcan la IA como ahorro de costos no muestran ningún ahorro
real medible al año fiscal siguiente. Débil (n chico, sin controles) y
más débil que antes de la corrección — hay que tratarla como
sugerente, no concluyente.

### 4. Intensidad de R&D por arquetipo (contemporánea, disclosed en el mismo documento) — posible confusor sectorial

| Arquetipo | R&D / revenue (mediana) |
|---|---|
| A cauteloso | 4,3% |
| B genérico | 7,9% |
| C cuantificador | 6,6% |
| D vocal | **14,1%** |

D tiene más del triple de intensidad de R&D que A. Esto es coherente
con "D tiene más sustancia real" (§ resultado 1), pero también es
exactamente lo que predeciría la composición sectorial de D (semis +
software, ya documentado en `01_...md` — NVDA, MSFT, ADBE, SNOW) sin
que el comportamiento de disclosure tenga nada que ver. **No se puede
todavía separar "D crece más porque divulga distinto" de "D crece más
porque es del sector que estructuralmente crece más" con este diseño**
— falta un control por sector/industria.

## Resultados: reacción de mercado al filing

Retorno crudo [-1, +5 días hábiles] alrededor del 10-K, por arquetipo
(sin cambios respecto a la versión anterior — esta parte no tenía el
bug de alineamiento):

| Arquetipo | Media | Mediana | Desv. est. | n |
|---|---|---|---|---|
| A cauteloso | −0,01% | 0,26% | 5,38% | 287 |
| B genérico | 0,08% | −0,11% | 6,69% | 495 |
| C cuantificador | −1,26% | −0,27% | 7,39% | 42 |
| D vocal | −0,16% | 0,11% | 6,58% | 360 |

Sin diferencias económicamente relevantes entre arquetipos (todas las
medias están dentro de ±1,3 p.p., con desviaciones de 5-7%, es decir,
ruido >> señal). Correlaciones directas:

- retorno vs. `promotional_rate` del filing: **r = −0,013**
- retorno vs. `specificity_index` del filing: **r = −0,054**
- retorno vs. volumen de menciones de IA (`n_frames`): **r = −0,042**

**Ninguna.** En esta ventana de 5 días y sin ajuste por mercado, el
tono o la especificidad del contenido de IA del filing no se relaciona
con el retorno alrededor de la publicación. Partiendo pre/post SEC 2024
(año ≥2024), la correlación `promotional_rate` vs. retorno pasa de
0,002 a 0,00 — tampoco hay evidencia de que el mercado empezara a
"castigar" el lenguaje promocional después del escrutinio SEC, al
menos con este diseño.

## Lectura conjunta

De los tres tipos de cruce (revenue, insumos de inversión, mercado),
**solo el de revenue muestra sustancia real** (débil pero consistente:
r=0,09 y D/C crecen más al año fiscal siguiente que A/B). Insumos
(`ai_investment`, `ai_infrastructure`) y mercado no muestran ninguna
relación, y `cost_outcome` apunta levemente en la dirección de
"washing" (afirma ahorro que no se materializa, aunque la señal se
debilitó tras la corrección de alineamiento). Esto es un resultado
calibrado, no negativo: si el diseño fuera capaz de detectar cualquier
relación espuria, vería relaciones espurias en todos los cruces por
igual — el hecho de que solo revenue muestre señal, y en la dirección
teóricamente correcta, es evidencia (débil) de que el instrumento SÍ
mide algo real, no solo ruido correlacionado con volumen de texto.

## Limitaciones (leer antes de citar cualquier número de esta sección)

- **Sin ajuste de mercado en los retornos** — son retornos crudos, no
  anormales. Con AI/tech en fuerte alza 2024-2026, cualquier cruce con
  arquetipo (más pesado en tech) puede confundirse con beta de mercado,
  no con contenido del filing. Próximo paso obligatorio: restar un
  benchmark (o CAPM de un factor) antes de interpretar magnitudes.
- **Sin control por sector/industria** en ninguno de los cruces
  financieros — el resultado de revenue y R&D de D/C puede ser
  composición sectorial, no comportamiento de disclosure. Necesita
  regresión con efectos fijos de sector (SIC) como mínimo.
- **Correlaciones a nivel empresa-año agregado, no panel con efectos
  fijos de empresa** — no se controla por tendencia propia de cada
  empresa (una empresa que siempre crece rápido y siempre habla de
  revenue contamina la correlación sin que haya relación causal
  filing→resultado).
- **`capex` tiene cobertura XBRL incompleta** (tag inconsistente entre
  filers) — los resultados de capex son los menos confiables de la
  sección.
- **Ventana de retorno fija en 5 días hábiles**, sin robustez probada
  con otras ventanas (1, 3, 10, 20 días) — un resultado nulo en una
  ventana no descarta reacción de mercado en otra.
- El alineamiento financiero usa el `period_end` XBRL más reciente con
  10 días de margen respecto al `filing_date` — filings con retraso
  inusual en el reporte XBRL respecto al 10-K en sí podrían quedar mal
  clasificados en el borde; no verificado sistemáticamente.
- 3 de 2.325 filas (0,13%) tienen `next_period_end` a 2-4 años del FY
  disclosed, no a 1 (huecos de datos XBRL en esos tickers, no el bug de
  calendario) — impacto despreciable en los resultados agregados pero
  no filtrado explícitamente; ver `leakage-checking.md`.
- Igual que el resto del proyecto: solo EE.UU., sin ponderar por
  `inclusion_weight`, población filtrada por el prefiltro.
