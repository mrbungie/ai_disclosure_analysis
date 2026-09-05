# Cruce con mercado y contabilidad (EE.UU.)

> **Recalculado con builders versionados (ver `10_builders_y_recalculo.md`).**
> El lado contable/mercado ya no viene de un script perdido: lo producen
> `build_firm_financials.py`, `build_market_factors.py` y `build_roic_wacc.py`
> (`make analytics`). Con mejor cobertura XBRL y el ERP corregido, varias
> cifras de este documento se movieron — las tablas de abajo son las de la
> corrida anterior; los deltas están listados en `10_...md`.

> **Recalculado 2026-09-05 con DEF 14A y 8-K.** El panel pasó de 1.229 a
> 1.363 empresas-año (429 → 454 empresas). Los insumos financieros y de
> mercado NO cambiaron — mismo XBRL, mismos precios; lo que se movió son
> las etiquetas de arquetipo y los `behavior_share_*`, porque K-means se
> re-ajustó sobre la población ampliada. Los parquets se rehacen con
> `scripts/analytics/build_firm_clusters.py` +
> `build_firm_panels.py`, dos scripts reconstruidos para esto (el
> generador original nunca se versionó); los anteriores están en
> `data/archive/processed/clusters/`. En el camino se corrigió un bug de
> reproducibilidad en la vista `gold_ai_frames` que hacía que cada
> consulta devolviera cifras distintas — ver `01_...md`.


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

> Recalculado 2026-09-05 sobre el panel ampliado (1.363 empresas-año, 454
> empresas, con DEF 14A y 8-K). Los insumos financieros no cambiaron —
> mismo XBRL, mismos precios; lo que cambió son las etiquetas de arquetipo
> y los `behavior_share_*`.

### 1. `revenue_outcome` (lo que dicen) vs. crecimiento de revenue real al año fiscal siguiente

Correlación entre `behavior_share_revenue_outcome` del año *t* y
`next_revenue_yoy`: **r = 0,091** (n=870, recortando outliers >300% YoY).
Prácticamente idéntica al 0,093 anterior, con 84 observaciones más.

Por arquetipo — crecimiento de revenue del FY **siguiente**:

| Arquetipo | Mediana next FY revenue YoY | Media | n |
|---|---|---|---|
| A cauteloso | 4,8% | 6,5% | 186 |
| B genérico | 7,2% | 9,3% | 294 |
| C cuantificador | 8,0% | **12,8%** | 113 |
| D vocal | **8,4%** | 9,9% | 277 |

El ranking ordinal A < B < C ≈ D se mantiene respecto de la versión
anterior. C y D quedan otra vez prácticamente empatados en mediana, con C
bastante arriba en media — el mismo patrón que reportaba la versión
original con el cluster C de 7 empresas, ahora con 113 observaciones
detrás en vez de 35.

### 2. `ai_investment` / `ai_infrastructure` vs. capex y R&D reales

| Comportamiento declarado (t) | vs. crecimiento real (FY t+1) | r | n |
|---|---|---|---|
| `ai_investment` | `capex_yoy` | 0,016 | 617 |
| `ai_investment` | `rd_expense_yoy` | 0,036 | 456 |
| `ai_infrastructure` | `capex_yoy` | **0,114** | 617 |
| `ai_infrastructure` | `rd_expense_yoy` | 0,014 | 456 |

El patrón central sobrevive: `ai_infrastructure` → `capex` sigue siendo
la única de las cuatro con magnitud apreciable (0,114, contra 0,113 en la
versión anterior — casi sin cambio). Las otras tres quedan en 0,014-0,036,
o sea nada.

La asimetría es el hallazgo: decir "invertimos en IA" no predice el capex
del año siguiente; decir "construimos infraestructura de IA" sí predice
algo, poco pero consistente entre poblaciones. Es la correlación más
estable de todo el cruce contable.

### 3. `cost_outcome` vs. SG&A real — la señal de "washing" desapareció

Correlación: **r = +0,011** (n=467), contra −0,027 en la versión anterior
— cambia de signo y queda en cero. Partiendo por intensidad de
`cost_outcome`:

| Grupo | SG&A YoY (FY t+1) mediana | n |
|---|---|---|
| Menos `cost_outcome` (mitad baja) | 5,1% | 357 |
| Más `cost_outcome` (mitad alta) | 5,2% | 110 |

La versión anterior reportaba esto como "la señal más cercana a washing
encontrada": quienes más enmarcaban la IA como ahorro de costos mostraban
SG&A creciendo MÁS (4,6% vs. 5,5%). **Con la población ampliada la
diferencia es de 0,1 p.p. y la correlación es cero.**

No es "antes había washing y ahora no". Es que **la señal nunca tuvo
fuerza suficiente para sobrevivir a un cambio de muestra** — era r=−0,027
sobre n=419. Es la lección metodológica más útil de esta actualización, y
`05_...md` la confirma: ese par queda último en el ranking FDR con
p=0,81.

(El corte por mediana queda desbalanceado, 357 vs. 110, porque la mayoría
de empresas-año tiene `cost_outcome` exactamente en 0 y cae del lado bajo.)

### 4. Intensidad de R&D por arquetipo (contemporánea) — posible confusor sectorial

| Arquetipo | R&D / revenue (mediana) |
|---|---|
| A cauteloso | 4,9% |
| B genérico | 7,0% |
| C cuantificador | 9,0% |
| D vocal | **13,0%** |

D casi triplica a A y el orden es monótono. Sigue en pie la advertencia:
es exactamente lo que predeciría la composición sectorial de D
(software/servicios) sin que el disclosure tenga nada que ver.
`04_...md` muestra que dentro de sector-año la brecha se reduce ~85%.

## Resultados: reacción de mercado al filing

Retorno crudo [-1, +5 días hábiles] alrededor del 10-K, por arquetipo:

| Arquetipo | Media | Mediana | Desv. est. | n |
|---|---|---|---|---|
| A cauteloso | 0,30% | 0,19% | 5,79% | 299 |
| B genérico | −0,22% | −0,06% | 6,57% | 479 |
| C cuantificador | 0,36% | −0,01% | 6,64% | 146 |
| D vocal | −0,15% | 0,26% | 6,09% | 375 |

Sin diferencias económicamente relevantes (medias dentro de ±0,4 p.p. con
desviaciones de 6-7%: ruido >> señal). El rango se estrechó respecto de
la versión anterior, donde C mostraba −1,26% de media con 42
observaciones de un cluster de 7 empresas.

Correlaciones directas:

- retorno vs. `promotional_rate` del filing: **r = −0,015**
- retorno vs. `specificity_index` del filing: **r = −0,030**
- retorno vs. volumen de menciones de IA (`n_frames`): **r = −0,054**

**Ninguna**, igual que antes y ahora con más datos.

Una nota de alcance que ahora importa más: `ret_m1_p5` se calcula sobre
la fecha de filing del **10-K**, pero el panel de texto mezcla frames de
10-K, DEF 14A y 8-K, así que el `promotional_rate` de una empresa-año
puede venir en parte de un proxy presentado en otra fecha. El cruce
texto→retorno quedó peor alineado que antes; separarlo por formulario es
trabajo pendiente.

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
