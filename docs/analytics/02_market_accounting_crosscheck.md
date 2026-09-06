# Cruce con mercado y contabilidad (EE.UU.)

Todas las cifras salen de la corrida vigente de `make analytics`
(`report_crosscheck_stats.py` reproduce las tablas numéricas) sobre el panel
de 1.426 empresas-año y 460 empresas: frames de 10-K, DEF 14A y 8-K,
población marcada por el prefiltro v2 (árboles, umbral 0,17 —
`prefilter_evaluation.md` §8.16), lado contable/mercado de
`build_firm_financials.py`, `build_market_factors.py` y `build_roic_wacc.py`
(ERP geométrico 6,48%, `10_builders_y_recalculo.md`).

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

Correlación entre `behavior_share_revenue_outcome` del año *t* y
`next_revenue_yoy`: **r = 0,049** (n=939, recortando outliers >300% YoY).
El test de permutación de `05_...md` la deja en p=0,12 — indistinguible de
cero.

Por arquetipo — crecimiento de revenue del FY **siguiente**:

| Arquetipo | Mediana next FY revenue YoY | Media | n |
|---|---|---|---|
| A cauteloso | 5,0% | 5,9% | 243 |
| B genérico | 7,1% | 9,9% | 309 |
| C cuantificador | 7,4% | **11,8%** | 105 |
| D vocal | **8,6%** | 11,2% | 282 |

El ranking ordinal es A < B < C ≈ D: C y D casi empatados en mediana, con
C arriba en media por su cola derecha (semis en años de expansión). El
orden por arquetipo existe aunque la correlación lineal no: la diferencia
está en los extremos (A contra D), no en el gradiente continuo.

### 2. `ai_investment` / `ai_infrastructure` vs. capex y R&D reales

| Comportamiento declarado (t) | vs. crecimiento real (FY t+1) | r | n |
|---|---|---|---|
| `ai_investment` | `capex_yoy` | −0,007 | 840 |
| `ai_investment` | `rd_expense_yoy` | 0,044 | 512 |
| `ai_infrastructure` | `capex_yoy` | **0,086** | 840 |
| `ai_infrastructure` | `rd_expense_yoy` | 0,019 | 512 |

`ai_infrastructure` → `capex` es la única de las cuatro con alguna
magnitud (0,086). Las otras tres quedan entre −0,007 y 0,044, o sea nada.
Con cobertura de capex parcial (sólo los filers que reportan el tag más
común) la correlación sube a ~0,11; con la cobertura completa de 87%
(`10_...md`) baja a 0,086 — se apoya en parte en qué filers tienen dato.

La asimetría es el hallazgo: decir "invertimos en IA" no predice el capex
del año siguiente; decir "construimos infraestructura de IA" sí predice
algo, poco. Es la mayor del cruce contable y no pasa el FDR de `05_...md`
(p=0,013 contra un umbral BH de 0,0045).

### 3. `cost_outcome` vs. SG&A real — la señal de "washing" desapareció

Correlación: **r = −0,011** (n=799): cero. Partiendo por intensidad de
`cost_outcome`:

| Grupo | SG&A YoY (FY t+1) mediana | n |
|---|---|---|
| Menos `cost_outcome` (mitad baja) | 5,7% | 585 |
| Más `cost_outcome` (mitad alta) | 6,0% | 214 |

La hipótesis de washing acá sería "quienes más enmarcan la IA como
ahorro de costos muestran SG&A creciendo MÁS". **La diferencia es de 0,3
p.p., la correlación es cero, y con n=799 no es cuestión de muestra
chica.** `05_...md` lo confirma: ese par queda noveno de once en el
ranking FDR con p=0,75. No hay señal de washing en el cruce de costos.

(El corte por mediana queda desbalanceado, 585 vs. 214, porque la mayoría
de empresas-año tiene `cost_outcome` exactamente en 0 y cae del lado bajo.)

### 4. Intensidad de R&D por arquetipo (contemporánea) — posible confusor sectorial

| Arquetipo | R&D / revenue (mediana) |
|---|---|
| A cauteloso | 5,3% |
| B genérico | 6,9% |
| C cuantificador | 8,6% |
| D vocal | **13,2%** |

D multiplica por 2,5 a A y el orden es monótono. Sigue en pie la advertencia:
es exactamente lo que predeciría la composición sectorial de D
(software/servicios) sin que el disclosure tenga nada que ver.
`04_...md` muestra que dentro de sector-año la brecha se reduce ~85%.

## Resultados: reacción de mercado al filing

Retorno crudo [-1, +5 días hábiles] alrededor del 10-K, por arquetipo:

| Arquetipo | Media | Mediana | Desv. est. | n |
|---|---|---|---|---|
| A cauteloso | 0,12% | 0,27% | 5,91% | 368 |
| B genérico | −0,16% | −0,11% | 6,54% | 497 |
| C cuantificador | 0,63% | 1,01% | 6,74% | 130 |
| D vocal | −0,17% | −0,11% | 6,35% | 363 |

Sin diferencias económicamente relevantes (medias dentro de ±0,7 p.p. con
desviaciones de 6-7%: ruido >> señal).

Correlaciones directas:

- retorno vs. `promotional_rate` del filing: **r = −0,018**
- retorno vs. `specificity_index` del filing: **r = −0,008**
- retorno vs. volumen de menciones de IA (`n_frames`): **r = −0,050**

**Ninguna.**

Una nota de alcance que ahora importa más: `ret_m1_p5` se calcula sobre
la fecha de filing del **10-K**, pero el panel de texto mezcla frames de
10-K, DEF 14A y 8-K, así que el `promotional_rate` de una empresa-año
puede venir en parte de un proxy presentado en otra fecha. El cruce
texto→retorno quedó peor alineado que antes; separarlo por formulario es
trabajo pendiente.

## Lectura conjunta

De los tres tipos de cruce (revenue, insumos de inversión, mercado),
**ninguno muestra una relación que sobreviva a la corrección por
comparaciones múltiples** (`05_...md`). Lo que queda de revenue es un
orden por arquetipo (D y C crecen más al año fiscal siguiente que A y
B, 8,6% y 7,4% contra 5,0% y 7,1% de mediana) con una correlación lineal
de 0,049 que la permutación no distingue de cero, y que `04_...md`
reduce a 0,019 dentro de sector-año. Insumos (`ai_investment`,
`ai_infrastructure`) y mercado no muestran ninguna relación, y
`cost_outcome` quedó en cero exacto. Es un resultado calibrado, no
negativo: el instrumento de texto no predice resultados financieros al
año siguiente, y ahora eso está medido sobre 799-939 observaciones con
cobertura XBRL decente (`10_...md`), no sobre la mitad de la muestra.

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
