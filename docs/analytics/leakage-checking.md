# Leakage checking: alineamiento temporal en cruces con datos externos

Registro del chequeo de leakage/alineamiento temporal hecho sobre
`02_market_accounting_crosscheck.md`, y checklist a repetir en
cualquier análisis futuro que cruce `gold_ai_frames`/el panel
empresa-año con datos externos fechados (XBRL, precios, u otra fuente).

## Pregunta que originó el chequeo

Al construir el cruce "talk vs. walk" (¿lo que las empresas dicen sobre
IA predice resultados financieros reales?), la primera versión alineó
"año siguiente" restando 1 al año calendario del `period_end` de XBRL
y volviendo a mergear por año. Antes de reportar cualquier resultado,
correspondía verificar explícitamente: ¿ese shift realmente aterriza en
datos que NO existían todavía al momento del filing (correcto, es
"futuro"), o hay alguna forma de que esté usando información ya
conocida al momento en que se escribió el frame (leakage)?

## Cómo se verificó (no se asumió)

Se tomaron dos empresas con calendarios fiscales deliberadamente
distintos y se comparó, con fechas reales, `filing_date` (10-K) vs.
`period_end` (XBRL) del mismo documento:

```sql
-- filing_manifest: fechas reales de filing
-- El filtro form_type='10-K' es OBLIGATORIO hoy y no lo era cuando se
-- escribió este chequeo: `filing_manifest` ahora UNIONa DEF 14A y 8-K
-- junto al 10-K. Sin el filtro esta consulta devuelve 86 filas para KO
-- en vez de 6, y el alineamiento a año fiscal deja de tener sentido:
-- un 8-K no reporta ningún FY.
SELECT ticker, filing_date FROM filing_manifest
WHERE country_code='us' AND form_type='10-K' AND ticker IN ('KO','NVDA')
ORDER BY ticker, filing_date;
```

```python
# xbrl: period_end de la cifra de revenue anual reportada en cada filing
df = pd.read_parquet('data/raw/xbrl_facts/us/<TICKER>.parquet')
ann = df[(df.concept=='us-gaap:Revenues') & (duration_days between 340 and 380)]
```

Resultado:

| Ticker | Cierre fiscal | `filing_date` | `period_end` del FY disclosed | ¿Mismo año calendario? |
|---|---|---|---|---|
| KO | diciembre | 2024-02-20 | 2023-12-31 | **No** — desfase de 1 año |
| NVDA | enero | 2025-02-26 | 2025-01-26 | **Sí** — sin desfase |

Esto confirma con datos reales (no con una suposición sobre "las
empresas cierran en diciembre") que el desfase entre año-calendario-de-
`period_end` y año-calendario-de-`filing_date` **depende del mes de
cierre fiscal de cada empresa**, no es una constante aplicable a todo
el panel con un solo shift.

## Qué tan grave era

**No era leakage** en el sentido estricto (nunca se usó un valor que no
existiera aún en la fecha del filing — el shift fijo de −1 apuntaba a
datos futuros en ambos casos, solo que a un horizonte distinto según la
empresa). El problema real:

- Para empresas de cierre no-enero (la mayoría, cierre Q4 sobre todo):
  la columna etiquetada "next_year" en realidad apuntaba **dos años
  fiscales adelante** del FY disclosed en el documento, no uno — un
  horizonte de predicción más largo y más ruidoso del que se pretendía
  medir.
- Para empresas de cierre enero-como-NVDA: el shift caía por
  casualidad en el año correcto.
- El resultado: el panel mezclaba horizontes de predicción distintos
  (1 año fiscal adelante vs. 2) según el mes de cierre de cada
  empresa, sin que el código lo supiera ni lo declarara — ruido
  sistemático, no aleatorio, y specific a la composición sectorial
  (los sectores con cierre no-calendario, como retail con cierre
  enero/febrero, quedaban alineados distinto que el resto).

## Corrección aplicada

Reemplazar el shift de año calendario por una búsqueda basada en fechas
reales, por ticker:

1. Ordenar los `period_end` anuales (duración 340-380 días) de CADA
   ticker.
2. Para cada `filing_date`, ubicar el `period_end` más reciente
   `<= filing_date` (+10 días de margen) → ese es el FY **disclosed
   en ese documento** — dato ya conocido, válido como covariable
   contemporánea pero NUNCA como "outcome futuro a predecir".
3. Tomar el `period_end` INMEDIATAMENTE siguiente de ese mismo ticker
   → ese es el FY genuinamente futuro (no había terminado todavía
   cuando se filed el documento) — el correcto para testear si el
   comportamiento declarado predice el resultado real.

```python
idx_disclosed = np.searchsorted(pe, filing_date + pd.Timedelta(days=10), side='right') - 1
disclosed = fg.iloc[idx_disclosed]          # ya conocido al filing — no usar como outcome
next_fy    = fg.iloc[idx_disclosed + 1]      # futuro real — sí usar como outcome
next_revenue_yoy = next_fy.revenue / disclosed.revenue - 1
```

Ver `data/processed/clusters/firm_year_financials.parquet` (v2) y el
script referenciado en `02_market_accounting_crosscheck.md`.

## Verificación del fix sobre TODO el panel, no solo KO/NVDA

KO y NVDA solo sirvieron para DEMOSTRAR que el bug existía. El fix
(`build_financials_v2.py`) corre la búsqueda por fecha real
independientemente para cada uno de los 429 tickers — pero eso hay que
verificarlo, no darlo por sentado solo porque el código "se ve"
correcto. Chequeo: sobre las 2.325 filas con FY siguiente disponible,
¿el gap entre `disclosed_period_end` y `next_period_end` es
consistentemente ~365 días (un año fiscal), para cualquier ticker?

```python
fin['gap_days'] = (fin['next_period_end'] - fin['disclosed_period_end']).dt.days
pd.cut(fin['gap_days'], [0,340,380,720,760,2000]).value_counts()
```

| Rango de gap | n filas |
|---|---|
| 340-380 días (correcto, ~1 año fiscal) | **2.319 (99,7%)** |
| 0-340 días | 3 |
| 380-720 días | 1 |
| 760-2000 días | 2 |

El fix generaliza: 99,7% de las filas quedan en la ventana correcta de
un año fiscal, para todos los meses de cierre fiscal del panel, no solo
para los dos casos usados para demostrar el bug original.

**3 excepciones residuales** (UAA, USB×2) con gaps de 455 a 1.461 días
— no es el bug del shift calendario (ya corregido), es un problema
distinto: **huecos de datos en XBRL** para esos tickers (años faltantes
entre medio), que hacen que "el próximo `period_end` disponible" esté
2-4 años adelante en vez de 1. Impacto: 3 de 2.325 filas (0,13%) —
no invalida la corrección, pero cualquier análisis futuro que use este
panel debería filtrar `gap_days` fuera de `[340, 380]` antes de tratar
`next_*_yoy` como "un año fiscal adelante" con confianza.

## Qué NO tenía este problema (verificado, no asumido)

- **Retornos de mercado** (`firm_year_filing_returns.parquet`): usan
  `filing_date` real de punta a punta (`searchsorted` sobre la propia
  serie de precios), nunca pasan por un bucket de año calendario. Sin
  desfase posible de este tipo.
- **El panel de arquetipos/comportamientos** (`firm_year_archetype_behaviors.parquet`,
  `01_ai_disclosure_analytics.md`): `year` = año de `filing_date` del
  documento que contiene cada frame — es la fecha de origen, no un
  dato derivado que pudiera desalinearse.
- **La sección de trend-break SEC/DeepSeek** (`01_...md`): usa
  trimestres calendario (`date_trunc('quarter', filing_date)`)
  directamente sobre `filing_date`, sin ningún paso de shift — no
  aplica este tipo de bug (aunque tiene sus propias limitaciones,
  documentadas ahí, sobre solapamiento de cortes).

## Checklist para el próximo cruce con datos fechados

Antes de reportar cualquier correlación "texto del año *t* → outcome
externo":

1. **¿El outcome usa una fecha real (`filing_date`, `period_end`,
   `date`) de punta a punta, o pasa por un bucket de año/trimestre
   calendario en algún punto?** Si pasa por un bucket, verificar con
   2+ casos reales de calendarios distintos (como aquí KO vs. NVDA)
   que el bucket signifnow lo mismo para todas las entidades del
   panel — nunca asumirlo.
2. **¿"Mismo período" realmente es contemporáneo, o ya es
   look-ahead/lookback por construcción?** Un 10-K reporta un FY que
   ya terminó — "mismo año que el filing" casi nunca es "el año que se
   está viviendo al momento del filing".
3. **¿"Año siguiente" es genuinamente un dato que no existía aún en la
   fecha del filing?** Verificar con la fecha real de cierre de ese
   período, no con aritmética de año calendario.
4. Si la fuente tiene calendarios heterogéneos por entidad (fiscal
   year end, distintos exchanges con distinto trading calendar, etc.),
   **el alineamiento debe hacerse por entidad**, nunca con un shift
   global.
5. **¿La consulta filtra por `form_type`?** (Agregado 2026-09-05.)
   `filing_manifest` dejó de ser una tabla de 10-K: hoy tiene 35.829
   8-K, 2.898 10-K y 2.883 DEF 14A para EE.UU. Cualquier cruce que
   alinee texto contra un período fiscal tiene que restringirse al
   formulario que efectivamente reporta ese período — el 8-K es un
   evento puntual y la DEF 14A sigue el calendario de la junta de
   accionistas, no el FY. Un JOIN sin filtro multiplica las filas por
   empresa-año en vez de fallar, así que el error es silencioso.
