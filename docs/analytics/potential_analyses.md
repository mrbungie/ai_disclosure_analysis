# Backlog de análisis pendientes

Ideas de análisis ya diseñadas (a veces con script parcialmente
corrido) que quedaron pendientes de terminar — para no perderlas entre
sesiones. Mover cada ítem a un doc numerado (`0N_...md`) cuando se
complete, y borrarlo de aquí.

## 1. Proyección año-a-año del cluster de comportamiento (verificar persistencia real de washing/sustancia callada)

**Por qué**: `06_voice_vs_behavior_clustering.md` y `07_...md`/`08_...md`
construyeron el cluster de comportamiento sobre el POOL de todos los
años por empresa, y luego se descubrió (ver corrección en `06_...md`)
que la lista de "candidatos a washing" no persiste bien cuando se mira
el ARQUETIPO DE VOZ año a año — la mayoría tiene 1-2 años de datos o
migra de arquetipo. (Actualizado 2026-09-05: con DEF 14A y 8-K la lista
pasó a 20 empresas y sólo 2 —AAPL, CL— sostienen la etiqueta en todos
sus años con ≥3 años de panel, así que el problema empeoró en vez de
resolverse. Ver `06_...md`.) Pero esa
verificación solo miró la voz por año; el CLUSTER DE COMPORTAMIENTO
nunca se proyectó por año — sigue siendo pooled. Falta hacer el mismo
ejercicio para comportamiento: proyectar cada firma-año sobre los
centroides de comportamiento ya entrenados (igual método que el panel
de arquetipos de voz en `01_...md`, sección "Panel empresa-año"), y
cruzar voz-por-año × comportamiento-por-año para tener una etiqueta de
washing/sustancia-callada verdaderamente año a año, no una mezcla de
pooled+año como se hizo hasta ahora.

**Estado**: script ya escrito y probado parcialmente
(`scratchpad/behavior_evolution.py` de la sesión, no versionado en el
repo — reconstruir desde esta descripción si se perdió):

```python
# 1. entrenar centroides de comportamiento sobre el pool (>=5 frames), como en 06_...md
# 2. agregar por (ticker, year) con umbral >=3 frames/año (mismo que el panel de voz)
# 3. proyectar cada firma-año sobre esos centroides (.transform + argmin), NO re-clusterizar
# 4. guardar en data/processed/clusters/firm_year_behavior_cluster.parquet
# 5. merge con firm_year_archetype_behaviors.parquet (voz por año) -> cruce año a año real
# 6. redefinir is_washing_year = (archetype_año=='D_vocal_leader') & (behavior_cluster_año=='minimo')
# 7. ver evolución de wy (empresas-año marcadas) y comparar contra la lista pooled original
```

**Bloqueado por**: la corrida se interrumpió porque `duckdb/thesis.duckdb`
estaba bloqueado por un proceso `build_duckdb.py --with-text-tables`
corriendo en paralelo (posiblemente relanzado más de una vez) — en un
momento la tabla `paragraphs` no existía (reemplazada por
`unique_paragraphs`), sugiriendo que el rebuild cambió el esquema del
que depende la vista `gold_ai_frames`. Antes de reintentar:
1. Confirmar que el rebuild terminó limpio (sin proceso
   `build_duckdb.py` corriendo).
2. Verificar que `gold_ai_frames` sigue siendo consultable
   (`SELECT COUNT(*) FROM gold_ai_frames LIMIT 1`) y que su definición
   no quedó rota por el cambio `paragraphs` → `unique_paragraphs`.
3. Recién ahí re-correr el script de proyección de comportamiento.

**Qué se espera encontrar**: probablemente una historia parecida a la
de la voz — GPC persistiendo como candidato "real", CCL/FE/NEM
demasiado poco dato para concluir, HII migrando. Pero también podría
revelar candidatos NUEVOS que la lista pooled no capturó (empresas
con voz D estable pero comportamiento que se degrada específicamente
en años recientes, invisible en el agregado pooled).

## 2. Placebo test para las otras 10 correlaciones de la tabla FDR

`05_circularity_and_robustness_checks.md` solo corrió el permutation
test para `revenue_outcome`→`next_revenue_yoy`. Repetir para las
otras 10 filas de esa tabla (especialmente `ai_infrastructure`→`capex_yoy`,
que resultó significativa tras FDR pero nunca se sometió a placebo)
daría el chequeo completo.

## 3. Efectos fijos de empresa para TODAS las relaciones de `02_...md`/`04_...md`

Solo se corrió efectos fijos de empresa para `revenue_outcome`→
`next_revenue_yoy` (§2 de `05_...md`). El mismo diseño (primeras
diferencias / demeaning por ticker) debería repetirse para
`ai_infrastructure`→`capex_yoy`, `cost_outcome`→`sga_expense_yoy`, y
el retorno de mercado (CAR) vs. `promotional_rate`/`specificity_index`
— para saber si esas relaciones también sobreviven o se explican por
heterogeneidad fija de empresa.

## 4. k del clustering de comportamiento y de voz — validar con más de una semilla/k

Tanto el arquetipo de voz (`01_...md`) como el de comportamiento
(`06_...md`) usan k=4 con seed=42 sin barrido sistemático de
estabilidad. Repetir con k=3,5,6 y varias semillas, reportar qué
tan estable es la partición (especialmente el cluster C de voz,
n=7, y el cluster 0 de comportamiento, n=26 — ambos chicos y
candidatos a ser artefacto de la semilla).

## 5. WACC con ERP dinámico y costo de deuda mejor estimado

`08_roic_wacc_value_creation.md` usa un ERP fijo (8,2%) para todo el
panel y un fallback grueso de costo de deuda (rf+2%) cuando falta
`InterestExpense`. Un ERP por año (o por década) y una estimación de
costo de deuda por rating/spread de crédito (si se consigue esa data)
reduciría el ruido de las comparaciones entre segmentos.
