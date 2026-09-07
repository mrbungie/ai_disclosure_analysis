# Brecha promocional entre canales: la misma empresa, el mismo ejercicio, call contra filing

Producido por `scripts/analytics/channel_gap_analysis.py`. Determinístico, sin
LLM. Es el diseño que `docs/pregunta_identificacion_sec.md` plantea como el
único capaz de separar el efecto del regulador del boom de IA, y el que
corresponde al mecanismo del caso Welltower: la SEC no objetó el 10-K por
promocional, objetó que la earnings call dijera "industry-leading" sin
respaldo proporcional en el 10-K. **El objeto es la brecha entre canales, no
el nivel en uno.**

**Modo de análisis final: margen extensivo.** Celda = empresa × ejercicio
fiscal con al menos una transcripción y un filing; outcome = intensidad por
1.000 párrafos de cada canal, **cero** cuando el canal no habla de IA. 2.281
celdas, 479 empresas, 449 a ambos lados de 2024; 1.021 celdas tienen calls sin
ningún frame de IA (5.238 de las 7.947 transcripciones no mencionan IA).

## El diseño

```
y[i, canal, t] = a[i, t] + b · (post[t] × es_filing[canal]) + e
```

Con efectos fijos empresa × período y exactamente dos canales por celda, el
estimador es la diferencia dentro de la celda:

```
gap[i, t] = y[i, call, t] − y[i, filing, t] = a_i + c · post[t] + e
```

Todo lo que es común a la empresa en ese período —el boom de IA, su sector,
su ciclo— se cancela en la resta. Efectos fijos de empresa sobre la brecha,
errores clusterizados por empresa, sólo empresas presentes a ambos lados del
corte; event study por ejercicio con base 2021 y test conjunto de tendencias
previas; estimación ponderada por documentos y sin celdas que mezclan
documentos de antes y después del 2024-03-01, como robustez.

**El período es el año fiscal, alineado por lo que cada documento cubre**,
no por la fecha en que se presenta:

| documento | año fiscal asignado |
|---|---|
| 10-K, 10-Q | el de `period_end_date`, con el mes de cierre de cada empresa |
| earnings call | el del `document_id` (`TICKER_YYYYQn` es el trimestre fiscal discutido) |
| DEF 14A, 8-K | el de presentación (no cubren un período) |

`post` = ejercicio 2024 en adelante (aviso de la SEC sobre AI-washing en
diciembre de 2023, enforcement en marzo de 2024). Las celdas que mezclan
documentos de antes y después del 5 de diciembre de 2023 se marcan y se
reporta la estimación sin ellas.

| | |
|---|---|
| frames | 46.075 (29.808 en filings, 16.267 en calls) |
| documentos | 48.038 filings y 7.947 transcripciones, todos con su conteo de párrafos |
| cobertura de las calls | ejercicios 2021-2025, ~3,2 transcripciones por empresa-año; la última es de mediados de 2025 (los filings llegan a 2026Q3) |
| celdas empresa × ejercicio | **2.281**, de 479 empresas: 469 / 466 / 462 / 454 / 430 por ejercicio 2021-2025 |
| empresas a ambos lados de 2024 | **449**, 2.215 celdas |

## Resultado 1: la call tiene 20 veces más promoción de IA por párrafo que el filing

Media sobre las 2.281 celdas, por 1.000 párrafos:

| dimensión | call | filing | brecha | t contra 0 |
|---|---:|---:|---:|---:|
| frames de IA | 39,4 | 4,0 | +35,4 | |
| **promocionales** | **8,6** | **0,4** | **+8,2** | |
| cuantificados | 8,1 | 0,2 | +7,9 | |
| gobernanza | 0,7 | 0,4 | +0,3 | |
| documentos con alguna IA (share) | 0,55 | 0,65 | −0,10 | |

**La misma empresa, en el mismo ejercicio, pone 20 veces más afirmaciones
promocionales de IA por párrafo en la call que en el filing**, y 40 veces más
afirmaciones cuantificadas. El filing tiene más documentos con alguna mención
de IA (65% contra 55%) pero a una densidad diez veces menor: la IA en el
filing es un factor de riesgo y una línea de gobernanza; en la call es el
producto. **Es una brecha promocional entre canales**, no evidencia de que
las afirmaciones sean falsas: es el tipo de desajuste entre canales que la SEC
señaló en el caso Welltower, medido en todo el corpus.

## Resultado 2: la brecha se disparó con el boom de 2023, y el escrutinio de 2024 no la tocó

Efecto de `post` sobre la brecha, efectos fijos de empresa, 449 empresas:

| brecha, por 1.000 párrafos | b (post) | p | ponderado | sin celdas mixtas | tendencias previas | event study vs. 2021 (22 / 23 / 24 / 25) |
|---|---:|---:|---:|---:|---|---|
| frames de IA | +31,8 | <0,001 | +32,1 | +31,3 | **falla** (p<0,01) | +0,4 / **+27,6** / +39,6 / +42,9 |
| **promocionales** | **+8,2** | <0,001 | +8,2 | +7,9 | **falla** (p<0,01) | −0,0 / **+5,9** / +9,1 / +11,3 |
| cuantificados | +9,3 | <0,001 | +9,2 | +10,1 | **falla** (p<0,01) | +0,1 / **+4,5** / +9,3 / +12,6 |
| gobernanza | +0,0 | 0,98 | +0,1 | −0,2 | falla (p=0,01) | −0,1 / +0,5 / +0,3 / −0,0 |
| documentos con IA (share) | −0,34 | <0,001 | −0,28 | −0,47 | falla | −0,06 / −0,16 / −0,29 / −0,54 |

Descomposición por canal (event study de cada canal por separado):

| promocionales por 1.000 párrafos | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| call | −0,1 | **+5,9** | +9,3 | +11,7 |
| filing | −0,1 | −0,0 | +0,2 | +0,4 |

**La brecha promocional se abre en el ejercicio 2023 —un año antes del
escrutinio— y sigue abriéndose después sin ningún quiebre en 2024.** La call
pasa de ~3 afirmaciones promocionales de IA por 1.000 párrafos en 2021-2022 a
+5,9 en 2023, +9,3 en 2024 y +11,7 en 2025 sobre esa base; el filing sube
+0,2 y +0,4. La única brecha que no crece es la de gobernanza: los filings
agregan gobernanza de IA al mismo ritmo que las calls agregan promoción, así
que la resta queda en cero.

El test de tendencias previas falla en todo, y tiene que fallar: la curva ya
subía en 2023. **El escrutinio de la SEC no es identificable sobre la
brecha**, igual que no lo es entre empresas (`07_shocks_sec_deepseek.md`). Lo que el
diseño sí dice, y con toda la precisión que se le puede pedir, es que la
distancia entre lo que la empresa vende en la call y lo que firma en el
filing se multiplicó por tres entre 2022 y 2025 y no se frenó cuando el
regulador anunció que perseguiría exactamente eso.

## Resultado 3: la brecha por empresa

Brecha promocional promedio por empresa (call − filing, por 1.000 párrafos)
con ≥2 celdas: 171 empresas con ≥3 frames por canal en al menos dos
ejercicios; sobre las 479 con celdas, la mediana es positiva y las 17 empresas
que no hablan de IA en ningún canal están en cero.

| empresa | brecha promocional 2021 → 2025 (tasa condicional, call − filing) |
|---|---|
| NVDA | +0,29 → +0,44, abriéndose desde 2023 |
| PLTR | +0,63 → +0,39, siempre arriba de +0,3 |
| TSLA | +0,33 → +0,27, plana |
| GOOGL | −0,01 → +0,19 |
| MSFT | −0,08 → +0,15 |
| HPE | +0,25 → −0,03, se cierra |
| WELL | una sola celda con frames en calls (ejercicio 2024, 3 frames): 26% promocional en la call contra 0% en el filing |

Welltower, el caso que motiva el diseño, casi no habla de IA en sus calls
antes de 2025 y las transcripciones terminan a mediados de 2025: no hay serie
que estimar. Es un caso, no una estimación.

La brecha por empresa **no se correlaciona con el score de exceso promocional
de `08_definiciones_de_washing.md`** (Spearman −0,07): ese score mide exceso DENTRO de
los filings dado lo que la empresa describe; la brecha mide cuánto más promete
en la call. Son dos formas de desajuste distintas; la segunda es la que
coincide con el mecanismo que la SEC señaló en la correspondencia con
Welltower.

## Resultado 4: la call cuenta las mismas actividades, con más cifras y nombres

¿La call sólo cambia el tono, o cambia también lo que la empresa dice estar
haciendo? Con las actividades divulgadas (`09_actividades_ia.md`), misma
empresa y mismo ejercicio, sobre las 459 celdas de 183 empresas con al
menos tres actividades en cada canal: proporción de las actividades del
canal en cada familia, call menos filing (`activity_grounding.py`,
`fig_brecha_actividades.png`).

| familia de actividad | % en la call | % en el filing | brecha (p.p.) | t |
|---|---:|---:|---:|---:|
| despliegue de cara al cliente | 43,5 | 41,3 | +2,2 | 1,8 |
| despliegue interno | 24,4 | 23,8 | +0,6 | 0,5 |
| IA propia | 71,8 | 70,4 | +1,5 | 1,0 |
| inversión en infraestructura | 5,9 | 7,9 | −2,0 | −3,0 |
| **resultado cuantificado** | **17,1** | **6,2** | **+11,0** | 13,3 |
| **con función de negocio declarada** | **76,5** | **64,8** | **+11,7** | 8,6 |
| proveedor, modelo o socio externo nombrado | 8,1 | 2,8 | +5,3 | 8,0 |
| con producto o proceso nombrado | 30,3 | 26,5 | +3,8 | 3,0 |
| piloto o exploración | 5,3 | 3,2 | +2,1 | 3,9 |
| talento o capacitación | 1,1 | 2,7 | −1,7 | −4,9 |
| gobernanza o restricción | 0,3 | 2,2 | −1,9 | −6,1 |

**La mezcla de actividades es la misma en los dos canales**: la proporción
de despliegue a clientes, despliegue interno e IA propia no difiere (brechas
de +0,6 a +2,2 p.p., ninguna distinguible de cero al 5%). Lo que cambia es la
evidencia y el detalle: en la call la misma actividad viene con función
declarada (+11,7 p.p.), con resultado cuantificado (+11,0: el filing casi
nunca pone la cifra), con proveedor o socio nombrado (+5,3, casi el triple)
y con producto nombrado (+3,8). El filing se queda con la infraestructura, el
talento y la gobernanza. La brecha entre canales, leída en actividades, no es
"cuenta cosas distintas según a quién" sino "cuenta lo mismo, con cifras y
nombres para el analista y sin ellas en lo que firma". Es estable por
ejercicio: la brecha de resultados cuantificados va de +9,0 a +11,8 p.p.
entre 2021 y 2025, la de función declarada crece de +5,0 a +16,7 y la de
producto nombrado de −1,4 a +8,8.

## Chequeo secundario: tasas condicionadas a hablar de IA

Normalizando por cuánto se habla —celdas con ≥3 frames en cada canal, 640
celdas de 251 empresas, outcome = proporción promocional entre los frames de
IA— la brecha es +10,0 p.p. (17,9% contra 7,9%) y **no se mueve después de
2024** (b=+0,015, p=0,36, tendencias previas planas, igual ponderando por
frames o sacando celdas mixtas). La proporción de gobernanza del filing sube
4 p.p. después de 2024. Es el mismo mensaje leído en tasas: la call no bajó
el tono cuando llegó el regulador; el filing agregó gobernanza. El script
imprime ambos márgenes; el extensivo es el análisis, éste es el chequeo de
que la conclusión no depende de cómo se normaliza.

## Qué diseños no aguanta la brecha

- **DiD con grupo tratado definido por la brecha previa** es reversión a la
  media y no se usa. Con tratados definidos por exposición o por la cola de
  `09`, las tendencias previas fallan o el efecto es cero.
- **Control sintético por empresa**: 4 puntos pre-evento por empresa a nivel
  de ejercicio y un pool de donantes con serie trimestral densa de 14. No hay
  serie que sintetizar.

## Qué queda para la tesis

1. **La brecha entre canales es el hecho estilizado más limpio del proyecto**:
   20 veces más promoción de IA por párrafo en la call que en el filing, sobre
   2.281 celdas de la misma empresa y el mismo ejercicio, con las calls que no
   hablan de IA adentro.
2. **Se abrió con el boom de 2023 y el escrutinio de la SEC de 2024 no la
   frenó.** El diseño no puede atribuirle al regulador ningún efecto, y en
   tasas condicionadas el efecto es cero.
3. La brecha es un **segundo score de washing**, ortogonal al de `08`, y es el
   que corresponde al mecanismo regulatorio.
4. En actividades, la call y el filing describen la misma mezcla de
   despliegues; la call agrega cifras, funciones, proveedores y nombres de
   producto; el filing se queda con infraestructura, talento y gobernanza.

## Limitaciones

- Ambos canales pasan por el mismo prefiltro y el mismo juez, con error de
  medición distinto: precisión 0,98 en 10-K/10-Q, 0,73 en proxy/8-K, 0,60 en
  calls (`prefilter_evaluation.md` §8.16). Parte del nivel de la brecha puede
  ser error diferencial; el event study dentro de empresa no lo sufre.
- `rhetoric_promotional` es una etiqueta de LLM sin validación humana
  (`docs/problemas_academicos.md` #1). `ui-validator/` existe para cerrar esto.
- Las transcripciones cubren ~3,2 de 4 calls por empresa-ejercicio y terminan
  a mediados de 2025: el ejercicio 2025 tiene menos calls por celda.
- La intensidad por párrafo depende del largo del documento: una call tiene
  ~70 párrafos y un 10-K ~500; la brecha en frames por 1.000 párrafos es
  también una brecha de densidad de género.
