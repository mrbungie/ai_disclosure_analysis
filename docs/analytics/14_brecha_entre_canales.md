# Brecha entre canales: la misma empresa, el mismo ejercicio, call contra filing

Producido por `scripts/analytics/channel_gap_analysis.py`. Determinístico, sin
LLM. Es el diseño que `docs/pregunta_identificacion_sec.md` plantea como el
único capaz de separar el efecto del regulador del boom de IA, y el que
corresponde al mecanismo del caso Welltower: la SEC no objetó el 10-K por
promocional, objetó que la earnings call dijera "industry-leading" sin
respaldo proporcional en el 10-K. **El objeto es la brecha entre canales, no
el nivel en uno.**

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
su ciclo, cuánto habla de IA— se cancela en la resta. Efectos fijos de
empresa sobre la brecha, errores clusterizados por empresa, sólo empresas
presentes a ambos lados del corte; event study por año con base 2021 y test
conjunto de tendencias previas.

**El período es el año fiscal, alineado por lo que cada documento cubre**,
no por la fecha en que se presenta: el 10-K de febrero de 2025 habla del
ejercicio 2024, y las calls de 2024 discuten los trimestres de 2024.

| documento | año fiscal asignado |
|---|---|
| 10-K, 10-Q | el de `period_end_date`, con el mes de cierre de cada empresa |
| earnings call | el del `document_id` (`TICKER_YYYYQn` es el trimestre fiscal discutido) |
| DEF 14A, 8-K | el de presentación (no cubren un período: el proxy mezcla compensación pasada y gobernanza actual) |

`post` = ejercicio 2024 en adelante (escrutinio de la SEC, marzo 2024). Una
celda puede mezclar documentos de antes y después del 2024-03-01 —el 10-K de
FY2023 presentado en febrero y el proxy de abril—; son 55 de 640 y se
reporta la estimación sin ellas. Como las tasas de celdas chicas son
ruidosas, se reporta también la estimación ponderada por frames (el mínimo
de los dos canales). Celda = empresa × ejercicio con ≥3 frames en cada canal.

| | |
|---|---|
| frames | 46.075 (29.808 en filings, 16.267 en calls) |
| cobertura de las calls | ejercicios 2021-2025, ~3,2 transcripciones por empresa-año; la última es de mediados de 2025 (los filings llegan a 2026Q3) |
| celdas empresa × ejercicio con ambos canales | **640**, de 251 empresas: 81 / 74 / 147 / 207 / 131 por ejercicio 2021-2025 |
| empresas a ambos lados de 2024 | **140**, 495 celdas |

## Resultado 1: la brecha existe, es grande y va en la dirección de Welltower

Media sobre las 640 celdas, call menos filing:

| dimensión | call | filing | brecha | t contra 0 |
|---|---:|---:|---:|---:|
| **promocional** | 17,9% | 7,9% | **+10,0 p.p.** | 13,9 |
| cuantificado | 16,9% | 4,7% | +12,2 p.p. | 16,2 |
| índice de especificidad | 0,238 | 0,151 | +0,087 | 17,8 |
| realizado | 70,6% | 69,9% | +0,8 p.p. | 0,8 |
| hipotético | 0,8% | 8,3% | −7,5 p.p. | −15,0 |
| gobernanza | 1,3% | 7,5% | −6,2 p.p. | −12,7 |

**La misma empresa, en el mismo ejercicio, es más del doble de promocional
en la call que en el filing.** Y no es que en la call diga vaguedades:
también cuantifica tres veces más y es más específica. En la call se vende el
producto con números; en el filing van los riesgos, la gobernanza y los
hipotéticos. Son dos géneros, y la brecha promocional es el rasgo de género
más estable de todos.

## Resultado 2: el escrutinio de la SEC no cerró la brecha promocional

Efecto de `post` sobre la brecha, efectos fijos de empresa, 140 empresas:

| brecha | b (post) | p | ponderado por frames | sin celdas mixtas | tendencias previas | event study vs. 2021 (22 / 23 / 24 / 25) |
|---|---:|---:|---:|---:|---|---|
| **promocional** | **+0,015** | **0,36** | +0,013 (p=0,52) | +0,018 (p=0,30) | pasa (p=0,28) | −0,03 / +0,02 / +0,02 / +0,02 |
| cuantificado | +0,032 | 0,046 | +0,058 (p<0,001) | +0,035 (p=0,047) | pasa (p=0,45) | −0,03 / +0,01 / +0,02 / +0,04 |
| índice de especificidad | +0,032 | 0,003 | +0,034 (p<0,001) | +0,033 (p=0,003) | pasa (p=0,16) | +0,02 / +0,03 / +0,05 / +0,06 |
| realizado | +0,034 | 0,10 | | | pasa (p=0,59) | −0,02 / +0,01 / +0,02 / +0,04 |
| hipotético | +0,001 | 0,94 | | | **falla** (p=0,002) | −0,01 / −0,05 / −0,03 / −0,03 |
| gobernanza | −0,040 | <0,001 | | | pasa (p=0,24) | −0,02 / −0,02 / −0,05 / −0,06 |

**La brecha promocional no se movió**: +0,015 con error estándar 0,016 sobre
una brecha de 10 p.p., tendencias previas planas, event study en cero los
cuatro años, y lo mismo ponderando por frames o sacando las celdas mixtas.
El intervalo de 95% excluye cualquier cierre mayor a 2 p.p. Es el mismo nulo
que `13_shocks.md` encontró entre empresas más y menos expuestas, pero en el
diseño que sí identifica: aquí el boom se resta dentro de la empresa-ejercicio.

## Resultado 3: la brecha de sustancia se abre, y hay que leer el event study para saber por qué

Tres brechas se mueven después de 2024. La descomposición por canal (event
study de cada canal por separado, mismas celdas, FE de empresa) dice de dónde
viene cada una:

| | call, 2022 / 23 / 24 / 25 | filing, 2022 / 23 / 24 / 25 |
|---|---|---|
| promocional | −0,02 / −0,01 / −0,00 / −0,00 | +0,00 / −0,03 / −0,03 / −0,02 |
| cuantificado | −0,02 / −0,01 / +0,01 / +0,04 | +0,01 / −0,01 / −0,01 / −0,01 |
| índice de especificidad | +0,01 / −0,02 / −0,01 / +0,01 | −0,01 / **−0,05** / **−0,06** / **−0,05** |

- **Especificidad: el filing se volvió menos específico sobre IA, pero desde
  el ejercicio 2023, no desde 2024.** La caída del filing es −0,05 ya en
  FY2023 (p<0,01) y se queda ahí; la call no cambia. El test conjunto de
  tendencias previas pasa (p=0,16) sólo porque 2022 es plano; el salto está
  en 2023, que es el año del boom de la IA generativa, no el del regulador.
  Es un hecho sobre el 10-K —a medida que todas las empresas hablan de IA,
  lo hacen en términos más genéricos—, no un efecto de la SEC.
- **Cuantificado: la call cuantifica más en 2025** (+0,04) con el filing
  quieto y las tendencias previas planas. Es el único movimiento que empieza
  después del evento, es chico y sólo es claro ponderando por frames
  (+5,8 p.p., p<0,001).
- **Gobernanza: el filing habla más de gobernanza de IA desde 2024** (brecha
  −0,05 y −0,06 contra −0,02 antes), con la call en cero. Es consistente con
  un regulador: la respuesta a un escrutinio sobre afirmaciones de IA es
  agregar gobernanza al documento que firma el abogado, no bajar el tono de
  la call.

Lo defendible: **después del escrutinio, las empresas no acercaron lo que
dicen en la call a lo que firman en el filing. Le agregaron gobernanza al
filing.**

## Resultado 4: los casos notorios

Brecha promocional por ejercicio (call − filing):

| empresa | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|
| NVDA | +0,29 | +0,11 | +0,18 | +0,27 | **+0,44** |
| PLTR | +0,63 | +0,33 | +0,42 | +0,42 | +0,39 |
| TSLA | +0,33 | +0,45 | +0,27 | +0,23 | +0,27 |
| ORCL | +0,00 | +0,37 | +0,35 | +0,29 | +0,21 |
| CRM | +0,35 | −0,22 | +0,09 | +0,38 | +0,28 |
| GOOGL | −0,01 | +0,08 | +0,00 | +0,12 | +0,19 |
| MSFT | −0,08 | −0,08 | +0,06 | +0,12 | +0,15 |
| META | +0,06 | +0,07 | +0,09 | +0,16 | +0,01 |
| PANW | −0,06 | −0,06 | −0,08 | −0,11 | +0,23 |
| HPE | +0,25 | +0,19 | +0,15 | −0,00 | −0,03 |
| IBM | −0,13 | −0,04 | −0,05 | +0,01 | +0,02 |
| WELL | — | — | — | +0,26 (3 frames en calls, 14 en filings) | — |

PLTR y TSLA viven en +0,3 a +0,4 los cinco años. NVDA, GOOGL y MSFT abren la
brecha después de 2024; HPE la cierra. Con doce empresas grandes como grupo
tratado, el DiD da +0,05 (p=0,40): dirección consistente, sin potencia.

**Welltower** tiene una sola celda, el ejercicio 2024, con 3 frames de IA en
sus calls. Antes de eso tenía un frame por año: no hablaba de IA. La carta de
la SEC (abril 2025) cae en un ejercicio para el que el corpus no tiene calls
suficientes. Es un caso, no una estimación.

## Resultado 5: la brecha por empresa mide otra cosa que el score de `09`

Brecha promocional promedio por empresa con ≥2 celdas (171 empresas): mediana
+9,2 p.p., casi todas positivas. Spearman con el exceso promocional de
`09_washing_score.md`: **−0,07** (n=171). El score de `09` mide exceso
promocional *dentro* de los filings dado lo que la empresa describe; la
brecha mide cuánto más promete en la call que en el filing. **No se
correlacionan.** Las 8 empresas de la cola de `09` no son las de mayor
brecha. Son dos formas de AI-washing distintas y la SEC persigue la segunda.

Por segmento (`11_segmentacion.md`): listadores de riesgo +12,2 p.p. (25
empresas), adoptantes con gobernanza +11,7 (70), desplegadores de producto
+8,0 (75). Las empresas cuyos filings son de riesgo y gobernanza son las que
más cambian de registro cuando hablan con analistas.

## Qué diseños no aguanta la brecha

- **DiD con grupo tratado definido por la brecha previa** (tercil alto contra
  bajo) da −14 p.p. y es reversión a la media: el grupo se define con la
  misma variable que después se mide, y el efecto desaparece en las demás
  dimensiones. No se usa. El antes/después con efectos fijos de empresa no
  selecciona sobre el resultado; su riesgo es el ruido de las tasas, y por
  eso se reporta ponderado por frames.
- **Control sintético por empresa**: hay 4 puntos pre-evento por empresa a
  nivel de ejercicio, y a nivel trimestral el pool de donantes con serie
  densa es de 14 con un error de ajuste del tamaño de la brecha. No hay serie
  que sintetizar; los casos notorios se leen uno por uno.

## Qué queda para la tesis

1. **La brecha entre canales es el hecho estilizado más limpio del proyecto**:
   +10 p.p. de promoción y +12 de cuantificación, en 640 celdas de la misma
   empresa y el mismo ejercicio. No depende de sectores, de composición ni
   del boom.
2. **El escrutinio de la SEC no la cerró** (b=+0,015, p=0,36, robusto a
   pesos y a celdas mixtas). Lo que cambió después de 2024 es que el filing
   agregó gobernanza de IA; la pérdida de especificidad del filing es del
   boom de 2023, no del regulador.
3. La brecha es un **segundo score de washing**, ortogonal al de `09`, y es el
   que corresponde al mecanismo regulatorio.

## Limitaciones

- Ambos canales pasan por el mismo prefiltro y el mismo juez, con error de
  medición distinto: precisión del prefiltro 0,98 en 10-K/10-Q, 0,73 en
  proxy/8-K, 0,60 en calls (`prefilter_evaluation.md` §8.16). Parte de la
  brecha de nivel puede ser error diferencial; el antes/después sobre la
  brecha no lo sufre, porque el error es constante en el tiempo dentro de la
  empresa.
- `rhetoric_promotional` y `specificity_*` son etiquetas de LLM sin
  validación humana (`docs/problemas_academicos.md` #1). `ui-validator/`
  existe para cerrar esto.
- Las transcripciones cubren ~3,2 de 4 calls por empresa-ejercicio y
  terminan a mediados de 2025 calendario (`01_fetch_transcripts.py`): el
  ejercicio 2025 tiene 131 celdas contra 207 de 2024. Con transcripciones de
  2025H2-2026 el post tendría un ejercicio más.
- 140 empresas a ambos lados del corte: potencia para detectar un cierre de
  ~2 p.p. en la brecha promocional, no menor.
- La entrada a la celda exige ≥3 frames en cada canal: sesgo hacia empresas
  que hablan de IA en los dos.
