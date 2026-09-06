# Brecha entre canales: la misma empresa, el mismo año, call contra filing

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
su ciclo, cuánto habla de IA— se cancela en la resta. `post` = 2024 en
adelante (escrutinio de la SEC, marzo 2024). Efectos fijos de empresa sobre
la brecha, errores clusterizados por empresa, sólo empresas presentes a ambos
lados del corte; event study por año con base 2021 y test conjunto de
tendencias previas.

**El período es el año calendario.** Los filings con frames de IA son
anuales —10-K en Q1, DEF 14A en Q2; el 10-Q casi nunca tiene frames— así que
parear por trimestre deja celdas sólo donde un 10-K coincide con una call.
Por año, la celda junta el 10-K, la DEF 14A, los 10-Q y las ~3-4 calls del
año. Celda = empresa × año con ≥3 frames en cada canal.

| | |
|---|---|
| frames | 46.075 (29.808 en filings, 16.267 en calls) |
| cobertura de las calls | 2020Q2–2025Q2 (~3,2 transcripciones por empresa-año; los filings llegan a 2026Q3) |
| celdas empresa × año con ambos canales | **619**, de 237 empresas: 69 / 73 / 111 / 179 / 187 por año 2021-2025 |
| empresas a ambos lados de 2024 | **115**, 439 celdas |

## Resultado 1: la brecha existe, es grande y va en la dirección de Welltower

Media sobre las 619 celdas, call menos filing:

| dimensión | call | filing | brecha | t contra 0 |
|---|---:|---:|---:|---:|
| **promocional** | 18,1% | 8,5% | **+9,6 p.p.** | 12,7 |
| cuantificado | 17,1% | 5,4% | +11,8 p.p. | 14,9 |
| índice de especificidad | 0,240 | 0,162 | +0,078 | 15,5 |
| realizado | 71,0% | 71,9% | −1,0 p.p. | −1,0 |
| hipotético | 0,7% | 6,6% | −5,9 p.p. | −13,5 |
| gobernanza | 1,5% | 9,6% | −8,1 p.p. | −13,7 |

**La misma empresa, en el mismo año, es el doble de promocional en la call
que en el filing.** Y no es que en la call diga vaguedades: también
cuantifica tres veces más y es más específica. En la call se vende el producto
con números; en el filing van los riesgos, la gobernanza y los hipotéticos.
Son dos géneros, y la brecha promocional es el rasgo de género más estable de
todos.

## Resultado 2: el escrutinio de la SEC no cerró la brecha promocional, y abrió la de especificidad

Efecto de `post` sobre la brecha, efectos fijos de empresa, 115 empresas:

| dimensión de la brecha | b (post) | SE | p | tendencias previas | event study vs. 2021 (2022 / 2023 / 2024 / 2025) |
|---|---:|---:|---:|---|---|
| **promocional** | **+0,017** | 0,018 | **0,339** | pasa (p=0,31) | −0,04 / +0,01 / +0,01 / +0,00 |
| cuantificado | +0,058 | 0,021 | 0,005 | pasa (p=0,40) | +0,03 / −0,01 / +0,05 / +0,08 |
| índice de especificidad | +0,048 | 0,012 | <0,001 | pasa (p=0,33) | +0,02 / −0,01 / +0,05 / +0,05 |
| realizado | +0,065 | 0,022 | 0,002 | pasa (p=0,10) | −0,02 / −0,08 / +0,02 / +0,03 |
| hipotético | −0,024 | 0,010 | 0,019 | pasa (p=0,70) | +0,01 / +0,01 / −0,02 / −0,01 |
| gobernanza | −0,039 | 0,012 | 0,001 | límite (p=0,06) | −0,02 / −0,04 / −0,05 / −0,08 |

**La brecha promocional no se movió**: +0,017 con error estándar 0,018 sobre
una brecha de 9,6 p.p., tendencias previas planas, event study en cero los
cuatro años. El intervalo de 95% excluye cualquier cierre mayor a 2 p.p.

**Lo que sí se movió es la brecha de sustancia, y en la dirección contraria a
la que el regulador querría.** Después de 2024 la call quedó 4,8 puntos de
índice más específica que el filing, y 5,8 p.p. más cuantificada, con
tendencias previas planas. La descomposición por canal dice de dónde viene:

| | call | filing (10-K, DEF 14A, 10-Q, 8-K) | filing, sólo 10-K |
|---|---:|---:|---:|
| especificidad 2021 → 2025 | 0,270 → 0,257 | 0,225 → 0,149 | 0,202 → 0,141 |
| event study 2024 / 2025 | −0,016 (p=0,19) / −0,011 (p=0,41) | **−0,062 (p<0,01) / −0,064 (p<0,01)** | **−0,044 / −0,046** |

**La call no se volvió más específica: el filing se volvió menos.** Con sólo
10-K en el canal filing (547 celdas, 216 empresas) el resultado es el mismo
(brecha de especificidad +0,043, p=0,001; cuantificado +0,060, p<0,001;
tendencias previas planas), así que no es composición de formularios. Tras
el escrutinio, las empresas siguieron vendiendo la IA con números en la call
y describieron menos concretamente su IA en el documento que firma el
abogado. Es la respuesta racional a un regulador que persigue afirmaciones
específicas no sustentadas: hacer el filing más vago, no la call más sobria.

Es el mismo nulo que `13_shocks.md` encontró entre empresas más y menos
expuestas, pero en el diseño que sí identifica —aquí el boom se resta dentro
de la empresa-año— y con un hallazgo adicional que el diseño de `13` no podía
ver, porque no compara canales.

## Resultado 3: los casos notorios

Brecha promocional por año (call − filing), con frames por canal:

| empresa | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|
| NVDA | +0,14 | +0,16 | +0,21 | +0,36 | **+0,64** |
| CRM | −0,14 | −0,10 | +0,24 | +0,25 | **+0,62** |
| PLTR | +0,63 | +0,30 | +0,50 | +0,32 | +0,41 |
| TSLA | — | +0,29 | +0,32 | +0,20 | +0,32 |
| ORCL | +0,39 | +0,17 | +0,18 | +0,34 | +0,22 |
| GOOGL | −0,01 | +0,36 | −0,06 | +0,03 | +0,22 |
| MSFT | +0,05 | −0,12 | +0,15 | +0,07 | +0,22 |
| META | +0,04 | +0,09 | +0,07 | +0,15 | +0,11 |
| PANW | −0,13 | −0,08 | −0,16 | −0,01 | +0,26 |
| HPE | +0,25 | +0,19 | +0,15 | −0,00 | −0,03 |
| IBM | −0,10 | −0,04 | −0,06 | −0,01 | −0,02 |
| WELL | — | — | — | — | +0,68 (4 frames en calls, 15 en filings) |

Los nombres grandes de la IA **abrieron** la brecha después de 2024: NVDA
pasa de +0,21 a +0,64, CRM de +0,24 a +0,62, PANW de negativa a +0,26. HPE
la cerró. Un DiD con esos doce como tratados contra el resto da +0,05 para
promocional (p=0,40) y +0,13 en 2025 (p=0,17): la dirección es consistente
pero doce empresas no dan potencia.

**Welltower** tiene un solo año con celda, 2025, con 62% de promocional en la
call contra 2% en el filing: es exactamente el mecanismo de la carta de la
SEC (abril 2025), pero antes de 2025 Welltower tenía un frame de IA por año
en sus calls —no hablaba de IA— y las transcripciones del corpus terminan en
2025Q2. No hay serie pre ni post que estimar: es un caso, no una estimación.

## Resultado 4: la brecha por empresa mide otra cosa que el score de `09`

Brecha promocional promedio por empresa con ≥2 celdas (171 empresas): mediana
+9,1 p.p., casi todas positivas. Spearman con el exceso promocional de
`09_washing_score.md`: **−0,06** (n=170). El score de `09` mide exceso
promocional *dentro* de los filings dado lo que la empresa describe; la
brecha mide cuánto más promete en la call que en el filing. **No se
correlacionan.** Las 8 empresas de la cola de `09` no son las de mayor
brecha. Son dos formas de AI-washing distintas y la SEC persigue la segunda.

Por segmento (`11_segmentacion.md`): adoptantes con gobernanza +10,8 p.p. (69
empresas), listadores de riesgo +11,1 (26), desplegadores de producto +8,2
(74). Las empresas cuyos filings son de gobernanza y riesgo son las que más
cambian de registro cuando hablan con analistas.

## Qué diseños no aguanta la brecha

- **DiD con grupo tratado** sobre la brecha: probado con tres definiciones.
  Tercil alto de brecha pre-2024 contra tercil bajo da −14 p.p. (p=0,004),
  que es reversión a la media por definir el grupo con la misma variable
  (el efecto desaparece en cuantificado, especificidad e hipotético); cola de
  `09` contra resto falla tendencias previas; exposición alta contra baja da
  cero. El antes/después con efectos fijos de arriba es lo defendible.
- **Control sintético por empresa**: a nivel trimestral sólo 15 empresas
  tienen serie densa, el pool de donantes queda en 14 y el mejor p por
  placebo posible es 0,07; el error de ajuste pre-evento (0,1-0,4) es del
  tamaño de la brecha. A nivel anual hay 4 puntos pre-evento por empresa.
  No hay serie que sintetizar; los casos notorios se leen uno por uno.

## Qué queda para la tesis

1. **La brecha entre canales es el hecho estilizado más limpio del proyecto**:
   +9,6 p.p. de promoción y +11,8 de cuantificación, en 619 celdas de la
   misma empresa y el mismo año. No depende de sectores, de composición ni
   del boom.
2. **El escrutinio de la SEC no la cerró** (b=+0,017, p=0,34, tendencias
   previas planas) **y abrió la brecha de especificidad**: el 10-K se volvió
   menos concreto sobre IA mientras la call no cambió.
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
- Las transcripciones terminan en 2025Q2 y cubren ~3,2 de 4 calls por
  empresa-año (`scripts/us/earnings_calls/01_fetch_transcripts.py`): el
  período post es 2024 y medio 2025. Con transcripciones de 2025H2-2026 el
  post tendría un año más.
- 115 empresas a ambos lados del corte: potencia para detectar un cierre de
  ~2 p.p. en la brecha promocional, no menor.
- La entrada a la celda exige ≥3 frames en cada canal: sesgo hacia empresas
  que hablan de IA en los dos.
