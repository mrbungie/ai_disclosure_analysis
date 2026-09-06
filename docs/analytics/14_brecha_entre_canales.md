# Brecha entre canales: la misma empresa, el mismo trimestre, call contra filing

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

Con efectos fijos empresa × trimestre y exactamente dos canales por celda, el
estimador es la diferencia dentro de la celda:

```
gap[i, t] = y[i, call, t] − y[i, filing, t] = a_i + c · post[t] + e
```

Todo lo que es común a la empresa en ese trimestre —el boom de IA, su sector,
su ciclo, cuánto habla de IA— se cancela en la resta. `post` es 2024Q2 en
adelante (escrutinio de la SEC, marzo 2024). Se estima con efectos fijos de
empresa sobre la brecha, errores clusterizados por empresa, ventana ±6
trimestres, y se reporta el event study por trimestre relativo con el test
conjunto de tendencias previas — la misma disciplina que `13_shocks.md`.

Trimestre = trimestre calendario de la fecha del documento. Celda = empresa ×
trimestre con ≥3 frames en cada canal. Filings = 10-K, 10-Q, DEF 14A y 8-K
juntos; el canal con responsabilidad legal contra el canal sin ella.

| | |
|---|---|
| frames | 46.075 (29.808 en filings, 16.267 en calls) |
| celdas empresa × trimestre con ambos canales | **590**, de 173 empresas, 2021Q1–2025Q2 |
| celdas en la ventana del DiD, con la empresa a ambos lados del corte | 399, de 86 empresas |

## Resultado 1: la brecha existe, es grande y va en la dirección de Welltower

Media sobre las 590 celdas, call menos filing:

| dimensión | call | filing | brecha | t contra 0 |
|---|---:|---:|---:|---:|
| **promocional** | 21,7% | 11,0% | **+10,7 p.p.** | 12,1 |
| cuantificado | 19,0% | 6,6% | +12,5 p.p. | 13,8 |
| índice de especificidad | 0,240 | 0,171 | +0,069 | 12,4 |
| realizado | 69,2% | 73,8% | −4,6 p.p. | −4,3 |
| hipotético | 0,8% | 5,1% | −4,3 p.p. | −9,8 |
| gobernanza | 1,6% | 8,9% | −7,4 p.p. | −10,7 |

**La misma empresa, en el mismo trimestre, es el doble de promocional en la
call que en el filing.** Y no es que en la call diga vaguedades: también
cuantifica tres veces más y es más específica. En la call se vende el producto
con números; en el filing se listan riesgos, gobernanza e hipotéticos. Son dos
géneros, y la brecha promocional es el rasgo de género más estable de todos:
590 celdas con t=12.

## Resultado 2: el escrutinio de la SEC no cerró la brecha

Efecto de `post` sobre la brecha, efectos fijos de empresa, 86 empresas
presentes a ambos lados del corte:

| dimensión de la brecha | b (post) | SE | p | tendencias previas |
|---|---:|---:|---:|---|
| **promocional** | **+0,001** | 0,019 | **0,958** | pasa (p=0,674) |
| cuantificado | +0,047 | 0,027 | 0,079 | pasa (p=0,264) |
| índice de especificidad | +0,036 | 0,013 | 0,005 | **falla** (p=0,012) |
| realizado | +0,031 | 0,023 | 0,173 | pasa (p=0,267) |
| hipotético | +0,018 | 0,009 | 0,048 | pasa (p=0,242) |
| gobernanza | −0,004 | 0,014 | 0,775 | pasa (p=0,574) |

**Cero, con precisión.** La brecha promocional entre lo que la empresa le dice
al mercado en la call y lo que firma en el filing es la misma antes y después
de que la SEC empezara a perseguir exactamente esa brecha: +0,001 con un error
estándar de 0,019 — el intervalo de 95% excluye cualquier cierre mayor a 4
p.p. sobre una brecha de 10,7. Las tendencias previas son planas (p=0,67).

Lo único que se mueve es la brecha de especificidad, y justo ahí el test de
tendencias previas falla: la call venía volviéndose relativamente más
específica desde antes de 2024, así que no se puede atribuir al evento. La
brecha de cuantificación (+4,7 p.p., p=0,08) y la de hipotético (+1,8 p.p.,
p=0,05) están en el borde, sin corrección múltiple y con seis outcomes; no se
reportan como efecto.

Es el mismo nulo que `13_shocks.md` encontró entre empresas más y menos
expuestas, pero ahora en el diseño que sí identifica: aquí no hace falta
suponer que el boom de IA afecta igual a tratados y controles, porque el boom
se resta dentro de la empresa-trimestre.

## Resultado 3: la brecha por empresa mide otra cosa que el score de `09`

Brecha promocional promedio por empresa, con ≥3 celdas (70 empresas):

| | |
|---|---|
| mediana | +9,2 p.p. |
| empresas con brecha positiva | prácticamente todas |
| top 5 | ORCL (+33), QCOM (+32), PLTR (+31), APH (+30), TSLA (+30) |
| Spearman con el exceso promocional de `09_washing_score.md` | **+0,10** (n=70) |

El score de `09` mide exceso promocional *dentro* de los filings dado lo que la
empresa describe; la brecha mide cuánto más promete en la call que en el
filing. **Casi no se correlacionan.** Las 8 empresas de la cola de `09` no son
las de mayor brecha: GOOGL, CRWD y PANW promocionan en los dos canales por
igual. Son dos formas de AI-washing distintas y la SEC persigue la segunda.

Por segmento (`11_segmentacion.md`): adoptantes con gobernanza +12,3 p.p. (16
empresas), desplegadores de producto +9,4 (51), listadores de riesgo +5,8 (3).
Las empresas cuyos filings son de gobernanza y riesgo son las que más cambian
de registro cuando hablan con analistas.

Welltower no entra: no tiene ninguna celda con ≥3 frames en ambos canales. El
caso que motiva el diseño está fuera de la muestra en la que se estima, lo que
es una limitación honesta, no un resultado.

## Qué queda para la tesis

1. **La brecha entre canales es el hecho estilizado más limpio del proyecto**:
   +10,7 p.p. de promoción, t=12, en 590 celdas de la misma empresa y el mismo
   trimestre. No depende de sectores, de composición ni del boom.
2. **El escrutinio de la SEC no la cerró.** Estimado con efectos fijos empresa ×
   trimestre, tendencias previas planas y un intervalo que excluye cierres
   mayores a 4 p.p.
3. La brecha es un **segundo score de washing**, ortogonal al de `09`, y es el
   que corresponde al mecanismo regulatorio. Su cola (ORCL, QCOM, PLTR, TSLA,
   JNPR, SPGI, NVDA, NOW, CRM) es distinta de la cola de `09`.

## Limitaciones

- Ambos canales pasan por el mismo prefiltro y el mismo juez, con error de
  medición distinto: precisión del prefiltro 0,98 en 10-K/10-Q, 0,73 en
  proxy/8-K, 0,60 en calls (`prefilter_evaluation.md` §8.16). Parte de la
  brecha de nivel puede ser error diferencial; el DiD sobre la brecha no lo
  sufre, porque el error es constante en el tiempo dentro de la empresa.
- `rhetoric_promotional` es una etiqueta de LLM sin validación humana
  (`docs/problemas_academicos.md` #1). El validador de `ui-validator/` existe
  para cerrar esto.
- 86 empresas a ambos lados del corte: potencia para detectar un cierre de ~4
  p.p., no menor.
- Filings pooled: un trimestre puede tener un 10-K y otro un 8-K. Separar por
  formulario deja celdas de 1-2 frames; el pool es el precio de tener 590.
- La entrada a la celda exige ≥3 frames en cada canal, así que la muestra es
  de empresas que hablan de IA en los dos: sesgo hacia las vocales.
