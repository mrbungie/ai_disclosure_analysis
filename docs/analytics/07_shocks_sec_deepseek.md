# Shocks: SEC 2024 y DeepSeek 2025

Responde las preguntas extendidas de `docs/thesis_proposal.md`: ¿cambió la
divulgación de IA después del escrutinio de la SEC sobre AI-washing? ¿Y
después de DeepSeek? Producido por `shock_analysis.py` (event study con
exposición continua) y `shock_did_simple.py` (dos grupos, pre/post). Panel:
todos los filings (10-K, 10-Q, DEF 14A, 8-K) por empresa-trimestre calendario
de presentación —11.302 empresa-trimestre, 510 empresas— con intensidades por
1.000 párrafos y cero cuando el trimestre no habla de IA. Figura:
`fig_sec_event_study.png`.

## SEC: resultado de no-identificación

**Fechas.** El primer aviso público es el discurso de Gensler sobre "AI
washing" del 5 de diciembre de 2023; el enforcement (Delphia, Global
Predictions) es del 18 de marzo de 2024. Los 10-K del ejercicio 2023 se
escriben en enero-febrero de 2024, después del aviso, así que el corte va al
inicio de 2024Q1 y 2023Q4 es la referencia.

**Diseño.** Event study con efectos fijos de empresa (within) y de trimestre,
ventana ±5, errores clusterizados por empresa; tratamiento = exposición
previa a IA (frames por 1.000 párrafos antes de 2023, en log, sobre todas las
empresas); controles de mezcla de formularios y log de párrafos. Los
coeficientes previos al evento se evalúan con un test conjunto F.

| outcome por 1.000 párrafos | test de tendencias previas | ¿efecto causal estimable? |
|---|---|---|
| promocionales | falla (p<0,001) | no |
| riesgo | falla (p<0,001) | no |
| hipotético | falla (p<0,001) | no |
| frames de IA | falla (p<0,001) | no |
| especificidad | falla (p<0,001) | no |

5.366 empresa-trimestre, 489 empresas. Los coeficientes previos no son cero
y tienen la forma de la temporada de 10-K (Q1 de cada año, positivo; el
resto, negativo): las empresas más expuestas ya divergían del resto antes del
aviso, y la estacionalidad de sus filings no la absorbe un efecto fijo de
trimestre común.

DiD simple (grupo = mitad más promocional por 1.000 párrafos antes de
diciembre de 2023; 108 tratadas, 382 controles; 5.374 empresa-trimestre):

| outcome por 1.000 párrafos | DiD | p | tendencias previas (p) |
|---|---:|---:|---:|
| promocionales *(dimensión del grupo)* | +0,91 | <0,001 | 0,060 |
| cuantificados | +0,33 | 0,082 | 0,389 |
| gobernanza | +0,57 | <0,001 | 0,087 |
| especificidad | +1,15 | <0,001 | 0,056 |

Las empresas más promocionales agregaron más gobernanza y más especificidad
por párrafo después de diciembre de 2023 que las demás —la dirección de una
respuesta a escrutinio—, pero con tendencias previas en el límite (p entre
0,06 y 0,09) y con la temporada de 10-K de 2024 concentrando el salto
(coeficientes de +0,76 y +1,75 en t=0 y +0,85 y +2,75 en t=+4, los dos Q1).

**Conclusión.** Las empresas más expuestas a IA ya venían separándose del
resto antes de diciembre de 2023; el aviso y el enforcement de la SEC caen en
medio de la difusión de la IA generativa y no hay grupo de control que no
estuviera en ella. **El efecto del regulador no se puede aislar del boom con
este diseño**, y lo que se observa en los filings después de 2024 —más riesgo,
más gobernanza, menos peso de la promoción (`01_evolucion_2021_2025.md`)— es
consistente tanto con una respuesta al escrutinio como con lo que cualquier
empresa agrega a su 10-K cuando una tecnología pasa a ser material. No se
intenta rescatar la identificación con controles sintéticos, matching ni
otros tratamientos.

## DeepSeek: sección corta, resultado nulo

Corte en 2025Q2 (20 de enero de 2025, mismo criterio). 5.287
empresa-trimestre, 484 empresas. Tendencias previas: fallan en especificidad,
riesgo, hipotético y frames (p<0,001) y están en el límite en promocionales
(p=0,079). El corte cae en plena difusión de la IA generativa y las empresas
expuestas se separaban del resto por su cuenta. **No hay discontinuidad
detectable separable de la tendencia.** Los outcomes propios del evento
(competencia, costo/capex, generativa) no se estiman: sin identificación no
hay nada que atribuir.

## Qué queda para la tesis

1. **Un nulo bien medido**, y decirlo es el resultado: un evento regulatorio
   universal en medio de un boom tecnológico no se identifica con un
   antes/después entre más y menos expuestos.
2. **La única forma de restar el boom es dentro de la empresa y del
   ejercicio**: la brecha entre canales (`06_brecha_entre_canales.md`). Ahí la
   brecha promocional tampoco se cerró.
3. DeepSeek queda fuera de alcance, y se dice.

## Limitaciones

- Trimestre calendario de presentación: la temporada de 10-K concentra la
  intensidad; el control de mezcla de formularios lo atenúa, no lo elimina.
- Etiquetas de un LLM sin validación humana (`docs/problemas_academicos.md` #1).
