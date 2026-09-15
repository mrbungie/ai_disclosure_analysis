# Problema de identificación: ¿el escrutinio de la SEC (marzo 2024) cambió la divulgación de IA?

Documento autocontenido para discutir el diseño empírico con alguien que no
conoce el proyecto. Todo lo que aparece acá está medido sobre el corpus real; las
cifras no son ilustrativas.

---

## 1. Contexto y pregunta

Tesis de magíster: *"AI Washing or Credible Disclosure? Patterns and Clusters in
Corporate AI Disclosures"*. Pregunta principal (descriptiva): en qué arquetipos
se agrupan las empresas según **cómo** divulgan IA — adopción, capacidades,
riesgos, gobernanza. Preguntas extendidas (aquí empieza el problema):

> ¿Las empresas con divulgación de IA vaga o promocional cambiaron su
> comportamiento **después del escrutinio de la SEC de 2024** sobre "AI
> washing"?

El evento: el **18 de marzo de 2024** la SEC anuncia acciones de enforcement por
AI-washing contra dos asesores de inversión (Delphia y Global Predictions), con
declaraciones públicas de su presidente advirtiendo a los emisores.

## 2. Datos

**Muestra**: 517 empresas estadounidenses — S&P 500 con membresía **congelada al
2021-12-31** (para que las que fueron adquiridas o quebraron sigan en el panel)
más 122 firmas agregadas a mano. 424 de ellas tienen 10-K en los seis años.

**Corpus**: 4,3 millones de párrafos únicos de 10-K, 10-Q, DEF 14A y 8-K,
2021-2026. Además, ya extraídos pero **sin procesar**: 483.214 párrafos de
transcripciones de earnings calls (481 empresas, 2020-2025).

**Medición**, en dos etapas:
1. Un prefiltro (gradient boosting sobre señales léxicas, semánticas y de forma
   del texto) marca los párrafos que mencionan IA: **20.778 textos únicos**,
   0,48% del corpus. Recall validado fuera de muestra: 0,99 en 10-K/10-Q, 0,97
   en DEF 14A, 0,92 en 8-K.
2. Un LLM (`qwen3.7-flash`) extrae de cada uno **"frames" semánticos**: una
   proposición sobre IA con sus atributos — sujeto, temporalidad
   (realizado/planeado/hipotético), dominio, conceptos (despliegue, resultados,
   inversión, riesgo, gobernanza...), banderas de especificidad, y dos banderas
   retóricas: `rhetoric_promotional` y `rhetoric_strategic_importance`.
   Resultado: **~24.000 frames** sobre 457 empresas.

**Variables candidatas a outcome**, todas como % de los frames de la empresa en
un trimestre: `promotional_rate`, `specificity_index`, `risk_share`,
`hypothetical_share`.

**Advertencia de medición, importante**: ninguna de esas etiquetas está validada
contra anotación humana. Lo único medido es acuerdo entre dos LLMs en la etapa
del prefiltro (κ=0,87), no en la extracción de frames.

## 3. Lo que se intentó y por qué falla

### Intento 1 — el original

Grupos definidos por "vaguedad" (`z(promotional_rate) − z(specificity_index)`)
sobre frames de 10-K pre-2024, cortados en la mediana; outcome = la misma métrica
medida en 10-Q trimestrales; regresión segmentada sobre **medias trimestrales por
grupo**, sin efectos fijos ni errores clusterizados.

Problemas: (a) sin efectos fijos, (b) la composición de cada grupo cambia cada
trimestre porque entran empresas nuevas al corpus todo el tiempo, (c) **el grupo
se define por el nivel pre-evento de la misma variable que después se mide**, lo
que garantiza reversión a la media.

### Intento 2 — event study con efectos fijos

Panel empresa-trimestre, efectos fijos de empresa y de trimestre, errores
estándar clusterizados por empresa, un coeficiente por trimestre relativo al
evento, panel balanceado, controles de composición documental.

**Todos los outcomes fallan el test conjunto de tendencias paralelas** (F=5,11
p=0,001; F=8,91 p<0,001; ...). Los coeficientes pre-evento son significativos
seis trimestres antes. Agregar tendencias lineales por grupo lo empeora. Es el
defecto (c) del intento 1: no es un problema de estimador.

### Intento 3 — tratamiento = exposición a IA (no el outcome)

Grupo definido por el **volumen** de frames de IA pre-2023 (no por cómo habla).
Con eso, `promotional_rate` **pasa** tendencias paralelas (F=1,21, p=0,32) y da
un efecto post de **+8,4 p.p.** (p<0,001); `risk_share` pasa y da cero.

**Pero no identifica nada**: las empresas más expuestas a IA antes de 2023 son
exactamente las más afectadas por el boom de IA generativa de 2024. El evento de
la SEC es un **shock común en tiempo calendario**, así que ese coeficiente recoge
todo lo que le pasó a las empresas expuestas a IA después de marzo de 2024. El
boom es la explicación alternativa obvia y no hay forma de separarla.

## 4. El problema, formulado

Se necesita variación transversal en la exposición al **enforcement**, que **no**
sea también variación en la exposición al **boom de IA**. Lo descartado hasta
ahora:

| variación | ¿separa SEC de boom? | por qué |
|---|---|---|
| vaguedad pre-evento | no | además es endógena al outcome |
| intensidad de IA pre-evento | **no** | es exactamente la exposición al boom |
| sección del documento (Item 1 vs 1A) | no | el boom actúa sobre las afirmaciones de producto |
| tema (IA vs no-IA en el mismo documento) | no | el boom es *sobre IA*: afecta lo mismo que la SEC |

Candidatos que parecen sobrevivir, y sus problemas:

**A. Canal: documento presentado vs. earnings call.** Mismo emisor, mismo
trimestre, dos canales. El boom entusiasma en los dos; la responsabilidad legal
(Sección 18, 10b-5) cae sólo sobre el filing — una call no es documento
presentado y tiene safe harbor del PSLRA. Con efectos fijos **empresa × trimestre**
el boom se absorbe entero y se identifica sobre la *brecha entre canales*.
Supuesto testeable con las brechas pre-evento.
*Costo*: los 483k párrafos de earnings calls necesitan embeddings, prefiltro y
extracción de frames (~US$0,30 y ~40 min de cómputo).
*Duda*: ¿es defendible tratar la call como "no regulada"? Reg FD aplica, y hay
jurisprudencia de 10b-5 sobre declaraciones en calls.

**B. Perímetro del enforcement.** Las dos acciones fueron contra **asesores de
inversión**. Empresas financieras reguladas (SIC 62xx) están dentro del
perímetro; el resto está advertido por analogía. Ambos grupos viven el boom.
*Duda*: el perímetro real de la advertencia era mucho más amplio que las dos
acciones; y los grupos difieren en todo lo demás.

**C. Riesgo de litigio ex-ante** (industrias de alto riesgo à la
Francis-Philbrick-Schipper). Empresas igualmente expuestas a IA responden
distinto según cuánto les cuesta una afirmación no sustanciable.
*Duda*: las industrias de alto riesgo de litigio son casi las mismas que las
tecnológicas, o sea las del boom.

**D. Atención previa del regulador**: 5.057 cartas de comentario UPLOAD/CORRESP
sobre 371 de las 517 empresas. Haber estado bajo revisión reciente como proxy de
sensibilidad al regulador.
*Duda*: recibir cartas correlaciona con tamaño y complejidad.

**E. Jurisdicción**: hay pipelines de Chile e Italia. Empresas no sujetas a la
SEC, mismo boom.
*Duda*: la medición no está validada fuera de EE.UU., los regímenes de
divulgación no son comparables y el idioma cambia.

## 4.bis. Encuadre: qué exige realmente la propuesta de tesis

La propuesta plantea las técnicas cuasi-causales como una **posibilidad** para
las preguntas extendidas ("timeseries analysis and quasi-causal techniques like
DiD **may** be used"), y su aporte esperado es un *practical disclosure-risk
framework* para managers, IR y directorios. El núcleo es medición → arquetipos →
evolución → análisis de shocks. **No es una tesis de econometría regulatoria cuyo
aporte sea un ATT limpio.**

Eso separa cuatro afirmaciones que conviene no colapsar:

| nivel | afirmación | exigencia |
|---|---|---|
| principal | qué arquetipos de divulgación existen | ninguna |
| temporal | cómo evolucionan | ninguna |
| shock | el grupo A cambió distinto que el B tras el evento | tratamiento exógeno al outcome, FE, SE clusterizados, pre-tendencias planas |
| causal | el enforcement causó el cambio | además, variación que separe el evento de todo lo demás de esa fecha |

El intento 1 falla incluso en el tercer nivel, y no por causalidad: su
tratamiento ES el outcome. El intento 3 sí sostiene el tercer nivel. Ninguno
sostiene el cuarto, y probablemente ninguno pueda con estos datos.

## 5. Las preguntas concretas

1. **¿Hasta qué nivel de la tabla de 4.bis conviene llegar?** Dado que la
   propuesta pide "cambio diferencial" y no un ATT, ¿alcanza con el intento 3
   reportado como asociativo, o vale la pena pagar el costo del diseño de canal
   para poder decir algo causal?
2. Si va DiD: de los candidatos A-E, ¿cuál aguanta un referato, y con qué
   supuesto explícito?
3. ¿Hay una fuente de variación que no se me ocurrió? (¿Timing escalonado por
   cierre fiscal? ¿Empresas con exposición diferencial a la SEC por tipo de
   registrante — foreign private issuers, emerging growth companies?)
4. Con n = 51-460 empresas según el corte y 6 años trimestrales, ¿hay potencia
   para algo de esto, o el diseño correcto es inviable por tamaño?
5. ¿Cambia la respuesta el hecho de que el outcome sea una **etiqueta de LLM sin
   validación humana**? ¿Hasta dónde se puede llevar una inferencia causal sobre
   una variable así?

## 6. Restricciones

- Los datos que faltan son caros en tiempo, no en dinero (el corpus de calls está
  extraído pero sin procesar; procesarlo cuesta centavos de LLM).
- No hay acceso a bases comerciales (Capital IQ, Bloomberg, Audit Analytics).
- Todo el pipeline es reproducible y versionado; agregar una variable nueva de
  una fuente pública es factible si es de bajo costo.
