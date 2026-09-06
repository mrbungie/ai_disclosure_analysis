# Preguntas extendidas: ¿cambió la divulgación de IA tras los shocks de 2024-2025?

Responde las preguntas extendidas de `docs/thesis_proposal.md`:

> ¿Cambiaron las empresas su divulgación después del escrutinio de la SEC sobre
> AI-washing (marzo 2024)? ¿Afectó DeepSeek (enero 2025) cómo enmarcan sus
> capacidades?

Producido por `scripts/analytics/shock_analysis.py` y `shock_did_simple.py`.
Determinístico, sin LLM.

**Modo de análisis final: margen extensivo.** El panel son todos los filings
(10-K, 10-Q, DEF 14A, 8-K) por empresa-trimestre calendario de presentación —
11.302 empresa-trimestre, 510 empresas— y los outcomes son **intensidades por
1.000 párrafos**: frames de IA, promocionales, específicos, de riesgo,
hipotéticos, con **cero** cuando el trimestre no habla de IA. La empresa que
deja de hablar de IA cuenta como cero; no sale del panel.

## El diseño

Event study con efectos fijos de empresa y de trimestre, ventana ±5
trimestres, referencia t−1, errores clusterizados por empresa. Controles: mezcla
de formularios del trimestre (share de párrafos de proxy y de 10-K) y log de
párrafos. Los coeficientes previos al evento se evalúan con un test conjunto F:
si no son cero, los grupos ya venían separándose y el "efecto" no es del evento.

**El tratamiento es la exposición previa a IA** —frames de IA por 1.000
párrafos antes de 2023, en logaritmo, sobre todas las empresas— no la vaguedad.
Definir el grupo con la misma métrica que después se mide produce convergencia
mecánica.

Cortes: SEC en 2024Q2 (el anuncio es del 18 de marzo de 2024; 2024Q1 mezcla y
queda como referencia), DeepSeek en 2025Q2 (20 de enero de 2025, mismo criterio).

## Resultado 1: el escrutinio de la SEC no es identificable en este panel

Ventana ±5, 5.355 empresa-trimestre, 488 empresas:

| outcome por 1.000 párrafos | tendencias previas | cambio post (media de coef.) |
|---|---|---:|
| promocionales | **falla** (F=2,84, p=0,024) | +0,40 |
| especificidad | pasa (F=1,50, p=0,20) | +0,70 |
| riesgo | **falla** (F=4,04, p=0,003) | +0,64 |
| hipotético | **falla** (F=5,89, p<0,001) | +0,44 |
| frames de IA | **falla** (F=4,18, p=0,002) | +3,6 |

**Las tendencias previas fallan en cuatro de cinco dimensiones.** Las empresas
más expuestas a IA venían aumentando su intensidad relativa de IA desde antes de
2024: el margen extensivo es donde vive el boom de la IA generativa —el 36% de
las empresas-año de 2021 hablaba de IA, el 92% de 2025— y un corte en marzo de
2024 cae en medio de esa curva. El único outcome con tendencias previas planas,
especificidad por párrafo (+0,70), tiene su salto en t+3, que es 2025Q1: la
temporada de 10-K, no el evento (+2,37 en t+3 contra +0,46 y +0,48 en t+1 y t+2).

Por segmento de divulgación (`11_segmentacion.md`, las 510 empresas), cambio
post en promocionales por 1.000 párrafos contra los listadores de riesgo:
desplegadores de producto +0,56 (p<0,001), adoptantes con gobernanza +0,04
(p=0,06). Los desplegadores aceleran más que el resto después de 2024, con la
misma advertencia: es la curva, no el corte.

**Lo defendible: no hay ninguna evidencia de que el escrutinio de la SEC haya
cambiado la divulgación de IA, y el diseño no puede producirla, porque no hay
grupo de control que no estuviera ya en el boom.** No es "la SEC no hizo nada":
es que un evento que cae en el medio de la difusión de una tecnología no se
identifica con un antes/después entre más y menos expuestos.

## Resultado 2: DiD simple, dos grupos, pre/post

`shock_did_simple.py`: `AltoRiesgo` = la mitad de empresas más promocional por
1.000 párrafos **antes de 2024**, `Post` = desde 2024Q2, efectos fijos de
empresa y trimestre, errores clusterizados, ventana ±5. 108 empresas de alto
riesgo, 381 de bajo, 5.364 empresa-trimestre. El grupo se define con una
dimensión y los outcomes se miden en otras; promocional se reporta como control
interno.

| outcome por 1.000 párrafos | DiD | SE | p | tendencias previas (p) |
|---|---:|---:|---:|---:|
| promocionales *(dimensión del grupo)* | +0,32 | 0,16 | 0,040 | **0,003** |
| cuantificados | +0,29 | 0,20 | 0,140 | 0,058 |
| gobernanza | +0,30 | 0,14 | 0,037 | **0,017** |
| especificidad | +0,60 | 0,22 | 0,007 | **<0,001** |

Tres de cuatro fallan tendencias previas; el cuarto no es significativo. Las
figuras (`data/processed/clusters/shock_did_<outcome>.png`) lo muestran: las
barras previas no cruzan el cero, el grupo de alto riesgo venía por debajo del
otro en todas las dimensiones y converge desde antes del corte. **Mismo
veredicto que arriba.**

## Resultado 3: sobre DeepSeek, menos todavía

Ventana ±5, 5.287 empresa-trimestre, 484 empresas. Tendencias previas: fallan
en especificidad (p<0,001), riesgo (p<0,001), hipotético (p<0,001) y frames
(p<0,001); límite en promocionales (p=0,079). El corte de enero de 2025 cae en
plena difusión de la IA generativa y las empresas expuestas se separaban del
resto por su cuenta. **Este diseño no puede decir nada sobre DeepSeek.**

## Qué queda para la tesis

1. **La respuesta a la pregunta extendida es que no es contestable con este
   diseño**, y decirlo es el resultado. Los tests de identificación fallan
   porque el evento cae en el medio del boom, y la única forma de separar
   regulador de boom es restar el boom dentro de la empresa: eso es la brecha
   entre canales (`14_brecha_entre_canales.md`), que en el margen extensivo
   tampoco identifica al regulador y en tasas condicionadas da cero.
2. **DeepSeek queda fuera de alcance**, y se dice.
3. Lo que el panel sí muestra sin ambigüedad es el boom: la intensidad de IA
   en los filings se multiplica por 11 entre 2021 y 2026 y la promocional por
   6,6 (`03_cohort_2021_summary.md`, tabla extensiva).

## Limitaciones

- Trimestre calendario de presentación: la temporada de 10-K (Q1) concentra
  la intensidad y produce el salto de t+3; el control de mezcla de formularios
  lo atenúa, no lo elimina.
- El outcome es una etiqueta de LLM sin validación humana
  (`docs/problemas_academicos.md` #1).
- Las transcripciones no entran a este panel (son otro canal, `14_...md`).
