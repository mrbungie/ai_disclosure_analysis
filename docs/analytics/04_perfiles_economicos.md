# Perfiles económicos de los segmentos

¿Los segmentos de `02_segmentacion.md` corresponden a tipos económicamente
coherentes de empresa, incluso dentro de sectores comparables? (RQ3.)
Producido por `economic_profiles.py` sobre todas las empresas-año con filings
2021-2025 (2.475, 508 empresas) y la asignación de segmento por empresa-año
(`firm_year_segments.parquet`): sin IA 1.047, desplegadores de producto 554,
adoptantes con gobernanza 512, listadores de riesgo 362. Financieros
winsorizados 1/99.

## Panel A — mediana cruda por segmento

| | sin IA | listadores de riesgo | adoptantes con gobernanza | desplegadores de producto |
|---|---:|---:|---:|---:|
| market cap (mediana) | $26B | $31B | $37B | $39B |
| R&D / ventas | 3,3% | 4,0% | 6,4% | **12,3%** |
| margen bruto | 39,3% | 42,3% | 43,6% | **59,5%** |
| margen operativo | 16,7% | 16,9% | 15,5% | 16,5% |
| crecimiento ingresos t+1 | 8,1% | 4,6% | 6,0% | 8,5% |
| beta | 0,87 | 0,74 | 0,86 | **1,06** |
| volatilidad pre-filing | 26,5% | 23,0% | 25,1% | **30,0%** |
| P/S | 2,6 | 2,7 | 2,7 | **4,0** |
| ROIC − WACC | +4,8% | +3,8% | +5,4% | +5,6% |

## Panel B — residualizado por sector × año

`X = a[sector×año] + γ·Segmento + e`, referencia = sin IA, SE cluster por
empresa. γ es la diferencia contra las empresas sin IA del mismo sector y año.

| | listadores de riesgo | adoptantes con gobernanza | desplegadores de producto |
|---|---:|---:|---:|
| log market cap | +0,06 (p=0,56) | +0,20 (p=0,03) | **+0,38 (p<0,001)** |
| R&D / ventas | +4,6 p.p. (p=0,01) | +2,5 p.p. (p=0,005) | **+5,7 p.p. (p<0,001)** |
| margen bruto | +3,4 p.p. (p=0,11) | +2,8 p.p. (p=0,14) | **+9,5 p.p. (p<0,001)** |
| margen operativo | −4,5 p.p. (p=0,05) | −1,3 p.p. (p=0,36) | −1,8 p.p. (p=0,34) |
| crecimiento ingresos t+1 | +0,9 p.p. (p=0,54) | −0,0 p.p. (p=0,96) | +0,9 p.p. (p=0,48) |
| beta | +0,02 (p=0,49) | +0,00 (p=0,92) | **+0,09 (p=0,006)** |
| volatilidad pre-filing | +0,0 p.p. (p=0,98) | −0,1 p.p. (p=0,88) | **+2,5 p.p. (p=0,03)** |
| P/S | +0,00 (p=0,99) | −0,14 (p=0,64) | **+0,90 (p=0,04)** |
| ROIC − WACC | −2,2 p.p. (p=0,31) | +1,7 p.p. (p=0,38) | −1,0 p.p. (p=0,61) |

## Lectura

- **Los desplegadores de producto son un tipo de empresa, también dentro de su
  sector y año**: más grandes (+46% de market cap), más I+D (+5,7 p.p.), más
  margen bruto (+9,5 p.p.), más beta, más volatilidad y más cara (P/S +0,9).
  Nada de eso es composición de industria.
- **Los listadores de riesgo y los adoptantes con gobernanza no se distinguen
  de las empresas sin IA de su sector** en tamaño, márgenes, riesgo ni
  valuación, salvo algo más de I+D. La diferencia entre ellos es de discurso,
  no de tipo económico.
- **Ningún segmento crece más ni crea más valor** que las empresas sin IA de su
  sector y año. El crecimiento y el ROIC−WACC no separan segmentos.
- ROIC−WACC se reporta por completitud; no es un resultado de cabecera.

## Limitaciones

- SIC-2 es granularidad gruesa; parte del residuo puede ser sub-sector.
- Los segmentos por empresa-año cambian con el año (el "sin IA" se vacía);
  el panel mezcla tipo de empresa con momento del boom, y el efecto fijo
  sector×año absorbe el momento sólo en promedio.
- Etiquetas de un LLM sin validación humana (`docs/problemas_academicos.md` #1).
