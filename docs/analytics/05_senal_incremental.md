# Señal incremental: ¿el contenido aporta información más allá del volumen y de los fundamentals?

Es el análisis central de la tesis (RQ4). Producido por
`scripts/analytics/incremental_signal.py` sobre todas las empresas-año con
filings, ejercicios 2021-2025 (`firm_year_master_v2.parquet`; 2.475 filas,
508 empresas, 59% con algún frame de IA). Financieros winsorizados 1/99.

## Diseño

Para cada outcome Y, cuatro modelos anidados con efectos fijos sector × año:

| modelo | qué sabe el modelo |
|---|---|
| M0 | fundamentals: log market cap, margen bruto, margen operativo, rotación de activos |
| M1 | M0 + **volumen**: log(1 + frames de IA por 1.000 párrafos) |
| M2 | M1 + **contenido**: log(1 + x por 1.000 párrafos) para realizado, despliegue, capacidad, riesgo, gobernanza, promocional, especificidad |
| M3 | M1 + dummies de segmento (`02_segmentacion.md`) |

Lo que se reporta es **ΔR² = R²(M2) − R²(M1)** —cuánto agregan las palabras
una vez que se sabe qué empresa es y cuánto habla de IA— y el R² parcial del
bloque semántico, (R²M2 − R²M1)/(1 − R²M1), con intervalo bootstrap por
empresa (300 réplicas). Cinco outcomes, ninguno más: beta, volatilidad
realizada pre-filing (proxy de idiosincrática), P/S, R&D/ventas, crecimiento
de ingresos t+1.

## Resultado

| outcome | n | R² M0 | R² M1 | R² M2 | ΔR² volumen | **ΔR² contenido** | IC 95% bootstrap | R² parcial | ΔR² segmentos |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| beta | 1.037 | 0,039 | 0,078 | 0,110 | +0,039 | **+0,032** | [+0,019, +0,085] | 3,5% | +0,010 |
| volatilidad | 1.039 | 0,145 | 0,166 | 0,188 | +0,021 | **+0,021** | [+0,010, +0,065] | 2,6% | +0,012 |
| P/S | 1.039 | 0,233 | 0,234 | 0,261 | +0,000 | **+0,027** | [+0,015, +0,079] | 3,5% | +0,001 |
| R&D / ventas | 649 | 0,453 | 0,497 | 0,519 | +0,044 | **+0,021** | [+0,011, +0,070] | 4,2% | +0,005 |
| crecimiento ingresos t+1 | 1.003 | 0,021 | 0,043 | 0,066 | +0,022 | **+0,023** | [+0,007, +0,088] | 2,4% | +0,012 |

**El contenido semántico agrega entre 2 y 4 puntos de R² parcial sobre
fundamentals y volumen, en los cinco outcomes, con intervalos bootstrap que
excluyen el cero.** Es del mismo orden que lo que agrega el volumen (0-4
puntos) y más que lo que agregan los segmentos como dummies (0-1 puntos).
Modesto, no cero, y consistente.

Qué dimensión lo carga (coeficientes estandarizados de M2, SE cluster por
empresa):

| outcome | dimensión con más peso | β estandarizado | p |
|---|---|---:|---:|
| beta | especificidad | +0,37 | <0,01 |
| volatilidad | especificidad | +0,26 | 0,01 |
| P/S | especificidad | +0,45 | <0,01 |
| R&D / ventas | despliegue (−), volumen (+) | −0,24 / +0,25 | 0,03 / 0,09 |
| crecimiento ingresos t+1 | especificidad | +0,37 | 0,01 |

**La señal es la especificidad**: cuánto de lo que la empresa dice de IA
viene con procesos, productos, proveedores, cifras o fechas. Dado el volumen,
la empresa que habla de IA en concreto es más riesgosa, más cara y crece más
que la que habla de IA en abstracto. La promoción no aporta (β ≈ −0,1 en P/S,
p=0,07; cero en el resto); la gobernanza resta beta (−0,12, p=0,04).

## Robustez

Mismo estimador, ΔR² del bloque semántico:

| outcome | principal | sólo 10-K | sin IT ni comunicaciones (SIC 35/36/48/73) | efectos fijos de empresa |
|---|---:|---:|---:|---:|
| beta | +0,032 | +0,031 | +0,056 | +0,022 |
| volatilidad | +0,021 | +0,021 | +0,085 | +0,009 |
| P/S | +0,027 | +0,015 | +0,012 | +0,006 |
| R&D / ventas | +0,021 | +0,021 | +0,028 | +0,008 |
| crecimiento ingresos t+1 | +0,023 | +0,034 | +0,007 | +0,012 |

- **Sólo 10-K**: igual. No es el proxy ni el 8-K.
- **Sin IT ni comunicaciones**: igual o mayor en riesgo (beta, volatilidad);
  menor en valuación y crecimiento. La señal no es "software vs. el resto".
- **Efectos fijos de empresa**: la mitad o menos. Como en todo el proyecto, la
  mayor parte de la información del texto es transversal —separa empresas—
  y una parte menor, pero no nula, es within-firm.
- Winsorización 1/99 en todo; sector × año en todo salvo la variante de
  efectos fijos de empresa.

## Lectura para la tesis

> El parseo semántico produce una taxonomía de divulgación coherente
> (`02_segmentacion.md`) y, además, señal económica incremental: 2-4 puntos de
> R² parcial sobre fundamentals y volumen en riesgo, valuación, I+D y
> crecimiento, concentrada en la especificidad de las afirmaciones. Es una
> señal de tipo de empresa más que de cambio dentro de la empresa.

El resultado sostiene la tesis con cualquier signo: si ΔR² fuera cero, la
conclusión sería "taxonomía sí, señal económica no". Salió positivo y chico,
y eso es lo que hay que escribir.

## Limitaciones

- R² bajos en beta, volatilidad y crecimiento: los outcomes son ruidosos y el
  ΔR² se lee en relación a ellos (por eso el R² parcial).
- La volatilidad realizada no es idiosincrática; `firm_year_market_factors`
  no trae la descomposición.
- R&D/ventas tiene 44% de cobertura XBRL; su fila tiene la mitad de la muestra.
- Etiquetas de un LLM sin validación humana (`docs/problemas_academicos.md`
  #1); la sensibilidad a colapsar campos con acuerdo mediocre queda pendiente
  de la anotación (`ui-validator/`).
