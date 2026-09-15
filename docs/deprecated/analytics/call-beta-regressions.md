# Earnings-call beta regressions

> Bugs de integridad de datos encontrados y corregidos en el panel que
> alimenta esta especificación (duplicados por dual-class shares y
> retailers con año fiscal no-calendario, merge sin `call_accession_number`
> en `attach_market()`, factores FF3 stale, `operating_margin`/`shares_out`
> sin fallback de concept-tag, ventana de beta a 63 vs 126 días) y una
> especificación alternativa (configuración + decoupling, as-of, sin
> leakage) al hallazgo de Governance-Led: ver `12_config_decoupling_asof.md`.

La unidad de observación es una earnings call agregada por firma y fecha. Las
calls que ocurren el mismo día para una firma se agregan antes de calcular
historiales. Los ceros de disclosure o substance son observaciones válidas.

La fecha exacta de la call es el ancla. `HistDisclosure` y `HistSubstance`
son promedios expanding de calls estrictamente anteriores; las sorpresas son
el valor actual menos ese historial propio. `Disclosure` son AI frames por
1,000 palabras. `Substance = log(1 + activities / 1,000 words) × grounding`.

`beta_pre` se estima con el modelo de mercado sobre ruedas `[-252, -21]` y
`beta_post` sobre `[+21, +126]`. Las ventanas no se solapan. Market cap,
Return60 y fundamentals se conocen antes de la call. Los fundamentals se
leen del último 10-K o 10-Q con `filing_date < call_date`, mediante los facts
XBRL del accession de ese filing.

## Baseline

$$
\beta^{post}_{i,t} = \alpha + \beta_1 HistD_{i,t^-} + \beta_2 HistS_{i,t^-}
+ \beta_3 SurpriseD_{i,t} + \beta_4 SurpriseS_{i,t}
+ \gamma \beta^{pre}_{i,t} + \delta X^{pre}_{i,t}
+ FE_{SIC2 \times CallYear} + \epsilon_{i,t}.
$$

La baseline oficial usa log market cap, Return60, beta pre-call e
idio-volatilidad pre-call. No incorpora SUE ni ratios contables: éstos no son
necesarios para identificar el cambio posterior de beta y restringen la
muestra. Los errores estándar se agrupan por firma. Se reportan coeficientes
estandarizados.

| Variable | Beta est. | IC95% | p |
|---|---:|---:|---:|
| Historical disclosure intensity | 0.0701 | [0.0207, 0.1196] | 0.005 |
| Historical substantive activity | 0.0082 | [-0.0440, 0.0603] | 0.759 |
| Call-specific disclosure surprise | 0.0286 | [0.0044, 0.0528] | 0.020 |
| Call-specific substance surprise | 0.0029 | [-0.0175, 0.0232] | 0.783 |

N = 6,794 calls; 447 firms; 262 SIC2 × CallYear cells; R² = 0.6919; median
prior calls = 9.

## Robustez de beta post

Un aumento de una desviación estándar en HistDisclosure (SD = 0.334, coef. no
estandarizado 0.103, p = 0.004) se asocia con 0.034 puntos de beta post; en
SurpriseDisclosure (SD = 0.448, coef. no estandarizado 0.030, p = 0.027), con
0.013 puntos. HistSubstance y SurpriseSubstance no son distinguibles de cero
en ninguna especificación (coef. no estandarizados 0.016 y 0.005; p = 0.788 y
0.794).

Los dos contrastes de igualdad disclosure-vs-substance no rechazan H0 a
niveles convencionales (HistDisclosure = HistSubstance: diferencia 0.064,
p = 0.193; SurpriseDisclosure = SurpriseSubstance: diferencia 0.025,
p = 0.229), a pesar de que solo los coeficientes de disclosure son
individualmente significativos. El respaldo de “voice matters, substance does
not” viene de los tests conjuntos, no de estos contrastes pareados: el bloque
de disclosure (HistDisclosure + SurpriseDisclosure) es conjuntamente
significativo (F = 5.15, p = 0.006), el de substance no lo es (F = 0.05,
p = 0.952). El bloque AI completo agrega 0.0051 de R² parcial sobre
controles + FE (N = 6,794).

**Ventana de beta post.** HistDisclosure es positivo y significativo en
`[+21,+63]` (0.082, p = 0.001), `[+21,+126]` (0.071, p = 0.004) y `[+21,+252]`
(0.076, p = 0.005). SurpriseDisclosure lo es en `[+21,+126]` y `[+21,+252]`
(p = 0.026 y 0.005) pero no en la ventana corta `[+21,+63]` (p = 0.089).
Substance permanece nulo en las tres ventanas.

**Ventana de beta pre.** Repitiendo con `[-126,-21]` y `[-252,-42]` como
control de beta pre, HistDisclosure y SurpriseDisclosure se mantienen
significativos (p < 0.03 en ambos casos); substance permanece nulo.

**DeltaBeta.** Usando `DeltaBeta = beta_post - beta_pre` como variable
dependiente, ninguna de las cuatro variables AI es significativa (p entre
0.27 y 0.41): la especificación ANCOVA —que controla por beta pre en vez de
diferenciarlo— es la que sostiene el resultado, y sigue siendo la
especificación principal.

**Contables.** Sobre la misma muestra (N = 4,580), agregar ROA y
liabilities/assets no mueve HistDisclosure (0.0631 → 0.0633 estandarizado,
p = 0.018 → 0.021) ni SurpriseDisclosure (0.0286 → 0.0275, p = 0.056 →
0.065); ninguno de los dos controles contables es significativo
(p = 0.093 y 0.453).

**Firm FE / within.** Reemplazar SIC2 × CallYear por efectos fijos de firma
cambia el resultado: HistDisclosure deja de ser significativo (0.027,
p = 0.589) y HistSubstance se vuelve negativo y significativo (-0.095,
p = 0.035). SurpriseDisclosure y SurpriseSubstance permanecen no
significativos. La identificación within-firm no sostiene el efecto de
HistDisclosure ni el signo nulo de HistSubstance del baseline entre-firmas.

**Bloques AI.** Solo histórico: HistDisclosure = 0.083 (p = 0.003). Solo
sorpresa: SurpriseDisclosure = 0.055 (p < 0.001), casi el doble que en la
especificación conjunta (0.027, p = 0.026). Los dos términos de disclosure se
canibalizan parcialmente al combinarse, consistente con su correlación de
0.43 (ver colinealidad).

**Colinealidad.** HistDisclosure y HistSubstance correlacionan 0.84;
SurpriseDisclosure y SurpriseSubstance, 0.61; los demás pares, entre 0.05 y
0.43. VIF: HistDisclosure 3.78, HistSubstance 3.55, SurpriseDisclosure 1.94,
SurpriseSubstance 1.66 — todos bajo el umbral convencional de 5.

**Influencia.** Winsorizando al 1%, HistDisclosure y SurpriseDisclosure se
mantienen significativos (p = 0.007 y 0.048). Excluyendo el 1% de extremos,
HistDisclosure se mantiene (0.084, p < 0.001) pero SurpriseDisclosure pierde
significancia (0.008, p = 0.525): su efecto depende en parte de un pequeño
número de observaciones extremas, a diferencia de HistDisclosure.

**Estimación de beta.** Exigir al menos 60 u 80 retornos por ventana produce
resultados idénticos (N = 6,794 en ambos casos): la cobertura mínima que ya
exige el panel excede holgadamente 80 observaciones para toda call retenida,
así que este umbral no es una restricción activa.

**Modelo de mercado.** Bajo FF3, HistDisclosure es algo menor que bajo CAPM
(0.062 vs. 0.071, p = 0.014 vs. 0.004) y SurpriseDisclosure es mayor y más
significativo (0.047 vs. 0.028, p = 0.002 vs. 0.026). Substance permanece
nulo bajo ambos modelos. FF5 y un benchmark amplio alternativo no están
disponibles en los datos de este repositorio (solo se dispone de factores
Fama-French de 3 factores) y no se corren.

**Concentración temporal.** Excluir cualquier año individual salvo 2024 deja
HistDisclosure y SurpriseDisclosure significativos. Excluyendo 2024,
HistDisclosure cae a 0.026 (p = 0.136) y SurpriseDisclosure a 0.004
(p = 0.732): el efecto de disclosure se concentra en 2024. Esto contradice
la expectativa inicial de que ningún año individual dominara y debe
reportarse como limitación, no como robustez confirmada.

**Concentración sectorial.** Excluyendo cualquiera de los sectores tech
(SIC2 35, 36 o 73) por separado, HistDisclosure y SurpriseDisclosure
permanecen significativos (p < 0.03); substance permanece nulo. El resultado
completo por sector de 2 dígitos SIC está en
`data/processed/clusters/call_beta_robustness_leave_one_sector_out.csv`.

**Concentración de calls.** Excluyendo el decil superior de firmas por
frecuencia de calls (46 firmas con más de 18 calls en la muestra),
HistDisclosure (0.071, p = 0.019) y SurpriseDisclosure (0.035, p = 0.015)
permanecen significativos.

**Placebo.** Regresando beta pre-call sobre el Disclosure y Substance de la
propia call (contemporáneo, no histórico ni sorpresa), Disclosure predice
fuertemente beta pre (coef. estandarizado 0.176, p < 0.001); Substance no
(p = 0.196). El placebo se esperaba débil o nulo y no lo es para Disclosure:
firmas que ya comunican más vía frames de IA tenían un beta más alto antes
de la call, lo que sugiere que el resultado de HistDisclosure refleja en
parte una característica estable de la firma y no solo una reacción
posterior a la call. Esto debe reportarse como limitación del diseño.

La tabla de paper, con los valores computados:

| Variable | Baseline | + Accounting | Firm FE / within | Delta beta |
|---|---:|---:|---:|---:|
| HistDisclosure | 0.0707 (p=0.004) | 0.0633 (p=0.021) | 0.0273 (p=0.589) | 0.0203 (p=0.321) |
| HistSubstance | 0.0071 (p=0.788) | 0.0168 (p=0.594) | -0.0947 (p=0.035) | 0.0192 (p=0.409) |
| SurpriseDisclosure | 0.0275 (p=0.026) | 0.0275 (p=0.065) | 0.0101 (p=0.442) | 0.0161 (p=0.382) |
| SurpriseSubstance | 0.0027 (p=0.794) | -0.0004 (p=0.976) | -0.0181 (p=0.106) | 0.0174 (p=0.265) |
| Beta pre | ✓ | ✓ | ✓ | — |
| Controles de mercado | ✓ | ✓ | ✓ | ✓ |
| ROA + liabilities/assets | — | ✓ | — | — |
| SIC2 × CallYear FE | ✓ | ✓ | — | ✓ |
| Firm FE | — | — | ✓ | — |
| N | 6,794 | 4,580 | 6,794 | 6,794 |

## Leverage robustness: matched samples

Para cada definición de leverage, M0 y M1 se corren sobre exactamente las
observaciones en que esa medida existe. Así, la diferencia entre modelos se
puede atribuir a agregar leverage y no a cambiar la muestra.

$$ M0:\ \beta^{post} \sim HistD + HistS + SurpriseD + SurpriseS +
\beta^{pre} + X + FE $$

$$ M1:\ M0 + Leverage $$

### Debt / Equity

| Variable | M0 | M1 |
|---|---:|---:|
| HistDisclosure, beta est. | 0.0532 | 0.0533 |
| HistDisclosure, p | 0.111 | 0.111 |
| SurpriseDisclosure, beta est. | 0.0344 | 0.0344 |
| SurpriseDisclosure, p | 0.074 | 0.074 |
| Leverage, beta est. | — | 0.0027 |
| Leverage, p | — | 0.599 |
| R² | 0.7290 | 0.7290 |

N = 3,223 calls; 255 firms; 214 FE cells. El control no mueve los
coeficientes; la diferencia frente al baseline proviene de la muestra.

### Liabilities / Assets

| Variable | M0 | M1 |
|---|---:|---:|
| HistDisclosure, beta est. | 0.0624 | 0.0577 |
| HistDisclosure, p | 0.045 | 0.060 |
| SurpriseDisclosure, beta est. | 0.0313 | 0.0298 |
| SurpriseDisclosure, p | 0.072 | 0.082 |
| Leverage, beta est. | — | -0.0383 |
| Leverage, p | — | 0.008 |
| R² | 0.7103 | 0.7113 |

N = 3,255 calls; 224 firms; 191 FE cells.

## Reproduction

```bash
source .venv/bin/activate
.venv/bin/python scripts/analytics/call_beta/call_car_regressions.py --outcome beta_post_126
```

`call_car_regressions.py` estima tanto la especificación de beta post-call
como la de CAR documentada en `call-car-regressions.md`; `--outcome
beta_post_126` selecciona la primera y no incluye SUE salvo que se pase
`--include-sue`. Outputs: `data/processed/clusters/call_beta_clean_regressions.csv`
y `data/processed/clusters/call_beta_clean_regression_samples.csv`.

La batería de robustez de la sección anterior se corre por separado:

```bash
source .venv/bin/activate
.venv/bin/python scripts/analytics/call_beta/call_beta_robustness.py
```

Recalcula beta e idio-vol desde precios crudos para cada ventana en vez de
confiar en columnas precomputadas, así que todas las variantes de ventana
usan el mismo estimador. Escribe un CSV por bloque de robustez
(`call_beta_robustness_<bloque>.csv`) y un JSON con los tests formales
(`call_beta_robustness_formal_tests.json`) en `data/processed/clusters/`.
