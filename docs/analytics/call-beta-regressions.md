# Earnings-call beta regressions

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

`X` contiene log market cap, Return60, operating margin y asset turnover. Los
errores estándar se agrupan por firma. Se reportan coeficientes
estandarizados.

| Variable | Beta est. | IC95% | p |
|---|---:|---:|---:|
| Historical disclosure intensity | 0.0775 | [0.0184, 0.1366] | 0.010 |
| Historical substantive activity | -0.0002 | [-0.0615, 0.0610] | 0.994 |
| Call-specific disclosure surprise | 0.0347 | [0.0058, 0.0636] | 0.019 |
| Call-specific substance surprise | 0.0032 | [-0.0217, 0.0282] | 0.799 |

N = 4,881 calls; 330 firms; 231 SIC2 × CallYear cells; R² = 0.7204; within
R² = 0.4592; median prior calls = 9; median financial-filing age = 90 days.

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
.venv/bin/python scripts/analytics/call_beta_regressions.py --model all
```

`--model baseline`, `--model debt_to_equity`, and
`--model liabilities_to_assets` run each family separately. Outputs are
written to `data/processed/clusters/call_beta_regressions.csv` and
`data/processed/clusters/call_beta_regression_samples.csv`.
