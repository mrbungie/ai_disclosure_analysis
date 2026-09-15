# Regresión CAR de earnings calls

`scripts/analytics/call_beta/call_car_regressions.py` estima una especificación
separada de la regresión de beta post-call. La variable dependiente es
`car_m1_p5`: suma de retornos anormales de los días 0 a +5, con el día -1
como precio de referencia, alrededor de la fecha exacta de la call.

El modelo de mercado se estima únicamente con retornos de las ruedas
`[-252,-21]` previas a cada call. De esa misma ventana salen `beta_pre_car`
e `idio_vol_pre`; las ventanas de estimación y de evento no se solapan.

La baseline es:

$$
CAR_{i,t} = AI_{i,t} + SUE_{i,t^-} + Return60_{i,t^-} +
\beta^{pre}_{i,t} + IdioVol^{pre}_{i,t} + \log(MarketCap_{i,t^-}) +
FE_{SIC2\times CallYear} + \epsilon_{i,t}.
$$

`AI` contiene `HistDisclosure`, `HistSubstance`, `SurpriseDisclosure` y
`SurpriseSubstance`. Los errores estándar son clustered por firma.

No hay consensus EPS local. Por eso `SUE` es un proxy contable as-of:

$$
SUE_{i,q}=\frac{EPS_{i,q}-EPS_{i,q-4}}
{sd_{\text{estrictamente previo}}(EPS_{i,q}-EPS_{i,q-4})}.
$$

Cada call recibe sólo el último 10-Q publicado estrictamente antes de ella.
La dispersión usa una ventana expanding desplazada: no incorpora el cambio
del trimestre actual ni datos futuros. Requiere cuatro cambios interanuales
previos.

Como robustez, el script estima M0 y M1 sobre la misma muestra con
contables observables. M1 agrega ROA (del último 10-K/10-Q as-of) y
liabilities/assets de los facts del último filing anterior a la call.

Archivos generados:

- `data/processed/clusters/call_car_panel.parquet`
- `data/processed/clusters/call_car_regressions.csv`
- `data/processed/clusters/call_car_regression_samples.csv`
