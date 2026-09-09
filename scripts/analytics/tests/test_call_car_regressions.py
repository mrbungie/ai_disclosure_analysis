import pandas as pd


def test_sue_uses_only_prior_earnings_changes():
    from scripts.analytics.call_car_regressions import standardized_unexpected_earnings

    eps = pd.DataFrame({
        "ticker": ["AAA"] * 9,
        "filing_date": pd.date_range("2023-01-01", periods=9, freq="90D"),
        "eps_diluted": [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0],
    })
    actual = standardized_unexpected_earnings(eps, min_history=2)

    # The first two quarter-on-year changes establish the prior dispersion;
    # the third can be scored.  Its denominator must exclude its own change.
    assert actual.loc[6, "sue"].round(6) == 5.656854
    assert actual.loc[5, "sue"] != actual.loc[5, "sue"]  # not enough prior history
