import ta


def adx_metric(high_series, low_series, close_series):
    adx_indicator = ta.trend.ADXIndicator(high=high_series, low=low_series, close=close_series, window=14)
    adx_val = adx_indicator.adx().iloc[-1]
    plus_di_val = adx_indicator.adx_pos().iloc[-1]
    minus_di_val = adx_indicator.adx_neg().iloc[-1]

    return adx_val, plus_di_val, minus_di_val