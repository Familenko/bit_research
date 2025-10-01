import ta


def atr_metric(high, low, close, window=30):
    atr_series = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=window).average_true_range()
    last_atr = atr_series.iloc[-1]
    
    return last_atr