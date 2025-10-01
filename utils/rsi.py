import ta


def rsi_metric(close_series, window=30):
    return ta.momentum.RSIIndicator(close=close_series, window=window).rsi().iloc[-1]