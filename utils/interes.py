import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def interes_metric(volume_series):
    cum_vol = volume_series.cumsum()
    cum_vol_norm = (cum_vol - cum_vol.min()) / (cum_vol.max() - cum_vol.min())
    time_norm = np.linspace(0, 1, len(cum_vol_norm))
    ideal_line = time_norm
    diff = cum_vol_norm.values - ideal_line
    interes = np.trapezoid(diff, time_norm) * -1
    if interes is None or np.isnan(interes):
        interes = 0.0

    return interes
