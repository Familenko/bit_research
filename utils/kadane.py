import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def kadane_subarray(open_series, window=100):
    rolling_mean = open_series.rolling(window=window, min_periods=1).mean()
    rolling_mean = rolling_mean * 1.1

    arr = open_series.values - rolling_mean.values

    max_sum = current_sum = arr[0]
    start = end = temp_start = 0

    for i in range(1, len(arr)):
        if arr[i] > current_sum + arr[i]:
            current_sum = arr[i]
            temp_start = i
        else:
            current_sum += arr[i]

        if current_sum >= max_sum:
            max_sum = current_sum
            start = temp_start
            end = i

    return open_series.index[start], open_series.index[end]
