def kadane_subarray(close_series, window=100):
    rolling_mean = close_series.rolling(window=window, min_periods=1).mean()
    rolling_mean = rolling_mean * 1.1

    arr = close_series.values - rolling_mean.values

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

    start_idx = close_series.index[start]
    end_idx = close_series.index[end]

    kadane_avg = close_series.loc[start_idx:end_idx].mean()
    kadane_coef = close_series.iloc[-1] / kadane_avg if end_idx == close_series.index[-1] else 0.0

    return start_idx, end_idx, kadane_coef
