import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet


def detect_top_changepoints(series, changepoint_n=3):
    series.index = pd.to_datetime(series.index)
    df = pd.DataFrame({"ds": series.index, "y": series.values})

    m = Prophet()
    m.fit(df)

    cps = m.changepoints
    deltas = m.params['delta'].mean(0)

    if len(cps) == 0:
        return []

    cp_df = pd.DataFrame({"date": cps, "strength": deltas})

    cp_df = cp_df.reindex(cp_df["strength"].abs().sort_values(ascending=False).index)
    top3 = cp_df.sort_values("date").tail(changepoint_n)

    results = []
    for _, row in top3.iterrows():
        mask = series.index >= row["date"]
        if not mask.any():
            continue
        closest_idx = series.index[mask][0]
        results.append(series.iloc[series.index.get_loc(closest_idx)])

    return results
