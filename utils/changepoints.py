import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet


def detect_top_changepoints(open_series, changepoint_n=3):
    open_series.index = pd.to_datetime(open_series.index)
    df = pd.DataFrame({"ds": open_series.index, "y": open_series.values})

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
        mask = open_series.index >= row["date"]
        if not mask.any():
            continue
        closest_idx = open_series.index[mask][0]
        results.append(open_series.iloc[open_series.index.get_loc(closest_idx)])

    return results
