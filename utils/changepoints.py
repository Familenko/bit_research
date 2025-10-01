import pandas as pd
from prophet import Prophet


def detect_top_changepoints(close_series, changepoint_n=3):
    close_series.index = pd.to_datetime(close_series.index)
    df = pd.DataFrame({"ds": close_series.index, "y": close_series.values})

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
        mask = close_series.index >= row["date"]
        if not mask.any():
            continue
        closest_idx = close_series.index[mask][0]
        results.append(close_series.iloc[close_series.index.get_loc(closest_idx)])

    return results
