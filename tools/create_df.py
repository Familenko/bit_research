from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import ta


TODAY = date.today().strftime('%Y-%m-%d')


def make_df(df_close, 
            symbols, 
            info_df, 
            cap_df, 
            volume_df, 
            df_open, 
            df_max, 
            df_min,
            last_days=180):
    
    result = pd.DataFrame()
    df_rsi = pd.DataFrame()

    last_days = max(last_days, 100)

    for symbol in symbols:
        series = df_close[symbol].iloc[-last_days:]
        volume_series = volume_df[symbol].iloc[-last_days:]

        # ==== Розрахунок інтересу ====
        cum_vol = volume_series.cumsum()
        cum_vol_norm = (cum_vol - cum_vol.min()) / (cum_vol.max() - cum_vol.min())
        time_norm = np.linspace(0, 1, len(cum_vol_norm))
        ideal_line = time_norm
        diff = cum_vol_norm.values - ideal_line
        interes = np.trapezoid(diff, time_norm) * -1
        if interes is None or np.isnan(interes):
            interes = 0.0

        # ==== RSI Series ====
        df_rsi[symbol] = ta.momentum.RSIIndicator(close=df_close[symbol], window=30).rsi()
        rsi_series = df_rsi[symbol].iloc[-last_days:]
        last_rsi = rsi_series.dropna().iloc[-1]

        # ==== 100-денна та 30-денна ковзна ====
        ma100_full = df_close[symbol].rolling(window=100).mean()
        ma100 = ma100_full.iloc[-last_days:]

        ma30_full = df_close[symbol].rolling(window=30).mean()
        ma30 = ma30_full.iloc[-last_days:]

        # ==== Лінії підтримки та опору ====
        symbol_cap = cap_df[cap_df['symbol'] == symbol]['cap'].values[0] / 1_000_000_000
        min_support_100 = info_df.loc[symbol, 'Min Support 100']
        max_resist_100 = info_df.loc[symbol, 'Max Resist 100']
        last_price = info_df.loc[symbol, 'Last Price']

        # ==== Розрахунок TP та SL ====
        atr_series = ta.volatility.AverageTrueRange(
            high=df_max[symbol],
            low=df_min[symbol],
            close=df_close[symbol],
            window=30
        ).average_true_range()
        last_atr = atr_series.dropna().iloc[-1]
        SL = last_price - last_atr * 2.5
        TP = last_price + last_atr * 2.5
        profit_pct = ((TP - last_price) / last_price) * 100

        # ==== CANDLE PATTERNS ====
        open_series = df_open[symbol].iloc[-last_days:]
        high_series = df_max[symbol].iloc[-last_days:]
        low_series = df_min[symbol].iloc[-last_days:]

        # Візьмемо останні 100 днів для голосування патернів
        last_100_dates = series.index[-100:]

        # Голосування за напрямок: рахунок "вгору" і "вниз"
        votes_up = 0.0
        votes_down = 0.0
        votes_neutral = 0.0

        # Одно-свічкові патерни (hammer, inverted hammer, doji)
        for date in last_100_dates:
            o = open_series.loc[date]
            h = high_series.loc[date]
            l = low_series.loc[date]
            c = series.loc[date]
            body = abs(c - o)
            candle_range = h - l
            lower_shadow = min(c, o) - l
            upper_shadow = h - max(c, o)
            vol = volume_series.loc[date]
            vol_mean_100 = volume_series[-100:].mean()
            vol_weight = vol / vol_mean_100 if vol_mean_100 > 0 else 1

            # hammer (позитивний сигнал)
            if candle_range > 0 and body < candle_range * 0.3 and lower_shadow > body * 2:
                votes_up += 1 * vol_weight

            # inverted hammer (негативний сигнал)
            if candle_range > 0 and body < candle_range * 0.3 and upper_shadow > body * 2 and lower_shadow < body * 0.5:
                votes_down += 1 * vol_weight

            # doji — сумнівний сигнал (не враховуємо в голосуванні)
            if candle_range > 0 and body < candle_range * 0.1:
                votes_neutral += 1 * vol_weight

        # Три-свічкові патерни (Morning Star, Evening Star)
        for i in range(2, len(last_100_dates)):
            date = last_100_dates[i]
            date_prev1 = last_100_dates[i-1]
            date_prev2 = last_100_dates[i-2]

            o1, c1 = open_series.loc[date_prev2], series.loc[date_prev2]
            o2, c2 = open_series.loc[date_prev1], series.loc[date_prev1]
            o3, c3 = open_series.loc[date],       series.loc[date]

            h1, l1 = high_series.loc[date_prev2], low_series.loc[date_prev2]
            h2, l2 = high_series.loc[date_prev1], low_series.loc[date_prev1]
            h3, l3 = high_series.loc[date],       low_series.loc[date]

            body1 = abs(c1 - o1)
            body2 = abs(c2 - o2)
            body3 = abs(c3 - o3)

            vol = volume_series.loc[date]
            vol_weight = vol / volume_series[-100:].mean() if volume_series[-100:].mean() > 0 else 1

            # Morning Star — сигнал на підйом
            if (c1 < o1 and body1 > (h1 - l1) * 0.5 and
                body2 < body1 * 0.3 and
                c3 > o3 and body3 > (h3 - l3) * 0.5 and
                c3 > ((c1 + o1)/2)):
                votes_up += 3 * vol_weight

            # Evening Star — сигнал на падіння
            if (c1 > o1 and body1 > (h1 - l1) * 0.5 and
                body2 < body1 * 0.3 and
                c3 < o3 and body3 > (h3 - l3) * 0.5 and
                c3 < ((c1 + o1)/2)):
                votes_down += 3 * vol_weight

        # Дво-свічкові патерни (Bullish Engulfing, Bearish Engulfing)
        for i in range(1, len(last_100_dates)):
            date = last_100_dates[i]
            date_prev = last_100_dates[i-1]

            h = high_series.loc[date]
            l = low_series.loc[date]

            o_prev, c_prev = open_series.loc[date_prev], series.loc[date_prev]
            o_curr, c_curr = open_series.loc[date],      series.loc[date]

            vol = volume_series.loc[date]
            vol_weight = vol / volume_series[-100:].mean() if volume_series[-100:].mean() > 0 else 1

            # Bullish Engulfing — вгору
            if (c_prev < o_prev and
                c_curr > o_curr and
                o_curr < c_prev and
                c_curr > o_prev):
                votes_up += 2 * vol_weight

            # Bearish Engulfing — вниз
            if (c_prev > o_prev and
                c_curr < o_curr and
                o_curr > c_prev and
                c_curr < o_prev):
                votes_down += 2 * vol_weight

        # ==== Визначення підсумкового напрямку з балами ====
        total_votes = int(abs(votes_up - votes_down) - votes_neutral + (interes * 100))
        scores = f"{total_votes} (U:{votes_up:.1f} D:{votes_down:.1f})"
        if votes_up > votes_down and total_votes > 0:
            direction = f"⬆️ Up {scores}"
        elif votes_down > votes_up and total_votes > 0:
            direction = f"⬇️ Down {scores}"
        else:
            direction = f"➡️ Sideways {scores}"

        # ==== Визначення сигналів на вхід та вихід ====
        entry_signal = False
        exit_signal = False

        rsi_overbought = 70
        rsi_oversold = 30

        if ((votes_up > votes_down) and 
            (total_votes > 0) and 
            (ma30.iloc[-1] > ma100.iloc[-1]) and 
            (last_rsi < rsi_overbought) and
            (last_price < max_resist_100 * 1.25) and 
            (volume_series[-7:].mean() > volume_series[-30:].mean())):
            entry_signal = True

        if ((votes_down > votes_up) and
            (total_votes > 0) and
            (ma30.iloc[-1] < ma100.iloc[-1]) and
            (last_rsi > rsi_oversold) and
            (last_price > min_support_100 * 0.75) and
            (volume_series[-7:].mean() > volume_series[-30:].mean())):
            exit_signal = True

        if entry_signal:
            signal_text = "BUY"
        elif exit_signal:
            signal_text = "SELL"
        else:
            signal_text = "HOLD"

        result[symbol] = pd.Series({
            'cap': float(symbol_cap),
            f'{TODAY}': last_price,
            'profit_pct': float(profit_pct),
            'SL': float(SL),
            'TP': float(TP),
            'direction': direction,
            'signal_text': signal_text,
            'last_rsi': last_rsi,
            'last_atr': last_atr,
            'votes_up': votes_up,
            'votes_down': votes_down,
            'total_votes': int(total_votes)
        })

    return result
