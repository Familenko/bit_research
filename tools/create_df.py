from datetime import date
import numpy as np
import pandas as pd
import ta
from tqdm import tqdm


class SymbolAnalyzer:
    def __init__(self, df, cap_df):
        self.TODAY = date.today().strftime('%Y-%m-%d')
        self.ignore_symbols = ['USDCUSDT', 'FDUSDUSDT', 'USD1USDT']
        self.cap_df = cap_df

        self.volume_df = df.pivot(index='timestamp', columns='symbol', values='volume')
        self.df_open = df.pivot(index='timestamp', columns='symbol', values='open')
        self.df_close = df.pivot(index='timestamp', columns='symbol', values='close')
        self.df_high = df.pivot(index='timestamp', columns='symbol', values='high')
        self.df_low = df.pivot(index='timestamp', columns='symbol', values='low')

        self.optimal_df = None
        self.analyze_df = None
        self.result_df = None

    def run(self, **kwargs):
        self.find_optimal_parameters(**kwargs)
        self.analyze(last_days=kwargs.get("last_days", 180))
        self.result_df = self.analyze_df.merge(
            self.optimal_df, left_index=True, right_index=True, suffixes=('', '_optimal')
        ).reset_index()
        self.result_df = self.result_df[~self.result_df['index'].isin(self.ignore_symbols)]
        return self.result_df

    def find_optimal_parameters(self, symbol_list=None,
                                min_last_days=60, max_last_days=180, step_day=10,
                                min_procent=0.0, max_procent=0.5, step_procent=0.05,
                                min_std_procent=0.0, max_std_procent=0.3, step_std=0.05):

        df_close = self.df_close
        optimal_dict = {}

        if symbol_list is None:
            symbol_list = df_close.columns.tolist()

        for symbol in tqdm(symbol_list, desc='Optimizing'):
            optimal = self._optimize_symbol(
                symbol, min_last_days, max_last_days, step_day,
                min_procent, max_procent, step_procent,
                min_std_procent, max_std_procent, step_std
            )

            if symbol != 'BTCUSDT':
                if (optimal[0] == max_procent and
                    optimal[1] == min_last_days and
                    optimal[2] == max_std_procent):
                    continue

            df = df_close[symbol]
            df_warning_100 = df.iloc[-101:-1]
            df_warning_30 = df.iloc[-31:-1]
            optimal_dict[symbol] = (
                *optimal,
                df_warning_100.min(), df_warning_30.min(), df.iloc[:-31].min(),
                df.iloc[:-31].max(), df_warning_100.max(), df_warning_30.max(),
                df.iloc[-1], df_warning_100.mean(), df_warning_30.mean()
            )

        self.optimal_df = pd.DataFrame(optimal_dict).T
        self.optimal_df.columns = [
            'Optimal Procent', 'Optimal Last Days', 'Optimal Std Procent',
            'Min Support 100', 'Min Support 30', 'Min Historical',
            'Max Historical', 'Max Resist 100', 'Max Resist 30',
            'Last Price', 'Mean 100', 'Mean 30'
        ]
        self.optimal_df.sort_values(by=['Optimal Procent', 'Optimal Last Days', 'Optimal Std Procent'],
                                    ascending=[True, False, False], inplace=True)

    def _optimize_symbol(self, symbol, min_last_days, max_last_days, step_day,
                         min_procent, max_procent, step_procent,
                         min_std_procent, max_std_procent, step_std):
        df = self.df_close[symbol]
        last_price = df.iloc[-1]
        optimal = (max_procent, min_last_days, max_std_procent)

        for last_days in range(min_last_days, max_last_days + 1, step_day):
            df_slice = df.iloc[-last_days:]
            mean_val = df_slice.mean()
            df_warning_30 = df.iloc[-31:-1]

            for std_procent in np.arange(max_std_procent, min_std_procent, -step_std):
                std_n = mean_val * std_procent

                for procent in np.arange(max_procent, min_procent, -step_procent):
                    min_hist = df.iloc[:-31].min()
                    min_hist_coeff = min_hist * (procent + 1.0)
                    min_supp_30 = df_warning_30.min()

                    if (df_slice.std() <= std_n and mean_val <= min_hist_coeff and last_price >= min_supp_30):
                        if (procent < optimal[0] or
                            (procent == optimal[0] and last_days > optimal[1]) or
                            (procent == optimal[0] and last_days == optimal[1] and std_procent < optimal[2])):
                            optimal = (procent, last_days, std_procent)
        return optimal

    def analyze(self, last_days=180):
        assert self.optimal_df is not None, "Спочатку викличте find_optimal_parameters()"
        analyze_df = {}

        for symbol in tqdm(self.optimal_df.index.tolist(), desc='Analyzing'):
            analyze_df[symbol] = self._analyze_symbol(symbol, last_days)

        self.analyze_df = pd.DataFrame(analyze_df).T

    def _analyze_symbol(self, symbol, last_days):
        volume_series = self.volume_df[symbol].iloc[-last_days:]
        interes = self._calculate_interest(volume_series)
        last_rsi = ta.momentum.RSIIndicator(close=self.df_close[symbol], window=30).rsi().iloc[-last_days:].dropna().iloc[-1]

        ma100 = self.df_close[symbol].rolling(window=100).mean().iloc[-last_days:]
        ma30 = self.df_close[symbol].rolling(window=30).mean().iloc[-last_days:]

        opt = self.optimal_df.loc[symbol]
        last_price = opt['Last Price']
        cap = self.cap_df[self.cap_df['symbol'] == symbol]['cap'].values[0] / 1_000_000_000

        atr_series = ta.volatility.AverageTrueRange(
            high=self.df_high[symbol],
            low=self.df_low[symbol],
            close=self.df_close[symbol],
            window=30
        ).average_true_range()
        last_atr = atr_series.dropna().iloc[-1]
        SL = last_price - last_atr * 2.5
        TP = last_price + last_atr * 2.5
        profit_pct = ((TP - last_price) / last_price) * 100

        votes_up, votes_down, votes_neutral = self._candle_votes(symbol, last_days)
        total_votes = int(abs(votes_up - votes_down) - votes_neutral + (interes * 100))

        direction = self._determine_direction(votes_up, votes_down, votes_neutral, total_votes)
        signal_text = self._generate_signal(votes_up, votes_down, total_votes, ma30, ma100, last_rsi, last_price,
                                            opt['Max Resist 100'], opt['Min Support 100'], volume_series)

        return pd.Series({
            'date': self.TODAY,
            'cap': float(cap),
            'profit_pct': float(profit_pct),
            'SL': float(SL),
            'TP': float(TP),
            'direction': direction,
            'votes_up': votes_up,
            'votes_down': votes_down,
            'total_votes': total_votes,
            'signal_text': signal_text,
            'last_rsi': last_rsi,
            'last_atr': last_atr
        })

    def _calculate_interest(self, volume_series):
        cum_vol = volume_series.cumsum()
        cum_vol_norm = (cum_vol - cum_vol.min()) / (cum_vol.max() - cum_vol.min())
        time_norm = np.linspace(0, 1, len(cum_vol_norm))
        diff = cum_vol_norm.values - time_norm
        interes = np.trapezoid(diff, time_norm) * -1
        return 0.0 if np.isnan(interes) else interes

    def _determine_direction(self, votes_up, votes_down, votes_neutral, total_votes):
        scores = f"{total_votes} (U:{votes_up:.1f} D:{votes_down:.1f})"
        if votes_up > votes_down and total_votes > 0:
            return f"⬆️ Up {scores}"
        elif votes_down > votes_up and total_votes > 0:
            return f"⬇️ Down {scores}"
        else:
            return f"➡️ Sideways {scores}"

    def _generate_signal(self, votes_up, votes_down, total_votes, ma30, ma100, last_rsi, last_price, max_resist_100, min_support_100, volume_series):
        entry_signal = (
            votes_up > votes_down and total_votes > 0 and
            ma30.iloc[-1] > ma100.iloc[-1] and
            last_rsi < 70 and
            last_price < max_resist_100 * 1.25 and
            volume_series[-7:].mean() > volume_series[-30:].mean()
        )
        exit_signal = (
            votes_down > votes_up and total_votes > 0 and
            ma30.iloc[-1] < ma100.iloc[-1] and
            last_rsi > 30 and
            last_price > min_support_100 * 0.75 and
            volume_series[-7:].mean() > volume_series[-30:].mean()
        )

        if entry_signal:
            return "BUY"
        elif exit_signal:
            return "SELL"
        return "HOLD"

    def _candle_votes(self, symbol, last_days):
        open_series = self.df_open[symbol].iloc[-last_days:]
        high_series = self.df_high[symbol].iloc[-last_days:]
        low_series = self.df_low[symbol].iloc[-last_days:]
        close_series = self.df_close[symbol].iloc[-last_days:]
        volume_series = self.volume_df[symbol].iloc[-last_days:]

        dates = close_series.index[-100:]
        vol_mean = volume_series[-100:].mean()

        votes_up = 0.0
        votes_down = 0.0
        votes_neutral = 0.0

        for date in dates:
            o = open_series.loc[date]
            h = high_series.loc[date]
            l = low_series.loc[date]
            c = close_series.loc[date]
            body = abs(c - o)
            candle_range = h - l
            lower_shadow = min(c, o) - l
            upper_shadow = h - max(c, o)
            vol = volume_series.loc[date]
            vol_weight = vol / vol_mean if vol_mean > 0 else 1

            if candle_range > 0 and body < candle_range * 0.3:
                if lower_shadow > body * 2:
                    votes_up += 1 * vol_weight  # hammer
                elif upper_shadow > body * 2 and lower_shadow < body * 0.5:
                    votes_down += 1 * vol_weight  # inverted hammer
                # elif body < candle_range * 0.1:
                #     votes_neutral += 1 * vol_weight  # doji
            if body < candle_range * 0.1:
                votes_neutral += 1 * vol_weight  # doji

        # Двосвічкові патерни: Engulfing
        for i in range(1, len(dates)):
            date = dates[i]
            prev_date = dates[i - 1]

            o_prev, c_prev = open_series.loc[prev_date], close_series.loc[prev_date]
            o_curr, c_curr = open_series.loc[date], close_series.loc[date]
            vol = volume_series.loc[date]
            vol_weight = vol / vol_mean if vol_mean > 0 else 1

            # Bullish Engulfing
            if c_prev < o_prev and c_curr > o_curr and o_curr < c_prev and c_curr > o_prev:
                votes_up += 2 * vol_weight

            # Bearish Engulfing
            if c_prev > o_prev and c_curr < o_curr and o_curr > c_prev and c_curr < o_prev:
                votes_down += 2 * vol_weight

        # Трисвічкові патерни: Morning Star / Evening Star
        for i in range(2, len(dates)):
            d1, d2, d3 = dates[i - 2], dates[i - 1], dates[i]
            o1, c1 = open_series.loc[d1], close_series.loc[d1]
            o2, c2 = open_series.loc[d2], close_series.loc[d2]
            o3, c3 = open_series.loc[d3], close_series.loc[d3]
            h1, l1 = high_series.loc[d1], low_series.loc[d1]
            h2, l2 = high_series.loc[d2], low_series.loc[d2]
            h3, l3 = high_series.loc[d3], low_series.loc[d3]
            body1 = abs(c1 - o1)
            body2 = abs(c2 - o2)
            body3 = abs(c3 - o3)
            vol = volume_series.loc[d3]
            vol_weight = vol / vol_mean if vol_mean > 0 else 1

            # Morning Star
            if (c1 < o1 and body1 > (h1 - l1) * 0.5 and
                body2 < body1 * 0.3 and
                c3 > o3 and body3 > (h3 - l3) * 0.5 and
                c3 > (c1 + o1) / 2):
                votes_up += 3 * vol_weight

            # Evening Star
            if (c1 > o1 and body1 > (h1 - l1) * 0.5 and
                body2 < body1 * 0.3 and
                c3 < o3 and body3 > (h3 - l3) * 0.5 and
                c3 < (c1 + o1) / 2):
                votes_down += 3 * vol_weight

        return votes_up, votes_down, votes_neutral

