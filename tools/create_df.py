from datetime import date
import numpy as np
import pandas as pd
import ta
from tqdm import tqdm

from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import ta


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

        self.result_df = self.analyze_df.merge(self.optimal_df, left_index=True, right_index=True).reset_index()
        self.result_df = self.result_df[~self.result_df['index'].isin(self.ignore_symbols)]

        self.result_df.rename(columns={'index': 'symbol'}, inplace=True)
        self.result_df = self.result_df.sort_values(['total_votes', 'cap',
                                                     'Optimal Procent', 'Optimal Last Days', 'Optimal Std Procent'], 
                                                     ascending=[False, False, True, False, False])
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

    def graph(self, last_days=180):
        assert self.result_df is not None, "Спочатку викличте run()"
        
        last_days = max(last_days, 100)

        num_symbols = len(self.result_df)
        fig, axes = plt.subplots(nrows=num_symbols, ncols=1, figsize=(16, 6 * num_symbols), sharex=False)

        if num_symbols == 1:
            axes = [axes]

        global_norm = pd.DataFrame()
        for symbol in self.result_df['symbol']:
            series = self.df_close[symbol].iloc[-last_days:]
            norm_series = (series - series.min()) / (series.max() - series.min())
            global_norm[symbol] = norm_series
        global_line = global_norm.mean(axis=1)

        for idx, symbol in enumerate(self.result_df['symbol']):
            series = self.df_close[symbol].iloc[-last_days:]
            volume_series = self.volume_df[symbol].iloc[-last_days:]
            ax = axes[idx]

            # ==== Графік інтересу ====
            cum_vol = volume_series.cumsum()
            cum_vol_norm = (cum_vol - cum_vol.min()) / (cum_vol.max() - cum_vol.min())
            time_norm = np.linspace(0, 1, len(cum_vol_norm))
            ideal_line = time_norm
            diff = cum_vol_norm.values - ideal_line
            interes = np.trapezoid(diff, time_norm) * -1
            if interes is None or np.isnan(interes):
                interes = 0.0

            ax_vol = ax.twinx()
            cum_vol.plot(ax=ax_vol, color='gray', linestyle='-', linewidth=0.5, label=f'Cumulative Volume ({interes:.2f})')
            ax_vol.set_ylabel('Cumulative Volume', color='gray')
            ax_vol.tick_params(axis='y', colors='gray')
            ax_vol.legend(loc='upper right', fontsize=8)

            # ==== Графік ціни ====
            series.plot(ax=ax, label='Price', color='gray', linewidth=1.5)

            # ==== Глобальна лінія ====
            global_scaled = global_line * (series.max() - series.min()) + series.min()
            global_scaled.plot(ax=ax, label='Global Mean', color='black', linestyle='--', linewidth=0.5)

            # ==== 200-денна ковзна ====
            ma200_full = self.df_close[symbol].rolling(window=200).mean()
            ma200 = ma200_full.iloc[-last_days:]
            ma200.plot(ax=ax, color='purple', linestyle='-', linewidth=1.2, label='MA 200')

            # ==== 100-денна ковзна ====
            ma100_full = self.df_close[symbol].rolling(window=100).mean()
            ma100 = ma100_full.iloc[-last_days:]
            ma100.plot(ax=ax, color='blue', linestyle='-', linewidth=1.2, label='MA 100')

            ma100_global = global_scaled.rolling(window=100).mean()
            ma100_global.plot(ax=ax, color='blue', linestyle='--', linewidth=0.5, label='Global MA 100')

            # ==== 30-денна ковзна ====
            ma30_full = self.df_close[symbol].rolling(window=30).mean()
            ma30 = ma30_full.iloc[-last_days:]
            ma30.plot(ax=ax, color='orange', linestyle='-', linewidth=1.2, label='MA 30')

            # ==== Лінії підтримки та опору ====
            symbol_cap = self.result_df.loc[self.result_df['symbol'] == symbol, 'cap'].values[0]
            min_support_100 = self.optimal_df.loc[symbol, 'Min Support 100']
            max_resist_100 = self.optimal_df.loc[symbol, 'Max Resist 100']
            min_support_30 = self.optimal_df.loc[symbol, 'Min Support 30']
            max_resist_30 = self.optimal_df.loc[symbol, 'Max Resist 30']
            last_price = self.optimal_df.loc[symbol, 'Last Price']
            max_historical = self.optimal_df.loc[symbol, 'Max Historical']

            # ==== Розрахунок TP та SL ====
            SL = self.result_df.loc[self.result_df['symbol'] == symbol, 'SL'].values[0]
            TP = self.result_df.loc[self.result_df['symbol'] == symbol, 'TP'].values[0]
            profit_pct = self.result_df.loc[self.result_df['symbol'] == symbol, 'profit_pct'].values[0]

            # ==== Фарбування ділянок підтримки та опору ====
            ax.axhspan(min_support_100, max_resist_100, color='lightgreen', alpha=0.1)
            ax.axhspan(max_resist_30, max_resist_100, color='red', alpha=0.1)

            ax.axhline(last_price, color='green', linestyle='--',
                    label=f"Last Price ({last_price:.2f})")
            ax.text(series.index[-1], last_price, f' {last_price:.2f}', 
                    verticalalignment='bottom', color='green', fontsize=10)
            
            ax.axhline(min_support_100, color='gray', linestyle='dotted',
                    label=f"Min Support 100 ({min_support_100:.2f})")
            ax.axhline(min_support_30, color='orange', linestyle='--',
                    label=f"Min Support 30 ({min_support_30:.2f})")

            ax.axhline(max_resist_100, color='gray', linestyle='dotted',
                    label=f"Max Resist 100 ({max_resist_100:.2f})")
            ax.axhline(max_resist_30, color='gray', linestyle='dotted',
                    label=f"Max Resist 30 ({max_resist_30:.2f})")

            ax.axhline(self.optimal_df.loc[symbol, 'Min Historical'], color='red', linestyle='--',
                       label=f"Min Historical ({self.optimal_df.loc[symbol, 'Min Historical']:.2f})")
            ax.axhline(max_historical, color='red', linestyle='--',
                       label=f"Max Historical ({max_historical:.2f})")
            
            if last_price > max_historical:
                ax.axhspan(max_historical, last_price, color='yellow', alpha=0.5, label='Above Max Historical')

            # ==== Найбільший обʼєм за останні 100 днів ====
            vol_max_idx_100 = volume_series[-100:].idxmax()
            vol_mean_100 = volume_series[-100:].mean()
            if volume_series[vol_max_idx_100] > vol_mean_100 * 2:
                price_at_vol_max_100 = series.loc[vol_max_idx_100]
                pos_100 = series.index.get_loc(vol_max_idx_100)

                ax.text(pos_100, price_at_vol_max_100, f' {price_at_vol_max_100:.2f}', 
                        verticalalignment='bottom', color='red', fontsize=10)

            # ==== Найбільший обʼєм за останні 30 днів ====
            vol_max_idx_30 = volume_series[-30:].idxmax()
            vol_mean_30 = volume_series[-30:].mean()
            if volume_series[vol_max_idx_30] > vol_mean_30 * 2:
                price_at_vol_max_30 = series.loc[vol_max_idx_30]
                pos_30 = series.index.get_loc(vol_max_idx_30)

                ax.text(pos_30, price_at_vol_max_30, f' {price_at_vol_max_30:.2f}', 
                        verticalalignment='bottom', color='red', fontsize=10)

            # ==== Лінії часу ====
            if len(series) >= 100:
                pos_30 = len(series) - 30
                pos_100 = len(series) - 100
                
                ax.axvline(pos_30, color='gray', linestyle=':', label='30 Days Ago')
                ax.axvline(pos_100, color='gray', linestyle=':', label='100 Days Ago')

            # ==== Накладання обʼєму ====
            lower_limit = min_support_100 * 0.8
            upper_limit = max_resist_100 * 1.2
            ax.set_ylim(lower_limit, upper_limit)
            vol_scaled = volume_series / volume_series.max() * (upper_limit - lower_limit) * 0.2 + lower_limit
            ax.fill_between(vol_scaled.index, vol_scaled, color='gray', alpha=0.3, label='Volume (scaled)')

            # ==== CANDLE PATTERNS ====
            open_series = self.df_open[symbol].iloc[-last_days:]
            high_series = self.df_high[symbol].iloc[-last_days:]
            low_series = self.df_low[symbol].iloc[-last_days:]
            close_series = self.df_close[symbol].iloc[-last_days:]

            # Візьмемо останні 100 днів для голосування патернів
            last_100_dates = series.index[-100:]

            # Одно-свічкові патерни (hammer, inverted hammer, doji)
            for date in last_100_dates:
                o = open_series.loc[date]
                h = high_series.loc[date]
                l = low_series.loc[date]
                c = close_series.loc[date]
                body = abs(c - o)
                candle_range = h - l
                lower_shadow = min(c, o) - l
                upper_shadow = h - max(c, o)

                # hammer (позитивний сигнал)
                if candle_range > 0 and body < candle_range * 0.3 and lower_shadow > body * 2:
                    ax.scatter(date, l, color='green', s=20, marker='P',
                            label='Hammer' if 'Hammer' not in ax.get_legend_handles_labels()[1] else "")

                # inverted hammer (негативний сигнал)
                if candle_range > 0 and body < candle_range * 0.3 and upper_shadow > body * 2 and lower_shadow < body * 0.5:
                    ax.scatter(date, h, color='red', s=20, marker='P',
                            label='Inverted Hammer' if 'Inverted Hammer' not in ax.get_legend_handles_labels()[1] else "")

                # doji — сумнівний сигнал (не враховуємо в голосуванні)
                if candle_range > 0 and body < candle_range * 0.1:
                    ax.scatter(date, c, color='orange', s=30, marker='D',
                            label='Doji' if 'Doji' not in ax.get_legend_handles_labels()[1] else "")

            # Три-свічкові патерни (Morning Star, Evening Star)
            for i in range(2, len(last_100_dates)):
                date = last_100_dates[i]
                date_prev1 = last_100_dates[i-1]
                date_prev2 = last_100_dates[i-2]

                o1, c1 = open_series.loc[date_prev2], close_series.loc[date_prev2]
                o2, c2 = open_series.loc[date_prev1], close_series.loc[date_prev1]
                o3, c3 = open_series.loc[date],       close_series.loc[date]

                h1, l1 = high_series.loc[date_prev2], low_series.loc[date_prev2]
                h2, l2 = high_series.loc[date_prev1], low_series.loc[date_prev1]
                h3, l3 = high_series.loc[date],       low_series.loc[date]

                body1 = abs(c1 - o1)
                body2 = abs(c2 - o2)
                body3 = abs(c3 - o3)

                # Morning Star — сигнал на підйом
                if (c1 < o1 and body1 > (h1 - l1) * 0.5 and
                    body2 < body1 * 0.3 and
                    c3 > o3 and body3 > (h3 - l3) * 0.5 and
                    c3 > ((c1 + o1)/2)):
                    ax.scatter(date, c3, color='green', s=80, marker='*',
                            label='Morning Star' if 'Morning Star' not in ax.get_legend_handles_labels()[1] else "")

                # Evening Star — сигнал на падіння
                if (c1 > o1 and body1 > (h1 - l1) * 0.5 and
                    body2 < body1 * 0.3 and
                    c3 < o3 and body3 > (h3 - l3) * 0.5 and
                    c3 < ((c1 + o1)/2)):
                    ax.scatter(date, c3, color='red', s=80, marker='*',
                            label='Evening Star' if 'Evening Star' not in ax.get_legend_handles_labels()[1] else "")

            # Дво-свічкові патерни (Bullish Engulfing, Bearish Engulfing)
            for i in range(1, len(last_100_dates)):
                date = last_100_dates[i]
                date_prev = last_100_dates[i-1]

                h = high_series.loc[date]
                l = low_series.loc[date]

                o_prev, c_prev = open_series.loc[date_prev], close_series.loc[date_prev]
                o_curr, c_curr = open_series.loc[date],     close_series.loc[date]

                # Bullish Engulfing — вгору
                if (c_prev < o_prev and
                    c_curr > o_curr and
                    o_curr < c_prev and
                    c_curr > o_prev):
                    ax.scatter(date, l, color='green', s=30, marker='^',
                            label='Engulfing Bullish' if 'Engulfing Bullish' not in ax.get_legend_handles_labels()[1] else "")

                # Bearish Engulfing — вниз
                if (c_prev > o_prev and
                    c_curr < o_curr and
                    o_curr > c_prev and
                    c_curr < o_prev):
                    ax.scatter(date, h, color='red', s=30, marker='v',
                            label='Engulfing Bearish' if 'Engulfing Bearish' not in ax.get_legend_handles_labels()[1] else "")

            direction = self.result_df.loc[self.result_df['symbol'] == symbol, 'direction'].values[0]
            signal_text = self.result_df.loc[self.result_df['symbol'] == symbol, 'signal_text'].values[0]
            last_rsi = self.result_df.loc[self.result_df['symbol'] == symbol, 'last_rsi'].values[0]
            last_atr = self.result_df.loc[self.result_df['symbol'] == symbol, 'last_atr'].values[0]
            ax.set_title(f"{symbol} | Cap: {symbol_cap:.2f}B USD | Profit: {profit_pct:.2f}% | "
                         f"SL: {SL:.2f} TP: {TP:.2f} | RSI: {last_rsi:.1f} ATR: {last_atr:.2f} | "
                         f"Trend: {direction} Signal: {signal_text}", fontsize=14)

            # ==== Налаштування графіка ====
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()
