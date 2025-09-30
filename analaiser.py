from datetime import date

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm

import ta

from utils.kadane import kadane_subarray
from utils.changepoints import detect_top_changepoints
from utils.adx import adx_metric
from utils.interes import interes_metric


class SymbolAnalyzer:
    def __init__(self, df, cap_df):
        self.TODAY = date.today().strftime('%Y-%m-%d')
        self.ignore_symbols = ['USDCUSDT', 'FDUSDUSDT', 'USD1USDT']

        self.data = {}
        self.data['capital'] = cap_df
        self.data['volume'] = df.pivot(index='timestamp', columns='symbol', values='volume')
        self.data['open'] = df.pivot(index='timestamp', columns='symbol', values='open')
        self.data['close'] = df.pivot(index='timestamp', columns='symbol', values='close')
        self.data['high'] = df.pivot(index='timestamp', columns='symbol', values='high')
        self.data['low'] = df.pivot(index='timestamp', columns='symbol', values='low')

        self.bullish_pattern = []

        self.result_df = None

    def _forming_result_df(self, analised_symbols):
        result_df = pd.DataFrame.from_dict(analised_symbols, orient='index')
        result_df.reset_index(inplace=True)
        result_df.rename(columns={'index': 'symbol'}, inplace=True)
        result_df = result_df[~result_df['symbol'].isin(self.ignore_symbols)]
        result_df = result_df.sort_values(
            ['total_votes', 'cap', 'max_procent', 'min_last_days', 'max_std_procent'],
            ascending=[False, False, True, False, False])
        
        desired_order = [
            'date', 'symbol', 'last_price', 'direction', 'signal_text', 'cap', 'SL', 'TP',
            'interes', 'last_rsi', 'last_atr', 'max_procent', 'min_last_days', 'max_std_procent',
            'min_support_100', 'max_resist_100', 'min_support_30', 'max_resist_30', 'min_historical', 'max_historical', 'mean_100', 'mean_30',
            'votes_up', 'votes_down', 'total_votes'
        ]
        result_df = result_df[[col for col in desired_order if col in result_df.columns]]

        return result_df

    def run(self, **kwargs):
        optimal_symbols = self.find_optimal_token(
            symbol_list=kwargs.get('symbol_list', 'all'),
        )
        analised_symbols = self.analyze(
            optimal_symbols=kwargs.get('symbol_list', optimal_symbols), 
            last_days=kwargs.get('last_days', 180)
            )

        self.result_df = self._forming_result_df(analised_symbols)
    
        return self.result_df

    def find_optimal_token(self, symbol_list='all',
                                min_last_days=60, max_last_days=180, step_day=10,
                                min_procent=0.0, max_procent=0.5, step_procent=0.05,
                                min_std_procent=0.0, max_std_procent=0.3, step_std=0.05):

        close_series = self.data['close']

        if symbol_list == 'all':
            symbol_list = close_series.columns.tolist()

        optimal_symbols = {}
        for symbol in tqdm(symbol_list, desc='Optimizing'):
            close_series_symbol = close_series[symbol]
            max_procent_found, min_last_days_found, max_std_procent_found = self._optimize_symbol(
                close_series_symbol, 
                min_last_days, max_last_days, step_day,
                min_procent, max_procent, step_procent,
                min_std_procent, max_std_procent, step_std
            )

            if symbol != 'BTCUSDT':
                if (max_procent_found == max_procent and
                    min_last_days_found == min_last_days and
                    max_std_procent_found == max_std_procent):
                    continue

            optimal_symbols[symbol] = {
                'max_procent': max_procent_found,
                'min_last_days': min_last_days_found,
                'max_std_procent': max_std_procent_found
            }

        return optimal_symbols

    def _optimize_symbol(self, close_series_symbol, min_last_days, max_last_days, step_day,
                         min_procent, max_procent, step_procent,
                         min_std_procent, max_std_procent, step_std):
        
        last_price = close_series_symbol.iloc[-1]
        optimal = (max_procent, min_last_days, max_std_procent)

        for last_days in range(min_last_days, max_last_days + 1, step_day):
            df_slice = close_series_symbol.iloc[-last_days:]
            mean_val = df_slice.mean()
            min_supp_30 = close_series_symbol.iloc[-31:-1].min()

            for std_procent in np.arange(max_std_procent, min_std_procent, -step_std):
                std_n = mean_val * std_procent

                for procent in np.arange(max_procent, min_procent, -step_procent):
                    min_hist = close_series_symbol.iloc[:-31].min()
                    min_hist_coeff = min_hist * (procent + 1.0)

                    if (df_slice.std() <= std_n and mean_val <= min_hist_coeff and last_price >= min_supp_30):
                        if (procent < optimal[0] or
                            (procent == optimal[0] and last_days > optimal[1]) or
                            (procent == optimal[0] and last_days == optimal[1] and std_procent < optimal[2])):
                            optimal = (procent, last_days, std_procent)

        return optimal

    def analyze(self, last_days=180, optimal_symbols='all'):

        analised_symbols = {}
        if isinstance(optimal_symbols, list):
            optimal_symbols = {symbol: {} for symbol in optimal_symbols}

        if optimal_symbols == 'all':
            optimal_symbols = {symbol: {} for symbol in self.data['close'].columns.tolist()}

        for symbol in tqdm(optimal_symbols.keys(), desc='Analyzing'):

            high_series = self.data['high'][symbol].iloc[-last_days:]
            low_series = self.data['low'][symbol].iloc[-last_days:]
            close_series = self.data['close'][symbol].iloc[-last_days:]
            volume_series = self.data['volume'][symbol].iloc[-last_days:]

            interes = interes_metric(volume_series)

            last_rsi = ta.momentum.RSIIndicator(close=close_series, window=30).rsi().iloc[-1]

            last_price = close_series.iloc[-1]
            atr_series = ta.volatility.AverageTrueRange(high=high_series, low=low_series, close=close_series, window=30).average_true_range()
            last_atr = atr_series.iloc[-1]
            SL = last_price - last_atr * 3
            TP = last_price * 1.2

            votes_up, votes_down, votes_neutral = self._candle_votes(symbol, last_days)
            total_votes = int(abs(votes_up - votes_down) - votes_neutral + (interes * 100))

            ma100 = self.data['close'][symbol].rolling(window=100).mean().iloc[-last_days:]
            ma30 = self.data['close'][symbol].rolling(window=30).mean().iloc[-last_days:]
            max_resist_100 = self.data['close'][symbol].iloc[-101:-1].max()
            min_support_100 = self.data['close'][symbol].iloc[-101:-1].min()

            direction = self._determine_direction(votes_up, votes_down, votes_neutral, total_votes)
            signal_text = self._generate_signal(votes_up, votes_down, total_votes, ma30, ma100, last_rsi, 
                                                last_price, max_resist_100, min_support_100, volume_series)

            cap = self.data['capital'][self.data['capital']['symbol'] == symbol]['cap'].values[0] / 1_000_000_000

            min_support_100 = close_series.iloc[-101:-1].min()
            max_resist_100 = close_series.iloc[-101:-1].max()
            min_support_30 = close_series.iloc[-31:-1].min()
            max_resist_30 = close_series.iloc[-31:-1].max()
            min_historical = close_series.iloc[:-31].min()
            max_historical = close_series.iloc[:-31].max()
            last_price = close_series.iloc[-1]
            mean_100 = close_series.iloc[-101:-1].mean()
            mean_30 = close_series.iloc[-31:-1].mean()

            analised_symbols[symbol] = {
                'date': self.TODAY,
                'cap': float(cap),
                'SL': float(SL),
                'TP': float(TP),
                'direction': direction,
                'votes_up': votes_up,
                'votes_down': votes_down,
                'total_votes': total_votes,
                'signal_text': signal_text,
                'last_rsi': last_rsi,
                'last_atr': last_atr,
                'interes': interes,
                'last_price': last_price,
                'min_support_100': min_support_100,
                'max_resist_100': max_resist_100,
                'min_support_30': min_support_30,
                'max_resist_30': max_resist_30,
                'min_historical': min_historical,
                'max_historical': max_historical,
                'mean_100': mean_100,
                'mean_30': mean_30,
                'max_procent': optimal_symbols[symbol].get('max_procent', np.nan),
                'min_last_days': optimal_symbols[symbol].get('min_last_days', np.nan),
                'max_std_procent': optimal_symbols[symbol].get('max_std_procent', np.nan)
            }

        return analised_symbols

    def _determine_direction(self, votes_up, votes_down, votes_neutral, total_votes):
        scores = f"{total_votes} (U:{votes_up:.1f} D:{votes_down:.1f} N:{votes_neutral:.1f})"
        if votes_up > votes_down and total_votes > 0:
            return f"⬆️{scores}"
        elif votes_down > votes_up and total_votes > 0:
            return f"⬇️{scores}"
        else:
            return f"➡️{scores}"

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

    def _candle_votes(self, symbol, last_days, ax=None):
        # ==== CANDLE PATTERNS ====
        open_series = self.data['open'][symbol].iloc[-last_days:]
        high_series = self.data['high'][symbol].iloc[-last_days:]
        low_series = self.data['low'][symbol].iloc[-last_days:]
        close_series = self.data['close'][symbol].iloc[-last_days:]
        volume_series = self.data['volume'][symbol].iloc[-last_days:]

        # Візьмемо останні 100 днів для голосування патернів
        last_100_dates = close_series.index[-100:]
        vol_mean_100 = self.data['volume'][symbol].iloc[-100:].mean()

        # Голосування за напрямок: рахунок "вгору" і "вниз"
        votes_up = 0.0
        votes_down = 0.0
        votes_neutral = 0.0

        engulfing, hammer, star = [], [], []

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
            vol = volume_series.loc[date]
            vol_weight = vol / vol_mean_100 if vol_mean_100 > 0 else 1

            # hammer (позитивний сигнал)
            if candle_range > 0 and body < candle_range * 0.3 and lower_shadow > body * 2:
                votes_up += 1 * vol_weight
                hammer.append(l)
                if ax:
                    ax.scatter(date, l, color='green', s=20, marker='P', label='Hammer' if 'Hammer' not in ax.get_legend_handles_labels()[1] else "")

            # inverted hammer (негативний сигнал)
            if candle_range > 0 and body < candle_range * 0.3 and upper_shadow > body * 2 and lower_shadow < body * 0.5:
                votes_down += 1 * vol_weight
                if ax:
                    ax.scatter(date, h, color='red', s=20, marker='P', label='Inverted Hammer' if 'Inverted Hammer' not in ax.get_legend_handles_labels()[1] else "")

            # doji — сумнівний сигнал
            if candle_range > 0 and body < candle_range * 0.1:
                votes_neutral += 1 * vol_weight
                if ax:
                    ax.scatter(date, c, color='orange', s=30, marker='D', label='Doji' if 'Doji' not in ax.get_legend_handles_labels()[1] else "")

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

            vol = volume_series.loc[date]
            vol_weight = vol / volume_series[-100:].mean() if volume_series[-100:].mean() > 0 else 1

            # Morning Star — сигнал на підйом
            if (c1 < o1 and body1 > (h1 - l1) * 0.5 and
                body2 < body1 * 0.3 and
                c3 > o3 and body3 > (h3 - l3) * 0.5 and
                c3 > ((c1 + o1)/2)):
                votes_up += 3 * vol_weight
                star.append(c3)
                if ax:
                    ax.scatter(date, c3, color='green', s=80, marker='*', label='Morning Star' if 'Morning Star' not in ax.get_legend_handles_labels()[1] else "")

            # Evening Star — сигнал на падіння
            if (c1 > o1 and body1 > (h1 - l1) * 0.5 and
                body2 < body1 * 0.3 and
                c3 < o3 and body3 > (h3 - l3) * 0.5 and
                c3 < ((c1 + o1)/2)):
                votes_down += 3 * vol_weight
                if ax:
                    ax.scatter(date, c3, color='red', s=80, marker='*', label='Evening Star' if 'Evening Star' not in ax.get_legend_handles_labels()[1] else "")

        # Дво-свічкові патерни (Bullish Engulfing, Bearish Engulfing)
        for i in range(1, len(last_100_dates)):
            date = last_100_dates[i]
            date_prev = last_100_dates[i-1]

            h = high_series.loc[date]
            l = low_series.loc[date]

            o_prev, c_prev = open_series.loc[date_prev], close_series.loc[date_prev]
            o_curr, c_curr = open_series.loc[date],     close_series.loc[date]

            vol = volume_series.loc[date]
            vol_weight = vol / volume_series[-100:].mean() if volume_series[-100:].mean() > 0 else 1

            # Bullish Engulfing — вгору
            if (c_prev < o_prev and
                c_curr > o_curr and
                o_curr < c_prev and
                c_curr > o_prev):
                votes_up += 2 * vol_weight
                engulfing.append(l)
                if ax:
                    ax.scatter(date, l, color='green', s=30, marker='^', label='Engulfing Bullish' if 'Engulfing Bullish' not in ax.get_legend_handles_labels()[1] else "")

            # Bearish Engulfing — вниз
            if (c_prev > o_prev and
                c_curr < o_curr and
                o_curr > c_prev and
                c_curr < o_prev):
                votes_down += 2 * vol_weight
                if ax:
                    ax.scatter(date, h, color='red', s=30, marker='v', label='Engulfing Bearish' if 'Engulfing Bearish' not in ax.get_legend_handles_labels()[1] else "")

        self.bullish_pattern =  hammer[-3:] + engulfing[-2:] + star[-1:]

        return votes_up, votes_down, votes_neutral
    

        open_series.index = pd.to_datetime(open_series.index)
        df = pd.DataFrame({
            "ds": open_series.index,
            "y": open_series.values
        })

        m = Prophet()
        m.fit(df)

        cps = m.changepoints
        deltas = m.params['delta'].mean(0)

        if len(cps) == 0:
            return []

        cp_df = pd.DataFrame({
            "date": cps,
            "strength": deltas
        })

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

    def graph(self, last_days=180, save_pdf=True):
        assert self.result_df is not None, "Спочатку викличте run()"
        last_days = max(last_days, 100)

        num_symbols = len(self.result_df)
        fig, axes = plt.subplots(nrows=num_symbols, ncols=1, figsize=(16, 8 * num_symbols), sharex=False)
        if num_symbols == 1:
            axes = [axes]

        # ==== Глобальна лінія ====
        global_norm = pd.DataFrame()
        for symbol in self.result_df['symbol']:
            open_series = self.data['close'][symbol].iloc[-last_days:]
            norm_series = (open_series - open_series.min()) / (open_series.max() - open_series.min())
            global_norm[symbol] = norm_series
        global_line = global_norm.mean(axis=1)

        # ==== ПОЧАТОК ЦИКЛУ ПО СИМВОЛАМ ====
        for idx, symbol in enumerate(self.result_df['symbol']):
            open_series = self.data['close'][symbol].iloc[-last_days:]
            high_series = self.data['high'][symbol].iloc[-last_days:]
            low_series = self.data['low'][symbol].iloc[-last_days:]
            close_series = self.data['close'][symbol].iloc[-last_days:]
            volume_series = self.data['volume'][symbol].iloc[-last_days:]

            if len(open_series) < last_days:
                continue

            ax = axes[idx]

            # ==== ADX ====
            adx_val, plus_di_val, minus_di_val = adx_metric(high_series, low_series, close_series)

            # ==== Графік інтересу ====
            cum_vol = volume_series.cumsum()
            interes = interes_metric(volume_series)

            ax_vol = ax.twinx()
            cum_vol.plot(ax=ax_vol, color='gray', linestyle='-', linewidth=0.5, label=f'Cumulative Volume ({interes:.2f})')
            ax_vol.set_ylabel('Cumulative Volume', color='gray')
            ax_vol.tick_params(axis='y', colors='gray')
            ax_vol.legend(loc='upper right', fontsize=8)

            # ==== Графік ціни ====
            open_series.plot(ax=ax, label='Price', color='gray', linewidth=1.5)

            # ==== Глобальна лінія ====
            global_scaled = global_line * (open_series.max() - open_series.min()) + open_series.min()
            global_scaled.plot(ax=ax, label='Global Mean', color='black', linestyle='--', linewidth=0.5)

            # ==== 200-денна ковзна ====
            ma200_full = self.data['close'][symbol].rolling(window=200).mean()
            ma200 = ma200_full.iloc[-last_days:]
            ma200.plot(ax=ax, color='purple', linestyle='-', linewidth=1.2, label='MA 200')

            # ==== 100-денна ковзна ====
            ma100_full = self.data['close'][symbol].rolling(window=100).mean()
            ma100 = ma100_full.iloc[-last_days:]
            ma100.plot(ax=ax, color='blue', linestyle='-', linewidth=1.2, label='MA 100')

            ma100_global = global_scaled.rolling(window=100).mean()
            ma100_global.plot(ax=ax, color='blue', linestyle='--', linewidth=0.5, label='Global MA 100')

            # ==== 30-денна ковзна ====
            ma30_full = self.data['close'][symbol].rolling(window=30).mean()
            ma30 = ma30_full.iloc[-last_days:]
            ma30.plot(ax=ax, color='orange', linestyle='-', linewidth=1.2, label='MA 30')

            # ==== Лінії підтримки та опору ====
            symbol_cap = self.result_df.loc[self.result_df['symbol'] == symbol, 'cap'].values[0]
            min_support_100 = self.result_df.loc[self.result_df['symbol'] == symbol, 'min_support_100'].values[0]
            max_resist_100 = self.result_df.loc[self.result_df['symbol'] == symbol, 'max_resist_100'].values[0]
            min_support_30 = self.result_df.loc[self.result_df['symbol'] == symbol, 'min_support_30'].values[0]
            max_resist_30 = self.result_df.loc[self.result_df['symbol'] == symbol, 'max_resist_30'].values[0]
            last_price = self.result_df.loc[self.result_df['symbol'] == symbol, 'last_price'].values[0]
            max_historical = self.result_df.loc[self.result_df['symbol'] == symbol, 'max_historical'].values[0]
            SL = self.result_df.loc[self.result_df['symbol'] == symbol, 'SL'].values[0]
            TP = self.result_df.loc[self.result_df['symbol'] == symbol, 'TP'].values[0]

            # ==== Фарбування ділянок підтримки та опору ====
            ax.axhspan(min_support_100, max_resist_100, color='lightgreen', alpha=0.1)
            ax.axhspan(max_resist_30, max_resist_100, color='red', alpha=0.1)

            ax.axhline(last_price, color='green', linestyle='--', label=f"Last Price ({last_price:.2f})")
            ax.text(open_series.index[-1], last_price, f' {last_price:.2f}', verticalalignment='bottom', color='green', fontsize=10)
            
            ax.axhline(min_support_100, color='gray', linestyle='dotted', label=f"Min Support 100 ({min_support_100:.2f})")
            ax.axhline(min_support_30, color='orange', linestyle='--', label=f"Min Support 30 ({min_support_30:.2f})")

            ax.axhline(max_resist_100, color='gray', linestyle='dotted', label=f"Max Resist 100 ({max_resist_100:.2f})")
            ax.axhline(max_resist_30, color='gray', linestyle='dotted', label=f"Max Resist 30 ({max_resist_30:.2f})")

            ax.axhline(self.result_df.loc[self.result_df['symbol'] == symbol, 'min_historical'].values[0], color='red', linestyle='--', label=f"Min Historical ({self.result_df.loc[self.result_df['symbol'] == symbol, 'min_historical'].values[0]:.2f})")
            ax.axhline(max_historical, color='red', linestyle='--', label=f"Max Historical ({max_historical:.2f})")
            
            if last_price > max_historical:
                ax.axhspan(max_historical, last_price, color='yellow', alpha=0.5, label='Above Max Historical')

            # ==== Найбільший обʼєм за останні 100 днів ====
            vol_max_idx_100 = volume_series[-100:].idxmax()
            vol_mean_100 = volume_series[-100:].mean()
            if volume_series[vol_max_idx_100] > vol_mean_100 * 2:
                price_at_vol_max_100 = open_series.loc[vol_max_idx_100]
                pos_100 = open_series.index.get_loc(vol_max_idx_100)

                ax.text(pos_100, price_at_vol_max_100, f' {price_at_vol_max_100:.2f}', 
                        verticalalignment='bottom', color='red', fontsize=10)

            # ==== Найбільший обʼєм за останні 30 днів ====
            vol_max_idx_30 = volume_series[-30:].idxmax()
            vol_mean_30 = volume_series[-30:].mean()
            if volume_series[vol_max_idx_30] > vol_mean_30 * 2:
                price_at_vol_max_30 = open_series.loc[vol_max_idx_30]
                pos_30 = open_series.index.get_loc(vol_max_idx_30)

                ax.text(pos_30, price_at_vol_max_30, f' {price_at_vol_max_30:.2f}', 
                        verticalalignment='bottom', color='red', fontsize=10)

            # ==== Лінії часу ====
            if len(open_series) >= 100:
                pos_7 = len(open_series) - 7
                pos_30 = len(open_series) - 30
                pos_100 = len(open_series) - 100
                
                ax.axvline(pos_7, color='gray', linestyle=':', label='7 Days Ago')
                ax.axvline(pos_30, color='gray', linestyle=':', label='30 Days Ago')
                ax.axvline(pos_100, color='gray', linestyle=':', label='100 Days Ago')

            # ==== Накладання обʼєму ====
            if last_price > max_resist_100 * 1.1:
                upper_limit = last_price * 1.1
            else:
                upper_limit = max_resist_100 * 1.1

            if last_price < min_support_100 * 0.9:
                lower_limit = last_price * 0.9
            else:
                lower_limit = min_support_100 * 0.9

            ax.set_ylim(lower_limit, upper_limit)
            vol_scaled = volume_series / volume_series.max() * (upper_limit - lower_limit) * 0.2 + lower_limit
            ax.fill_between(vol_scaled.index, vol_scaled, color='gray', alpha=0.3, label='Volume')

            # ==== Визначення та відображення Bullrun підмасиву ====
            start_idx, end_idx = kadane_subarray(open_series)
            ax.axvline(start_idx, color='blue', linestyle='--', linewidth=1.5, label='Bull Start')
            ax.axvline(end_idx, color='red', linestyle='--', linewidth=1.5, label='Bull End')

            kadane_avg = open_series.loc[start_idx:end_idx].mean()
            kadane_coef = last_price / kadane_avg if end_idx == open_series.index[-1] else 0.0

            # ==== Визначення та відображення топ точок зміни тренду ====
            top3_cps = detect_top_changepoints(open_series, changepoint_n=3)
            for cp in top3_cps:
                ax.axhline(cp, color='orange', linestyle='--', linewidth=1.0)

            # ==== CANDLE PATTERNS ====
            self._candle_votes(symbol, last_days, ax=ax)

            # ==== Заголовок графіка ====
            direction = self.result_df.loc[self.result_df['symbol'] == symbol, 'direction'].values[0]
            signal_text = self.result_df.loc[self.result_df['symbol'] == symbol, 'signal_text'].values[0]
            last_rsi = self.result_df.loc[self.result_df['symbol'] == symbol, 'last_rsi'].values[0]
            last_atr = self.result_df.loc[self.result_df['symbol'] == symbol, 'last_atr'].values[0]

            ax.set_title(
                f"| Cap: {symbol_cap:.2f}B USD | RSI: {last_rsi:.1f} ATR: {last_atr:.2f} Kadane: {kadane_coef:.2f} "
                f"| Trend: {direction} | ADX: {adx_val:.1f} (+DI: {plus_di_val:.1f}, -DI: {minus_di_val:.1f}) |",
                fontsize=12
            )

            # entry price
            vibrated_price = last_price - last_atr
            self.bullish_pattern += [ma30.iloc[-1], ma100.iloc[-1], vibrated_price]
            valid_entry_levels = [p for p in self.bullish_pattern if p < last_price]
            entry_price = max(valid_entry_levels) if valid_entry_levels else -1

            buttom_text = f"{symbol} | {signal_text} | SL: {SL:.2f} TP: {TP:.2f} | Entry: {entry_price:.2f}"
            color_map = {"BUY": "green", "SELL": "red", "HOLD": "gray"}
            signal_color = color_map.get(signal_text, "black")
            ax.set_xlabel(buttom_text, color=signal_color, fontsize=12)

            # ==== Налаштування графіка ====
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.subplots_adjust(hspace=0.75)

        if save_pdf:
            self.save_graph()

        plt.tight_layout()
        plt.show()

    def save_graph(self):
        plt.savefig(f'pdf_store/{self.TODAY}.pdf', dpi=300, bbox_inches='tight')
        try:
            plt.savefig(f'/Users/aleksejkitajskij/Library/Mobile Documents/com~apple~CloudDocs/bit_research/{self.TODAY}.pdf', dpi=300, bbox_inches='tight')
        except Exception as e:
            print("Error saving PDF to iCloud")
