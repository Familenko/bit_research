from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import ta


TODAY = date.today().strftime('%Y-%m-%d')


def paint_result(df, 
                 result, 
                 info_df, 
                 cap_df, 
                 volume_df, 
                 df_open, 
                 df_max, 
                 df_min,
                 last_days=180):
    
    collect_predict = pd.DataFrame()
    
    last_days = max(last_days, 100)

    if 'BTCUSDT' not in result:
        result.append('BTCUSDT')
    num_symbols = len(result)
    fig, axes = plt.subplots(nrows=num_symbols, ncols=1, figsize=(16, 6 * num_symbols), sharex=False)

    if num_symbols == 1:
        axes = [axes]

    global_norm = pd.DataFrame()
    for symbol in result:
        open_series = df[symbol].iloc[-last_days:]
        norm_series = (open_series - open_series.min()) / (open_series.max() - open_series.min())
        global_norm[symbol] = norm_series
    global_line = global_norm.mean(axis=1)

    df_rsi = pd.DataFrame()
    for symbol in result:
        df_rsi[symbol] = ta.momentum.RSIIndicator(close=df[symbol], window=30).rsi()

    for idx, symbol in enumerate(result):
        open_series = df[symbol].iloc[-last_days:]
        volume_series = volume_df[symbol].iloc[-last_days:]
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

        # ==== RSI open_series ====
        rsi_series = df_rsi[symbol].iloc[-last_days:]
        last_rsi = rsi_series.dropna().iloc[-1]

        # ==== Графік ціни ====
        open_series.plot(ax=ax, label='Price', color='gray', linewidth=1.5)

        # ==== Глобальна лінія ====
        global_scaled = global_line * (open_series.max() - open_series.min()) + open_series.min()
        global_scaled.plot(ax=ax, label='Global Mean', color='black', linestyle='--', linewidth=0.5)

        # ==== 200-денна ковзна ====
        ma200_full = df[symbol].rolling(window=200).mean()
        ma200 = ma200_full.iloc[-last_days:]
        ma200.plot(ax=ax, color='purple', linestyle='-', linewidth=1.2, label='MA 200')

        # ==== 100-денна ковзна ====
        ma100_full = df[symbol].rolling(window=100).mean()
        ma100 = ma100_full.iloc[-last_days:]
        ma100.plot(ax=ax, color='blue', linestyle='-', linewidth=1.2, label='MA 100')

        ma100_global = global_scaled.rolling(window=100).mean()
        ma100_global.plot(ax=ax, color='blue', linestyle='--', linewidth=0.5, label='Global MA 100')
        # ==== 30-денна ковзна ====
        ma30_full = df[symbol].rolling(window=30).mean()
        ma30 = ma30_full.iloc[-last_days:]
        ma30.plot(ax=ax, color='orange', linestyle='-', linewidth=1.2, label='MA 30')

        # ==== Лінії підтримки та опору ====
        symbol_cap = cap_df[cap_df['symbol'] == symbol]['cap'].values[0] / 1_000_000_000
        min_support_100 = info_df.loc[symbol, 'Min Support 100']
        max_resist_100 = info_df.loc[symbol, 'Max Resist 100']
        min_support_30 = info_df.loc[symbol, 'Min Support 30']
        max_resist_30 = info_df.loc[symbol, 'Max Resist 30']
        last_price = info_df.loc[symbol, 'Last Price']
        max_historical = info_df.loc[symbol, 'Max Historical']

        # ==== Розрахунок TP та SL ====
        atr_series = ta.volatility.AverageTrueRange(
            high=df_max[symbol],
            low=df_min[symbol],
            close=df[symbol],
            window=30
        ).average_true_range()
        last_atr = atr_series.dropna().iloc[-1]
        SL = last_price - last_atr * 2.5
        TP = last_price + last_atr * 2.5
        profit_pct = ((TP - last_price) / last_price) * 100

        # ==== Фарбування ділянок підтримки та опору ====
        ax.axhspan(min_support_100, max_resist_100, color='lightgreen', alpha=0.1)
        ax.axhspan(max_resist_30, max_resist_100, color='red', alpha=0.1)

        ax.axhline(last_price, color='green', linestyle='--',
                   label=f"Last Price ({last_price:.2f})")
        ax.text(open_series.index[-1], last_price, f' {last_price:.2f}', 
                verticalalignment='bottom', color='green', fontsize=10)
        
        ax.axhline(min_support_100, color='gray', linestyle='dotted',
                   label=f"Min Support 100 ({min_support_100:.2f})")
        ax.axhline(min_support_30, color='orange', linestyle='--',
                   label=f"Min Support 30 ({min_support_30:.2f})")

        ax.axhline(max_resist_100, color='gray', linestyle='dotted',
                   label=f"Max Resist 100 ({max_resist_100:.2f})")
        ax.axhline(max_resist_30, color='gray', linestyle='dotted',
                   label=f"Max Resist 30 ({max_resist_30:.2f})")
        
        ax.axhline(info_df.loc[symbol, 'Min Historical'], color='red', linestyle='--',
                   label=f"Min Historical ({info_df.loc[symbol, 'Min Historical']:.2f})")
        ax.axhline(info_df.loc[symbol, 'Max Historical'], color='red', linestyle='--',
                   label=f"Max Historical ({info_df.loc[symbol, 'Max Historical']:.2f})")
        
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
            pos_30 = len(open_series) - 30
            pos_100 = len(open_series) - 100
            
            ax.axvline(pos_30, color='gray', linestyle=':', label='30 Days Ago')
            ax.axvline(pos_100, color='gray', linestyle=':', label='100 Days Ago')

        # ==== Накладання обʼєму ====
        lower_limit = min_support_100 * 0.8
        upper_limit = max_resist_100 * 1.2
        ax.set_ylim(lower_limit, upper_limit)
        vol_scaled = volume_series / volume_series.max() * (upper_limit - lower_limit) * 0.2 + lower_limit
        ax.fill_between(vol_scaled.index, vol_scaled, color='gray', alpha=0.3, label='Volume (scaled)')

        # ==== CANDLE PATTERNS ====
        open_series = df_open[symbol].iloc[-last_days:]
        high_series = df_max[symbol].iloc[-last_days:]
        low_series = df_min[symbol].iloc[-last_days:]
        close_series = open_series

        # Візьмемо останні 100 днів для голосування патернів
        last_100_dates = open_series.index[-100:]

        # Голосування за напрямок: рахунок "вгору" і "вниз"
        votes_up = 0.0
        votes_down = 0.0
        votes_neutral = 0.0

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
            vol_mean_100 = volume_series[-100:].mean()
            vol_weight = vol / vol_mean_100 if vol_mean_100 > 0 else 1

            # hammer (позитивний сигнал)
            if candle_range > 0 and body < candle_range * 0.3 and lower_shadow > body * 2:
                ax.scatter(date, l, color='green', s=20, marker='P',
                        label='Hammer' if 'Hammer' not in ax.get_legend_handles_labels()[1] else "")
                votes_up += 1 * vol_weight

            # inverted hammer (негативний сигнал)
            if candle_range > 0 and body < candle_range * 0.3 and upper_shadow > body * 2 and lower_shadow < body * 0.5:
                ax.scatter(date, h, color='red', s=20, marker='P',
                        label='Inverted Hammer' if 'Inverted Hammer' not in ax.get_legend_handles_labels()[1] else "")
                votes_down += 1 * vol_weight

            # doji — сумнівний сигнал (не враховуємо в голосуванні)
            if candle_range > 0 and body < candle_range * 0.1:
                ax.scatter(date, c, color='orange', s=30, marker='D',
                           label='Doji' if 'Doji' not in ax.get_legend_handles_labels()[1] else "")
                votes_neutral += 1 * vol_weight

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
                ax.scatter(date, c3, color='green', s=80, marker='*',
                        label='Morning Star' if 'Morning Star' not in ax.get_legend_handles_labels()[1] else "")

            # Evening Star — сигнал на падіння
            if (c1 > o1 and body1 > (h1 - l1) * 0.5 and
                body2 < body1 * 0.3 and
                c3 < o3 and body3 > (h3 - l3) * 0.5 and
                c3 < ((c1 + o1)/2)):
                votes_down += 3 * vol_weight
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

            vol = volume_series.loc[date]
            vol_weight = vol / volume_series[-100:].mean() if volume_series[-100:].mean() > 0 else 1

            # Bullish Engulfing — вгору
            if (c_prev < o_prev and
                c_curr > o_curr and
                o_curr < c_prev and
                c_curr > o_prev):
                votes_up += 2 * vol_weight
                ax.scatter(date, l, color='green', s=30, marker='^',
                        label='Engulfing Bullish' if 'Engulfing Bullish' not in ax.get_legend_handles_labels()[1] else "")

            # Bearish Engulfing — вниз
            if (c_prev > o_prev and
                c_curr < o_curr and
                o_curr > c_prev and
                c_curr < o_prev):
                votes_down += 2 * vol_weight
                ax.scatter(date, h, color='red', s=30, marker='v',
                        label='Engulfing Bearish' if 'Engulfing Bearish' not in ax.get_legend_handles_labels()[1] else "")

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

        # ==== Додаємо напрямок у заголовок та df ====
        ax.set_title(f"{symbol} ({symbol_cap:.2f}B USD) | Profit: {profit_pct:.2f}% SL: {SL:.2f} TP: {TP:.2f} | "
                     f"RSI: {last_rsi:.1f} ATR: {last_atr:.2f} | Trend: {direction} Signal: {signal_text}", fontsize=14)
        
        collect_predict[symbol] = pd.open_series({
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

        # ==== Налаштування графіка ====
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'pdf_store/{TODAY}.pdf', dpi=300, bbox_inches='tight')
    try:
        plt.savefig(f'/Users/aleksejkitajskij/Library/Mobile Documents/com~apple~CloudDocs/bit_research/{TODAY}.pdf', dpi=300, bbox_inches='tight')
    except Exception as e:
        print("Error saving PDF to iCloud")
    plt.show()

    return collect_predict.T.sort_values(['total_votes', 'cap', 'profit_pct'], ascending=[False, False, False])


def paint_all(df, symbols, last_days=365):
    data = df[symbols].iloc[-last_days:].copy()

    for symbol in symbols:
        open_series = data[symbol]
        min_val = open_series.min()
        max_val = open_series.max()
        data[symbol] = (open_series - min_val) / (max_val - min_val)

    plt.figure(figsize=(16,8))
    x = range(len(data))
    for symbol in symbols:
        plt.plot(x, data[symbol], label=symbol, linewidth=1.0)
    
    plt.title(f'Normalized Price Chart for {len(symbols)} Symbols (last {last_days} days)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price (0-1)')
    plt.legend()
    plt.show()
