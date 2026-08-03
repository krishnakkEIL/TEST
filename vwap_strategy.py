#!/usr/bin/env python3
"""
Nifty VWAP Strategy (5-minute)
------------------------------
Index   : Nifty 50 (^NSEI)
Timeframe: 5 minutes

Buy when Close > VWAP.
Sell when Close < VWAP.

Educational backtester only — not live trading advice.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Yahoo Finance symbol for Nifty 50
NIFTY_SYMBOL = "^NSEI"
DEFAULT_INTERVAL = "5m"
DEFAULT_PERIOD = "5d"

# NSE cash-market session (IST)
NSE_SESSION_START = "09:15"
NSE_SESSION_END = "15:30"
IST = "Asia/Kolkata"


@dataclass
class Trade:
    side: str  # BUY or SELL
    timestamp: pd.Timestamp
    price: float
    units: int
    vwap: float


@dataclass
class BacktestResult:
    symbol: str
    trades: list[Trade]
    equity_curve: pd.Series
    final_equity: float
    total_return_pct: float
    buy_hold_return_pct: float
    num_trades: int
    win_rate_pct: float


def resolve_symbol(symbol: str) -> str:
    """Map friendly names to Yahoo tickers."""
    aliases = {
        "NIFTY": NIFTY_SYMBOL,
        "NIFTY50": NIFTY_SYMBOL,
        "NIFTY 50": NIFTY_SYMBOL,
        "NSEI": NIFTY_SYMBOL,
    }
    key = symbol.strip().upper()
    return aliases.get(key, symbol.strip())


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["High"] + df["Low"] + df["Close"]) / 3.0


def ensure_volume(df: pd.DataFrame) -> pd.Series:
    """
    Nifty cash index often has zero/missing volume on Yahoo Finance.
    Fall back to a range-based volume proxy so session VWAP is defined.
    """
    vol = df["Volume"].astype(float).copy()
    if vol.fillna(0).sum() > 0:
        return vol.replace(0, np.nan).ffill().fillna(1.0)

    # Proxy: larger bar range => more participation
    rng = (df["High"] - df["Low"]).abs()
    proxy = rng.replace(0, np.nan)
    proxy = proxy.fillna(df["Close"] * 0.0001)
    return proxy


def compute_vwap(df: pd.DataFrame, reset: str = "D") -> pd.Series:
    """
    Session VWAP (resets each IST trading day by default).

    VWAP = Σ(TypicalPrice × Volume) / Σ(Volume)
    """
    tp = typical_price(df)
    vol = ensure_volume(df)
    pv = tp * vol

    if reset.lower() == "none":
        return pv.cumsum() / vol.cumsum().replace(0, np.nan)

    day = df.index.tz_convert(IST).normalize() if df.index.tz is not None else df.index.normalize()
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = vol.groupby(day).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


def filter_nse_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only NSE regular session bars in IST."""
    out = df.copy()
    if out.index.tz is None:
        # Yahoo usually returns exchange-local / UTC-aware; assume IST if naive
        out.index = out.index.tz_localize(IST)
    else:
        out.index = out.index.tz_convert(IST)

    times = out.index.time
    start = pd.Timestamp(NSE_SESSION_START).time()
    end = pd.Timestamp(NSE_SESSION_END).time()
    mask = (times >= start) & (times <= end)
    return out.loc[mask]


def generate_signals(df: pd.DataFrame, vwap: pd.Series) -> pd.Series:
    """
    +1 = long (price above VWAP)
     0 = flat (price below VWAP)
    """
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[df["Close"] > vwap] = 1
    return signal


def backtest(
    df: pd.DataFrame,
    symbol: str,
    initial_cash: float = 100_000.0,
    units: Optional[int] = None,
    commission: float = 0.0,
    square_off_eod: bool = True,
) -> BacktestResult:
    """
    Long-only Nifty VWAP strategy on 5-min bars:
      - Enter long when Close is above VWAP
      - Exit long when Close is below VWAP
      - Optionally square off at session end
    """
    data = df.copy()
    data["VWAP"] = compute_vwap(data)
    data["signal"] = generate_signals(data, data["VWAP"])
    # Next-bar execution to avoid look-ahead bias
    data["position"] = data["signal"].shift(1).fillna(0).astype(int)

    if square_off_eod:
        # Force flat on the last bar of each IST session day
        day = pd.Series(data.index.normalize(), index=data.index)
        is_last = day != day.shift(-1)
        data.loc[is_last.fillna(True), "position"] = 0

    cash = float(initial_cash)
    position = 0
    trades: list[Trade] = []
    equity = []

    for ts, row in data.iterrows():
        price = float(row["Close"])
        vwap = float(row["VWAP"]) if pd.notna(row["VWAP"]) else price
        desired = int(row["position"])

        if desired == 1 and position == 0:
            qty = units if units is not None else max(1, int(cash // price))
            cost = qty * price + commission
            if qty > 0 and cost <= cash:
                cash -= cost
                position = qty
                trades.append(Trade("BUY", ts, price, qty, vwap))

        elif desired == 0 and position > 0:
            proceeds = position * price - commission
            cash += proceeds
            trades.append(Trade("SELL", ts, price, position, vwap))
            position = 0

        equity.append(cash + position * price)

    equity_curve = pd.Series(equity, index=data.index, name="equity")

    wins = 0
    round_trips = 0
    entry_price = None
    for t in trades:
        if t.side == "BUY":
            entry_price = t.price
        elif t.side == "SELL" and entry_price is not None:
            round_trips += 1
            if t.price > entry_price:
                wins += 1
            entry_price = None

    final_equity = float(equity_curve.iloc[-1])
    total_return = (final_equity / initial_cash - 1.0) * 100.0
    buy_hold = (float(data["Close"].iloc[-1]) / float(data["Close"].iloc[0]) - 1.0) * 100.0
    win_rate = (wins / round_trips * 100.0) if round_trips else 0.0

    return BacktestResult(
        symbol=symbol,
        trades=trades,
        equity_curve=equity_curve,
        final_equity=final_equity,
        total_return_pct=total_return,
        buy_hold_return_pct=buy_hold,
        num_trades=len(trades),
        win_rate_pct=win_rate,
    )


def fetch_nifty_5m(symbol: str = NIFTY_SYMBOL, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Download Nifty OHLCV on 5-minute bars and keep NSE session only."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=DEFAULT_INTERVAL, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} ({period}, {DEFAULT_INTERVAL})")

    df = df.rename(columns=str.title)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for {symbol}: {sorted(missing)}")

    df = df[list(required)].dropna(subset=["Open", "High", "Low", "Close"])
    df = filter_nse_session(df)
    if df.empty:
        raise ValueError(f"No NSE session bars for {symbol}")
    return df


def make_demo_nifty_5m(days: int = 3, seed: int = 42) -> pd.DataFrame:
    """Synthetic Nifty-like 5-minute bars for NSE sessions (offline demo)."""
    rng = np.random.default_rng(seed)
    # 09:15 to 15:30 IST inclusive on 5m => 76 bars/day
    frames = []
    price = 24_500.0
    start_day = pd.Timestamp("2026-07-29", tz=IST)

    for d in range(days):
        day = start_day + pd.Timedelta(days=d)
        if day.weekday() >= 5:
            continue
        idx = pd.date_range(
            f"{day.date()} {NSE_SESSION_START}",
            f"{day.date()} {NSE_SESSION_END}",
            freq="5min",
            tz=IST,
        )
        rets = rng.normal(0.00002, 0.0009, size=len(idx))
        close = price * np.exp(np.cumsum(rets))
        high = close * (1 + rng.uniform(0.0001, 0.0015, size=len(idx)))
        low = close * (1 - rng.uniform(0.0001, 0.0015, size=len(idx)))
        open_ = np.roll(close, 1)
        open_[0] = price
        volume = rng.integers(50_000, 250_000, size=len(idx))
        frames.append(
            pd.DataFrame(
                {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
                index=idx,
            )
        )
        price = float(close[-1])

    return pd.concat(frames)


def print_report(result: BacktestResult) -> None:
    print(f"\n=== Nifty VWAP Strategy (5m): {result.symbol} ===")
    print(f"Final equity     : ₹{result.final_equity:,.2f}")
    print(f"Strategy return  : {result.total_return_pct:+.2f}%")
    print(f"Buy & hold return: {result.buy_hold_return_pct:+.2f}%")
    print(f"Trades           : {result.num_trades}")
    print(f"Win rate         : {result.win_rate_pct:.1f}%")
    print("\nRecent trades:")
    for t in result.trades[-10:]:
        print(
            f"  {t.timestamp}  {t.side:<4}  "
            f"{t.units} @ {t.price:.2f}  (VWAP {t.vwap:.2f})"
        )


def plot_result(df: pd.DataFrame, result: BacktestResult, out_path: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt

    vwap = compute_vwap(df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(df.index, df["Close"], label="Nifty Close", linewidth=1.2)
    ax.plot(df.index, vwap, label="VWAP", linewidth=1.2, linestyle="--")

    buys = [t for t in result.trades if t.side == "BUY"]
    sells = [t for t in result.trades if t.side == "SELL"]
    if buys:
        ax.scatter([t.timestamp for t in buys], [t.price for t in buys], marker="^", s=60, label="Buy", zorder=5)
    if sells:
        ax.scatter([t.timestamp for t in sells], [t.price for t in sells], marker="v", s=60, label="Sell", zorder=5)

    ax.set_title(f"{result.symbol} 5m — Buy Above VWAP / Sell Below VWAP")
    ax.set_ylabel("Nifty")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    axes[1].plot(result.equity_curve.index, result.equity_curve.values, color="tab:green", linewidth=1.2)
    axes[1].set_ylabel("Equity (₹)")
    axes[1].set_xlabel("Time (IST)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"\nChart saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nifty 5m VWAP buy-above / sell-below backtester")
    p.add_argument("--symbol", default="NIFTY", help="Index symbol (default: NIFTY -> ^NSEI)")
    p.add_argument("--period", default=DEFAULT_PERIOD, help="yfinance period (e.g. 5d, 10d, 60d)")
    p.add_argument("--cash", type=float, default=100_000.0, help="Starting capital in INR")
    p.add_argument("--units", type=int, default=None, help="Fixed units per trade (default: all-in)")
    p.add_argument("--no-eod-squareoff", action="store_true", help="Keep overnight positions")
    p.add_argument("--demo", action="store_true", help="Use synthetic Nifty 5m data")
    p.add_argument("--plot", action="store_true", help="Show / save price + equity chart")
    p.add_argument("--out", default="nifty_vwap_5m.png", help="Chart output path when --plot is set")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbol = resolve_symbol(args.symbol)

    if args.demo:
        display = "NIFTY-DEMO"
        df = make_demo_nifty_5m()
        print("Using synthetic Nifty 5-minute demo data (NSE session)")
    else:
        display = "NIFTY 50" if symbol == NIFTY_SYMBOL else symbol
        print(f"Fetching {display} ({symbol}) — period={args.period}, interval=5m ...")
        try:
            df = fetch_nifty_5m(symbol=symbol, period=args.period)
        except Exception as exc:
            print(f"Data fetch failed ({exc}); falling back to demo data.")
            display = "NIFTY-DEMO"
            df = make_demo_nifty_5m()

    print(f"Bars loaded: {len(df)} | Session: {NSE_SESSION_START}-{NSE_SESSION_END} IST")

    result = backtest(
        df,
        symbol=display,
        initial_cash=args.cash,
        units=args.units,
        square_off_eod=not args.no_eod_squareoff,
    )
    print_report(result)

    if args.plot:
        plot_result(df, result, out_path=args.out)


if __name__ == "__main__":
    main()
