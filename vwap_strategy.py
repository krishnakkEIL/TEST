#!/usr/bin/env python3
"""
VWAP Momentum Strategy
----------------------
Buy when price trades above VWAP.
Sell when price trades below VWAP.

Educational backtester only — not live trading advice.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    side: str  # BUY or SELL
    timestamp: pd.Timestamp
    price: float
    shares: int
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


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["High"] + df["Low"] + df["Close"]) / 3.0


def compute_vwap(df: pd.DataFrame, reset: str = "D") -> pd.Series:
    """
    Volume-Weighted Average Price.

    reset:
      - "D" session VWAP (resets each calendar day)
      - "none" cumulative VWAP over the whole series
    """
    tp = typical_price(df)
    pv = tp * df["Volume"]

    if reset.lower() == "none":
        return pv.cumsum() / df["Volume"].cumsum().replace(0, np.nan)

    # Group by trading day for intraday / multi-day session VWAP
    day = df.index.normalize()
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["Volume"].groupby(day).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


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
    initial_cash: float = 10_000.0,
    shares: Optional[int] = None,
    commission: float = 0.0,
) -> BacktestResult:
    """
    Long-only VWAP strategy:
      - Enter long when Close crosses above VWAP
      - Exit long when Close crosses below VWAP
    """
    data = df.copy()
    data["VWAP"] = compute_vwap(data)
    data["signal"] = generate_signals(data, data["VWAP"])
    data["position"] = data["signal"].shift(1).fillna(0).astype(int)

    cash = float(initial_cash)
    position = 0
    trades: list[Trade] = []
    equity = []

    for ts, row in data.iterrows():
        price = float(row["Close"])
        vwap = float(row["VWAP"]) if pd.notna(row["VWAP"]) else price
        desired = int(row["position"])

        # Enter long
        if desired == 1 and position == 0:
            qty = shares if shares is not None else max(1, int(cash // price))
            cost = qty * price + commission
            if qty > 0 and cost <= cash:
                cash -= cost
                position = qty
                trades.append(Trade("BUY", ts, price, qty, vwap))

        # Exit long
        elif desired == 0 and position > 0:
            proceeds = position * price - commission
            cash += proceeds
            trades.append(Trade("SELL", ts, price, position, vwap))
            position = 0

        equity.append(cash + position * price)

    equity_curve = pd.Series(equity, index=data.index, name="equity")

    # Round-trip win rate
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


def fetch_ohlcv(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} ({period}, {interval})")
    df = df.rename(columns=str.title)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for {symbol}: {sorted(missing)}")
    return df[list(required)].dropna()


def make_demo_data(bars: int = 390, seed: int = 42) -> pd.DataFrame:
    """Synthetic 1-minute session for offline demos."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-08-01 09:30", periods=bars, freq="min")
    rets = rng.normal(0.00005, 0.0012, size=bars)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0.0002, 0.002, size=bars))
    low = close * (1 - rng.uniform(0.0002, 0.002, size=bars))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(1_000, 20_000, size=bars)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def print_report(result: BacktestResult) -> None:
    print(f"\n=== VWAP Strategy Backtest: {result.symbol} ===")
    print(f"Final equity     : ${result.final_equity:,.2f}")
    print(f"Strategy return  : {result.total_return_pct:+.2f}%")
    print(f"Buy & hold return: {result.buy_hold_return_pct:+.2f}%")
    print(f"Trades           : {result.num_trades}")
    print(f"Win rate         : {result.win_rate_pct:.1f}%")
    print("\nRecent trades:")
    for t in result.trades[-10:]:
        print(
            f"  {t.timestamp}  {t.side:<4}  "
            f"{t.shares} @ {t.price:.2f}  (VWAP {t.vwap:.2f})"
        )


def plot_result(df: pd.DataFrame, result: BacktestResult, out_path: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt

    vwap = compute_vwap(df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(df.index, df["Close"], label="Close", linewidth=1.2)
    ax.plot(df.index, vwap, label="VWAP", linewidth=1.2, linestyle="--")

    buys = [t for t in result.trades if t.side == "BUY"]
    sells = [t for t in result.trades if t.side == "SELL"]
    if buys:
        ax.scatter([t.timestamp for t in buys], [t.price for t in buys], marker="^", s=60, label="Buy", zorder=5)
    if sells:
        ax.scatter([t.timestamp for t in sells], [t.price for t in sells], marker="v", s=60, label="Sell", zorder=5)

    ax.set_title(f"{result.symbol} — Buy Above VWAP / Sell Below VWAP")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    axes[1].plot(result.equity_curve.index, result.equity_curve.values, color="tab:green", linewidth=1.2)
    axes[1].set_ylabel("Equity ($)")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"\nChart saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VWAP buy-above / sell-below strategy backtester")
    p.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    p.add_argument("--period", default="5d", help="yfinance period (e.g. 1d, 5d, 1mo)")
    p.add_argument("--interval", default="5m", help="yfinance interval (e.g. 1m, 5m, 15m, 1h)")
    p.add_argument("--cash", type=float, default=10_000.0, help="Starting cash")
    p.add_argument("--shares", type=int, default=None, help="Fixed share size (default: all-in)")
    p.add_argument("--demo", action="store_true", help="Use synthetic data instead of Yahoo Finance")
    p.add_argument("--plot", action="store_true", help="Show / save price + equity chart")
    p.add_argument("--out", default="vwap_backtest.png", help="Chart output path when --plot is set")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo:
        symbol = "DEMO"
        df = make_demo_data()
        print("Using synthetic demo data")
    else:
        symbol = args.symbol.upper()
        print(f"Fetching {symbol} ({args.period}, {args.interval})...")
        try:
            df = fetch_ohlcv(symbol, period=args.period, interval=args.interval)
        except Exception as exc:
            print(f"Data fetch failed ({exc}); falling back to demo data.")
            symbol = "DEMO"
            df = make_demo_data()

    result = backtest(df, symbol=symbol, initial_cash=args.cash, shares=args.shares)
    print_report(result)

    if args.plot:
        plot_result(df, result, out_path=args.out)


if __name__ == "__main__":
    main()
