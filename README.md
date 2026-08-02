# VWAP Trading Strategy

Buy when price is **above VWAP**. Sell when price is **below VWAP**.

This is a long-only backtester for research and learning — not live brokerage execution or financial advice.

## Strategy

| Condition | Action |
|-----------|--------|
| Close > VWAP | Buy / stay long |
| Close < VWAP | Sell / stay flat |

VWAP is session-based (resets each calendar day):

```text
VWAP = Σ(TypicalPrice × Volume) / Σ(Volume)
TypicalPrice = (High + Low + Close) / 3
```

Signals are executed on the next bar to avoid look-ahead bias.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Demo data (no internet required):

```bash
python3 vwap_strategy.py --demo --plot
```

Live Yahoo Finance data:

```bash
python3 vwap_strategy.py --symbol AAPL --period 5d --interval 5m --plot
```

Useful options:

```bash
python3 vwap_strategy.py --symbol MSFT --period 1d --interval 1m --cash 25000 --shares 10
```

## Tests

```bash
python3 -m unittest test_vwap_strategy.py -v
```

## Files

- `vwap_strategy.py` — VWAP calculation, signals, backtest, CLI
- `test_vwap_strategy.py` — unit tests
- `requirements.txt` — dependencies
