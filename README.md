# Nifty VWAP Strategy (5-minute)

Buy Nifty when price is **above VWAP**. Sell when price is **below VWAP**.

| Setting | Value |
|---------|-------|
| Index | Nifty 50 (`^NSEI`) |
| Timeframe | 5 minutes |
| Session | NSE 09:15–15:30 IST |
| Style | Long-only intraday (EOD square-off by default) |

Educational backtester only — not live brokerage execution or financial advice.

## Rules

| Condition | Action |
|-----------|--------|
| Close > VWAP | Buy / stay long |
| Close < VWAP | Sell / stay flat |

```text
VWAP = Σ(TypicalPrice × Volume) / Σ(Volume)
TypicalPrice = (High + Low + Close) / 3
```

- VWAP resets each trading day
- Signals execute on the **next** 5m bar (no look-ahead)
- Positions are squared off at session end by default
- If Yahoo has no index volume, a range-based volume proxy is used

## Setup

```bash
pip install -r requirements.txt
```

## Run

Nifty 5m (default):

```bash
python3 vwap_strategy.py --plot
```

Demo data (offline):

```bash
python3 vwap_strategy.py --demo --plot
```

More history / fixed size:

```bash
python3 vwap_strategy.py --period 10d --cash 200000 --units 1 --plot
```

## Tests

```bash
python3 -m unittest test_vwap_strategy.py -v
```

## Files

- `vwap_strategy.py` — Nifty 5m VWAP strategy + backtester
- `test_vwap_strategy.py` — unit tests
- `requirements.txt` — dependencies
