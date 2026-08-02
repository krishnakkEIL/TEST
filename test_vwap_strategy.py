#!/usr/bin/env python3
"""Unit tests for VWAP strategy helpers."""

import unittest

import pandas as pd

from vwap_strategy import backtest, compute_vwap, generate_signals, make_demo_data


class TestVwapStrategy(unittest.TestCase):
    def test_vwap_is_volume_weighted(self):
        idx = pd.date_range("2026-08-01 09:30", periods=3, freq="min")
        df = pd.DataFrame(
            {
                "Open": [10, 20, 30],
                "High": [10, 20, 30],
                "Low": [10, 20, 30],
                "Close": [10, 20, 30],
                "Volume": [100, 100, 100],
            },
            index=idx,
        )
        vwap = compute_vwap(df)
        # Equal volume + equal weights on bars => average of typical prices so far
        self.assertAlmostEqual(vwap.iloc[0], 10.0)
        self.assertAlmostEqual(vwap.iloc[1], 15.0)
        self.assertAlmostEqual(vwap.iloc[2], 20.0)

    def test_signals_buy_above_sell_below(self):
        idx = pd.date_range("2026-08-01 09:30", periods=4, freq="min")
        close = pd.Series([101, 99, 105, 95], index=idx)
        vwap = pd.Series([100, 100, 100, 100], index=idx)
        df = pd.DataFrame({"Close": close})
        signal = generate_signals(df, vwap)
        self.assertEqual(list(signal), [1, 0, 1, 0])

    def test_backtest_runs_on_demo_data(self):
        df = make_demo_data(bars=120, seed=7)
        result = backtest(df, symbol="DEMO", initial_cash=10_000)
        self.assertGreater(result.final_equity, 0)
        self.assertEqual(len(result.equity_curve), len(df))
        # Position rules: first trade if any should be a BUY
        if result.trades:
            self.assertEqual(result.trades[0].side, "BUY")


if __name__ == "__main__":
    unittest.main()
