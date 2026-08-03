#!/usr/bin/env python3
"""Unit tests for Nifty 5m VWAP strategy."""

import unittest

import pandas as pd

from vwap_strategy import (
    backtest,
    compute_vwap,
    filter_nse_session,
    generate_signals,
    make_demo_nifty_5m,
    resolve_symbol,
)


class TestNiftyVwapStrategy(unittest.TestCase):
    def test_resolve_nifty_alias(self):
        self.assertEqual(resolve_symbol("nifty"), "^NSEI")
        self.assertEqual(resolve_symbol("NIFTY50"), "^NSEI")

    def test_vwap_is_volume_weighted(self):
        idx = pd.date_range("2026-08-01 09:15", periods=3, freq="5min", tz="Asia/Kolkata")
        df = pd.DataFrame(
            {
                "Open": [100, 200, 300],
                "High": [100, 200, 300],
                "Low": [100, 200, 300],
                "Close": [100, 200, 300],
                "Volume": [100, 100, 100],
            },
            index=idx,
        )
        vwap = compute_vwap(df)
        self.assertAlmostEqual(vwap.iloc[0], 100.0)
        self.assertAlmostEqual(vwap.iloc[1], 150.0)
        self.assertAlmostEqual(vwap.iloc[2], 200.0)

    def test_signals_buy_above_sell_below(self):
        idx = pd.date_range("2026-08-01 09:15", periods=4, freq="5min", tz="Asia/Kolkata")
        df = pd.DataFrame({"Close": [101, 99, 105, 95]}, index=idx)
        vwap = pd.Series([100, 100, 100, 100], index=idx)
        signal = generate_signals(df, vwap)
        self.assertEqual(list(signal), [1, 0, 1, 0])

    def test_nse_session_filter(self):
        idx = pd.date_range("2026-08-01 09:00", periods=20, freq="5min", tz="Asia/Kolkata")
        df = pd.DataFrame(
            {
                "Open": 1.0,
                "High": 1.0,
                "Low": 1.0,
                "Close": 1.0,
                "Volume": 1.0,
            },
            index=idx,
        )
        filtered = filter_nse_session(df)
        self.assertTrue(all(t.time() >= pd.Timestamp("09:15").time() for t in filtered.index))
        self.assertTrue(all(t.time() <= pd.Timestamp("15:30").time() for t in filtered.index))

    def test_backtest_runs_on_demo_nifty(self):
        df = make_demo_nifty_5m(days=2, seed=7)
        result = backtest(df, symbol="NIFTY-DEMO", initial_cash=100_000)
        self.assertGreater(result.final_equity, 0)
        self.assertEqual(len(result.equity_curve), len(df))
        if result.trades:
            self.assertEqual(result.trades[0].side, "BUY")


if __name__ == "__main__":
    unittest.main()
