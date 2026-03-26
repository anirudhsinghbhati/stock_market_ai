from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, List, Literal, TypedDict, cast

import numpy as np
import pandas as pd


class BacktestResult(TypedDict):
    engine: str
    sharpe_ratio: float
    max_drawdown: float
    profit_pct: float
    equity_curve: List[Dict[str, Any]]
    trade_log: List[Dict[str, Any]]


class Backtester:
    """Simple class-based backtesting engine.

    Accepts a DataFrame containing market data and model predictions, then
    simulates directional trades and computes portfolio performance metrics.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float,
        prediction_column: str = "prediction",
        price_column: str = "Close",
        probability_column: str = "probability",
        target_column: str = "target",
        stop_loss_column: str = "stop_loss",
        atr_column: str = "ATR",
        capital_per_trade_pct: float = 0.10,
        brokerage_per_trade: float = 20.0,
        slippage_pct: float = 0.001,
    ) -> None:
        if data is None or data.empty:
            raise ValueError("Input DataFrame is empty.")
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0.")
        if price_column not in data.columns:
            raise ValueError(f"DataFrame must contain '{price_column}' column.")
        if prediction_column not in data.columns:
            raise ValueError(f"DataFrame must contain '{prediction_column}' column.")
        if capital_per_trade_pct <= 0 or capital_per_trade_pct > 1:
            raise ValueError("capital_per_trade_pct must be in (0, 1].")
        if brokerage_per_trade < 0:
            raise ValueError("brokerage_per_trade must be >= 0.")
        if slippage_pct < 0:
            raise ValueError("slippage_pct must be >= 0.")

        self.data = data.copy().reset_index(drop=True)
        self.initial_capital = float(initial_capital)
        self.balance = float(initial_capital)
        self.prediction_column = prediction_column
        self.price_column = price_column
        self.probability_column = probability_column
        self.target_column = target_column
        self.stop_loss_column = stop_loss_column
        self.atr_column = atr_column
        self.capital_per_trade_pct = float(capital_per_trade_pct)
        self.brokerage_per_trade = float(brokerage_per_trade)
        self.slippage_pct = float(slippage_pct)

        self.current_position: str | None = None  # None or BUY
        self.quantity: float = 0.0
        self.entry_price: float | None = None
        self.entry_date: Any | None = None
        self.entry_index: int | None = None
        self.current_target: float | None = None
        self.current_stop_loss: float | None = None

        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.metrics: Dict[str, float] = {}

    def execute_trade(self, row: pd.Series, index: int) -> None:
        """Execute entry/exit logic for one row of data and update account state."""
        price_val = pd.to_numeric(pd.Series([row[self.price_column]]), errors="coerce").iloc[0]
        if pd.isna(price_val):
            return
        price = float(price_val)
        if price <= 0:
            return

        high_val = pd.to_numeric(pd.Series([row.get("High", price)]), errors="coerce").iloc[0]
        low_val = pd.to_numeric(pd.Series([row.get("Low", price)]), errors="coerce").iloc[0]
        high = float(high_val) if pd.notna(high_val) else price
        low = float(low_val) if pd.notna(low_val) else price

        pred_raw = str(row[self.prediction_column]).strip().upper()
        signal_up = pred_raw in {"UP", "BUY", "LONG", "1", "TRUE"}

        row_target: float | None = None
        row_stop_loss: float | None = None

        atr_source = self.atr_column if self.atr_column in row.index else "atr"
        atr_val = pd.to_numeric(pd.Series([row.get(atr_source)]), errors="coerce").iloc[0]
        atr = float(atr_val) if pd.notna(atr_val) else None

        # ATR-based levels:
        # stop_loss = entry_price - (1.5 * ATR)
        # target = entry_price + (2 * ATR)
        if atr is not None and atr > 0:
            row_stop_loss = price - (1.5 * atr)
            row_target = price + (2.0 * atr)
        else:
            if self.target_column in self.data.columns:
                value = pd.to_numeric(pd.Series([row[self.target_column]]), errors="coerce").iloc[0]
                row_target = float(value) if pd.notna(value) else None
            if self.stop_loss_column in self.data.columns:
                value = pd.to_numeric(pd.Series([row[self.stop_loss_column]]), errors="coerce").iloc[0]
                row_stop_loss = float(value) if pd.notna(value) else None

        row_date = row.get("Date", index)

        # No open position: check entry condition.
        if self.current_position is None:
            if not signal_up:
                return
            capital_per_trade = self.balance * self.capital_per_trade_pct
            quantity = capital_per_trade / price if capital_per_trade > 0 else 0.0
            if quantity <= 0:
                return
            self.current_position = "BUY"
            self.quantity = float(quantity)
            self.entry_price = price
            self.entry_date = row_date
            self.entry_index = int(index)
            self.current_target = row_target
            self.current_stop_loss = row_stop_loss
            return

        # Position open: check exit conditions.
        if self.entry_price is None:
            return

        target_hit = self.current_target is not None and high >= float(self.current_target)
        stop_hit = self.current_stop_loss is not None and low <= float(self.current_stop_loss)
        prediction_flip = not signal_up

        exit_price: float | None = None
        exit_reason: str | None = None
        if stop_hit:
            exit_price = float(self.current_stop_loss) if self.current_stop_loss is not None else price
            exit_reason = "stop_loss_hit"
        elif target_hit:
            exit_price = float(self.current_target) if self.current_target is not None else price
            exit_reason = "target_hit"
        elif prediction_flip:
            exit_price = price
            exit_reason = "prediction_flip"

        if exit_price is None:
            return

        gross_profit = (float(exit_price) - float(self.entry_price)) * self.quantity
        slippage_cost = ((float(self.entry_price) + float(exit_price)) * self.quantity) * self.slippage_pct
        total_cost = self.brokerage_per_trade + slippage_cost
        profit = gross_profit - total_cost
        self.balance += float(profit)
        self.trades.append(
            {
                "entry_date": self.entry_date,
                "exit_date": row_date,
                "entry_price": float(self.entry_price),
                "exit_price": float(exit_price),
                "gross_profit": float(gross_profit),
                "brokerage": float(self.brokerage_per_trade),
                "slippage_cost": float(slippage_cost),
                "trading_cost": float(total_cost),
                "profit": float(profit),
                "pnl": float(profit),
                "quantity": float(self.quantity),
                "exit_reason": exit_reason,
            }
        )

        self.current_position = None
        self.quantity = 0.0
        self.entry_price = None
        self.entry_date = None
        self.entry_index = None
        self.current_target = None
        self.current_stop_loss = None

    def run_backtest(self) -> Dict[str, Any]:
        """Run backtest over input DataFrame and return trades/equity/metrics."""
        df = self.data
        prices = pd.to_numeric(df[self.price_column], errors="coerce").ffill()

        for idx in range(len(df)):
            row = df.iloc[idx]
            self.execute_trade(row, idx)

            # Track current capital after each day/step.
            self.equity_curve.append(float(self.balance))

        # Close any open position at last price.
        if self.current_position is not None and not df.empty:
            last_price = float(prices.iloc[-1])
            final_row = df.iloc[-1].copy()
            final_row[self.price_column] = last_price
            final_row[self.prediction_column] = "DOWN"
            self.execute_trade(final_row, int(len(df) - 1))
            self.equity_curve.append(float(self.balance))

        self.calculate_metrics()
        return {
            "trade_history": self.trades,
            "final_capital": float(self.balance),
            "trades": self.trades,
            "balance": float(self.balance),
            "equity_curve": self.equity_curve,
            "metrics": self.metrics,
        }

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics from equity curve/trades."""
        final_capital = float(self.balance)
        total_profit = final_capital - self.initial_capital
        total_trades = len(self.trades)
        profitable_trades = sum(1 for trade in self.trades if float(trade.get("profit", trade.get("pnl", 0.0))) > 0)
        win_rate = (profitable_trades / total_trades) if total_trades > 0 else 0.0

        if not self.equity_curve:
            self.metrics = {
                "initial_capital": float(self.initial_capital),
                "final_capital": final_capital,
                "total_profit": float(total_profit),
                "total_return_pct": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": float(win_rate),
                "profitable_trades": float(profitable_trades),
                "total_trades": float(total_trades),
            }
            return self.metrics

        equity = pd.Series(self.equity_curve, dtype=float)
        returns = equity.pct_change().dropna()
        if returns.empty or returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = float(returns.mean() / returns.std())

        running_max = equity.cummax()
        drawdown = (running_max - equity) / running_max.replace(0, np.nan)
        max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0

        total_return_pct = ((final_capital / self.initial_capital) - 1.0) * 100.0
        self.metrics = {
            "initial_capital": float(self.initial_capital),
            "final_capital": final_capital,
            "total_profit": float(total_profit),
            "total_return_pct": float(total_return_pct),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe),
            "win_rate": float(win_rate),
            "profitable_trades": float(profitable_trades),
            "total_trades": float(total_trades),
        }
        return self.metrics


def _calc_metrics(equity_curve: pd.Series) -> BacktestResult:
    """Compute Sharpe ratio, max drawdown, and total profit percentage."""
    if equity_curve.empty:
        return {
            "engine": "none",
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_pct": 0.0,
            "equity_curve": [],
            "trade_log": [],
        }

    returns = equity_curve.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float((returns.mean() / returns.std()) * sqrt(252.0))

    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    start_value = float(equity_curve.iloc[0])
    end_value = float(equity_curve.iloc[-1])
    profit_pct = ((end_value / start_value) - 1.0) * 100.0 if start_value > 0 else 0.0

    return {
        "engine": "custom",
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "profit_pct": float(profit_pct),
        "equity_curve": [],
        "trade_log": [],
    }


def _prepare_signals(
    probabilities: pd.Series,
    buy_threshold: float,
    sell_threshold: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create leakage-safe long/short entry and exit signals from model probabilities."""
    probs = probabilities.astype(float).clip(0.0, 1.0)

    long_entries = (probs > buy_threshold).shift(1, fill_value=False).astype(bool)
    long_exits = (probs < sell_threshold).shift(1, fill_value=False).astype(bool)
    short_entries = (probs < sell_threshold).shift(1, fill_value=False).astype(bool)
    short_exits = (probs > buy_threshold).shift(1, fill_value=False).astype(bool)
    return long_entries, long_exits, short_entries, short_exits


def _run_vectorbt_backtest(
    prices: pd.Series,
    probabilities: pd.Series,
    buy_threshold: float,
    sell_threshold: float,
    sl_stop: float,
    tp_stop: float,
) -> BacktestResult:
    """Run a vectorbt portfolio simulation and return risk/performance metrics."""
    import vectorbt as vbt

    long_entries, long_exits, short_entries, short_exits = _prepare_signals(
        probabilities,
        buy_threshold,
        sell_threshold,
    )

    portfolio = vbt.Portfolio.from_signals(
        close=prices,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        fees=0.0005,
        freq="1D",
    )

    values = portfolio.value()
    metrics = _calc_metrics(values)
    metrics["engine"] = "vectorbt"

    equity_df = values.reset_index(drop=True).rename("equity").to_frame()
    equity_df.insert(0, "step", range(len(equity_df)))
    metrics["equity_curve"] = cast(List[Dict[str, Any]], equity_df.to_dict(orient="records"))

    try:
        trades_obj = getattr(portfolio, "trades", None)
        records_readable = getattr(trades_obj, "records_readable", None)
        if records_readable is None:
            metrics["trade_log"] = []
            return metrics
        trades_df = records_readable.copy()
        rename_map = {
            "Entry Timestamp": "entry_time",
            "Exit Timestamp": "exit_time",
            "Entry Price": "entry_price",
            "Exit Price": "exit_price",
            "PnL": "pnl",
            "Return": "return",
            "Direction": "direction",
            "Size": "size",
            "Status": "status",
        }
        for src, dst in rename_map.items():
            if src in trades_df.columns:
                trades_df = trades_df.rename(columns={src: dst})

        keep_cols = [
            column
            for column in [
                "entry_time",
                "exit_time",
                "direction",
                "size",
                "entry_price",
                "exit_price",
                "pnl",
                "return",
                "status",
            ]
            if column in trades_df.columns
        ]
        if keep_cols:
            trades_df = trades_df[keep_cols]
        metrics["trade_log"] = cast(List[Dict[str, Any]], trades_df.to_dict(orient="records"))
    except Exception:
        metrics["trade_log"] = []

    return metrics


@dataclass
class _BTContext:
    prices: List[float]
    probs: List[float]
    buy_threshold: float
    sell_threshold: float
    sl_stop: float
    tp_stop: float


def _run_backtrader_backtest(
    prices: pd.Series,
    probabilities: pd.Series,
    buy_threshold: float,
    sell_threshold: float,
    sl_stop: float,
    tp_stop: float,
) -> BacktestResult:
    """Run a simplified backtrader simulation and return key metrics."""
    import backtrader as bt

    data = pd.DataFrame(
        {
            "datetime": pd.to_datetime(prices.index),
            "open": prices.values,
            "high": prices.values,
            "low": prices.values,
            "close": prices.values,
            "volume": np.ones(len(prices)),
            "openinterest": np.zeros(len(prices)),
            "prob": probabilities.values,
        }
    ).set_index("datetime")

    class ProbData(bt.feeds.PandasData):
        lines = ("prob",)
        params = (("prob", -1),)

    class ProbStrategy(bt.Strategy):
        def __init__(self) -> None:
            self.entry_price: float | None = None
            self.direction: int = 0
            self.values: List[float] = []

        def next(self) -> None:
            price = float(self.data.close[0])
            prob = float(self.data.prob[0])

            if self.direction != 0 and self.entry_price is not None:
                if self.direction > 0:
                    hit_sl = price <= self.entry_price * (1.0 - sl_stop)
                    hit_tp = price >= self.entry_price * (1.0 + tp_stop)
                else:
                    hit_sl = price >= self.entry_price * (1.0 + sl_stop)
                    hit_tp = price <= self.entry_price * (1.0 - tp_stop)

                if hit_sl or hit_tp:
                    self.close()
                    self.entry_price = None
                    self.direction = 0

            if self.direction == 0:
                if prob > buy_threshold:
                    self.buy()
                    self.entry_price = price
                    self.direction = 1
                elif prob < sell_threshold:
                    self.sell()
                    self.entry_price = price
                    self.direction = -1
            elif self.direction > 0 and prob < sell_threshold:
                self.close()
                self.entry_price = None
                self.direction = 0
            elif self.direction < 0 and prob > buy_threshold:
                self.close()
                self.entry_price = None
                self.direction = 0

            self.values.append(float(self.broker.getvalue()))

    cerebro = bt.Cerebro()
    cerebro.addstrategy(ProbStrategy)
    data_feed = ProbData(dataname=data)  # type: ignore[call-arg]
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0005)
    strategies = cerebro.run()

    if not strategies:
        return {
            "engine": "backtrader",
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_pct": 0.0,
            "equity_curve": [],
            "trade_log": [],
        }

    values = pd.Series(strategies[0].values)
    metrics = _calc_metrics(values)
    metrics["engine"] = "backtrader"
    equity_df = values.reset_index(drop=True).rename("equity").to_frame()
    equity_df.insert(0, "step", range(len(equity_df)))
    metrics["equity_curve"] = cast(List[Dict[str, Any]], equity_df.to_dict(orient="records"))
    metrics["trade_log"] = []
    return metrics


def run_backtest(
    data: pd.DataFrame,
    probability_series: pd.Series,
    engine: Literal["vectorbt", "backtrader"] = "vectorbt",
    buy_threshold: float = 0.7,
    sell_threshold: float = 0.3,
    sl_stop: float = 0.02,
    tp_stop: float = 0.04,
) -> BacktestResult:
    """Run trading backtest from model probabilities and return key metrics.

    Metrics:
    - Sharpe Ratio
    - Max Drawdown
    - Profit %
    """
    if data is None or data.empty:
        raise ValueError("Backtest data is empty.")
    if "Close" not in data.columns:
        raise ValueError("Backtest data must include 'Close' column.")

    prices = pd.Series(data["Close"].astype(float).to_numpy(), index=pd.RangeIndex(len(data)))
    probs = pd.Series(probability_series.astype(float).to_numpy(), index=prices.index)

    if engine == "vectorbt":
        return _run_vectorbt_backtest(prices, probs, buy_threshold, sell_threshold, sl_stop, tp_stop)
    return _run_backtrader_backtest(prices, probs, buy_threshold, sell_threshold, sl_stop, tp_stop)
