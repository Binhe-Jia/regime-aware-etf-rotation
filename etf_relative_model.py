from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for the ETF-relative residual trading model."""

    signal_mode: str = "mean_reversion"
    proxy_lookback: int = 60
    residual_z_lookback: int = 60
    rsi_window: int = 14
    market_ma_window: int = 200
    stock_ma_window: int = 200
    atr_window: int = 14
    entry_z: float = -2.0
    exit_z: float = -0.5
    entry_rsi: float = 35.0
    momentum_entry_z: float = 0.50
    momentum_exit_z: float = -0.75
    momentum_entry_rsi: float = 50.0
    base_position_fraction: float = 0.0
    max_hold_days: int = 10
    risk_per_trade: float = 0.01
    stop_atr_multiple: float = 2.0
    min_stop_pct: float = 0.02
    max_position_fraction: float = 1.0
    transaction_cost_bps: float = 2.0
    slippage_bps: float = 3.0
    use_adf_filter: bool = False
    adf_lookback: int = 252
    adf_pvalue_threshold: float = 0.10


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style Relative Strength Index."""

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    value = 100.0 - (100.0 / (1.0 + rs))
    return value.fillna(50.0)


def average_true_range(
    high: pd.Series | None,
    low: pd.Series | None,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average true range; falls back to close-to-close volatility if OHLC is unavailable."""

    if high is None or low is None:
        return close.pct_change().rolling(window).std() * close

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _as_frame(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return data.loc[:, list(columns)].astype(float)


def rolling_proxy_residuals(
    data: pd.DataFrame,
    stock_return_col: str,
    proxy_return_cols: list[str],
    lookback: int,
) -> pd.Series:
    """
    Estimate stock-specific return after removing market and sector ETF effects.

    For each date, the regression is fit only on prior observations, then the current
    stock return is compared with the ETF-proxy prediction.
    """

    returns = _as_frame(data, [stock_return_col, *proxy_return_cols])
    residual = pd.Series(np.nan, index=returns.index, name="residual_return")

    for i in range(lookback, len(returns)):
        train = returns.iloc[i - lookback : i].dropna()
        current = returns.iloc[i]
        if len(train) < max(20, len(proxy_return_cols) + 5) or current.isna().any():
            continue

        x_train = np.column_stack(
            [np.ones(len(train)), train.loc[:, proxy_return_cols].to_numpy()]
        )
        y_train = train.loc[:, stock_return_col].to_numpy()
        beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)

        x_current = np.r_[1.0, current.loc[proxy_return_cols].to_numpy(dtype=float)]
        residual.iloc[i] = current.loc[stock_return_col] - float(x_current @ beta)

    return residual


def rolling_adf_pvalues(series: pd.Series, lookback: int) -> pd.Series:
    """
    Rolling Augmented Dickey-Fuller p-values for residual stationarity checks.

    If statsmodels is unavailable, the caller gets NaNs and can decide whether to
    block trades or run without the filter.
    """

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        adfuller = None

    pvalues = pd.Series(np.nan, index=series.index, name="residual_adf_pvalue")
    min_periods = max(40, lookback // 2)
    for i in range(lookback, len(series)):
        window = series.iloc[i - lookback : i].dropna()
        if len(window) < min_periods or window.nunique() < 5:
            continue
        try:
            if adfuller is not None:
                pvalues.iloc[i] = float(adfuller(window, autolag="AIC")[1])
            else:
                pvalues.iloc[i] = approximate_adf_pvalue(window)
        except (ValueError, np.linalg.LinAlgError):
            continue
    return pvalues


def approximate_adf_pvalue(window: pd.Series) -> float:
    """Small dependency-free Dickey-Fuller approximation used when statsmodels is absent."""

    values = window.to_numpy(dtype=float)
    y_lag = values[:-1]
    delta_y = np.diff(values)
    x = np.column_stack([np.ones(len(y_lag)), y_lag])
    beta, *_ = np.linalg.lstsq(x, delta_y, rcond=None)
    residual = delta_y - x @ beta
    degrees = max(len(delta_y) - x.shape[1], 1)
    sigma2 = float((residual @ residual) / degrees)
    covariance = sigma2 * np.linalg.pinv(x.T @ x)
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    if standard_error <= 0.0:
        return 1.0
    t_stat = float(beta[1] / standard_error)
    if t_stat <= -3.43:
        return 0.01
    if t_stat <= -2.86:
        return 0.05
    if t_stat <= -2.57:
        return 0.10
    return 0.50


class ETFRelativeMeanReversionModel:
    """
    Implements the final PDF framework:

    ETF proxy + residual z-score + RSI + regime filter + volatility sizing +
    transaction-cost-aware backtest.
    """

    def __init__(
        self,
        stock_close: str = "stock_close",
        market_close: str = "market_close",
        sector_close: str = "sector_close",
        stock_high: str | None = "stock_high",
        stock_low: str | None = "stock_low",
        config: StrategyConfig | None = None,
    ) -> None:
        self.stock_close = stock_close
        self.market_close = market_close
        self.sector_close = sector_close
        self.stock_high = stock_high
        self.stock_low = stock_low
        self.config = config or StrategyConfig()

    def build_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        required = [self.stock_close, self.market_close, self.sector_close]
        data = _as_frame(prices, required).copy()

        data["stock_return"] = data[self.stock_close].pct_change()
        data["market_return"] = data[self.market_close].pct_change()
        data["sector_return"] = data[self.sector_close].pct_change()
        data["rsi"] = rsi(data[self.stock_close], self.config.rsi_window)
        data["market_ma"] = (
            data[self.market_close]
            .rolling(self.config.market_ma_window, min_periods=self.config.market_ma_window)
            .mean()
        )
        data["stock_ma"] = (
            data[self.stock_close]
            .rolling(self.config.stock_ma_window, min_periods=self.config.stock_ma_window)
            .mean()
        )
        data["risk_on"] = data[self.market_close] > data["market_ma"]
        data["stock_uptrend"] = data[self.stock_close] > data["stock_ma"]

        data["residual_return"] = rolling_proxy_residuals(
            data,
            "stock_return",
            ["market_return", "sector_return"],
            self.config.proxy_lookback,
        )
        residual_mean = data["residual_return"].rolling(
            self.config.residual_z_lookback, min_periods=20
        ).mean()
        residual_std = data["residual_return"].rolling(
            self.config.residual_z_lookback, min_periods=20
        ).std()
        data["residual_z"] = (data["residual_return"] - residual_mean) / residual_std
        data["residual_adf_pvalue"] = rolling_adf_pvalues(
            data["residual_return"],
            self.config.adf_lookback,
        )
        data["residual_stationary"] = (
            data["residual_adf_pvalue"] <= self.config.adf_pvalue_threshold
        )

        high = prices[self.stock_high].astype(float) if self.stock_high in prices else None
        low = prices[self.stock_low].astype(float) if self.stock_low in prices else None
        data["atr"] = average_true_range(
            high,
            low,
            data[self.stock_close],
            self.config.atr_window,
        )
        data["atr_pct"] = data["atr"] / data[self.stock_close]
        return data

    def position_size(self, features: pd.DataFrame) -> pd.Series:
        stop_pct = (
            features["atr_pct"].fillna(features["stock_return"].rolling(20).std())
            * self.config.stop_atr_multiple
        )
        stop_pct = stop_pct.clip(lower=self.config.min_stop_pct)
        fraction = self.config.risk_per_trade / stop_pct
        return fraction.clip(lower=0.0, upper=self.config.max_position_fraction).fillna(0.0)

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        features = self.build_features(prices)
        sizes = self.position_size(features)

        mode = self.config.signal_mode.lower().replace("-", "_")
        if mode not in {"mean_reversion", "trend", "hybrid", "buy_hold"}:
            raise ValueError(
                "signal_mode must be one of: mean_reversion, trend, hybrid, buy_hold"
            )

        if mode == "buy_hold":
            output = features.copy()
            output["position"] = self.config.max_position_fraction
            output["target_size"] = self.config.max_position_fraction
            output["entry_signal"] = False
            output["exit_signal"] = False
            if len(output) > 0:
                output.iloc[0, output.columns.get_loc("entry_signal")] = True
            return output

        position = pd.Series(0.0, index=features.index, name="position")
        entry_flag = pd.Series(False, index=features.index, name="entry_signal")
        exit_flag = pd.Series(False, index=features.index, name="exit_signal")

        state = 0.0
        holding_days = 0

        for i, idx in enumerate(features.index):
            position.iloc[i] = state
            if state > 0:
                holding_days += 1

            row = features.iloc[i]
            adf_ok = (
                True
                if not self.config.use_adf_filter
                else bool(row["residual_stationary"])
            )
            can_evaluate = pd.notna(row["residual_z"]) and pd.notna(row["rsi"]) and adf_ok

            trend_active = (
                can_evaluate
                and bool(row["risk_on"])
                and bool(row["stock_uptrend"])
            )
            mean_reversion_entry = (
                can_evaluate
                and bool(row["risk_on"])
                and row["residual_z"] < self.config.entry_z
                and row["rsi"] < self.config.entry_rsi
                and sizes.iloc[i] > 0
            )

            if mode == "mean_reversion":
                should_exit = (
                    state > 0
                    and can_evaluate
                    and (
                        row["residual_z"] > self.config.exit_z
                        or holding_days >= self.config.max_hold_days
                        or not bool(row["risk_on"])
                    )
                )
                should_enter = state == 0 and mean_reversion_entry
                next_state = float(sizes.iloc[i]) if should_enter else state
            else:
                trend_size = (
                    min(self.config.base_position_fraction, self.config.max_position_fraction)
                    if trend_active
                    else 0.0
                )
                tactical_size = float(sizes.iloc[i]) if mean_reversion_entry else 0.0
                if mode == "trend":
                    desired_state = trend_size
                else:
                    desired_state = min(
                        self.config.max_position_fraction,
                        max(trend_size, trend_size + tactical_size),
                    )

                should_enter = state == 0 and desired_state > 0
                should_exit = state > 0 and desired_state == 0
                next_state = desired_state

            if should_exit:
                exit_flag.iloc[i] = True
                state = 0.0
                holding_days = 0
            elif should_enter:
                entry_flag.iloc[i] = True
                state = next_state
                holding_days = 0
            elif mode in {"trend", "hybrid"}:
                state = next_state

        output = features.copy()
        output["position"] = position
        output["target_size"] = sizes
        output["entry_signal"] = entry_flag
        output["exit_signal"] = exit_flag
        return output

    def backtest(self, prices: pd.DataFrame, initial_equity: float = 100_000.0) -> BacktestResult:
        curve = self.generate_positions(prices)
        turnover = curve["position"].diff().abs().fillna(curve["position"].abs())
        cost_rate = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10_000.0
        curve["turnover"] = turnover
        curve["cost_return"] = turnover * cost_rate
        curve["strategy_return"] = (
            curve["position"].shift(1).fillna(0.0) * curve["stock_return"].fillna(0.0)
            - curve["cost_return"]
        )
        curve["equity"] = initial_equity * (1.0 + curve["strategy_return"]).cumprod()
        curve["buy_hold_equity"] = initial_equity * (
            1.0 + curve["stock_return"].fillna(0.0)
        ).cumprod()

        trades = self._extract_trades(curve, initial_equity)
        metrics = self._metrics(curve, trades)
        return BacktestResult(equity_curve=curve, trades=trades, metrics=metrics)

    def _extract_trades(self, curve: pd.DataFrame, initial_equity: float) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        in_trade = False
        entry_idx = None
        entry_equity = None

        for idx, row in curve.iterrows():
            if not in_trade and row["entry_signal"]:
                in_trade = True
                entry_idx = idx
                entry_equity = row["equity"]
            elif in_trade and row["exit_signal"]:
                exit_equity = row["equity"]
                rows.append(
                    {
                        "entry_date": entry_idx,
                        "exit_date": idx,
                        "return": (exit_equity / entry_equity) - 1.0
                        if entry_equity not in (None, 0)
                        else 0.0,
                    }
                )
                in_trade = False
                entry_idx = None
                entry_equity = None

        if in_trade and entry_idx is not None and entry_equity not in (None, 0):
            rows.append(
                {
                    "entry_date": entry_idx,
                    "exit_date": curve.index[-1],
                    "return": (curve["equity"].iloc[-1] / entry_equity) - 1.0,
                }
            )

        trades = pd.DataFrame(rows)
        if trades.empty:
            return pd.DataFrame(columns=["entry_date", "exit_date", "return"])
        return trades

    def _metrics(self, curve: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
        strategy_returns = curve["strategy_return"].dropna()
        total_return = curve["equity"].iloc[-1] / curve["equity"].iloc[0] - 1.0
        years = max(len(strategy_returns) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
        volatility = strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = (
            strategy_returns.mean() / strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            if strategy_returns.std() > 0
            else 0.0
        )
        drawdown = curve["equity"] / curve["equity"].cummax() - 1.0

        trade_returns = trades["return"] if not trades.empty else pd.Series(dtype=float)
        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "annual_volatility": float(volatility),
            "sharpe": float(sharpe),
            "max_drawdown": float(drawdown.min()),
            "exposure": float((curve["position"] > 0).mean()),
            "turnover": float(curve["turnover"].sum()),
            "trade_count": float(len(trades)),
            "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else 0.0,
            "avg_trade_return": float(trade_returns.mean()) if len(trade_returns) else 0.0,
            "buy_hold_return": float(
                curve["buy_hold_equity"].iloc[-1] / curve["buy_hold_equity"].iloc[0] - 1.0
            ),
        }
