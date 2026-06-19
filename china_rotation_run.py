from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path

import numpy as np
import pandas as pd

from china_market_run import normalize_china_symbol


DEFAULT_UNIVERSE = [
    "600519.SS",  # Kweichow Moutai
    "000858.SZ",  # Wuliangye
    "002594.SZ",  # BYD
    "300750.SZ",  # CATL
    "000333.SZ",  # Midea
    "600036.SS",  # China Merchants Bank
    "601318.SS",  # Ping An
    "601888.SS",  # China Tourism Group Duty Free
    "600276.SS",  # Hengrui Medicine
    "000651.SZ",  # Gree Electric
    "300760.SZ",  # Mindray
    "601012.SS",  # LONGi
]

CHINA_ETF_PRESETS = {
    "china_broad_etfs": [
        "510300.SS",  # CSI 300 ETF
        "510500.SS",  # CSI 500 ETF
        "512100.SS",  # CSI 1000 ETF
        "159915.SZ",  # ChiNext ETF
        "588000.SS",  # STAR 50 ETF
    ],
    "china_style_etfs": [
        "510300.SS",  # CSI 300 ETF
        "510050.SS",  # SSE 50 ETF
        "510500.SS",  # CSI 500 ETF
        "512100.SS",  # CSI 1000 ETF
        "159915.SZ",  # ChiNext ETF
        "515000.SS",  # technology ETF
    ],
}


@dataclass(frozen=True)
class RotationConfig:
    allocation_mode: str = "equal_weight"
    top_n: int = 4
    rebalance: str = "M"
    rebalance_tolerance: float = 0.0
    core_weight: float = 0.70
    market_ma_window: int = 200
    stock_ma_window: int = 200
    momentum_fast: int = 63
    momentum_medium: int = 126
    momentum_slow: int = 252
    skip_recent_days: int = 20
    volatility_window: int = 63
    momentum_top_n: int = 1
    momentum_score_mode: str = "raw"
    momentum_confirmation: str = "none"
    defensive_asset: str | None = None
    breadth_adjusted: bool = False
    breadth_full_threshold: float = 0.60
    breadth_partial_threshold: float = 0.40
    breadth_partial_exposure: float = 0.50
    regime_filter: bool = False
    regime_stress_exposure: float = 0.50
    regime_defensive_asset: str | None = None
    regime_train_years: int = 3
    regime_n_components: int = 3
    regime_n_regimes: int = 2
    regime_retrain_frequency: str = "M"
    regime_signal_lag: int = 1
    max_annual_volatility: float = 0.80
    max_gross_exposure: float = 1.0
    risk_mode: str = "none"
    quality_min_roe: float | None = None
    quality_max_debt_to_equity: float | None = None
    quality_min_profit_margin: float | None = None
    quality_require_positive_fcf: bool = False
    transaction_cost_bps: float = 6.0
    slippage_bps: float = 8.0
    rebalance_tolerance_by_asset: dict[str, float] | None = None
    rebalance_to: str = "target"


def parse_universe(text: str | None) -> list[str]:
    if not text:
        return DEFAULT_UNIVERSE
    raw = [item.strip() for item in text.split(",") if item.strip()]
    if not raw:
        return DEFAULT_UNIVERSE
    return [normalize_china_symbol(item) for item in raw]


def load_point_in_time_universe(path: Path, snapshot_date: str) -> tuple[list[str], pd.Timestamp]:
    data = pd.read_csv(path)
    lower_columns = {column.lower().strip(): column for column in data.columns}
    date_col = lower_columns.get("date") or lower_columns.get("snapshot_date")
    ticker_col = lower_columns.get("ticker") or lower_columns.get("symbol")
    if date_col is None or ticker_col is None:
        raise SystemExit("Universe file must include date and ticker columns.")

    data = data.rename(columns={date_col: "date", ticker_col: "ticker"}).copy()
    data["date"] = pd.to_datetime(data["date"])
    requested = pd.to_datetime(snapshot_date)
    available_dates = data.loc[data["date"] <= requested, "date"]
    if available_dates.empty:
        raise SystemExit(f"No universe snapshot on or before {snapshot_date}.")

    chosen_date = available_dates.max()
    tickers = (
        data.loc[data["date"] == chosen_date, "ticker"]
        .dropna()
        .astype(str)
        .map(normalize_china_symbol)
        .drop_duplicates()
        .tolist()
    )
    if not tickers:
        raise SystemExit(f"Universe snapshot {chosen_date.date()} has no tickers.")
    return tickers, chosen_date


def parse_tolerance_values(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values or [0.0, 0.05, 0.10, 0.15, 0.20]


def parse_tolerance_band_map(
    text: str | None,
    normalizer=normalize_china_symbol,
) -> dict[str, float]:
    if not text:
        return {}
    bands = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(
                "Tolerance bands must use ticker=value pairs, for example 510300.SS=0.025,SPY=0.075."
            )
        ticker, value = item.split("=", 1)
        bands[normalizer(ticker.strip())] = float(value)
    return bands


def load_quality_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None

    data = pd.read_csv(path)
    lower_columns = {column.lower().strip(): column for column in data.columns}
    ticker_col = lower_columns.get("ticker") or lower_columns.get("symbol")
    if ticker_col is None:
        raise SystemExit("Quality CSV must include a ticker or symbol column.")

    renamed = data.rename(columns={ticker_col: "ticker"}).copy()
    alias_map = {
        "return_on_equity": "roe",
        "roe": "roe",
        "debt_to_equity": "debt_to_equity",
        "debt/equity": "debt_to_equity",
        "profit_margin": "profit_margin",
        "net_margin": "profit_margin",
        "free_cash_flow": "free_cash_flow",
        "fcf": "free_cash_flow",
    }
    for raw_name, canonical in alias_map.items():
        if raw_name in lower_columns and canonical not in renamed.columns:
            renamed = renamed.rename(columns={lower_columns[raw_name]: canonical})

    renamed["ticker"] = renamed["ticker"].map(lambda value: normalize_china_symbol(str(value)))
    renamed = renamed.set_index("ticker")
    for column in ["roe", "debt_to_equity", "profit_margin", "free_cash_flow"]:
        if column in renamed:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    return renamed


def quality_candidates(
    stock_cols: list[str],
    quality_data: pd.DataFrame | None,
    config: RotationConfig,
) -> list[str]:
    if quality_data is None:
        raise ValueError("quality_equal_weight requires --quality-csv.")

    available = quality_data.reindex(stock_cols)
    mask = pd.Series(True, index=available.index)
    if config.quality_min_roe is not None and "roe" in available:
        mask &= available["roe"] >= config.quality_min_roe
    if config.quality_max_debt_to_equity is not None and "debt_to_equity" in available:
        mask &= available["debt_to_equity"] <= config.quality_max_debt_to_equity
    if config.quality_min_profit_margin is not None and "profit_margin" in available:
        mask &= available["profit_margin"] >= config.quality_min_profit_margin
    if config.quality_require_positive_fcf and "free_cash_flow" in available:
        mask &= available["free_cash_flow"] > 0

    selected = mask[mask].index.tolist()
    if not selected:
        raise ValueError("Quality filter removed every stock. Relax the thresholds.")
    return selected


def _extract_yfinance_close(downloaded: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(downloaded.columns, pd.MultiIndex):
        if ticker in downloaded.columns.get_level_values(0):
            return downloaded[(ticker, "Close")]
        return downloaded[("Close", ticker)]
    return downloaded["Close"]


def load_yfinance_closes(
    tickers: list[str],
    period: str,
    interval: str,
    start: str | None,
    end: str | None,
    cache_dir: Path | None,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit(
            "yfinance is not installed. Install it with: python -m pip install yfinance"
        ) from exc

    cache_path = cache_dir or (Path(__file__).resolve().parent / ".yfinance-cache")
    cache_path.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_path))

    download_kwargs = {
        "tickers": tickers,
        "interval": interval,
        "auto_adjust": True,
        "group_by": "ticker",
        "progress": False,
        "threads": False,
    }
    if start or end:
        download_kwargs["start"] = start
        download_kwargs["end"] = end
    else:
        download_kwargs["period"] = period

    downloaded = yf.download(**download_kwargs)
    if downloaded.empty:
        raise SystemExit(f"No data returned for symbols: {', '.join(tickers)}")

    closes = pd.DataFrame(
        {ticker: _extract_yfinance_close(downloaded, ticker) for ticker in tickers}
    )
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index().dropna(how="all")
    valid = closes.columns[closes.notna().sum() >= 260].tolist()
    closes = closes.loc[:, valid].ffill().dropna(how="any")
    if len(closes) < 260:
        raise SystemExit(
            f"Only {len(closes)} aligned rows were downloaded. Use a longer period or fewer tickers."
        )
    if len(closes.columns) < 2:
        raise SystemExit("Need at least two usable stock series for rotation.")
    return closes


def make_synthetic_rotation_data(rows: int = 1_500, assets: int = 10, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-02", periods=rows)
    market_return = rng.normal(0.0002, 0.012, rows)

    data = {"510300.SS": 4.0 * np.cumprod(1.0 + market_return)}
    for i in range(assets):
        drift = rng.normal(0.00005, 0.00025)
        beta = rng.uniform(0.6, 1.2)
        noise = rng.normal(0.0, rng.uniform(0.012, 0.026), rows)
        trend = np.sin(np.linspace(0, rng.uniform(8, 16), rows) + rng.uniform(0, 3)) * 0.001
        returns = drift + beta * market_return + noise + trend
        data[f"000{i + 1:03d}.SZ"] = 20.0 * np.cumprod(1.0 + returns)

    return pd.DataFrame(data, index=index)


def month_or_week_rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    period = index.to_period("W-FRI" if frequency.upper().startswith("W") else "M")
    marker = pd.Series(period, index=index)
    return marker.index[marker != marker.shift(-1)]


def build_scores(closes: pd.DataFrame, market_col: str, config: RotationConfig) -> pd.DataFrame:
    stock_closes = closes.drop(columns=[market_col])
    fast = stock_closes / stock_closes.shift(config.momentum_fast) - 1.0
    medium = stock_closes / stock_closes.shift(config.momentum_medium) - 1.0
    slow = stock_closes.shift(config.skip_recent_days) / stock_closes.shift(
        config.momentum_slow
    ) - 1.0
    volatility = stock_closes.pct_change().rolling(config.volatility_window).std() * np.sqrt(252)
    relative = stock_closes.pct_change(config.momentum_medium).sub(
        closes[market_col].pct_change(config.momentum_medium), axis=0
    )

    score = 0.25 * fast + 0.30 * medium + 0.35 * slow + 0.10 * relative
    score = score.where(stock_closes > stock_closes.rolling(config.stock_ma_window).mean())
    score = score.where(volatility <= config.max_annual_volatility)
    return score


def generate_rotation_weights(
    closes: pd.DataFrame,
    market_col: str,
    config: RotationConfig,
    quality_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    stock_cols = [column for column in closes.columns if column != market_col]
    defensive_assets = {
        asset
        for asset in [config.defensive_asset, config.regime_defensive_asset]
        if asset is not None
    }
    allocation_cols = [column for column in stock_cols if column not in defensive_assets] or stock_cols
    stock_closes = closes.loc[:, allocation_cols]
    stock_ma = stock_closes.rolling(config.stock_ma_window).mean()
    scores = build_scores(closes, market_col, config).reindex(columns=allocation_cols)
    volatility = stock_closes.pct_change().rolling(config.volatility_window).std() * np.sqrt(252)
    returns_3m = stock_closes / stock_closes.shift(63) - 1.0
    returns_6m = stock_closes / stock_closes.shift(126) - 1.0
    returns_12m = stock_closes / stock_closes.shift(config.momentum_slow) - 1.0
    vol_6m = stock_closes.pct_change().rolling(126).std() * np.sqrt(252)
    vol_12m = stock_closes.pct_change().rolling(config.momentum_slow).std() * np.sqrt(252)
    ma_100 = stock_closes.rolling(100).mean()
    weights = pd.DataFrame(0.0, index=closes.index, columns=stock_cols)

    market_ma = closes[market_col].rolling(config.market_ma_window).mean()
    if config.risk_mode == "none":
        risk_on = pd.Series(True, index=closes.index)
    elif config.risk_mode == "market":
        risk_on = closes[market_col] > market_ma
    else:
        raise ValueError("risk_mode must be one of: market, none")
    rebalance_dates = month_or_week_rebalance_dates(closes.index, config.rebalance)

    allocation_mode = config.allocation_mode.lower().replace("-", "_")
    valid_allocations = {
        "top_n",
        "equal_weight",
        "ivy_trend_filter",
        "dual_momentum",
        "protective_dual_momentum",
        "core_satellite",
        "inverse_vol",
        "trend_equal_weight",
        "score_weighted",
        "quality_equal_weight",
    }
    if allocation_mode not in valid_allocations:
        raise ValueError(f"allocation_mode must be one of: {', '.join(sorted(valid_allocations))}")

    current = pd.Series(0.0, index=stock_cols)
    first_date = closes.index[0] if len(closes.index) else None
    for date in closes.index:
        if date in rebalance_dates or date == first_date:
            current[:] = 0.0
            if bool(risk_on.loc[date]):
                if allocation_mode == "equal_weight":
                    current.loc[allocation_cols] = config.max_gross_exposure / len(allocation_cols)
                elif allocation_mode == "ivy_trend_filter":
                    trend = stock_closes.loc[date] > stock_ma.loc[date]
                    current.loc[trend[trend].index] = config.max_gross_exposure / len(allocation_cols)
                elif allocation_mode in {"dual_momentum", "protective_dual_momentum"}:
                    momentum_cols = [
                        column for column in allocation_cols if column != config.defensive_asset
                    ]
                    trailing_return = returns_12m.loc[date].reindex(momentum_cols)
                    eligible = trailing_return > 0.0
                    if config.momentum_confirmation == "ma200":
                        eligible &= stock_closes.loc[date].reindex(momentum_cols) > stock_ma.loc[
                            date
                        ].reindex(momentum_cols)
                    elif config.momentum_confirmation == "fast":
                        eligible &= returns_3m.loc[date].reindex(momentum_cols) > 0.0
                        eligible &= stock_closes.loc[date].reindex(momentum_cols) > stock_ma.loc[
                            date
                        ].reindex(momentum_cols)
                    elif config.momentum_confirmation == "aggressive":
                        eligible &= returns_6m.loc[date].reindex(momentum_cols) > 0.0
                        eligible &= stock_closes.loc[date].reindex(momentum_cols) > ma_100.loc[
                            date
                        ].reindex(momentum_cols)
                        eligible &= stock_closes.loc[date].reindex(momentum_cols) > stock_ma.loc[
                            date
                        ].reindex(momentum_cols)
                    elif config.momentum_confirmation != "none":
                        raise ValueError(
                            "momentum_confirmation must be one of: none, ma200, fast, aggressive"
                        )

                    if config.momentum_score_mode == "raw":
                        momentum_score = trailing_return
                    elif config.momentum_score_mode == "risk_adjusted":
                        momentum_score = trailing_return / vol_12m.loc[date].reindex(
                            momentum_cols
                        ).replace(0.0, np.nan)
                    elif config.momentum_score_mode == "blended_risk_adjusted":
                        score_6m = returns_6m.loc[date].reindex(momentum_cols) / vol_6m.loc[
                            date
                        ].reindex(momentum_cols).replace(0.0, np.nan)
                        score_12m = trailing_return / vol_12m.loc[date].reindex(
                            momentum_cols
                        ).replace(0.0, np.nan)
                        momentum_score = 0.5 * score_6m + 0.5 * score_12m
                    else:
                        raise ValueError(
                            "momentum_score_mode must be one of: raw, risk_adjusted, blended_risk_adjusted"
                        )

                    ranked = momentum_score.where(eligible).dropna().sort_values(ascending=False)
                    selected = ranked.head(max(1, config.momentum_top_n)).index
                    risk_exposure = config.max_gross_exposure
                    use_breadth = config.breadth_adjusted or allocation_mode == "protective_dual_momentum"
                    if use_breadth:
                        breadth = float((trailing_return > 0.0).mean()) if len(trailing_return) else 0.0
                        if len(selected) == 0:
                            risk_exposure = 0.0
                        elif breadth >= config.breadth_full_threshold:
                            risk_exposure = config.max_gross_exposure
                        elif breadth >= config.breadth_partial_threshold:
                            risk_exposure = (
                                config.max_gross_exposure * config.breadth_partial_exposure
                            )
                        else:
                            risk_exposure = 0.0
                    if len(selected) > 0:
                        current.loc[selected] = risk_exposure / len(selected)
                    defensive_exposure = config.max_gross_exposure - float(current.sum())
                    if (
                        defensive_exposure > 0.0
                        and config.defensive_asset
                        and config.defensive_asset in current.index
                    ):
                        current.loc[config.defensive_asset] = defensive_exposure
                elif allocation_mode == "quality_equal_weight":
                    selected = quality_candidates(allocation_cols, quality_data, config)
                    current.loc[selected] = config.max_gross_exposure / len(selected)
                elif allocation_mode == "inverse_vol":
                    inv_vol = 1.0 / volatility.loc[date].replace(0.0, np.nan)
                    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(inv_vol) > 0:
                        current.loc[inv_vol.index] = (
                            config.max_gross_exposure * inv_vol / inv_vol.sum()
                        )
                elif allocation_mode == "trend_equal_weight":
                    trend = stock_closes.loc[date] > stock_ma.loc[date]
                    selected = trend[trend].index
                    if len(selected) > 0:
                        current.loc[selected] = config.max_gross_exposure / len(selected)
                elif allocation_mode == "score_weighted":
                    positive_scores = scores.loc[date].dropna().clip(lower=0.0)
                    positive_scores = positive_scores[positive_scores > 0.0]
                    if len(positive_scores) > 0:
                        current.loc[positive_scores.index] = (
                            config.max_gross_exposure
                            * positive_scores
                            / positive_scores.sum()
                        )
                else:
                    ranked = scores.loc[date].dropna().sort_values(ascending=False)
                    selected = ranked[ranked > 0].head(config.top_n).index

                    if allocation_mode == "top_n":
                        if len(selected) > 0:
                            current.loc[selected] = config.max_gross_exposure / len(selected)
                    else:
                        core_weight = float(np.clip(config.core_weight, 0.0, 1.0))
                        satellite_weight = 1.0 - core_weight
                        current.loc[allocation_cols] = (
                            config.max_gross_exposure * core_weight / len(allocation_cols)
                        )
                        if len(selected) > 0 and satellite_weight > 0:
                            current.loc[selected] += (
                                config.max_gross_exposure
                                * satellite_weight
                                / len(selected)
                            )
        weights.loc[date] = current

    return weights


def backtest_rotation(
    closes: pd.DataFrame,
    market_col: str,
    config: RotationConfig,
    initial_equity: float = 100_000.0,
    quality_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    targets = generate_rotation_weights(closes, market_col, config, quality_data)
    stock_returns = closes.drop(columns=[market_col]).pct_change().fillna(0.0)
    regime_stress = pd.Series(False, index=closes.index)
    if config.regime_filter:
        from regime_filter import RegimeConfig, generate_regime_signals

        regime_defensive_assets = {
            asset
            for asset in [config.defensive_asset, config.regime_defensive_asset]
            if asset is not None
        }
        regime_asset_cols = [
            column
            for column in stock_returns.columns
            if column not in regime_defensive_assets
        ]
        regime_signals = generate_regime_signals(
            closes,
            market_col,
            asset_cols=regime_asset_cols,
            config=RegimeConfig(
                n_components=config.regime_n_components,
                n_regimes=config.regime_n_regimes,
                train_years=config.regime_train_years,
                retrain_frequency=config.regime_retrain_frequency,
            ),
        )
        regime_stress = (
            regime_signals["regime_stress"]
            .shift(config.regime_signal_lag)
            .reindex(closes.index)
            .ffill()
            .infer_objects(copy=False)
            .fillna(False)
            .astype(bool)
        )
        stress_exposure = float(np.clip(config.regime_stress_exposure, 0.0, 1.0))
        risky_cols = [
            column
            for column in targets.columns
            if column not in regime_defensive_assets
        ]
        stress_rows = regime_stress[regime_stress].index
        if len(stress_rows) > 0 and risky_cols:
            targets.loc[stress_rows, risky_cols] *= stress_exposure
            defensive_asset = config.regime_defensive_asset or config.defensive_asset
            if defensive_asset and defensive_asset in targets.columns:
                defensive_weight = (
                    config.max_gross_exposure - targets.loc[stress_rows].sum(axis=1)
                ).clip(lower=0.0)
                targets.loc[stress_rows, defensive_asset] += defensive_weight
    cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000.0

    benchmark_cols = [
        column
        for column in stock_returns.columns
        if column not in {config.defensive_asset, config.regime_defensive_asset}
    ] or stock_returns.columns.tolist()
    equal_weight = stock_returns.loc[:, benchmark_cols].mean(axis=1)
    market_return = closes[market_col].pct_change().fillna(0.0)
    rebalance_dates = set(month_or_week_rebalance_dates(closes.index, config.rebalance))
    if len(closes.index) > 0:
        rebalance_dates.add(closes.index[0])

    weights = pd.DataFrame(0.0, index=closes.index, columns=stock_returns.columns)
    turnover = pd.Series(0.0, index=closes.index)
    strategy_return = pd.Series(0.0, index=closes.index)

    equity = float(initial_equity)
    equity_values: list[float] = []
    current_weights = pd.Series(0.0, index=stock_returns.columns)

    for i, date in enumerate(closes.index):
        previous_equity = equity
        asset_return = stock_returns.loc[date]
        portfolio_return = float((current_weights * asset_return).sum())
        equity *= 1.0 + portfolio_return

        denominator = 1.0 + portfolio_return
        if denominator != 0:
            drifted_weights = current_weights * (1.0 + asset_return) / denominator
        else:
            drifted_weights = current_weights * 0.0

        daily_turnover = 0.0
        if date in rebalance_dates:
            target = targets.loc[date].fillna(0.0)
            drift = (target - drifted_weights).abs()
            if config.rebalance_tolerance_by_asset:
                tolerance = pd.Series(
                    {
                        asset: config.rebalance_tolerance_by_asset.get(
                            asset,
                            config.rebalance_tolerance,
                        )
                        for asset in drift.index
                    }
                )
            else:
                tolerance = pd.Series(config.rebalance_tolerance, index=drift.index)
            should_trade = i == 0 or bool((drift > tolerance).any())
            if should_trade:
                if i == 0 or config.rebalance_to == "target":
                    desired_weights = target.copy()
                elif config.rebalance_to == "corridor":
                    desired_weights = drifted_weights.copy()
                    upper = target + tolerance
                    lower = (target - tolerance).clip(lower=0.0)
                    over_upper = drifted_weights > upper
                    under_lower = drifted_weights < lower
                    desired_weights.loc[over_upper] = upper.loc[over_upper]
                    desired_weights.loc[under_lower] = lower.loc[under_lower]
                    desired_weights.loc[target <= 0.0] = 0.0
                    if desired_weights.sum() > config.max_gross_exposure:
                        desired_weights *= config.max_gross_exposure / desired_weights.sum()
                else:
                    raise ValueError("rebalance_to must be one of: target, corridor")

                daily_turnover = float((desired_weights - drifted_weights).abs().sum())
                equity *= 1.0 - daily_turnover * cost_rate
                current_weights = desired_weights.copy()
            else:
                current_weights = drifted_weights
        else:
            current_weights = drifted_weights

        turnover.loc[date] = daily_turnover
        weights.loc[date] = current_weights
        equity_values.append(equity)
        strategy_return.loc[date] = equity / previous_equity - 1.0 if previous_equity else 0.0

    curve = pd.DataFrame(
        {
            "strategy_return": strategy_return,
            "market_return": market_return,
            "equal_weight_return": equal_weight,
            "turnover": turnover,
            "exposure": weights.sum(axis=1),
            "regime_stress": regime_stress.astype(float),
        },
        index=closes.index,
    )
    curve["equity"] = equity_values
    curve["market_equity"] = initial_equity * (1.0 + curve["market_return"]).cumprod()
    curve["equal_weight_equity"] = initial_equity * (
        1.0 + curve["equal_weight_return"]
    ).cumprod()
    curve.attrs["initial_equity"] = initial_equity

    metrics = summarize(curve)
    return curve, weights, metrics


def summarize(curve: pd.DataFrame) -> dict[str, float]:
    returns = curve["strategy_return"]
    initial_equity = float(curve.attrs.get("initial_equity", curve["equity"].iloc[0]))
    total_return = curve["equity"].iloc[-1] / initial_equity - 1.0
    years = max(len(returns) / 252, 1 / 252)
    volatility = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
    drawdown = curve["equity"] / curve["equity"].cummax() - 1.0
    market_return = curve["market_equity"].iloc[-1] / curve["market_equity"].iloc[0] - 1.0
    equal_weight_return = (
        curve["equal_weight_equity"].iloc[-1] / curve["equal_weight_equity"].iloc[0] - 1.0
    )
    return {
        "total_return": float(total_return),
        "cagr": float((1.0 + total_return) ** (1.0 / years) - 1.0),
        "annual_volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "average_exposure": float(curve["exposure"].mean()),
        "regime_stress_fraction": float(curve.get("regime_stress", pd.Series(0.0, index=curve.index)).mean()),
        "turnover": float(curve["turnover"].sum()),
        "market_return": float(market_return),
        "equal_weight_return": float(equal_weight_return),
        "excess_vs_equal_weight": float(total_return - equal_weight_return),
    }


def run_grid_search(
    closes: pd.DataFrame,
    market_col: str,
    initial_equity: float,
    quality_data: pd.DataFrame | None = None,
    quality_options: dict[str, float | bool | None] | None = None,
    rebalance_tolerance: float = 0.0,
    rebalance_tolerance_by_asset: dict[str, float] | None = None,
    sort_by: str = "total_return",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    allocations = [
        "equal_weight",
        "ivy_trend_filter",
        "dual_momentum",
        "inverse_vol",
        "trend_equal_weight",
        "score_weighted",
        "top_n",
        "core_satellite",
    ]
    if quality_data is not None:
        allocations.append("quality_equal_weight")
    risk_modes = ["none", "market"]

    for allocation in allocations:
        top_ns = [4] if allocation not in {"top_n", "core_satellite"} else [2, 3, 4, 6, 8]
        core_weights = [0.70] if allocation != "core_satellite" else [0.50, 0.70, 0.85]
        for risk_mode in risk_modes:
            for top_n in top_ns:
                for core_weight in core_weights:
                    config = RotationConfig(
                        allocation_mode=allocation,
                        top_n=top_n,
                        core_weight=core_weight,
                        risk_mode=risk_mode,
                        rebalance_tolerance=rebalance_tolerance,
                        rebalance_tolerance_by_asset=rebalance_tolerance_by_asset,
                        rebalance_to="target",
                        momentum_top_n=1,
                        momentum_score_mode="raw",
                        momentum_confirmation="none",
                        **(quality_options or {}),
                    )
                    curve, _, metrics = backtest_rotation(
                        closes,
                        market_col,
                        config,
                        initial_equity=initial_equity,
                        quality_data=quality_data,
                    )
                    rows.append(
                        {
                            "allocation": allocation,
                            "risk_mode": risk_mode,
                            "top_n": top_n,
                            "core_weight": core_weight,
                            "final_equity": curve["equity"].iloc[-1],
                            **metrics,
                            "excess_vs_equal": metrics["total_return"]
                            - metrics["equal_weight_return"],
                        }
                    )

    result = pd.DataFrame(rows)
    if sort_by not in result.columns:
        sort_by = "total_return"
    return result.sort_values(sort_by, ascending=False).reset_index(drop=True)


def run_tolerance_sweep(
    closes: pd.DataFrame,
    market_col: str,
    args: argparse.Namespace,
    values: list[float],
    quality_data: pd.DataFrame | None = None,
    quality_options: dict[str, float | bool | None] | None = None,
) -> pd.DataFrame:
    rows = []
    baseline_equity = None
    baseline_turnover = None
    for tolerance in values:
        config = RotationConfig(
            allocation_mode=args.allocation,
            top_n=args.top_n,
            rebalance=args.rebalance,
            rebalance_tolerance=tolerance,
            rebalance_tolerance_by_asset=args.tolerance_band_map,
            rebalance_to=args.rebalance_to,
            core_weight=args.core_weight,
            momentum_top_n=args.momentum_top_n,
            momentum_score_mode=args.momentum_score_mode,
            momentum_confirmation=args.momentum_confirmation,
            defensive_asset=normalize_china_symbol(args.defensive_asset) if args.defensive_asset else None,
            breadth_adjusted=args.breadth_adjusted,
            breadth_full_threshold=args.breadth_full_threshold,
            breadth_partial_threshold=args.breadth_partial_threshold,
            breadth_partial_exposure=args.breadth_partial_exposure,
            max_gross_exposure=args.max_exposure,
            risk_mode=args.risk_mode,
            **(quality_options or {}),
        )
        curve, _, metrics = backtest_rotation(
            closes,
            market_col,
            config,
            initial_equity=args.initial_capital,
            quality_data=quality_data,
        )
        final_equity = float(curve["equity"].iloc[-1])
        if tolerance == 0.0:
            baseline_equity = final_equity
            baseline_turnover = metrics["turnover"]
        rows.append(
            {
                "rebalance_tolerance": tolerance,
                "final_equity": final_equity,
                "total_return": metrics["total_return"],
                "excess_vs_equal_weight": metrics["excess_vs_equal_weight"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "turnover": metrics["turnover"],
                "turnover_saved_vs_0": np.nan,
                "equity_added_vs_0": np.nan,
            }
        )

    result = pd.DataFrame(rows)
    if baseline_equity is not None:
        result["equity_added_vs_0"] = result["final_equity"] - baseline_equity
    if baseline_turnover is not None:
        result["turnover_saved_vs_0"] = baseline_turnover - result["turnover"]
    return result.sort_values("final_equity", ascending=False).reset_index(drop=True)


def _metric_value(metrics: dict[str, float], metric: str) -> float:
    value = float(metrics.get(metric, np.nan))
    if metric in {"max_drawdown"}:
        return -abs(value)
    return value


def run_walk_forward_tolerance(
    closes: pd.DataFrame,
    market_col: str,
    base_config: RotationConfig,
    initial_equity: float,
    values: list[float],
    train_years: int = 3,
    test_years: int = 1,
    optimize_metric: str = "sharpe",
    mode: str = "anchored",
    quality_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if train_years < 1 or test_years < 1:
        raise ValueError("train_years and test_years must both be at least 1.")
    if mode not in {"anchored", "rolling"}:
        raise ValueError("walk-forward mode must be one of: anchored, rolling")

    start = closes.index.min()
    final_date = closes.index.max()
    first_test_start = start + pd.DateOffset(years=train_years)
    test_start = first_test_start
    equity = float(initial_equity)
    rows = []

    while test_start < final_date:
        test_end = min(test_start + pd.DateOffset(years=test_years), final_date + pd.Timedelta(days=1))
        train_start = start if mode == "anchored" else test_start - pd.DateOffset(years=train_years)
        train = closes.loc[(closes.index >= train_start) & (closes.index < test_start)]
        test = closes.loc[(closes.index >= test_start) & (closes.index < test_end)]
        if len(train) < 252 or len(test) < 20:
            test_start = test_end
            continue

        candidates = []
        for tolerance in values:
            config = dataclass_replace(base_config, rebalance_tolerance=tolerance)
            train_curve, _, train_metrics = backtest_rotation(
                train,
                market_col,
                config,
                initial_equity=initial_equity,
                quality_data=quality_data,
            )
            candidates.append(
                {
                    "tolerance": tolerance,
                    "score": _metric_value(train_metrics, optimize_metric),
                    "train_final_equity": float(train_curve["equity"].iloc[-1]),
                    "train_total_return": train_metrics["total_return"],
                    "train_sharpe": train_metrics["sharpe"],
                    "train_max_drawdown": train_metrics["max_drawdown"],
                }
            )

        candidates_df = pd.DataFrame(candidates).sort_values(
            ["score", "train_final_equity"],
            ascending=False,
        )
        best = candidates_df.iloc[0]
        chosen_tolerance = float(best["tolerance"])
        config = dataclass_replace(base_config, rebalance_tolerance=chosen_tolerance)
        test_curve, _, test_metrics = backtest_rotation(
            test,
            market_col,
            config,
            initial_equity=equity,
            quality_data=quality_data,
        )
        equity = float(test_curve["equity"].iloc[-1])
        rows.append(
            {
                "train_start": train.index[0].date(),
                "train_end": train.index[-1].date(),
                "test_start": test.index[0].date(),
                "test_end": test.index[-1].date(),
                "chosen_tolerance": chosen_tolerance,
                "train_score": float(best["score"]),
                "train_total_return": float(best["train_total_return"]),
                "train_sharpe": float(best["train_sharpe"]),
                "train_max_drawdown": float(best["train_max_drawdown"]),
                "oos_total_return": test_metrics["total_return"],
                "oos_sharpe": test_metrics["sharpe"],
                "oos_max_drawdown": test_metrics["max_drawdown"],
                "oos_turnover": test_metrics["turnover"],
                "oos_final_equity": equity,
            }
        )
        test_start = test_end

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a China A-share momentum and relative-strength rotation strategy."
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated China tickers. Six-digit codes are auto-suffixed.",
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=None,
        help="CSV with date,ticker rows for point-in-time universe snapshots.",
    )
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="Snapshot date to use with --universe-file. Defaults to --start if provided.",
    )
    parser.add_argument(
        "--etf-preset",
        choices=sorted(CHINA_ETF_PRESETS),
        default=None,
        help="Use a predefined China ETF universe instead of the default current stock leaders.",
    )
    parser.add_argument(
        "--market",
        default="510300.SS",
        help="Market regime proxy. Default: 510300.SS.",
    )
    parser.add_argument("--period", default="max", help="yfinance period. Default: max.")
    parser.add_argument("--interval", default="1d", help="yfinance interval. Default: 1d.")
    parser.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=4,
        help="Number of stocks to hold at each rebalance. Default: 4.",
    )
    parser.add_argument(
        "--momentum-top-n",
        type=int,
        default=1,
        help="For dual_momentum, hold this many positive-momentum winners. Default: 1.",
    )
    parser.add_argument(
        "--momentum-score-mode",
        choices=["raw", "risk_adjusted", "blended_risk_adjusted"],
        default="raw",
        help="For dual_momentum, rank by raw return or volatility-adjusted momentum. Default: raw.",
    )
    parser.add_argument(
        "--momentum-confirmation",
        choices=["none", "ma200", "fast", "aggressive"],
        default="none",
        help="For dual_momentum, require extra recent trend confirmation. Default: none.",
    )
    parser.add_argument(
        "--defensive-asset",
        default=None,
        help="Optional ticker to hold when dual_momentum has no positive eligible asset.",
    )
    parser.add_argument(
        "--breadth-adjusted",
        action="store_true",
        help="Scale dual-momentum exposure using the share of assets with positive 12-month momentum.",
    )
    parser.add_argument(
        "--breadth-full-threshold",
        type=float,
        default=0.60,
        help="Breadth needed for full risk exposure. Default: 0.60.",
    )
    parser.add_argument(
        "--breadth-partial-threshold",
        type=float,
        default=0.40,
        help="Breadth needed for partial risk exposure. Default: 0.40.",
    )
    parser.add_argument(
        "--breadth-partial-exposure",
        type=float,
        default=0.50,
        help="Risk exposure when breadth is between partial and full thresholds. Default: 0.50.",
    )
    parser.add_argument(
        "--allocation",
        choices=[
            "top_n",
            "equal_weight",
            "ivy_trend_filter",
            "dual_momentum",
            "protective_dual_momentum",
            "core_satellite",
            "inverse_vol",
            "trend_equal_weight",
            "score_weighted",
            "quality_equal_weight",
        ],
        default="equal_weight",
        help="Portfolio allocation mode. Default: equal_weight.",
    )
    parser.add_argument(
        "--core-weight",
        type=float,
        default=0.70,
        help="Equal-weight core sleeve for core_satellite mode. Default: 0.70.",
    )
    parser.add_argument(
        "--rebalance",
        choices=["M", "W"],
        default="M",
        help="Rebalance frequency: monthly or weekly. Default: M.",
    )
    parser.add_argument(
        "--rebalance-tolerance",
        type=float,
        default=0.0,
        help="Only rebalance when a holding drifts this far from target weight. Default: 0.",
    )
    parser.add_argument(
        "--rebalance-to",
        choices=["target", "corridor"],
        default="target",
        help="Trade breached holdings back to target or only to the tolerance corridor edge. Default: target.",
    )
    parser.add_argument(
        "--tolerance-bands",
        default=None,
        help="Optional asset-specific bands as ticker=value pairs, for example 510300.SS=0.025,159915.SZ=0.075.",
    )
    parser.add_argument(
        "--max-exposure",
        type=float,
        default=1.0,
        help="Maximum gross long exposure. Default: 1.0.",
    )
    parser.add_argument(
        "--risk-mode",
        choices=["market", "none"],
        default="none",
        help="Use market trend cash filter or stay invested. Default: none.",
    )
    parser.add_argument(
        "--yf-cache-dir",
        type=Path,
        default=None,
        help="Optional yfinance cache directory. Default: .yfinance-cache in this project.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of yfinance.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Starting account value for the backtest. Default: 100000.",
    )
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=None,
        help="Optional fundamentals CSV with ticker plus columns such as roe, debt_to_equity, profit_margin, free_cash_flow.",
    )
    parser.add_argument(
        "--quality-min-roe",
        type=float,
        default=None,
        help="Minimum ROE for quality_equal_weight, for example 0.10.",
    )
    parser.add_argument(
        "--quality-max-debt-to-equity",
        type=float,
        default=None,
        help="Maximum debt-to-equity for quality_equal_weight.",
    )
    parser.add_argument(
        "--quality-min-profit-margin",
        type=float,
        default=None,
        help="Minimum profit margin for quality_equal_weight, for example 0.05.",
    )
    parser.add_argument(
        "--quality-require-positive-fcf",
        action="store_true",
        help="Require positive free_cash_flow for quality_equal_weight.",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Compare allocation modes and parameters, then rank them.",
    )
    parser.add_argument(
        "--tolerance-sweep",
        action="store_true",
        help="Compare rebalance tolerance values net of transaction costs.",
    )
    parser.add_argument(
        "--walk-forward-tolerance",
        action="store_true",
        help="Optimize tolerance on prior data and report next-period out-of-sample results.",
    )
    parser.add_argument(
        "--walk-forward-mode",
        choices=["anchored", "rolling"],
        default="anchored",
        help="Use expanding anchored training or rolling training. Default: anchored.",
    )
    parser.add_argument(
        "--walk-forward-train-years",
        type=int,
        default=3,
        help="Training years before each out-of-sample test. Default: 3.",
    )
    parser.add_argument(
        "--walk-forward-test-years",
        type=int,
        default=1,
        help="Out-of-sample test years per step. Default: 1.",
    )
    parser.add_argument(
        "--walk-forward-metric",
        default="sharpe",
        help="Training metric used to choose tolerance, for example sharpe or total_return.",
    )
    parser.add_argument(
        "--tolerance-values",
        default="0,0.02,0.05,0.10,0.15,0.20,0.25",
        help="Comma-separated tolerances for --tolerance-sweep.",
    )
    parser.add_argument(
        "--point-in-time-universe",
        action="store_true",
        help="Confirm that --tickers is a point-in-time universe for the requested start date.",
    )
    parser.add_argument(
        "--sort-by",
        default="total_return",
        help="Metric for --grid-search ranking, for example total_return, sharpe, max_drawdown.",
    )
    args = parser.parse_args()
    args.tolerance_band_map = parse_tolerance_band_map(args.tolerance_bands)

    market = normalize_china_symbol(args.market)
    defensive_asset = normalize_china_symbol(args.defensive_asset) if args.defensive_asset else None
    universe_source = "current China leader basket"
    point_in_time_snapshot = None
    if args.universe_file is not None:
        snapshot_date = args.snapshot_date or args.start
        if snapshot_date is None:
            raise SystemExit("--universe-file requires --snapshot-date or --start.")
        universe, point_in_time_snapshot = load_point_in_time_universe(
            args.universe_file,
            snapshot_date,
        )
        universe_source = f"point-in-time file snapshot {point_in_time_snapshot.date()}"
        args.point_in_time_universe = True
    elif args.etf_preset is not None:
        universe = CHINA_ETF_PRESETS[args.etf_preset]
        universe_source = f"ETF preset {args.etf_preset}"
        args.point_in_time_universe = True
    else:
        universe = parse_universe(args.tickers)
    if defensive_asset and defensive_asset != market and defensive_asset not in universe:
        universe = [*universe, defensive_asset]
    tickers = [market, *[ticker for ticker in universe if ticker != market]]
    survivor_universe = not args.synthetic and not args.point_in_time_universe

    if args.synthetic:
        closes = make_synthetic_rotation_data()
        market = "510300.SS"
        data_source = "synthetic China rotation data"
        universe_source = "synthetic generated universe"
    else:
        closes = load_yfinance_closes(
            tickers=tickers,
            period=args.period,
            interval=args.interval,
            start=args.start,
            end=args.end,
            cache_dir=args.yf_cache_dir,
        )
        if market not in closes:
            raise SystemExit(f"Market proxy {market} was not downloaded successfully.")
        data_source = f"yfinance China rotation: {len(closes.columns) - 1} stocks vs {market}"

    quality_data = load_quality_csv(args.quality_csv)
    quality_options = {
        "quality_min_roe": args.quality_min_roe,
        "quality_max_debt_to_equity": args.quality_max_debt_to_equity,
        "quality_min_profit_margin": args.quality_min_profit_margin,
        "quality_require_positive_fcf": args.quality_require_positive_fcf,
    }

    if args.tolerance_sweep:
        result = run_tolerance_sweep(
            closes,
            market,
            args,
            parse_tolerance_values(args.tolerance_values),
            quality_data=quality_data,
            quality_options=quality_options,
        )
        display_cols = [
            "rebalance_tolerance",
            "final_equity",
            "total_return",
            "excess_vs_equal_weight",
            "sharpe",
            "max_drawdown",
            "turnover",
            "turnover_saved_vs_0",
            "equity_added_vs_0",
        ]
        print(f"Data source: {data_source}")
        print(f"Rows: {len(closes)} | Start: {closes.index[0].date()} | End: {closes.index[-1].date()}")
        print(f"Universe source: {universe_source}")
        print(f"Initial capital: {args.initial_capital:,.2f}")
        if survivor_universe:
            print("Bias warning: current-survivor universe, not point-in-time constituents.")
        print("Ranked by: final_equity")
        print()
        print(result.loc[:, display_cols].round(4).to_string(index=False))
        return

    if args.walk_forward_tolerance:
        base_config = RotationConfig(
            allocation_mode=args.allocation,
            top_n=args.top_n,
            rebalance=args.rebalance,
            rebalance_tolerance=args.rebalance_tolerance,
            rebalance_tolerance_by_asset=args.tolerance_band_map,
            rebalance_to=args.rebalance_to,
            core_weight=args.core_weight,
            momentum_top_n=args.momentum_top_n,
            momentum_score_mode=args.momentum_score_mode,
            momentum_confirmation=args.momentum_confirmation,
            defensive_asset=defensive_asset,
            breadth_adjusted=args.breadth_adjusted,
            breadth_full_threshold=args.breadth_full_threshold,
            breadth_partial_threshold=args.breadth_partial_threshold,
            breadth_partial_exposure=args.breadth_partial_exposure,
            max_gross_exposure=args.max_exposure,
            risk_mode=args.risk_mode,
            **quality_options,
        )
        result = run_walk_forward_tolerance(
            closes,
            market,
            base_config,
            initial_equity=args.initial_capital,
            values=parse_tolerance_values(args.tolerance_values),
            train_years=args.walk_forward_train_years,
            test_years=args.walk_forward_test_years,
            optimize_metric=args.walk_forward_metric,
            mode=args.walk_forward_mode,
            quality_data=quality_data,
        )
        print(f"Data source: {data_source}")
        print(f"Rows: {len(closes)} | Start: {closes.index[0].date()} | End: {closes.index[-1].date()}")
        print(f"Universe source: {universe_source}")
        print(f"Initial capital: {args.initial_capital:,.2f}")
        print(
            "Walk-forward tolerance: "
            f"{args.walk_forward_mode}, train={args.walk_forward_train_years}y, "
            f"test={args.walk_forward_test_years}y, metric={args.walk_forward_metric}"
        )
        if survivor_universe:
            print("Bias warning: current-survivor universe, not point-in-time constituents.")
        if result.empty:
            print("No walk-forward windows were available. Use a longer date range.")
        else:
            display_cols = [
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "chosen_tolerance",
                "train_score",
                "oos_total_return",
                "oos_sharpe",
                "oos_max_drawdown",
                "oos_turnover",
                "oos_final_equity",
            ]
            print()
            print(result.loc[:, display_cols].round(4).to_string(index=False))
            print()
            final_equity = float(result["oos_final_equity"].iloc[-1])
            print(f"Compounded OOS final equity: {final_equity:,.2f}")
            print(f"Compounded OOS total return: {final_equity / args.initial_capital - 1.0:.4f}")
        return

    if args.grid_search:
        result = run_grid_search(
            closes,
            market,
                initial_equity=args.initial_capital,
                quality_data=quality_data,
                quality_options=quality_options,
                rebalance_tolerance=args.rebalance_tolerance,
                rebalance_tolerance_by_asset=args.tolerance_band_map,
            sort_by=args.sort_by,
        )
        display_cols = [
            "allocation",
            "risk_mode",
            "top_n",
            "core_weight",
            "final_equity",
            "total_return",
            "excess_vs_equal",
            "sharpe",
            "max_drawdown",
            "average_exposure",
            "turnover",
        ]
        print(f"Data source: {data_source}")
        print(f"Rows: {len(closes)} | Start: {closes.index[0].date()} | End: {closes.index[-1].date()}")
        print(f"Universe source: {universe_source}")
        print(f"Initial capital: {args.initial_capital:,.2f}")
        if survivor_universe:
            print("Bias warning: current-survivor universe, not point-in-time constituents.")
        print(f"Ranked by: {args.sort_by}")
        print()
        print(result.loc[:, display_cols].head(20).round(4).to_string(index=False))
        return

    config = RotationConfig(
        allocation_mode=args.allocation,
        top_n=args.top_n,
        rebalance=args.rebalance,
        rebalance_tolerance=args.rebalance_tolerance,
        rebalance_tolerance_by_asset=args.tolerance_band_map,
        rebalance_to=args.rebalance_to,
        core_weight=args.core_weight,
        momentum_top_n=args.momentum_top_n,
        momentum_score_mode=args.momentum_score_mode,
        momentum_confirmation=args.momentum_confirmation,
        defensive_asset=defensive_asset,
        breadth_adjusted=args.breadth_adjusted,
        breadth_full_threshold=args.breadth_full_threshold,
        breadth_partial_threshold=args.breadth_partial_threshold,
        breadth_partial_exposure=args.breadth_partial_exposure,
        max_gross_exposure=args.max_exposure,
        risk_mode=args.risk_mode,
        **quality_options,
    )
    curve, weights, metrics = backtest_rotation(
        closes,
        market,
        config,
        initial_equity=args.initial_capital,
        quality_data=quality_data,
    )
    stock_returns = closes.drop(columns=[market]).iloc[-1] / closes.drop(columns=[market]).iloc[0] - 1.0
    best_stock = stock_returns.idxmax()
    worst_stock = stock_returns.idxmin()
    current_holdings = weights.iloc[-1]
    current_holdings = current_holdings[current_holdings > 0].sort_values(ascending=False)

    print(f"Data source: {data_source}")
    print(f"Rows: {len(closes)} | Start: {closes.index[0].date()} | End: {closes.index[-1].date()}")
    print(f"Universe source: {universe_source}")
    print(
        f"Allocation: {config.allocation_mode} | Risk mode: {config.risk_mode} | "
        f"Rebalance to: {config.rebalance_to}"
    )
    if survivor_universe:
        print("Bias warning: current-survivor universe, not point-in-time constituents.")
    print(
        f"Initial capital: {args.initial_capital:,.2f} | "
        f"Final equity: {curve['equity'].iloc[-1]:,.2f}"
    )
    print(f"Universe: {', '.join([column for column in closes.columns if column != market])}")
    print(f"Best stock buy-hold: {best_stock} ({stock_returns.loc[best_stock]:.4f})")
    print(f"Worst stock buy-hold: {worst_stock} ({stock_returns.loc[worst_stock]:.4f})")
    print()
    print("Metrics")
    for key, value in metrics.items():
        print(f"{key:>22}: {value: .4f}")

    print("\nRecent portfolio state")
    recent = curve.loc[:, ["equity", "exposure", "turnover", "strategy_return"]].tail(12)
    print(recent.round(4).to_string())

    print("\nCurrent holdings")
    if current_holdings.empty:
        print("Cash")
    else:
        print(current_holdings.round(4).to_string())


if __name__ == "__main__":
    main()
