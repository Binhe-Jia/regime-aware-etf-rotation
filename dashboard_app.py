from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from china_market_run import normalize_china_symbol
from china_rotation_run import (
    CHINA_ETF_PRESETS,
    DEFAULT_UNIVERSE as DEFAULT_CHINA_UNIVERSE,
    RotationConfig,
    backtest_rotation,
    load_yfinance_closes,
)
from us_rotation_run import DEFAULT_US_UNIVERSE, ETF_PRESETS as US_ETF_PRESETS


st.set_page_config(
    page_title="Equal-Weight Rotation Dashboard",
    layout="wide",
)


US_PRESETS = {
    "US leader basket": DEFAULT_US_UNIVERSE,
    "US sectors ETF basket": US_ETF_PRESETS["us_sectors"],
    "US asset-class ETF basket": US_ETF_PRESETS["us_asset_classes"],
}

CHINA_PRESETS = {
    "China leader basket": DEFAULT_CHINA_UNIVERSE,
    "China broad ETF basket": CHINA_ETF_PRESETS["china_broad_etfs"],
    "China style ETF basket": CHINA_ETF_PRESETS["china_style_etfs"],
}


def format_pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def format_money(value: float, currency: str) -> str:
    return f"{currency}{value:,.0f}"


def parse_custom_universe(text: str, market: str, is_china: bool) -> list[str]:
    if not text.strip():
        return []
    raw = [item.strip() for item in text.split(",") if item.strip()]
    if is_china:
        tickers = [normalize_china_symbol(item) for item in raw]
    else:
        tickers = [item.upper() for item in raw]
    return [ticker for ticker in tickers if ticker != market]


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_closes(
    tickers: tuple[str, ...],
    period: str,
    interval: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    return load_yfinance_closes(
        tickers=list(tickers),
        period=period,
        interval=interval,
        start=start or None,
        end=end or None,
        cache_dir=Path(".yfinance-cache"),
    )


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def make_tolerance_sweep(
    closes: pd.DataFrame,
    market: str,
    base_config: RotationConfig,
    initial_capital: float,
    values: list[float],
) -> pd.DataFrame:
    rows = []
    baseline_equity = None
    baseline_turnover = None
    for tolerance in values:
        config = replace(base_config, rebalance_tolerance=tolerance)
        curve, _, metrics = backtest_rotation(
            closes,
            market,
            config,
            initial_equity=initial_capital,
        )
        final_equity = float(curve["equity"].iloc[-1])
        if tolerance == 0.0:
            baseline_equity = final_equity
            baseline_turnover = metrics["turnover"]
        rows.append(
            {
                "tolerance": tolerance,
                "final_equity": final_equity,
                "total_return": metrics["total_return"],
                "excess_vs_equal_weight": metrics["excess_vs_equal_weight"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "turnover": metrics["turnover"],
            }
        )

    result = pd.DataFrame(rows)
    if baseline_equity is not None:
        result["equity_added_vs_0"] = result["final_equity"] - baseline_equity
    if baseline_turnover is not None:
        result["turnover_saved_vs_0"] = baseline_turnover - result["turnover"]
    return result.sort_values("final_equity", ascending=False).reset_index(drop=True)


def make_volatility_tolerance_map(
    closes: pd.DataFrame,
    market: str,
    base_tolerance: float,
    min_tolerance: float,
    max_tolerance: float,
) -> dict[str, float]:
    returns = closes.drop(columns=[market]).pct_change().dropna(how="all")
    annual_vol = returns.std() * np.sqrt(252)
    median_vol = float(annual_vol.median())
    if not np.isfinite(median_vol) or median_vol <= 0:
        return {asset: base_tolerance for asset in annual_vol.index}

    tolerance = base_tolerance * median_vol / annual_vol.replace(0.0, np.nan)
    tolerance = tolerance.replace([np.inf, -np.inf], np.nan).fillna(base_tolerance)
    tolerance = tolerance.clip(lower=min_tolerance, upper=max_tolerance)
    return tolerance.to_dict()


def metrics_from_returns(
    returns: pd.Series,
    equity: pd.Series,
    initial_capital: float,
) -> dict[str, float]:
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown(equity).min()),
    }


def make_policy_comparison(
    closes: pd.DataFrame,
    market: str,
    current_curve: pd.DataFrame,
    current_metrics: dict[str, float],
    current_config: RotationConfig,
    initial_capital: float,
) -> pd.DataFrame:
    strict_config = replace(
        current_config,
        rebalance_tolerance=0.0,
        rebalance_tolerance_by_asset=None,
        rebalance_to="target",
    )
    strict_curve, _, strict_metrics = backtest_rotation(
        closes,
        market,
        strict_config,
        initial_equity=initial_capital,
    )
    daily_metrics = metrics_from_returns(
        current_curve["equal_weight_return"],
        current_curve["equal_weight_equity"],
        initial_capital,
    )
    market_metrics = metrics_from_returns(
        current_curve["market_return"],
        current_curve["market_equity"],
        initial_capital,
    )
    rows = [
        {
            "policy": "Daily frictionless equal-weight",
            "final_equity": daily_metrics["final_equity"],
            "total_return": daily_metrics["total_return"],
            "excess_vs_daily_equal": 0.0,
            "sharpe": daily_metrics["sharpe"],
            "max_drawdown": daily_metrics["max_drawdown"],
            "turnover": np.nan,
        },
        {
            "policy": "Calendar rebalance with costs",
            "final_equity": float(strict_curve["equity"].iloc[-1]),
            "total_return": strict_metrics["total_return"],
            "excess_vs_daily_equal": strict_metrics["total_return"] - daily_metrics["total_return"],
            "sharpe": strict_metrics["sharpe"],
            "max_drawdown": strict_metrics["max_drawdown"],
            "turnover": strict_metrics["turnover"],
        },
        {
            "policy": "Selected tolerance policy",
            "final_equity": float(current_curve["equity"].iloc[-1]),
            "total_return": current_metrics["total_return"],
            "excess_vs_daily_equal": current_metrics["total_return"] - daily_metrics["total_return"],
            "sharpe": current_metrics["sharpe"],
            "max_drawdown": current_metrics["max_drawdown"],
            "turnover": current_metrics["turnover"],
        },
        {
            "policy": "Market proxy",
            "final_equity": market_metrics["final_equity"],
            "total_return": market_metrics["total_return"],
            "excess_vs_daily_equal": market_metrics["total_return"] - daily_metrics["total_return"],
            "sharpe": market_metrics["sharpe"],
            "max_drawdown": market_metrics["max_drawdown"],
            "turnover": np.nan,
        },
    ]
    return pd.DataFrame(rows)


def make_dual_momentum_ablation(
    closes: pd.DataFrame,
    market: str,
    initial_capital: float,
    base_config: RotationConfig,
) -> pd.DataFrame:
    default_defensive = (
        base_config.defensive_asset
        if base_config.defensive_asset and base_config.defensive_asset in closes.columns
        else None
    )
    variants = [
        {
            "variant": "A original top 1",
            "top_n": 1,
            "score": "raw",
            "confirmation": "none",
        },
        {"variant": "B top 2", "top_n": 2, "score": "raw", "confirmation": "none"},
        {"variant": "C top 3", "top_n": 3, "score": "raw", "confirmation": "none"},
        {
            "variant": "D top 1 + MA200",
            "top_n": 1,
            "score": "raw",
            "confirmation": "ma200",
        },
        {
            "variant": "E top 1 risk-adjusted",
            "top_n": 1,
            "score": "risk_adjusted",
            "confirmation": "none",
        },
        {"variant": "G top 2 + MA200", "top_n": 2, "score": "raw", "confirmation": "ma200"},
        {
            "variant": "H top 1 + breadth",
            "top_n": 1,
            "score": "raw",
            "confirmation": "none",
            "breadth_adjusted": True,
            "defensive_asset": None,
        },
    ]
    if default_defensive:
        variants.append(
            {
                "variant": "I top 1 + breadth + defensive",
                "top_n": 1,
                "score": "raw",
                "confirmation": "none",
                "breadth_adjusted": True,
                "defensive_asset": default_defensive,
            }
        )
    rows = []
    for variant in variants:
        config = replace(
            base_config,
            allocation_mode="dual_momentum",
            momentum_top_n=variant["top_n"],
            momentum_score_mode=variant["score"],
            momentum_confirmation=variant["confirmation"],
            breadth_adjusted=bool(variant.get("breadth_adjusted", False)),
            defensive_asset=variant.get("defensive_asset"),
        )
        curve, _, metrics = backtest_rotation(
            closes,
            market,
            config,
            initial_equity=initial_capital,
        )
        rows.append(
            {
                "variant": variant["variant"],
                "top_n": variant["top_n"],
                "score": variant["score"],
                "confirmation": variant["confirmation"],
                "breadth_adjusted": bool(variant.get("breadth_adjusted", False)),
                "defensive_asset": variant.get("defensive_asset") or "cash",
                "final_equity": float(curve["equity"].iloc[-1]),
                "total_return": metrics["total_return"],
                "excess_vs_equal_weight": metrics["excess_vs_equal_weight"],
                "vs_market": metrics["total_return"] - metrics["market_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "turnover": metrics["turnover"],
            }
        )
    return pd.DataFrame(rows).sort_values("final_equity", ascending=False).reset_index(drop=True)


st.title("Equal-Weight Rotation Dashboard")

with st.sidebar:
    st.header("Portfolio")
    market_region = st.radio(
        "Market",
        ["China A-share", "U.S."],
        index=1,
        horizontal=True,
    )
    is_china = market_region == "China A-share"
    currency = "RMB " if is_china else "$"
    preset_map = CHINA_PRESETS if is_china else US_PRESETS
    default_market = "510300.SS" if is_china else "SPY"

    preset_name = st.selectbox("Universe preset", list(preset_map), index=1 if is_china else 2)
    market = st.text_input("Market proxy", value=default_market)
    market = normalize_china_symbol(market) if is_china else market.upper()

    custom_tickers = st.text_area(
        "Custom tickers",
        placeholder="Optional comma-separated override",
        height=80,
    )
    custom_universe = parse_custom_universe(custom_tickers, market, is_china)
    universe = custom_universe or [ticker for ticker in preset_map[preset_name] if ticker != market]

    st.header("Backtest")
    start = st.text_input("Start date", value="2020-01-01" if is_china else "2018-01-01")
    end = st.text_input("End date", value="")
    period = st.selectbox("Fallback period", ["max", "10y", "5y", "2y", "1y"], index=0)
    initial_capital = st.number_input(
        "Initial capital",
        min_value=1_000.0,
        value=50_000.0,
        step=5_000.0,
    )

    st.header("Strategy")
    allocation = st.selectbox(
        "Allocation",
        [
            "equal_weight",
            "ivy_trend_filter",
            "dual_momentum",
            "protective_dual_momentum",
            "top_n",
            "core_satellite",
            "inverse_vol",
            "trend_equal_weight",
            "score_weighted",
        ],
        index=2,
    )
    if allocation in {"dual_momentum", "protective_dual_momentum"}:
        st.caption(
            "Dual momentum: rank assets by momentum, hold the top positive candidates, "
            "or use the defensive/cash sleeve when none qualify."
        )
        if allocation == "protective_dual_momentum":
            st.caption(
                "Protective dual momentum keeps top-1 concentration but cuts exposure when universe breadth weakens."
            )
    elif allocation == "ivy_trend_filter":
        st.caption("Ivy trend filter: each asset keeps its sleeve only when above its 200-day average.")
    risk_mode = st.selectbox("Risk mode", ["none", "market"], index=0)
    rebalance = st.selectbox("Rebalance", ["M", "W"], index=0)
    rebalance_tolerance = st.slider("Rebalance tolerance", 0.0, 0.30, 0.10, 0.01)
    rebalance_to = st.selectbox(
        "Rebalance destination",
        ["target", "corridor"],
        index=0,
        help="Target trades fully back to desired weights; corridor trades only to the tolerance boundary.",
    )
    use_volatility_tolerance = st.checkbox(
        "Use volatility-adjusted tolerance",
        value=False,
        help="High-volatility assets get tighter bands; lower-volatility assets get wider bands.",
    )
    min_asset_tolerance = st.slider("Minimum asset tolerance", 0.0, 0.15, 0.02, 0.01)
    max_asset_tolerance = st.slider("Maximum asset tolerance", 0.02, 0.30, 0.15, 0.01)
    top_n = st.slider("Top N", 1, max(1, min(12, len(universe))), min(4, max(1, len(universe))))
    momentum_top_n = st.slider(
        "Momentum Top N",
        1,
        max(1, min(6, len(universe))),
        1,
        help="For dual momentum, hold the top N positive-momentum assets instead of only top 1.",
    )
    momentum_score_mode = st.selectbox(
        "Momentum score",
        ["raw", "risk_adjusted", "blended_risk_adjusted"],
        index=0,
        help="Raw return, 12-month return per unit volatility, or blended 6/12-month risk-adjusted momentum.",
    )
    momentum_confirmation = st.selectbox(
        "Momentum confirmation",
        ["none", "ma200", "fast", "aggressive"],
        index=0,
        help="Optional recent-trend and moving-average confirmation for dual momentum.",
    )
    breadth_adjusted = st.checkbox(
        "Breadth-adjusted exposure",
        value=allocation == "protective_dual_momentum",
        help="Use the share of assets with positive 12-month momentum to scale exposure.",
    )
    breadth_full_threshold = st.slider("Full-risk breadth threshold", 0.0, 1.0, 0.60, 0.05)
    breadth_partial_threshold = st.slider("Partial-risk breadth threshold", 0.0, 1.0, 0.40, 0.05)
    breadth_partial_exposure = st.slider("Partial-risk exposure", 0.0, 1.0, 0.50, 0.05)
    defensive_asset = st.text_input(
        "Defensive asset",
        value="" if is_china else "BIL",
        help="Optional asset to hold when no dual-momentum candidate is eligible.",
    )
    st.subheader("Regime Filter")
    regime_filter = st.checkbox(
        "Use regime filter",
        value=False,
        help="Detect stress regimes with PCA + k-means + classifier and reduce risk exposure.",
    )
    regime_stress_exposure = st.slider("Stress exposure", 0.0, 1.0, 0.50, 0.05)
    regime_defensive_asset = st.text_input(
        "Regime defensive asset",
        value="" if is_china else "BIL",
        help="Optional sleeve for capital removed from risk assets during detected stress regimes.",
    )
    regime_train_years = st.slider("Regime train years", 1, 8, 3, 1)
    regime_components = st.slider("Regime PCA components", 1, 6, 3, 1)
    regime_signal_lag = st.slider(
        "Regime signal lag",
        0,
        5,
        1,
        1,
        help="Lag the regime signal to reduce same-day lookahead.",
    )
    core_weight = st.slider("Core weight", 0.0, 1.0, 0.70, 0.05)
    transaction_cost_bps = st.number_input("Transaction cost bps", 0.0, 100.0, 6.0, 1.0)
    slippage_bps = st.number_input("Slippage bps", 0.0, 100.0, 8.0, 1.0)
    run_sweep = st.checkbox("Show tolerance sweep", value=True)

defensive_symbol = (
    normalize_china_symbol(defensive_asset) if is_china else defensive_asset.upper()
) if defensive_asset else None
regime_defensive_symbol = (
    normalize_china_symbol(regime_defensive_asset)
    if is_china
    else regime_defensive_asset.upper()
) if regime_defensive_asset else None
for extra_symbol in [defensive_symbol, regime_defensive_symbol]:
    if extra_symbol and extra_symbol != market and extra_symbol not in universe:
        universe = [*universe, extra_symbol]

tickers = [market, *[ticker for ticker in universe if ticker != market]]
survivor_biased = "leader basket" in preset_name.lower() and not custom_universe

base_config = RotationConfig(
    allocation_mode=allocation,
    top_n=top_n,
    rebalance=rebalance,
    rebalance_tolerance=rebalance_tolerance,
    rebalance_to=rebalance_to,
    core_weight=core_weight,
    momentum_top_n=momentum_top_n,
    momentum_score_mode=momentum_score_mode,
    momentum_confirmation=momentum_confirmation,
    defensive_asset=defensive_symbol,
    breadth_adjusted=breadth_adjusted,
    breadth_full_threshold=breadth_full_threshold,
    breadth_partial_threshold=breadth_partial_threshold,
    breadth_partial_exposure=breadth_partial_exposure,
    regime_filter=regime_filter,
    regime_stress_exposure=regime_stress_exposure,
    regime_defensive_asset=regime_defensive_symbol,
    regime_train_years=regime_train_years,
    regime_n_components=regime_components,
    regime_signal_lag=regime_signal_lag,
    max_gross_exposure=1.0,
    risk_mode=risk_mode,
    transaction_cost_bps=transaction_cost_bps,
    slippage_bps=slippage_bps,
)

if survivor_biased:
    st.warning(
        "This preset uses a current leader basket. Treat long backtests as behavior checks, "
        "not point-in-time proof."
    )

try:
    with st.spinner("Downloading prices and running backtest..."):
        closes = cached_closes(tuple(tickers), period, "1d", start, end)
        if market not in closes:
            st.error(f"Market proxy {market} was not downloaded successfully.")
            st.stop()
        config = base_config
        tolerance_map = None
        if use_volatility_tolerance:
            tolerance_map = make_volatility_tolerance_map(
                closes,
                market,
                base_tolerance=rebalance_tolerance,
                min_tolerance=min_asset_tolerance,
                max_tolerance=max_asset_tolerance,
            )
            config = replace(base_config, rebalance_tolerance_by_asset=tolerance_map)
        curve, weights, metrics = backtest_rotation(
            closes,
            market,
            config,
            initial_equity=initial_capital,
        )
except (Exception, SystemExit) as exc:
    st.error(str(exc))
    st.stop()
    raise SystemExit(1) from exc

final_equity = float(curve["equity"].iloc[-1])
start_date = closes.index[0].date()
end_date = closes.index[-1].date()
stock_columns = [column for column in closes.columns if column != market]

st.caption(
    f"{market_region} | {preset_name} | {len(stock_columns)} assets vs {market} | "
    f"{start_date} to {end_date}"
)
if config.regime_filter:
    st.caption(
        f"Regime filter active | stress exposure {config.regime_stress_exposure:.0%} | "
        f"defensive asset {config.regime_defensive_asset or 'cash'} | "
        f"stress days {metrics.get('regime_stress_fraction', 0.0):.1%}"
    )

metric_cols = st.columns(6)
metric_cols[0].metric("Final equity", format_money(final_equity, currency))
metric_cols[1].metric("Total return", format_pct(metrics["total_return"]))
metric_cols[2].metric("Vs equal-weight", format_pct(metrics["excess_vs_equal_weight"]))
metric_cols[3].metric("Vs market", format_pct(metrics["total_return"] - metrics["market_return"]))
metric_cols[4].metric("Sharpe", f"{metrics['sharpe']:.2f}")
metric_cols[5].metric("Max drawdown", format_pct(metrics["max_drawdown"]))

tab_overview, tab_compare, tab_ablation, tab_risk, tab_holdings, tab_sweep = st.tabs(
    ["Overview", "Policy Compare", "Dual Momentum Tests", "Risk", "Holdings", "Tolerance Sweep"]
)

with tab_overview:
    equity_view = curve.loc[:, ["equity", "market_equity", "equal_weight_equity"]].rename(
        columns={
            "equity": "Strategy",
            "market_equity": "Market proxy",
            "equal_weight_equity": "Frictionless equal-weight",
        }
    )
    st.subheader("Equity Curve")
    st.line_chart(equity_view)

    recent_returns = curve.loc[:, ["strategy_return", "market_return", "equal_weight_return"]].tail(252)
    recent_returns = (1.0 + recent_returns).cumprod() - 1.0
    recent_returns = recent_returns.rename(
        columns={
            "strategy_return": "Strategy",
            "market_return": "Market proxy",
            "equal_weight_return": "Frictionless equal-weight",
        }
    )
    st.subheader("Trailing One-Year Growth")
    st.line_chart(recent_returns)

with tab_risk:
    risk_cols = st.columns(2)
    drawdown_view = pd.DataFrame(
        {
            "Strategy": drawdown(curve["equity"]),
            "Market proxy": drawdown(curve["market_equity"]),
            "Frictionless equal-weight": drawdown(curve["equal_weight_equity"]),
        }
    )
    with risk_cols[0]:
        st.subheader("Drawdown")
        st.line_chart(drawdown_view)
    with risk_cols[1]:
        st.subheader("Exposure and Turnover")
        risk_lines = curve.loc[:, ["exposure", "turnover"]].copy()
        if "regime_stress" in curve:
            risk_lines["regime_stress"] = curve["regime_stress"]
        st.line_chart(risk_lines)

    summary = pd.Series(metrics).rename("value").to_frame()
    st.dataframe(summary.style.format("{:.4f}"), use_container_width=True)

with tab_compare:
    st.subheader("Execution Policy Comparison")
    policy_comparison = make_policy_comparison(
        closes,
        market,
        curve,
        metrics,
        config,
        initial_capital,
    )
    st.bar_chart(policy_comparison.set_index("policy")["final_equity"])
    numeric_columns = policy_comparison.select_dtypes(include=[np.number]).columns
    st.dataframe(
        policy_comparison.style.format("{:.4f}", subset=numeric_columns),
        use_container_width=True,
    )

with tab_ablation:
    st.subheader("Dual Momentum Ablation")
    ablation = make_dual_momentum_ablation(closes, market, initial_capital, base_config)
    st.bar_chart(ablation.set_index("variant")["final_equity"])
    numeric_columns = ablation.select_dtypes(include=[np.number]).columns
    st.dataframe(ablation.style.format("{:.4f}", subset=numeric_columns), use_container_width=True)

with tab_holdings:
    current_holdings = weights.iloc[-1]
    current_holdings = current_holdings[current_holdings > 0].sort_values(ascending=False)
    if current_holdings.empty:
        st.info("Current model position is cash.")
    else:
        st.subheader("Current Weights")
        st.bar_chart(current_holdings.rename("weight"))

    returns = closes.drop(columns=[market]).iloc[-1] / closes.drop(columns=[market]).iloc[0] - 1.0
    asset_table = pd.DataFrame(
        {
            "current_weight": weights.iloc[-1].reindex(stock_columns).fillna(0.0),
            "buy_hold_return": returns.reindex(stock_columns),
        }
    ).sort_values("current_weight", ascending=False)
    st.dataframe(asset_table.style.format("{:.4f}"), use_container_width=True)
    if use_volatility_tolerance and tolerance_map:
        tolerance_table = (
            pd.Series(tolerance_map, name="asset_tolerance")
            .reindex(stock_columns)
            .to_frame()
            .sort_values("asset_tolerance")
        )
        st.subheader("Asset-Specific Tolerance Bands")
        st.dataframe(tolerance_table.style.format("{:.4f}"), use_container_width=True)

with tab_sweep:
    if run_sweep:
        values = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
        sweep = make_tolerance_sweep(closes, market, config, initial_capital, values)
        st.subheader("Rebalance Tolerance Sweep")
        chart_data = sweep.sort_values("tolerance").set_index("tolerance")
        st.bar_chart(chart_data["final_equity"])
        st.dataframe(sweep.style.format("{:.4f}"), use_container_width=True)
    else:
        st.info("Enable tolerance sweep in the sidebar.")
