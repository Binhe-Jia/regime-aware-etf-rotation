from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from china_rotation_run import (
    RotationConfig,
    backtest_rotation,
    load_yfinance_closes,
    parse_tolerance_band_map,
    parse_tolerance_values,
    run_grid_search,
    run_walk_forward_tolerance,
)


DEFAULT_US_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "BRK-B",
    "JPM",
    "LLY",
    "AVGO",
    "TSLA",
    "COST",
]

ETF_PRESETS = {
    "us_sectors": [
        "XLK",
        "XLF",
        "XLV",
        "XLY",
        "XLI",
        "XLE",
        "XLP",
        "XLU",
        "XLB",
        "XLRE",
        "XLC",
    ],
    "us_asset_classes": [
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "EEM",
        "TLT",
        "IEF",
        "GLD",
        "DBC",
        "VNQ",
    ],
}


def parse_universe(text: str | None) -> list[str]:
    if not text:
        return DEFAULT_US_UNIVERSE
    tickers = [item.strip().upper() for item in text.split(",") if item.strip()]
    return tickers or DEFAULT_US_UNIVERSE


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
        .str.upper()
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


def run_tolerance_sweep(
    closes: pd.DataFrame,
    market: str,
    args: argparse.Namespace,
    values: list[float],
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
            defensive_asset=args.defensive_asset.upper() if args.defensive_asset else None,
            breadth_adjusted=args.breadth_adjusted,
            breadth_full_threshold=args.breadth_full_threshold,
            breadth_partial_threshold=args.breadth_partial_threshold,
            breadth_partial_exposure=args.breadth_partial_exposure,
            regime_filter=args.regime_filter,
            regime_stress_exposure=args.regime_stress_exposure,
            regime_defensive_asset=args.regime_defensive_asset.upper()
            if args.regime_defensive_asset
            else None,
            regime_train_years=args.regime_train_years,
            regime_n_components=args.regime_components,
            regime_signal_lag=args.regime_signal_lag,
            max_gross_exposure=args.max_exposure,
            risk_mode=args.risk_mode,
        )
        curve, _, metrics = backtest_rotation(
            closes,
            market,
            config,
            initial_equity=args.initial_capital,
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


def run_dual_momentum_ablation(
    closes: pd.DataFrame,
    market: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    variants = [
        {
            "variant": "A_original_top1",
            "momentum_top_n": 1,
            "momentum_score_mode": "raw",
            "momentum_confirmation": "none",
        },
        {
            "variant": "B_top2",
            "momentum_top_n": 2,
            "momentum_score_mode": "raw",
            "momentum_confirmation": "none",
        },
        {
            "variant": "C_top3",
            "momentum_top_n": 3,
            "momentum_score_mode": "raw",
            "momentum_confirmation": "none",
        },
        {
            "variant": "D_top1_ma200",
            "momentum_top_n": 1,
            "momentum_score_mode": "raw",
            "momentum_confirmation": "ma200",
        },
        {
            "variant": "E_top1_risk_adjusted",
            "momentum_top_n": 1,
            "momentum_score_mode": "risk_adjusted",
            "momentum_confirmation": "none",
        },
        {
            "variant": "G_top2_ma200",
            "momentum_top_n": 2,
            "momentum_score_mode": "raw",
            "momentum_confirmation": "ma200",
        },
        {
            "variant": "H_top1_breadth",
            "momentum_top_n": 1,
            "momentum_score_mode": "raw",
            "momentum_confirmation": "none",
            "breadth_adjusted": True,
            "defensive_asset": None,
        },
        {
            "variant": "I_top1_breadth_defensive",
            "momentum_top_n": 1,
            "momentum_score_mode": "raw",
            "momentum_confirmation": "none",
            "breadth_adjusted": True,
            "defensive_asset": args.defensive_asset.upper() if args.defensive_asset else None,
        },
    ]
    rows = []
    for variant in variants:
        variant_defensive = variant.get("defensive_asset", None)
        config = RotationConfig(
            allocation_mode="dual_momentum",
            rebalance=args.rebalance,
            rebalance_tolerance=args.rebalance_tolerance,
            rebalance_tolerance_by_asset=args.tolerance_band_map,
            rebalance_to=args.rebalance_to,
            core_weight=args.core_weight,
            momentum_top_n=variant["momentum_top_n"],
            momentum_score_mode=variant["momentum_score_mode"],
            momentum_confirmation=variant["momentum_confirmation"],
            defensive_asset=variant_defensive,
            breadth_adjusted=bool(variant.get("breadth_adjusted", False)),
            breadth_full_threshold=args.breadth_full_threshold,
            breadth_partial_threshold=args.breadth_partial_threshold,
            breadth_partial_exposure=args.breadth_partial_exposure,
            regime_filter=args.regime_filter,
            regime_stress_exposure=args.regime_stress_exposure,
            regime_defensive_asset=args.regime_defensive_asset.upper()
            if args.regime_defensive_asset
            else None,
            regime_train_years=args.regime_train_years,
            regime_n_components=args.regime_components,
            regime_signal_lag=args.regime_signal_lag,
            max_gross_exposure=args.max_exposure,
            risk_mode=args.risk_mode,
        )
        curve, _, metrics = backtest_rotation(
            closes,
            market,
            config,
            initial_equity=args.initial_capital,
        )
        rows.append(
            {
                **variant,
                "breadth_adjusted": bool(variant.get("breadth_adjusted", False)),
                "defensive_asset": variant_defensive or "cash",
                "final_equity": float(curve["equity"].iloc[-1]),
                **metrics,
                "vs_market": metrics["total_return"] - metrics["market_return"],
            }
        )
    return pd.DataFrame(rows).sort_values("final_equity", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the US leader basket equal-weight/tolerance strategy."
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated US tickers. Default: frozen US leader basket.",
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
        choices=sorted(ETF_PRESETS),
        default=None,
        help="Use a predefined ETF universe instead of the default current stock leaders.",
    )
    parser.add_argument("--market", default="SPY", help="Market proxy. Default: SPY.")
    parser.add_argument("--period", default="max", help="yfinance period. Default: max.")
    parser.add_argument("--interval", default="1d", help="yfinance interval. Default: 1d.")
    parser.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD.")
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
        ],
        default="equal_weight",
        help="Portfolio allocation mode. Default: equal_weight.",
    )
    parser.add_argument("--top-n", type=int, default=4, help="Top-N count. Default: 4.")
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
        "--core-weight",
        type=float,
        default=0.70,
        help="Equal-weight core sleeve for core_satellite mode. Default: 0.70.",
    )
    parser.add_argument(
        "--rebalance",
        choices=["M", "W"],
        default="M",
        help="Rebalance frequency. Default: monthly.",
    )
    parser.add_argument(
        "--rebalance-tolerance",
        type=float,
        default=0.10,
        help="Only rebalance if a holding drifts this far from target. Default: 0.10.",
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
        help="Optional asset-specific bands as ticker=value pairs, for example SPY=0.075,AAPL=0.05.",
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
        "--regime-filter",
        action="store_true",
        help="Use PCA/k-means/classifier regime detection to reduce exposure in stress regimes.",
    )
    parser.add_argument(
        "--regime-stress-exposure",
        type=float,
        default=0.50,
        help="Risk-asset exposure during detected stress regimes. Default: 0.50.",
    )
    parser.add_argument(
        "--regime-defensive-asset",
        default=None,
        help="Optional defensive ticker to hold with reduced risk exposure, for example BIL.",
    )
    parser.add_argument(
        "--regime-train-years",
        type=int,
        default=3,
        help="Years of prior data for each regime model retrain. Default: 3.",
    )
    parser.add_argument(
        "--regime-components",
        type=int,
        default=3,
        help="PCA components used by the regime model. Default: 3.",
    )
    parser.add_argument(
        "--regime-signal-lag",
        type=int,
        default=1,
        help="Days to lag regime signals to avoid same-day lookahead. Default: 1.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Starting account value for the backtest. Default: 100000.",
    )
    parser.add_argument(
        "--yf-cache-dir",
        type=Path,
        default=None,
        help="Optional yfinance cache directory. Default: .yfinance-cache.",
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
        "--dual-momentum-ablation",
        action="store_true",
        help="Compare original, top-N, MA200, and risk-adjusted dual momentum variants.",
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
        help="Metric for --grid-search ranking, for example total_return or sharpe.",
    )
    args = parser.parse_args()
    args.tolerance_band_map = parse_tolerance_band_map(
        args.tolerance_bands,
        normalizer=lambda value: value.upper(),
    )
    if args.dual_momentum_ablation and args.defensive_asset is None:
        args.defensive_asset = "BIL"

    market = args.market.upper()
    defensive_asset = args.defensive_asset.upper() if args.defensive_asset else None
    regime_defensive_asset = (
        args.regime_defensive_asset.upper() if args.regime_defensive_asset else None
    )
    universe_source = "current US leader basket"
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
        universe = ETF_PRESETS[args.etf_preset]
        universe_source = f"ETF preset {args.etf_preset}"
        args.point_in_time_universe = True
    else:
        universe = parse_universe(args.tickers)
    for extra_asset in [defensive_asset, regime_defensive_asset]:
        if extra_asset and extra_asset != market and extra_asset not in universe:
            universe = [*universe, extra_asset]

    tickers = [market, *[ticker for ticker in universe if ticker != market]]
    survivor_universe = not args.point_in_time_universe
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

    data_source = f"yfinance US rotation: {len(closes.columns) - 1} stocks vs {market}"
    if args.tolerance_sweep:
        result = run_tolerance_sweep(
            closes,
            market,
            args,
            parse_tolerance_values(args.tolerance_values),
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
            regime_filter=args.regime_filter,
            regime_stress_exposure=args.regime_stress_exposure,
            regime_defensive_asset=regime_defensive_asset,
            regime_train_years=args.regime_train_years,
            regime_n_components=args.regime_components,
            regime_signal_lag=args.regime_signal_lag,
            max_gross_exposure=args.max_exposure,
            risk_mode=args.risk_mode,
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

    if args.dual_momentum_ablation:
        result = run_dual_momentum_ablation(closes, market, args)
        display_cols = [
            "variant",
            "momentum_top_n",
            "momentum_score_mode",
            "momentum_confirmation",
            "breadth_adjusted",
            "defensive_asset",
            "final_equity",
            "total_return",
            "excess_vs_equal_weight",
            "vs_market",
            "sharpe",
            "max_drawdown",
            "turnover",
        ]
        print(f"Data source: {data_source}")
        print(f"Rows: {len(closes)} | Start: {closes.index[0].date()} | End: {closes.index[-1].date()}")
        print(f"Universe source: {universe_source}")
        print(f"Initial capital: {args.initial_capital:,.2f}")
        print("Ranked by: final_equity")
        print()
        print(result.loc[:, display_cols].round(4).to_string(index=False))
        return

    if args.grid_search:
        result = run_grid_search(
            closes,
            market,
            initial_equity=args.initial_capital,
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
        regime_filter=args.regime_filter,
        regime_stress_exposure=args.regime_stress_exposure,
        regime_defensive_asset=regime_defensive_asset,
        regime_train_years=args.regime_train_years,
        regime_n_components=args.regime_components,
        regime_signal_lag=args.regime_signal_lag,
        max_gross_exposure=args.max_exposure,
        risk_mode=args.risk_mode,
    )
    curve, weights, metrics = backtest_rotation(
        closes,
        market,
        config,
        initial_equity=args.initial_capital,
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
