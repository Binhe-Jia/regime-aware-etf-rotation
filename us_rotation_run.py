from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from china_rotation_run import (
    RotationConfig,
    backtest_rotation,
    load_yfinance_closes,
    run_grid_search,
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


def parse_universe(text: str | None) -> list[str]:
    if not text:
        return DEFAULT_US_UNIVERSE
    tickers = [item.strip().upper() for item in text.split(",") if item.strip()]
    return tickers or DEFAULT_US_UNIVERSE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the US leader basket equal-weight/tolerance strategy."
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated US tickers. Default: frozen US leader basket.",
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

    market = args.market.upper()
    universe = parse_universe(args.tickers)
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
    if args.grid_search:
        result = run_grid_search(
            closes,
            market,
            initial_equity=args.initial_capital,
            rebalance_tolerance=args.rebalance_tolerance,
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
        core_weight=args.core_weight,
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
    print(f"Allocation: {config.allocation_mode} | Risk mode: {config.risk_mode}")
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
