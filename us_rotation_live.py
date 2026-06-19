from __future__ import annotations

import argparse
from pathlib import Path

from china_rotation_live import (
    LiveConfig,
    append_journal,
    build_live_orders,
    load_holdings,
    load_latest_market_data,
    parse_tolerance_bands,
)
from us_rotation_run import DEFAULT_US_UNIVERSE, parse_universe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manual rebalance orders for the frozen US leader basket."
    )
    parser.add_argument("--holdings", type=Path, required=True, help="CSV with ticker,shares rows.")
    parser.add_argument(
        "--universe",
        default=None,
        help="Optional comma-separated ticker universe. Defaults to frozen US leader basket.",
    )
    parser.add_argument(
        "--rebalance-tolerance",
        type=float,
        default=0.10,
        help="Maximum allowed absolute weight drift before rebalancing. Default: 0.10.",
    )
    parser.add_argument(
        "--rebalance-to",
        choices=["target", "corridor"],
        default="target",
        help="Suggest trades back to target or only to the tolerance corridor edge. Default: target.",
    )
    parser.add_argument(
        "--tolerance-bands",
        default=None,
        help="Optional asset-specific bands as ticker=value pairs, for example SPY=0.075,AAPL=0.05.",
    )
    parser.add_argument(
        "--allow-cash-drift",
        action="store_true",
        help="For corridor mode, do not offset net sells/buys with counter-trades.",
    )
    parser.add_argument(
        "--min-trade-value",
        type=float,
        default=25.0,
        help="Ignore suggested trades below this USD value. Default: 25.",
    )
    parser.add_argument(
        "--max-single-order-fraction",
        type=float,
        default=0.10,
        help="Cap each order to this fraction of portfolio value. Default: 0.10.",
    )
    parser.add_argument(
        "--lot-size",
        type=int,
        default=1,
        help="Round suggested shares down to this lot size. Default: 1.",
    )
    parser.add_argument(
        "--redistribute-skipped-cash",
        action="store_true",
        help="Use leftover cash from skipped/rounded buys to add valid buy lots within order and concentration caps.",
    )
    parser.add_argument(
        "--period",
        default="10d",
        help="yfinance lookback period for latest price and previous close. Default: 10d.",
    )
    parser.add_argument(
        "--yf-cache-dir",
        type=Path,
        default=None,
        help="Optional yfinance cache directory. Default: .yfinance-cache.",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=None,
        help="Optional CSV journal path to append this run and suggested orders.",
    )
    parser.add_argument("--note", default="", help="Optional journal note.")
    args = parser.parse_args()

    universe = parse_universe(args.universe) if args.universe else DEFAULT_US_UNIVERSE
    holdings, cash = load_holdings(args.holdings)
    market_data = load_latest_market_data(universe, args.period, args.yf_cache_dir)
    config = LiveConfig(
        rebalance_tolerance=args.rebalance_tolerance,
        tolerance_bands=parse_tolerance_bands(args.tolerance_bands, normalizer=lambda value: value.upper()),
        min_trade_value=args.min_trade_value,
        max_single_order_fraction=args.max_single_order_fraction,
        lot_size=args.lot_size,
        rebalance_to=args.rebalance_to,
        cash_neutral_corridor=not args.allow_cash_drift,
        redistribute_skipped_cash=args.redistribute_skipped_cash,
        price_limit_warning_pct=0.20,
    )
    current, orders, skipped, concentration_warnings, summary = build_live_orders(
        universe,
        holdings,
        cash,
        market_data,
        config,
    )
    append_journal(args.journal, summary, orders, args.note)

    print(f"Portfolio value: {summary['portfolio_value']:,.2f} USD")
    print(f"Cash: {summary['cash']:,.2f} USD ({summary['cash_pct']:.2%})")
    print(f"Cash drift: {summary['cash_drift']:.2%}")
    print(
        f"Projected cash after suggested orders: {summary['projected_cash']:,.2f} USD "
        f"({summary['projected_cash_pct']:.2%})"
    )
    print(f"Max drift: {summary['max_drift']:.2%}")
    print(f"Max tolerance breach: {summary['max_tolerance_breach']:.2%}")
    print(f"Rebalance required: {'Yes' if summary['rebalance_required'] else 'No'}")
    print(f"Rebalance to: {summary['rebalance_to']}")
    print(f"Estimated turnover: {summary['estimated_turnover']:.2%}")
    print(f"Estimated transaction/slippage cost: {summary['estimated_cost']:,.2f} USD")
    print(f"Largest position: {summary['largest_weight']:.2%}")
    print(f"Smallest position: {summary['smallest_weight']:.2%}")
    print(f"Projected largest position: {summary['projected_largest_weight']:.2%}")
    print(f"Projected smallest position: {summary['projected_smallest_weight']:.2%}")
    print(f"Concentration warnings: {summary['concentration_warning_count']}")

    print("\nCurrent weights")
    display_current = current.loc[
        :,
        [
            "shares",
            "price",
            "market_value",
            "current_weight",
            "target_weight",
            "trade_to_weight",
            "tolerance_band",
            "drift",
        ],
    ].copy()
    print(display_current.round(4).to_string())

    print("\nSuggested orders")
    if orders.empty:
        print("No trades suggested.")
    else:
        display_orders = orders.loc[
            :,
            [
                "ticker",
                "action",
                "current_weight",
                "target_weight",
                "trade_to_weight",
                "projected_weight_after_order",
                "tolerance_band",
                "drift",
                "price",
                "estimated_shares",
                "trade_value",
                "estimated_cost",
                "warnings",
            ],
        ].copy()
        print(display_orders.round(4).to_string(index=False))

    print("\nSkipped desired trades")
    if skipped.empty:
        print("None")
    else:
        print(skipped.round(4).to_string(index=False))

    print("\nConcentration warnings")
    if concentration_warnings.empty:
        print("None")
    else:
        print(concentration_warnings.round(4).to_string(index=False))

    if args.journal is not None:
        print(f"\nJournal updated: {args.journal}")


if __name__ == "__main__":
    main()
