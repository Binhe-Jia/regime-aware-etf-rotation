from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from china_market_run import normalize_china_symbol
from china_rotation_run import DEFAULT_UNIVERSE


CASH_LABEL = "cash"


@dataclass(frozen=True)
class LiveConfig:
    rebalance_tolerance: float = 0.10
    tolerance_bands: dict[str, float] | None = None
    min_trade_value: float = 500.0
    max_single_order_fraction: float = 0.10
    concentration_warning_weight: float = 0.20
    small_position_warning_weight: float = 0.02
    transaction_cost_bps: float = 6.0
    slippage_bps: float = 8.0
    lot_size: int = 100
    price_limit_warning_pct: float = 0.095
    rebalance_to: str = "target"
    cash_neutral_corridor: bool = True
    redistribute_skipped_cash: bool = False


def parse_universe(text: str | None) -> list[str]:
    if not text:
        return DEFAULT_UNIVERSE
    tickers = [item.strip() for item in text.split(",") if item.strip()]
    return [normalize_china_symbol(ticker) for ticker in tickers]


def parse_tolerance_bands(
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


def load_holdings(path: Path) -> tuple[pd.Series, float]:
    data = pd.read_csv(path)
    lower_columns = {column.lower().strip(): column for column in data.columns}
    ticker_col = lower_columns.get("ticker") or lower_columns.get("symbol")
    shares_col = lower_columns.get("shares") or lower_columns.get("quantity")
    if ticker_col is None or shares_col is None:
        raise SystemExit("Holdings CSV must include ticker and shares columns.")

    holdings = {}
    cash = 0.0
    for _, row in data.iterrows():
        ticker_raw = str(row[ticker_col]).strip()
        shares = float(row[shares_col])
        if ticker_raw.lower() == CASH_LABEL:
            cash += shares
        else:
            holdings[normalize_china_symbol(ticker_raw)] = shares

    return pd.Series(holdings, dtype=float), cash


def load_latest_market_data(
    tickers: list[str],
    period: str,
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

    downloaded = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=False,
    )
    if downloaded.empty:
        raise SystemExit(f"No market data returned for: {', '.join(tickers)}")

    rows = []
    for ticker in tickers:
        try:
            if isinstance(downloaded.columns, pd.MultiIndex):
                frame = downloaded[ticker].copy()
            else:
                frame = downloaded.copy()
        except KeyError:
            rows.append(
                {
                    "ticker": ticker,
                    "price": np.nan,
                    "previous_price": np.nan,
                    "volume": np.nan,
                    "as_of": None,
                }
            )
            continue

        frame = frame.dropna(subset=["Close"])
        if frame.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "price": np.nan,
                    "previous_price": np.nan,
                    "volume": np.nan,
                    "as_of": None,
                }
            )
            continue

        latest = frame.iloc[-1]
        previous = frame.iloc[-2] if len(frame) > 1 else latest
        rows.append(
            {
                "ticker": ticker,
                "price": float(latest["Close"]),
                "previous_price": float(previous["Close"]),
                "volume": float(latest.get("Volume", np.nan)),
                "as_of": frame.index[-1],
            }
        )

    return pd.DataFrame(rows).set_index("ticker")


def tolerance_series(
    assets: pd.Index,
    default_tolerance: float,
    tolerance_bands: dict[str, float] | None,
) -> pd.Series:
    tolerance = pd.Series(default_tolerance, index=assets, dtype=float)
    for asset, band in (tolerance_bands or {}).items():
        if asset in tolerance.index:
            tolerance.loc[asset] = float(band)
    return tolerance


def evaluate_asymmetric_corridor_policy(
    current_weights: pd.Series,
    target_weights: pd.Series,
    tolerance_bands: pd.Series,
    rebalance_to: str,
    cash_pct: float,
    target_cash_pct: float,
    default_cash_tolerance: float,
    cash_neutral: bool = True,
) -> tuple[pd.Series, pd.Series, bool]:
    drift_from_target = current_weights - target_weights
    cash_drift = cash_pct - target_cash_pct
    breach = drift_from_target.abs() > tolerance_bands
    rebalance_required = bool(breach.any() or abs(cash_drift) > default_cash_tolerance)

    if not rebalance_required:
        return current_weights.copy(), pd.Series(0.0, index=current_weights.index), False

    if rebalance_to == "target" or cash_drift > default_cash_tolerance:
        trade_to_weights = target_weights.copy()
    elif rebalance_to == "corridor":
        trade_to_weights = current_weights.copy()
        upper = target_weights + tolerance_bands
        lower = (target_weights - tolerance_bands).clip(lower=0.0)
        above = current_weights > upper
        below = current_weights < lower
        trade_to_weights.loc[above] = upper.loc[above]
        trade_to_weights.loc[below] = lower.loc[below]
    else:
        raise SystemExit("--rebalance-to must be one of: target, corridor.")

    trade_delta = trade_to_weights - current_weights
    if rebalance_to == "corridor" and cash_neutral:
        net_delta = float(trade_delta.sum())
        if net_delta < 0.0:
            buy_capacity = (target_weights - (current_weights + trade_delta)).clip(lower=0.0)
            capacity_sum = float(buy_capacity.sum())
            if capacity_sum > 0.0:
                trade_delta += buy_capacity / capacity_sum * min(-net_delta, capacity_sum)
        elif net_delta > cash_pct:
            required_sell = net_delta - cash_pct
            sell_capacity = ((current_weights + trade_delta) - target_weights).clip(lower=0.0)
            capacity_sum = float(sell_capacity.sum())
            if capacity_sum > 0.0:
                trade_delta -= sell_capacity / capacity_sum * min(required_sell, capacity_sum)

    trade_to_weights = (current_weights + trade_delta).clip(lower=0.0)
    return trade_to_weights, trade_delta, rebalance_required


def is_china_symbol(ticker: str) -> bool:
    return ticker.endswith((".SS", ".SZ")) or ticker.isdigit()


def round_share_order(
    ticker: str,
    target_share_move: float,
    lot_size: int,
    current_shares: float,
) -> int | float:
    if is_china_symbol(ticker):
        if target_share_move > 0:
            shares = int(target_share_move // max(lot_size, 1)) * max(lot_size, 1)
        else:
            shares = int(np.fix(target_share_move))
            shares = max(shares, -int(current_shares))
        return shares
    if lot_size <= 1:
        return round(target_share_move, 4)
    shares = int(abs(target_share_move) // lot_size) * lot_size
    return shares if target_share_move > 0 else -shares


def generate_corridor_orders(
    current_holdings: pd.Series,
    target_trades: pd.Series,
    asset_prices: pd.Series,
    total_portfolio_value: float,
    lot_size: int,
) -> dict[str, dict[str, float | str]]:
    order_sheet = {}
    for asset, weight_delta in target_trades.items():
        if abs(weight_delta) < 1e-5:
            continue
        price = float(asset_prices.loc[asset])
        target_cash_move = float(weight_delta * total_portfolio_value)
        target_share_move = target_cash_move / price
        shares_to_order = round_share_order(
            asset,
            target_share_move,
            lot_size,
            float(current_holdings.reindex([asset]).fillna(0.0).iloc[0]),
        )
        if shares_to_order:
            order_sheet[asset] = {
                "action": "BUY" if shares_to_order > 0 else "SELL",
                "shares": abs(shares_to_order),
                "estimated_value": abs(float(shares_to_order) * price),
                "trade_value": float(shares_to_order) * price,
            }
    return order_sheet


def build_live_orders(
    universe: list[str],
    holdings: pd.Series,
    cash: float,
    market_data: pd.DataFrame,
    config: LiveConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | bool]]:
    prices = market_data["price"].reindex(universe)
    missing = prices[prices.isna()].index.tolist()
    if missing:
        raise SystemExit(f"Missing prices for: {', '.join(missing)}")

    shares = holdings.reindex(universe).fillna(0.0)
    values = shares * prices
    portfolio_value = float(values.sum() + cash)
    if portfolio_value <= 0:
        raise SystemExit("Portfolio value must be positive.")

    current_weights = values / portfolio_value
    target_weights = pd.Series(1.0 / len(universe), index=universe)
    drift = target_weights - current_weights
    cash_pct = cash / portfolio_value
    target_cash_pct = max(0.0, 1.0 - float(target_weights.sum()))
    cash_drift = cash_pct - target_cash_pct
    asset_tolerance = tolerance_series(
        current_weights.index,
        config.rebalance_tolerance,
        config.tolerance_bands,
    )
    trade_to_weights, trade_drift, rebalance_required = evaluate_asymmetric_corridor_policy(
        current_weights,
        target_weights,
        asset_tolerance,
        config.rebalance_to,
        cash_pct,
        target_cash_pct,
        config.rebalance_tolerance,
        cash_neutral=config.cash_neutral_corridor,
    )
    max_drift = float(max(drift.abs().max(), abs(cash_drift)))
    max_tolerance_breach = float(max((drift.abs() - asset_tolerance).max(), 0.0))

    current = pd.DataFrame(
        {
            "shares": shares,
            "price": prices,
            "market_value": values,
            "current_weight": current_weights,
            "target_weight": target_weights,
            "trade_to_weight": trade_to_weights,
            "tolerance_band": asset_tolerance,
            "drift": drift,
        }
    )

    concentration_warnings = []
    for ticker in universe:
        weight = float(current_weights.loc[ticker])
        if weight > config.concentration_warning_weight:
            concentration_warnings.append(
                {
                    "ticker": ticker,
                    "current_weight": weight,
                    "warning": "above_concentration_limit",
                }
            )
        elif 0.0 < weight < config.small_position_warning_weight:
            concentration_warnings.append(
                {
                    "ticker": ticker,
                    "current_weight": weight,
                    "warning": "below_small_position_limit",
                }
            )

    orders = []
    skipped = []
    if rebalance_required:
        max_order_value = portfolio_value * config.max_single_order_fraction
        order_sheet = generate_corridor_orders(
            shares,
            trade_drift,
            prices,
            portfolio_value,
            config.lot_size,
        )
        for ticker in universe:
            raw_trade_value = float(trade_drift.loc[ticker] * portfolio_value)
            if abs(raw_trade_value) < 1e-10:
                continue
            action = "BUY" if raw_trade_value > 0 else "SELL"
            capped_trade_value = float(
                np.clip(raw_trade_value, -max_order_value, max_order_value)
            )
            if abs(capped_trade_value) < config.min_trade_value:
                skipped.append(
                    {
                        "ticker": ticker,
                        "action": action,
                        "desired_trade_value": raw_trade_value,
                        "reason": "below_min_trade_value",
                    }
                )
                continue

            price = float(prices.loc[ticker])
            capped_weight_delta = capped_trade_value / portfolio_value
            if abs(capped_trade_value - raw_trade_value) > 1e-8:
                order_sheet = generate_corridor_orders(
                    shares,
                    pd.Series({ticker: capped_weight_delta}),
                    prices,
                    portfolio_value,
                    config.lot_size,
                )
            order = order_sheet.get(ticker)
            estimated_shares = float(order["shares"]) if order else 0.0
            if estimated_shares <= 0:
                skipped.append(
                    {
                        "ticker": ticker,
                        "action": action,
                        "desired_trade_value": raw_trade_value,
                        "reason": "below_lot_size",
                    }
                )
                continue

            trade_value = float(order["trade_value"])
            action = str(order["action"])
            estimated_cost = abs(trade_value) * (
                config.transaction_cost_bps + config.slippage_bps
            ) / 10_000.0
            latest = market_data.loc[ticker]
            previous_price = float(latest["previous_price"])
            daily_return = price / previous_price - 1.0 if previous_price else 0.0
            warnings = []
            if pd.notna(latest["volume"]) and latest["volume"] <= 0:
                warnings.append("zero_volume")
            if abs(daily_return) >= config.price_limit_warning_pct:
                warnings.append("price_limit_review")

            orders.append(
                {
                    "ticker": ticker,
                    "action": action,
                    "current_weight": current_weights.loc[ticker],
                    "target_weight": target_weights.loc[ticker],
                    "trade_to_weight": trade_to_weights.loc[ticker],
                    "tolerance_band": asset_tolerance.loc[ticker],
                    "drift": drift.loc[ticker],
                    "price": price,
                    "estimated_shares": estimated_shares,
                    "trade_value": trade_value,
                    "estimated_cost": estimated_cost,
                    "daily_return": daily_return,
                    "warnings": ",".join(warnings),
                }
            )

        if config.redistribute_skipped_cash and orders:
            order_by_ticker = {str(order["ticker"]): order for order in orders}
            buy_candidates = [
                ticker
                for ticker in universe
                if float(trade_drift.loc[ticker]) > 0.0
                and ticker in order_by_ticker
                and order_by_ticker[ticker]["action"] == "BUY"
            ]
            if buy_candidates:
                cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000.0
                concentration_cap_value = (
                    portfolio_value * config.concentration_warning_weight
                )
                for _ in range(len(universe) * 3):
                    net_trade_value = sum(float(order["trade_value"]) for order in orders)
                    estimated_cost = sum(float(order["estimated_cost"]) for order in orders)
                    available_cash = cash - net_trade_value - estimated_cost
                    if available_cash < config.min_trade_value:
                        break

                    added_lot = False
                    for ticker in sorted(buy_candidates, key=lambda item: prices.loc[item]):
                        order = order_by_ticker[ticker]
                        price = float(prices.loc[ticker])
                        lot = max(config.lot_size, 1)
                        lot_value = price * lot
                        lot_cost = lot_value * cost_rate
                        if lot_value + lot_cost > available_cash:
                            continue
                        current_order_value = abs(float(order["trade_value"]))
                        if current_order_value + lot_value > max_order_value:
                            continue
                        projected_position_value = (
                            float(values.loc[ticker]) + current_order_value + lot_value
                        )
                        if projected_position_value > concentration_cap_value:
                            continue

                        order["estimated_shares"] = float(order["estimated_shares"]) + lot
                        order["trade_value"] = float(order["trade_value"]) + lot_value
                        order["estimated_cost"] = abs(float(order["trade_value"])) * cost_rate
                        existing_warnings = str(order.get("warnings", ""))
                        warnings = [item for item in existing_warnings.split(",") if item]
                        if "redistributed_cash" not in warnings:
                            warnings.append("redistributed_cash")
                        order["warnings"] = ",".join(warnings)
                        added_lot = True
                        break

                    if not added_lot:
                        break

    if orders:
        for order in orders:
            ticker = str(order["ticker"])
            projected_value = float(values.loc[ticker]) + float(order["trade_value"])
            order["projected_weight_after_order"] = projected_value / portfolio_value

    orders_df = pd.DataFrame(orders)
    skipped_df = pd.DataFrame(skipped)
    estimated_turnover = (
        float(orders_df["trade_value"].abs().sum() / portfolio_value)
        if not orders_df.empty
        else 0.0
    )
    estimated_cost = float(orders_df["estimated_cost"].sum()) if not orders_df.empty else 0.0
    net_trade_value = float(orders_df["trade_value"].sum()) if not orders_df.empty else 0.0
    projected_cash = cash - net_trade_value - estimated_cost
    projected_values = values.copy()
    if not orders_df.empty:
        for _, order in orders_df.iterrows():
            projected_values.loc[order["ticker"]] += float(order["trade_value"])
    projected_weights = projected_values / portfolio_value
    summary = {
        "portfolio_value": portfolio_value,
        "cash": cash,
        "cash_pct": cash_pct,
        "cash_drift": cash_drift,
        "projected_cash": projected_cash,
        "projected_cash_pct": projected_cash / portfolio_value,
        "max_drift": max_drift,
        "max_tolerance_breach": max_tolerance_breach,
        "rebalance_required": rebalance_required,
        "rebalance_to": config.rebalance_to,
        "estimated_turnover": estimated_turnover,
        "estimated_cost": estimated_cost,
        "largest_weight": float(current_weights.max()),
        "smallest_weight": float(current_weights.min()),
        "projected_largest_weight": float(projected_weights.max()),
        "projected_smallest_weight": float(projected_weights.min()),
        "concentration_warning_count": len(concentration_warnings),
    }
    return current, orders_df, skipped_df, pd.DataFrame(concentration_warnings), summary


def append_journal(
    path: Path | None,
    summary: dict[str, float | bool],
    orders: pd.DataFrame,
    note: str,
) -> None:
    if path is None:
        return

    timestamp = datetime.now().isoformat(timespec="seconds")
    if orders.empty:
        rows = [
            {
                "timestamp": timestamp,
                "portfolio_value": summary["portfolio_value"],
                "max_drift": summary["max_drift"],
                "rebalance_required": summary["rebalance_required"],
                "ticker": "",
                "action": "NO_TRADE",
                "estimated_shares": 0,
                "trade_value": 0.0,
                "estimated_cost": 0.0,
                "notes": note,
            }
        ]
    else:
        rows = []
        for _, order in orders.iterrows():
            rows.append(
                {
                    "timestamp": timestamp,
                    "portfolio_value": summary["portfolio_value"],
                    "max_drift": summary["max_drift"],
                    "rebalance_required": summary["rebalance_required"],
                    "ticker": order["ticker"],
                    "action": order["action"],
                    "estimated_shares": order["estimated_shares"],
                    "trade_value": order["trade_value"],
                    "estimated_cost": order["estimated_cost"],
                    "notes": note,
                }
            )

    output = pd.DataFrame(rows)
    output.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manual rebalance orders for the frozen China leader basket."
    )
    parser.add_argument("--holdings", type=Path, required=True, help="CSV with ticker,shares rows.")
    parser.add_argument(
        "--universe",
        default=None,
        help="Optional comma-separated ticker universe. Defaults to frozen China leader basket.",
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
        help="Optional asset-specific bands as ticker=value pairs, for example 510300.SS=0.025,159915.SZ=0.075.",
    )
    parser.add_argument(
        "--allow-cash-drift",
        action="store_true",
        help="For corridor mode, do not offset net sells/buys with counter-trades.",
    )
    parser.add_argument(
        "--min-trade-value",
        type=float,
        default=500.0,
        help="Ignore suggested trades below this RMB value. Default: 500.",
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
        default=100,
        help="Round suggested shares down to this lot size. Default: 100.",
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

    universe = parse_universe(args.universe)
    holdings, cash = load_holdings(args.holdings)
    market_data = load_latest_market_data(universe, args.period, args.yf_cache_dir)
    config = LiveConfig(
        rebalance_tolerance=args.rebalance_tolerance,
        tolerance_bands=parse_tolerance_bands(args.tolerance_bands),
        min_trade_value=args.min_trade_value,
        max_single_order_fraction=args.max_single_order_fraction,
        lot_size=args.lot_size,
        rebalance_to=args.rebalance_to,
        cash_neutral_corridor=not args.allow_cash_drift,
        redistribute_skipped_cash=args.redistribute_skipped_cash,
    )
    current, orders, skipped, concentration_warnings, summary = build_live_orders(
        universe,
        holdings,
        cash,
        market_data,
        config,
    )
    append_journal(args.journal, summary, orders, args.note)

    print(f"Portfolio value: {summary['portfolio_value']:,.2f} RMB")
    print(f"Cash: {summary['cash']:,.2f} RMB ({summary['cash_pct']:.2%})")
    print(f"Cash drift: {summary['cash_drift']:.2%}")
    print(
        f"Projected cash after suggested orders: {summary['projected_cash']:,.2f} RMB "
        f"({summary['projected_cash_pct']:.2%})"
    )
    print(f"Max drift: {summary['max_drift']:.2%}")
    print(f"Max tolerance breach: {summary['max_tolerance_breach']:.2%}")
    print(f"Rebalance required: {'Yes' if summary['rebalance_required'] else 'No'}")
    print(f"Rebalance to: {summary['rebalance_to']}")
    print(f"Estimated turnover: {summary['estimated_turnover']:.2%}")
    print(f"Estimated transaction/slippage cost: {summary['estimated_cost']:,.2f} RMB")
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
