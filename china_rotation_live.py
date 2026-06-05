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
    min_trade_value: float = 500.0
    max_single_order_fraction: float = 0.10
    concentration_warning_weight: float = 0.20
    small_position_warning_weight: float = 0.02
    transaction_cost_bps: float = 6.0
    slippage_bps: float = 8.0
    lot_size: int = 100
    price_limit_warning_pct: float = 0.095


def parse_universe(text: str | None) -> list[str]:
    if not text:
        return DEFAULT_UNIVERSE
    tickers = [item.strip() for item in text.split(",") if item.strip()]
    return [normalize_china_symbol(ticker) for ticker in tickers]


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
    max_drift = float(max(drift.abs().max(), abs(cash_drift)))
    rebalance_required = max_drift > config.rebalance_tolerance

    current = pd.DataFrame(
        {
            "shares": shares,
            "price": prices,
            "market_value": values,
            "current_weight": current_weights,
            "target_weight": target_weights,
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
        for ticker in universe:
            raw_trade_value = float(drift.loc[ticker] * portfolio_value)
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
            raw_shares = abs(capped_trade_value) / price
            if config.lot_size > 1:
                estimated_shares = np.floor(raw_shares / config.lot_size) * config.lot_size
            else:
                estimated_shares = np.floor(raw_shares)
            estimated_shares = int(estimated_shares)
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

            if action == "SELL":
                estimated_shares = min(estimated_shares, int(shares.loc[ticker]))
                if estimated_shares <= 0:
                    skipped.append(
                        {
                            "ticker": ticker,
                            "action": action,
                            "desired_trade_value": raw_trade_value,
                            "reason": "no_shares_to_sell",
                        }
                    )
                    continue

            trade_value = estimated_shares * price * (1 if action == "BUY" else -1)
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
                    "drift": drift.loc[ticker],
                    "price": price,
                    "estimated_shares": estimated_shares,
                    "trade_value": trade_value,
                    "estimated_cost": estimated_cost,
                    "daily_return": daily_return,
                    "warnings": ",".join(warnings),
                }
            )

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
    summary = {
        "portfolio_value": portfolio_value,
        "cash": cash,
        "cash_pct": cash_pct,
        "cash_drift": cash_drift,
        "projected_cash": projected_cash,
        "projected_cash_pct": projected_cash / portfolio_value,
        "max_drift": max_drift,
        "rebalance_required": rebalance_required,
        "estimated_turnover": estimated_turnover,
        "estimated_cost": estimated_cost,
        "largest_weight": float(current_weights.max()),
        "smallest_weight": float(current_weights.min()),
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
        min_trade_value=args.min_trade_value,
        max_single_order_fraction=args.max_single_order_fraction,
        lot_size=args.lot_size,
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
    print(f"Rebalance required: {'Yes' if summary['rebalance_required'] else 'No'}")
    print(f"Estimated turnover: {summary['estimated_turnover']:.2%}")
    print(f"Estimated transaction/slippage cost: {summary['estimated_cost']:,.2f} RMB")
    print(f"Largest position: {summary['largest_weight']:.2%}")
    print(f"Smallest position: {summary['smallest_weight']:.2%}")
    print(f"Concentration warnings: {summary['concentration_warning_count']}")

    print("\nCurrent weights")
    display_current = current.loc[
        :,
        ["shares", "price", "market_value", "current_weight", "target_weight", "drift"],
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
