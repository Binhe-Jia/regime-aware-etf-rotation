from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from etf_relative_model import ETFRelativeMeanReversionModel, StrategyConfig


def make_synthetic_data(rows: int = 900, seed: int = 7) -> pd.DataFrame:
    """Create a small stock/market/sector dataset with occasional residual selloffs."""

    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=rows)

    market_return = rng.normal(0.00035, 0.010, rows)
    sector_return = 0.75 * market_return + rng.normal(0.00015, 0.009, rows)

    residual = np.zeros(rows)
    for i in range(1, rows):
        residual[i] = -0.35 * residual[i - 1] + rng.normal(0.0, 0.012)
    for i in range(240, rows, 120):
        residual[i : i + 3] -= np.array([0.055, 0.030, 0.015])

    stock_return = 0.55 * market_return + 0.65 * sector_return + residual

    market_close = 420.0 * np.cumprod(1.0 + market_return)
    sector_close = 180.0 * np.cumprod(1.0 + sector_return)
    stock_close = 100.0 * np.cumprod(1.0 + stock_return)
    daily_range = np.maximum(0.006, np.abs(stock_return) * 0.7 + 0.01)

    return pd.DataFrame(
        {
            "stock_close": stock_close,
            "stock_high": stock_close * (1.0 + daily_range / 2),
            "stock_low": stock_close * (1.0 - daily_range / 2),
            "market_close": market_close,
            "sector_close": sector_close,
        },
        index=index,
    )


def load_prices(path: Path | None) -> pd.DataFrame:
    if path is None:
        return make_synthetic_data()

    data = pd.read_csv(path)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")
    return data.sort_index()


def _extract_yfinance_field(downloaded: pd.DataFrame, ticker: str, field: str) -> pd.Series:
    if isinstance(downloaded.columns, pd.MultiIndex):
        if ticker in downloaded.columns.get_level_values(0):
            return downloaded[(ticker, field)]
        return downloaded[(field, ticker)]
    return downloaded[field]


def load_yfinance_prices(
    ticker: str,
    market: str,
    sector: str,
    period: str = "max",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    cache_dir: Path | None = None,
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

    symbols = [ticker, market, sector]
    download_kwargs = {
        "tickers": symbols,
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
        raise SystemExit(f"No data returned for symbols: {', '.join(symbols)}")

    prices = pd.DataFrame(
        {
            "stock_close": _extract_yfinance_field(downloaded, ticker, "Close"),
            "stock_high": _extract_yfinance_field(downloaded, ticker, "High"),
            "stock_low": _extract_yfinance_field(downloaded, ticker, "Low"),
            "market_close": _extract_yfinance_field(downloaded, market, "Close"),
            "sector_close": _extract_yfinance_field(downloaded, sector, "Close"),
        }
    )
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna()
    if len(prices) < 260:
        raise SystemExit(
            f"Only {len(prices)} usable rows were downloaded. Use a longer --period."
        )
    return prices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ETF-relative residual mean-reversion model."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV with stock_close, market_close, sector_close, and optional stock_high/stock_low.",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Stock ticker to download with yfinance, for example NVDA.",
    )
    parser.add_argument(
        "--market",
        default="QQQ",
        help="Market/proxy ETF ticker for yfinance mode. Default: QQQ.",
    )
    parser.add_argument(
        "--sector",
        default="SMH",
        help="Sector ETF ticker for yfinance mode. Default: SMH.",
    )
    parser.add_argument(
        "--period",
        default="max",
        help="yfinance lookback period, for example 2y, 5y, 10y, max. Ignored if --start or --end is set. Default: max.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance bar interval, for example 1d, 1h, 15m. Default: 1d.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional yfinance start date in YYYY-MM-DD format, for example 2010-01-01.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional yfinance end date in YYYY-MM-DD format. yfinance treats this as exclusive.",
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
        help="Use synthetic demo data even if no CSV or ticker is provided.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Starting account value for the backtest. Default: 100000.",
    )
    parser.add_argument(
        "--adf-filter",
        action="store_true",
        help="Require rolling residual stationarity before opening residual mean-reversion trades.",
    )
    parser.add_argument(
        "--adf-lookback",
        type=int,
        default=252,
        help="Rolling window for the residual ADF test. Default: 252.",
    )
    parser.add_argument(
        "--adf-pvalue",
        type=float,
        default=0.10,
        help="Maximum ADF p-value allowed for residual trades. Default: 0.10.",
    )
    args = parser.parse_args()

    if args.csv is not None:
        prices = load_prices(args.csv)
        data_source = f"CSV: {args.csv}"
    elif args.ticker is not None:
        prices = load_yfinance_prices(
            ticker=args.ticker.upper(),
            market=args.market.upper(),
            sector=args.sector.upper(),
            period=args.period,
            interval=args.interval,
            start=args.start,
            end=args.end,
            cache_dir=args.yf_cache_dir,
        )
        data_source = f"yfinance: {args.ticker.upper()} vs {args.market.upper()} + {args.sector.upper()}"
    else:
        prices = make_synthetic_data()
        data_source = "synthetic demo data"

    config = StrategyConfig(
        proxy_lookback=60,
        residual_z_lookback=60,
        market_ma_window=200,
        entry_z=-2.0,
        entry_rsi=35.0,
        max_hold_days=10,
        risk_per_trade=0.01,
        max_position_fraction=0.75,
        transaction_cost_bps=2.0,
        slippage_bps=3.0,
        use_adf_filter=args.adf_filter,
        adf_lookback=args.adf_lookback,
        adf_pvalue_threshold=args.adf_pvalue,
    )
    model = ETFRelativeMeanReversionModel(config=config)
    result = model.backtest(prices, initial_equity=args.initial_capital)

    print(f"Data source: {data_source}")
    print(f"Rows: {len(prices)} | Start: {prices.index[0].date()} | End: {prices.index[-1].date()}")
    print(
        f"Initial capital: {args.initial_capital:,.2f} | "
        f"Final equity: {result.equity_curve['equity'].iloc[-1]:,.2f}"
    )
    print()
    print("Metrics")
    for key, value in result.metrics.items():
        print(f"{key:>20}: {value: .4f}")

    print("\nRecent signals")
    columns = [
        "stock_close",
        "residual_z",
        "residual_adf_pvalue",
        "residual_stationary",
        "rsi",
        "risk_on",
        "position",
        "entry_signal",
        "exit_signal",
        "equity",
    ]
    print(result.equity_curve.loc[:, columns].tail(12).round(4).to_string())

    if not result.trades.empty:
        print("\nLast trades")
        print(result.trades.tail(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
