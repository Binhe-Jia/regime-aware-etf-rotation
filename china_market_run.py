from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from etf_relative_model import ETFRelativeMeanReversionModel, StrategyConfig


DEFAULT_CHINA_STOCK = "600519.SS"
DEFAULT_CHINA_MARKET_PROXY = "510300.SS"
DEFAULT_CHINA_SECTOR_PROXY = "159915.SZ"


def normalize_china_symbol(symbol: str) -> str:
    """
    Convert common China ticker inputs into Yahoo Finance symbols.

    Examples:
    600519 -> 600519.SS
    000001 -> 000001.SZ
    510300 -> 510300.SS
    159915 -> 159915.SZ
    """

    clean = symbol.strip().upper()
    if "." in clean or clean.startswith("^"):
        return clean
    if not clean.isdigit() or len(clean) != 6:
        return clean

    shanghai_prefixes = ("5", "6", "9")
    if clean.startswith(shanghai_prefixes):
        return f"{clean}.SS"
    return f"{clean}.SZ"


def make_china_like_synthetic_data(rows: int = 1_200, seed: int = 88) -> pd.DataFrame:
    """Create a China-market-flavored daily dataset for offline testing."""

    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-04", periods=rows)

    market_return = rng.normal(0.00015, 0.012, rows)
    sector_return = 0.65 * market_return + rng.normal(0.00010, 0.014, rows)

    residual = np.zeros(rows)
    for i in range(1, rows):
        residual[i] = -0.42 * residual[i - 1] + rng.normal(0.0, 0.018)
    for i in range(260, rows, 140):
        residual[i : i + 4] -= np.array([0.060, 0.035, 0.020, 0.010])

    stock_return = 0.50 * market_return + 0.55 * sector_return + residual

    market_close = 4.0 * np.cumprod(1.0 + market_return)
    sector_close = 2.2 * np.cumprod(1.0 + sector_return)
    stock_close = 100.0 * np.cumprod(1.0 + stock_return)
    daily_range = np.maximum(0.008, np.abs(stock_return) * 0.8 + 0.012)

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
        return make_china_like_synthetic_data()

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
        description="Run the China-market ETF-relative residual mean-reversion model."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV with stock_close, market_close, sector_close, and optional stock_high/stock_low.",
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_CHINA_STOCK,
        help="China stock ticker. Six-digit codes are auto-suffixed as .SS or .SZ. Default: 600519.SS.",
    )
    parser.add_argument(
        "--market",
        default=DEFAULT_CHINA_MARKET_PROXY,
        help="China market proxy. Default: 510300.SS, a CSI 300 ETF proxy.",
    )
    parser.add_argument(
        "--sector",
        default=DEFAULT_CHINA_SECTOR_PROXY,
        help="China sector/style proxy. Default: 159915.SZ, a ChiNext ETF proxy.",
    )
    parser.add_argument(
        "--period",
        default="max",
        help="yfinance lookback period, for example 2y, 5y, 10y, max. Ignored if --start or --end is set. Default: max.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance bar interval. Daily bars are recommended for China A-shares. Default: 1d.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional yfinance start date in YYYY-MM-DD format, for example 2015-01-01.",
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
        "--strategy",
        choices=["mean_reversion", "trend", "hybrid", "buy_hold"],
        default="hybrid",
        help="Signal mode. hybrid holds trend exposure and adds residual mean-reversion trades. Default: hybrid.",
    )
    parser.add_argument(
        "--base-position",
        type=float,
        default=0.75,
        help="Base long exposure for trend/hybrid modes, as a portfolio fraction. Default: 0.75.",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=1.00,
        help="Maximum portfolio fraction invested in the stock. Default: 1.00.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use China-like synthetic demo data instead of yfinance.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Starting account value for the backtest. Default: 100000.",
    )
    args = parser.parse_args()

    if args.csv is not None:
        prices = load_prices(args.csv)
        data_source = f"CSV: {args.csv}"
    elif args.synthetic:
        prices = make_china_like_synthetic_data()
        data_source = "China-like synthetic demo data"
    else:
        ticker = normalize_china_symbol(args.ticker)
        market = normalize_china_symbol(args.market)
        sector = normalize_china_symbol(args.sector)
        prices = load_yfinance_prices(
            ticker=ticker,
            market=market,
            sector=sector,
            period=args.period,
            interval=args.interval,
            start=args.start,
            end=args.end,
            cache_dir=args.yf_cache_dir,
        )
        data_source = f"yfinance China: {ticker} vs {market} + {sector}"

    config = StrategyConfig(
        signal_mode=args.strategy,
        proxy_lookback=80,
        residual_z_lookback=80,
        market_ma_window=200,
        stock_ma_window=200,
        entry_z=-2.0,
        entry_rsi=35.0,
        momentum_entry_z=0.50,
        momentum_exit_z=-0.75,
        momentum_entry_rsi=50.0,
        base_position_fraction=args.base_position,
        max_hold_days=10,
        risk_per_trade=0.008,
        max_position_fraction=args.max_position,
        transaction_cost_bps=4.0,
        slippage_bps=6.0,
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
