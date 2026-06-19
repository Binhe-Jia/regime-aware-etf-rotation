# ETF-Relative Trading Model

This project implements the framework described in the last part of
`GARCH-LSTM for Stock Forecasting.pdf`:

```text
ETF proxy + residual signal + regime filter + risk sizing + cost-aware execution
```

It is intentionally not a black-box price predictor. The model asks whether a stock is
unusually weak relative to market and sector ETF proxies, confirms the setup with RSI,
trades only when the market regime is acceptable, sizes positions by volatility, and
includes transaction and slippage costs in the backtest.

## Files

- `etf_relative_model.py` - reusable model, feature engineering, sizing, and backtest code.
- `example_run.py` - runnable example using live yfinance data, synthetic data, or your own CSV.
- `china_market_run.py` - China-market copy with A-share defaults and `.SS`/`.SZ` ticker handling.
- `china_rotation_run.py` - China A-share stock-selection and rotation strategy.
- `china_rotation_live.py` - deployment helper that reads current holdings and generates manual rebalance orders.
- `us_rotation_run.py` - U.S. leader basket backtester using the same rotation engine.
- `us_rotation_live.py` - U.S. deployment helper for holdings/order checks.
- `dashboard_app.py` - Streamlit dashboard with equity curves, drawdowns, holdings, and tolerance sweeps.
- `STRATEGY_SPEC.md` - frozen production rule for paper trading.
- `US_STRATEGY_SPEC.md` - frozen U.S. production rule for paper trading.
- `holdings_sample.csv` - template for live/paper portfolio state.
- `china_point_in_time_universe_template.csv` - example China universe snapshot format.
- `us_holdings_sample.csv` - U.S. empty starting template.
- `us_point_in_time_universe_template.csv` - example U.S. universe snapshot format.
- `us_paper_holdings.csv` - U.S. simulated paper portfolio after initial fills.

## Data Format

Use daily data with these columns:

```text
date,stock_close,market_close,sector_close,stock_high,stock_low
```

`stock_high` and `stock_low` are optional, but recommended because they improve ATR-based
risk sizing. For example:

```text
date,stock_close,stock_high,stock_low,market_close,sector_close
2025-01-02,101.2,102.0,100.4,480.1,212.3
```

For Nvidia, a reasonable proxy pair could be `QQQ` as the market/growth proxy and `SMH`
as the semiconductor sector ETF.

## Run

Synthetic demo:

```bash
python example_run.py
```

Live/historical market data from yfinance:

```bash
python example_run.py --ticker NVDA --market QQQ --sector SMH --period max
```

Set starting capital:

```bash
python example_run.py --ticker NVDA --initial-capital 50000
```

Preset windows also work:

```bash
python example_run.py --ticker NVDA --market QQQ --sector SMH --period 10y
```

For a precise backtest window, use `--start` and `--end`. When either is provided,
`--period` is ignored:

```bash
python example_run.py --ticker NVDA --market QQQ --sector SMH --start 2011-01-01 --end 2026-01-01
```

By default, the script stores yfinance's local cache in `.yfinance-cache` inside this
project. You can override it:

```bash
python example_run.py --ticker NVDA --yf-cache-dir path/to/cache
```

Your own CSV:

```bash
python example_run.py --csv path/to/prices.csv
```

Install dependencies first if needed:

```bash
python -m pip install -r requirements.txt
```

## Dashboard

For screenshots, demos, or a LinkedIn project post, launch the Streamlit dashboard:

```bash
streamlit run dashboard_app.py
```

The dashboard defaults to the current strongest tested setup:

```text
U.S. asset-class ETF universe + dual_momentum + 2018 start date
Momentum Top N = 1 + raw 12-month momentum
```

It also includes a protective dual-momentum variant:

```text
protective_dual_momentum
Top-1 raw 12-month momentum
Full exposure when at least 60% of the universe has positive 12-month momentum
Half exposure when breadth is between 40% and 60%
Defensive sleeve or cash when breadth is below 40%
```

This variant is designed as risk control, not guaranteed return enhancement. In the
2018-to-2026 U.S. asset-class ETF test, the original concentrated top-1 dual momentum
still produced the highest final equity, while the breadth-adjusted version improved
drawdown and Sharpe in exchange for lower total return.

The dashboard lets you switch between China and U.S. universes, choose ETF or leader
baskets, adjust capital, rebalance tolerance, costs, and risk mode, then view:

```text
equity curve vs market and frictionless equal-weight
calendar rebalance vs tolerance-band execution comparison
dual-momentum ablation tests
drawdown comparison
current portfolio weights
asset-specific volatility-adjusted tolerance bands
asset buy-and-hold returns
rebalance tolerance sweep
```

The main dashboard framing is smart-beta execution research rather than short-term price
prediction. The strategy targets equal-weight exposure, then studies how calendar
rebalancing, tolerance bands, transaction costs, and slippage change the realized return
relative to a frictionless daily equal-weight benchmark.

The dashboard also supports two rebalance destinations:

```text
target    - after a breach, trade all the way back to target weight
corridor  - after a breach, trade only back to the tolerance corridor edge
```

`corridor` mode is useful for testing whether smaller corrective trades reduce turnover
enough to offset the tracking gap versus strict equal-weight. Real execution choices such
as MOC orders for U.S. ETFs or TWAP/VWAP execution for China ETFs are not directly
observable in daily yfinance data, so model them by lowering the slippage assumption and
then validate fills in paper trading.

The yfinance path downloads adjusted OHLC data for the stock ticker plus two proxy ETFs.
It is suitable for daily or intraday research data, but it is not a direct broker feed and
should not be treated as execution-grade real-time market data.

The residual ETF-relative model can require rolling residual stationarity before it opens
mean-reversion trades:

```bash
python example_run.py --ticker NVDA --market QQQ --sector SMH --period max --adf-filter
python china_market_run.py --ticker 002594 --market 510300 --sector 159915 --period max --strategy hybrid --adf-filter
```

When `statsmodels` is installed, the filter uses its Augmented Dickey-Fuller p-value.
Without `statsmodels`, the project falls back to a lightweight Dickey-Fuller
approximation so the feature remains runnable from a minimal install.

## China Market Version

The China-focused copy uses the same model, but defaults to China A-share style symbols
and more conservative costs:

```bash
python china_market_run.py
```

Default symbols:

```text
stock:  600519.SS
market: 510300.SS
sector: 159915.SZ
```

You can pass six-digit China tickers without suffixes. The script will infer `.SS` for
Shanghai-style prefixes and `.SZ` for Shenzhen-style prefixes:

```bash
python china_market_run.py --ticker 002594 --market 510300 --sector 159915 --period max
```

Set starting capital:

```bash
python china_market_run.py --ticker 002594 --initial-capital 50000
```

The China runner defaults to `hybrid` mode because a strict residual mean-reversion
strategy can stay in cash for years. Compare modes directly:

```bash
python china_market_run.py --ticker 002594 --market 510300 --sector 159915 --period max --strategy mean_reversion
python china_market_run.py --ticker 002594 --market 510300 --sector 159915 --period max --strategy trend
python china_market_run.py --ticker 002594 --market 510300 --sector 159915 --period max --strategy hybrid
python china_market_run.py --ticker 002594 --market 510300 --sector 159915 --period max --strategy buy_hold
```

For trend and hybrid modes, adjust market participation with:

```bash
python china_market_run.py --ticker 002594 --base-position 1.0 --max-position 1.0
```

Explicit Yahoo symbols also work:

```bash
python china_market_run.py --ticker 600519.SS --market 510300.SS --sector 512690.SS --start 2015-01-01
```

Offline China-like synthetic data:

```bash
python china_market_run.py --synthetic
```

## China Rotation Strategy

The rotation runner is the more direct implementation of the suggested upgrade: instead
of timing one stock, it selects the strongest names from a China A-share basket and holds
them only when the market proxy is in an uptrend.

Default basket:

```text
600519.SS, 000858.SZ, 002594.SZ, 300750.SZ, 000333.SZ, 600036.SS,
601318.SS, 601888.SS, 600276.SS, 000651.SZ, 300760.SZ, 601012.SS
```

Run the default equal-weight basket strategy:

```bash
python china_rotation_run.py --period max
```

Set starting capital:

```bash
python china_rotation_run.py --period max --initial-capital 50000
```

For the current default China basket, equal-weight has been stronger than the tested
top-N and core-satellite variants. The script therefore defaults to staying invested in
the equal-weight basket, while still reporting the market proxy and equal-weight benchmark
for comparison.

Compare allocation modes:

```bash
python china_rotation_run.py --period max --allocation equal_weight
python china_rotation_run.py --period max --allocation ivy_trend_filter
python china_rotation_run.py --period max --allocation dual_momentum
python china_rotation_run.py --period max --allocation protective_dual_momentum
python china_rotation_run.py --period max --allocation top_n --top-n 4
python china_rotation_run.py --period max --allocation core_satellite --top-n 4 --core-weight 0.7
python china_rotation_run.py --period max --allocation inverse_vol
python china_rotation_run.py --period max --allocation trend_equal_weight
python china_rotation_run.py --period max --allocation score_weighted
python china_rotation_run.py --period max --allocation quality_equal_weight --quality-csv quality.csv --quality-min-roe 0.10
```

Run a grid search to rank strategy variants against equal-weight:

```bash
python china_rotation_run.py --period max --initial-capital 50000 --grid-search
```

Rank by Sharpe instead of total return:

```bash
python china_rotation_run.py --period max --grid-search --sort-by sharpe
```

Quality-filtered equal-weight needs a fundamentals CSV. Supported columns include
`ticker`, `roe`, `debt_to_equity`, `profit_margin`, and `free_cash_flow`:

```text
ticker,roe,debt_to_equity,profit_margin,free_cash_flow
600519,0.31,0.18,0.48,1000000000
002594,0.18,0.85,0.07,500000000
```

Example:

```bash
python china_rotation_run.py --allocation quality_equal_weight --quality-csv quality.csv --quality-min-roe 0.10 --quality-max-debt-to-equity 1.5 --quality-require-positive-fcf
```

The `ivy_trend_filter` mode is an Ivy Portfolio-style monthly trend filter. Each asset
keeps its equal-weight sleeve only when it is above its 200-day moving average; otherwise
that sleeve stays in cash. Unlike `trend_equal_weight`, it does not redistribute inactive
asset weights to the remaining winners.

The `dual_momentum` mode is a simple public rules-based relative/absolute momentum model.
At each rebalance, it ranks assets by trailing 12-month return, holds the strongest asset
if that return is positive, and otherwise holds cash. This is more concentrated than
equal-weight and should be judged against drawdown and turnover, not only final return.

Dual momentum can be diversified and risk-adjusted:

```bash
python us_rotation_run.py --etf-preset us_asset_classes --start 2018-01-01 --allocation dual_momentum --momentum-top-n 3
python us_rotation_run.py --etf-preset us_asset_classes --start 2018-01-01 --allocation dual_momentum --momentum-top-n 3 --momentum-score-mode blended_risk_adjusted --momentum-confirmation fast
```

The protective dual-momentum variant keeps top-1 raw momentum but scales exposure by
market breadth. A defensive ETF can replace idle cash:

```bash
python us_rotation_run.py --etf-preset us_asset_classes --start 2018-01-01 --initial-capital 50000 --allocation protective_dual_momentum --defensive-asset BIL
python us_rotation_run.py --etf-preset us_asset_classes --start 2018-01-01 --initial-capital 50000 --allocation dual_momentum --breadth-adjusted --defensive-asset BIL
```

An optional regime filter implements the PCA -> k-means -> classifier framework from
the regime-switching paper. It uses price-derived market, volatility, drawdown, and
breadth features, then reduces risk exposure during detected stress regimes:

```bash
python us_rotation_run.py --etf-preset us_asset_classes --start 2018-01-01 --initial-capital 50000 --allocation dual_momentum --regime-filter --regime-defensive-asset BIL --regime-stress-exposure 0.50
```

In the dashboard, enable this from the sidebar under `Regime Filter`. The Risk tab then
plots the detected `regime_stress` series alongside exposure and turnover.

Run a clean ablation table:

```bash
python us_rotation_run.py --etf-preset us_asset_classes --start 2018-01-01 --initial-capital 50000 --dual-momentum-ablation
```

Validate tolerance choices out of sample with anchored walk-forward testing:

```bash
python us_rotation_run.py --etf-preset us_asset_classes --start 2018-01-01 --initial-capital 50000 --allocation equal_weight --walk-forward-tolerance --walk-forward-metric sharpe
```

This optimizes the tolerance only on prior data, then applies the chosen value to the
next out-of-sample year. It is meant to validate whether a tolerance rule is robust, not
to guarantee that it beats the best fixed tolerance chosen with hindsight.

`--momentum-top-n` holds the top positive-momentum assets instead of only the top one.
`--momentum-score-mode risk_adjusted` ranks by 12-month return per unit of realized
volatility. `--momentum-score-mode blended_risk_adjusted` combines 6-month and 12-month
risk-adjusted momentum. `--momentum-confirmation fast` requires positive recent momentum
and price above its 200-day moving average.

Use tolerance bands to reduce rebalancing trades. A value of `0.05` means the strategy
only trades at a rebalance date if at least one holding is more than five percentage
points away from its target weight:

```bash
python china_rotation_run.py --period max --allocation equal_weight --rebalance-tolerance 0.05
```

To test corridor-to-target trading instead of full target rebalancing:

```bash
python china_rotation_run.py --period max --allocation equal_weight --rebalance-tolerance 0.05 --rebalance-to corridor
```

To test asymmetric asset-specific tolerance bands:

```bash
python china_rotation_run.py --period max --allocation equal_weight --rebalance-to corridor --tolerance-bands 510300.SS=0.025,159915.SZ=0.075
```

To decide whether rebalancing is worth the transaction cost, run a tolerance sweep:

```bash
python china_rotation_run.py --period max --initial-capital 50000 --allocation equal_weight --risk-mode none --tolerance-sweep
```

You can also include the same tolerance in a grid search:

```bash
python china_rotation_run.py --period max --initial-capital 50000 --rebalance-tolerance 0.05 --grid-search
```

To test a defensive version that holds cash when the CSI 300 proxy is below its 200-day
moving average:

```bash
python china_rotation_run.py --period max --risk-mode market
```

Use your own universe:

```bash
python china_rotation_run.py --tickers 002594,300750,600519,000858,000333,600036 --top-n 3 --period max
```

Long `--period max` tests on the default China stock basket are survivor-biased because
the basket is made of current leaders. Use those runs as behavior checks, not proof that
the same strategy was selectable years ago. If your manual ticker list is genuinely
point-in-time for the start date, mark it:

```bash
python china_rotation_run.py --tickers 600519,000858,000333,600036,601318,600276 --start 2018-01-01 --point-in-time-universe
```

Cleaner alternatives are ETF universes or a point-in-time universe file:

```bash
python china_rotation_run.py --etf-preset china_broad_etfs --start 2020-01-01 --initial-capital 50000
python china_rotation_run.py --universe-file china_point_in_time_universe_template.csv --start 2018-01-01 --initial-capital 50000
```

The China template file is only an example format. For a serious bias-controlled
backtest, fill it with constituents that were actually knowable on each snapshot date.

Use a precise historical window:

```bash
python china_rotation_run.py --start 2016-01-01 --end 2026-01-01 --top-n 4
```

Try weekly rebalancing:

```bash
python china_rotation_run.py --rebalance W --top-n 4
```

The rotation strategy reports the strategy return, CSI 300 proxy return, and equal-weight
universe return. That makes it easier to see whether the selection/risk filter adds value
over simply holding the basket.

`equal_weight_return` is a frictionless daily equal-weight benchmark. The strategy can
underperform it when tolerance bands, trading costs, or drifted weights reduce exposure to
the best-performing names. Use `excess_vs_equal_weight` to see this gap directly.

## Paper Deployment

The frozen paper-trading rule is:

```text
China leader basket + equal-weight target + monthly check + 10% rebalance tolerance + no market timing
```

Create a holdings file:

```text
ticker,shares
600519.SS,100
000858.SZ,200
cash,1200
```

Generate manual rebalance suggestions:

```bash
python china_rotation_live.py --holdings holdings_sample.csv --rebalance-tolerance 0.10
```

For corridor-to-target live order suggestions:

```bash
python china_rotation_live.py --holdings holdings_sample.csv --rebalance-tolerance 0.10 --rebalance-to corridor
```

For asymmetric live bands:

```bash
python china_rotation_live.py --holdings holdings_sample.csv --rebalance-to corridor --tolerance-bands 600519.SS=0.025,510300.SS=0.075
```

Append the run to a paper-trading journal:

```bash
python china_rotation_live.py --holdings holdings_sample.csv --rebalance-tolerance 0.10 --journal paper_trading_journal.csv --note "monthly check"
```

After the initial paper fill, use `paper_holdings.csv`:

```bash
python china_rotation_live.py --holdings paper_holdings.csv --rebalance-tolerance 0.10 --lot-size 1 --journal paper_trading_journal.csv --note "weekly paper check"
```

Track actual paper decisions and fills in `paper_fills_template.csv`:

```text
date,ticker,action,suggested_shares,filled_shares,suggested_price,fill_price,slippage,reason
```

For real A-share lot-size feasibility, run:

```bash
python china_rotation_live.py --holdings holdings_sample.csv --rebalance-tolerance 0.10 --lot-size 100
```

With a 50,000 RMB sample portfolio, this currently leaves many desired trades skipped as
`below_lot_size`, which means a real-money version would need more capital, fewer/lower
priced stocks, or ETF proxies.

If the account has enough room under its order and concentration caps, you can let the
live helper reallocate skipped/rounded buy cash into valid lower-priced buy lots:

```bash
python china_rotation_live.py --holdings holdings_sample.csv --rebalance-tolerance 0.10 --lot-size 100 --redistribute-skipped-cash
```

The live helper prints portfolio value, cash percentage, current weights, target weights,
max drift, rebalance decision, estimated turnover, estimated cost, and suggested orders.
It does not place trades.

China A-shares usually trade in 100-share buy lots. With small capital, equal-weighting
all 12 names may be impossible because higher-priced names can fall below one lot at the
target weight. The live helper reports these as `below_lot_size`. For paper-only
fractional testing, use:

```bash
python china_rotation_live.py --holdings holdings_sample.csv --lot-size 1
```

## US Paper Deployment

The U.S. counterpart uses the same deployment pattern:

```text
US leader basket + equal-weight target + monthly check + 10% rebalance tolerance + no market timing
```

Backtest:

```bash
python us_rotation_run.py --period 5y --initial-capital 50000 --allocation equal_weight --risk-mode none --rebalance-tolerance 0.10
```

Long `--period max` tests on the default U.S. basket are survivor-biased because the
universe is made of current winners/leaders. Use those runs as behavior checks, not proof
that the strategy would have been selectable years ago. If you provide a true
point-in-time universe for the chosen start date, mark it explicitly:

```bash
python us_rotation_run.py --tickers AAPL,MSFT,JPM,... --start 2018-01-01 --point-in-time-universe
```

Cleaner alternatives:

```bash
python us_rotation_run.py --etf-preset us_sectors --start 2018-01-01 --initial-capital 50000
python us_rotation_run.py --etf-preset us_asset_classes --start 2018-01-01 --initial-capital 50000
```

Or provide a point-in-time universe file with `date,ticker` rows:

```bash
python us_rotation_run.py --universe-file us_point_in_time_universe_template.csv --start 2018-01-01 --initial-capital 50000
```

To decide whether trading is worth the transaction cost, run a tolerance sweep:

```bash
python us_rotation_run.py --universe-file us_point_in_time_universe_template.csv --start 2018-01-01 --initial-capital 50000 --allocation equal_weight --risk-mode none --tolerance-sweep
```

This ranks rebalance tolerances by final equity and reports turnover saved versus the
zero-tolerance baseline. Use this as a research diagnostic; do not keep retuning the
deployment threshold without out-of-sample or paper-trading validation.

The template file is only an example format. For a serious bias-controlled backtest, fill
it with constituents that were actually knowable on each snapshot date.

After using a point-in-time universe, it is normal for the deployed strategy to trail
`equal_weight_return`. That benchmark is frictionless, while the strategy includes costs
and tolerance-band drift.

Generate initial paper orders from cash:

```bash
python us_rotation_live.py --holdings us_holdings_sample.csv --rebalance-tolerance 0.10
```

Run weekly/monthly paper checks after simulated fills:

```bash
python us_rotation_live.py --holdings us_paper_holdings.csv --rebalance-tolerance 0.10 --journal us_paper_trading_journal.csv --note "weekly paper check"
```

Run a U.S. grid search:

```bash
python us_rotation_run.py --period 5y --initial-capital 50000 --rebalance-tolerance 0.10 --grid-search
```

## Trading Logic

Entry:

```text
residual_z < -2
RSI < 35
market_close > 200-day moving average
```

Exit:

```text
residual_z > -0.5
or max holding period is reached
or market regime turns risk-off
```

Position size:

```text
position_fraction = risk_per_trade / max(ATR stop distance, minimum stop distance)
```

The backtest reports total return, CAGR, Sharpe, max drawdown, exposure, turnover, trade
count, win rate, average trade return, and buy-and-hold return.
