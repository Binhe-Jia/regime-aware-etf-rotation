# US Leader Basket Deployment Spec

This strategy is frozen for paper-trading deployment.

## Universe

```text
AAPL
MSFT
NVDA
AMZN
GOOGL
META
BRK-B
JPM
LLY
AVGO
TSLA
COST
```

## Allocation

Target equal weight across all 12 stocks.

## Rebalance Rule

- Check monthly.
- Target weight per stock: `1 / 12`.
- Rebalance only if any holding is more than 10 percentage points away from target.
- No market timing filter.

## Execution Rule

- Generate suggested orders only; review manually before trading.
- Prefer limit orders for less liquid names or volatile days.
- Avoid chasing the market open.
- Ignore small trades below the configured minimum trade value.
- Cap a single order at the configured maximum order fraction of account value.

## Deployment Stage

Paper trade first. Log each run, suggested order, fill decision, fill price, slippage, and notes before considering real-money deployment.

## Backtest Limitation

The default U.S. universe is a current leader basket. Long historical backtests on this
fixed list are survivor-biased and should not be read as evidence that the same strategy
was available in 2012. Treat them as a stress/behavior test for the frozen basket, not as
a clean point-in-time strategy proof.

For a less biased historical strategy test, use a point-in-time universe file that contains
the constituents that would have been knowable at the chosen start date.
