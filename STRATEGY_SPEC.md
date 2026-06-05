# China Leader Basket Deployment Spec

This strategy is frozen for paper-trading deployment.

## Universe

```text
600519.SS
000858.SZ
002594.SZ
300750.SZ
000333.SZ
600036.SS
601318.SS
601888.SS
600276.SS
000651.SZ
300760.SZ
601012.SS
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
- Prefer patient limit orders.
- Avoid chasing the open and close.
- Skip or manually review names with missing data, zero volume, or likely price-limit moves.
- Ignore small trades below the configured minimum trade value.
- Cap a single order at the configured maximum order fraction of account value.

## Deployment Stage

Paper trade first. Log each run, suggested order, fill decision, fill price, slippage, and notes before considering real-money deployment.
