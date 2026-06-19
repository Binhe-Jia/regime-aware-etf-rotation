from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    n_components: int = 3
    n_regimes: int = 2
    train_years: int = 3
    retrain_frequency: str = "M"
    random_state: int = 42


def build_regime_features(
    closes: pd.DataFrame,
    market_col: str,
    asset_cols: list[str] | None = None,
) -> pd.DataFrame:
    assets = asset_cols or [column for column in closes.columns if column != market_col]
    market = closes[market_col]
    market_returns = market.pct_change()
    asset_returns = closes.loc[:, assets].pct_change() if assets else pd.DataFrame(index=closes.index)
    equal_weight = asset_returns.mean(axis=1) if len(asset_returns.columns) else market_returns
    breadth_200 = (
        closes.loc[:, assets].gt(closes.loc[:, assets].rolling(200).mean()).mean(axis=1)
        if assets
        else pd.Series(np.nan, index=closes.index)
    )

    features = pd.DataFrame(
        {
            "market_return_21": market.pct_change(21),
            "market_return_63": market.pct_change(63),
            "market_return_126": market.pct_change(126),
            "market_volatility_21": market_returns.rolling(21).std() * np.sqrt(252),
            "market_volatility_63": market_returns.rolling(63).std() * np.sqrt(252),
            "market_drawdown_252": market / market.rolling(252).max() - 1.0,
            "equal_weight_return_21": equal_weight.rolling(21).apply(
                lambda values: np.prod(1.0 + values) - 1.0,
                raw=True,
            ),
            "equal_weight_return_63": equal_weight.rolling(63).apply(
                lambda values: np.prod(1.0 + values) - 1.0,
                raw=True,
            ),
            "asset_breadth_200": breadth_200,
            "cross_asset_volatility": asset_returns.rolling(21).std().mean(axis=1) * np.sqrt(252),
        },
        index=closes.index,
    )
    return features.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def _retrain_dates(index: pd.DatetimeIndex, frequency: str) -> set[pd.Timestamp]:
    period = index.to_period("W-FRI" if frequency.upper().startswith("W") else "M")
    marker = pd.Series(period, index=index)
    return set(marker.index[marker != marker.shift(1)])


def _stress_label(features: pd.DataFrame, labels: np.ndarray) -> int:
    labelled = features.copy()
    labelled["label"] = labels
    grouped = labelled.groupby("label").agg(
        {
            "market_return_63": "mean",
            "market_volatility_63": "mean",
            "market_drawdown_252": "mean",
            "asset_breadth_200": "mean",
        }
    )
    score = (
        -grouped["market_return_63"].fillna(0.0)
        + grouped["market_volatility_63"].fillna(0.0)
        - grouped["market_drawdown_252"].fillna(0.0)
        - grouped["asset_breadth_200"].fillna(0.5)
    )
    return int(score.idxmax())


def generate_regime_signals(
    closes: pd.DataFrame,
    market_col: str,
    asset_cols: list[str] | None = None,
    config: RegimeConfig | None = None,
) -> pd.DataFrame:
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Regime filtering requires scikit-learn. Install scikit-learn.") from exc

    cfg = config or RegimeConfig()
    features = build_regime_features(closes, market_col, asset_cols)
    output = pd.DataFrame(
        {
            "regime": np.nan,
            "regime_stress": False,
            "regime_confidence": np.nan,
        },
        index=closes.index,
    )
    if features.empty:
        return output

    train_days = max(int(cfg.train_years * 252), 252)
    retrain_dates = _retrain_dates(features.index, cfg.retrain_frequency)
    fitted = None
    stress_cluster = None

    for i, date in enumerate(features.index):
        if i < train_days:
            continue
        if fitted is None or date in retrain_dates:
            train = features.iloc[:i].dropna()
            if len(train) < train_days:
                continue
            scaler = StandardScaler()
            scaled = scaler.fit_transform(train)
            n_components = max(
                1,
                min(cfg.n_components, scaled.shape[1], scaled.shape[0] - 1),
            )
            pca = PCA(n_components=n_components, random_state=cfg.random_state)
            pca_train = pca.fit_transform(scaled)
            n_regimes = max(2, min(cfg.n_regimes, len(train)))
            kmeans = KMeans(n_clusters=n_regimes, random_state=cfg.random_state, n_init=100)
            labels = kmeans.fit_predict(pca_train)
            stress_cluster = _stress_label(train, labels)
            classifier = AdaBoostClassifier(random_state=cfg.random_state, n_estimators=50)
            classifier.fit(pca_train, labels)
            fitted = (scaler, pca, classifier)

        if fitted is None or stress_cluster is None:
            continue
        scaler, pca, classifier = fitted
        row = features.loc[[date]]
        pca_row = pca.transform(scaler.transform(row))
        label = int(classifier.predict(pca_row)[0])
        output.loc[date, "regime"] = label
        output.loc[date, "regime_stress"] = label == stress_cluster
        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(pca_row)[0]
            output.loc[date, "regime_confidence"] = float(probabilities.max())

    output["regime_stress"] = output["regime_stress"].fillna(False).astype(bool)
    return output.ffill()
