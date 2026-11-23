"""Fair-value model: estimate what a Madrid flat should rent for, then flag listings that sit outside a prediction interval.

Run from the repo root:  python -m src.model.fair_value

How a flag is made
------------------
1. The best model predicts log rent for every flat out-of-fold (5-fold), so no flat is scored by a model that saw it.
2. The interval width varies by flat. A scale model learns how large the typical error is for each flat's features
   (fitted on out-of-fold residuals of the training split). Split-conformal calibration on a held-out split then sets a
   multiplier q, so each flat's half-width is q * sigma(x). Coverage is checked on a separate test split.
3. Each flat gets an interval [pred * exp(-h), pred * exp(h)] with h = q * sigma(x), and a score = log(actual / pred) / h.
   Score > 1 means the rent sits above the interval (overpriced); score < -1 means below it (below typical).
   The score is a ranking measure, not a probability.

Flags are relative to comparable flats in this dataset, not to true market value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from src.data.data_acquire import load_df
    from src.data.preprocess import TARGET, build_preprocessor, prepare_xy
except ModuleNotFoundError:
    from data.data_acquire import load_df
    from data.preprocess import TARGET, build_preprocessor, prepare_xy


SEED = 0
N_FOLDS = 5
ALPHA = 0.10            # 1 - ALPHA = 90% prediction interval
CAL_FRAC = 0.20         # share of flats held out to calibrate the interval
TEST_FRAC = 0.20        # share held out to check its coverage
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "fair_value_labels.csv"


def make_models() -> dict[str, Pipeline]:
    ridge = Pipeline(
        [
            ("pre", build_preprocessor()),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-2, 3, 30))),
        ]
    )
    gbm = Pipeline(
        [
            ("pre", build_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=15,
                    min_samples_leaf=10,
                    l2_regularization=1.0,
                    random_state=SEED,
                ),
            ),
        ]
    )
    baseline = Pipeline([("pre", build_preprocessor()), ("model", DummyRegressor(strategy="median"))])
    return {"baseline_median": baseline, "ridge": ridge, "gradient_boosting": gbm}


def euro_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - actual
    return {
        "MAE_eur": float(np.mean(np.abs(err))),
        "MedAE_eur": float(np.median(np.abs(err))),
        "MAPE_pct": float(np.mean(np.abs(err) / actual) * 100),
        "R2_log": float(1 - np.sum((np.log(pred) - np.log(actual)) ** 2) / np.sum((np.log(actual) - np.log(actual).mean()) ** 2)),
    }


def conformal_halfwidth(abs_log_resid: np.ndarray, alpha: float = ALPHA) -> float:
    """Split-conformal quantile with the finite-sample correction ceil((n + 1)(1 - alpha)) / n."""
    n = len(abs_log_resid)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(abs_log_resid)[min(k, n) - 1])


SCALE_EPS = 1e-3


def make_scale_model() -> Pipeline:
    """Predicts log|out-of-fold log residual| from the flat's features: a learned, per-flat error scale."""
    return Pipeline(
        [
            ("pre", build_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05, max_iter=150, max_leaf_nodes=15, min_samples_leaf=30,
                    l2_regularization=1.0, random_state=SEED,
                ),
            ),
        ]
    )


def coverage_by_group(covered: np.ndarray, pred: np.ndarray, groups: int = 4) -> pd.DataFrame:
    """Coverage within bands of predicted rent: a flat-wide average can hide under-covered price ranges."""
    edges = np.quantile(pred, np.linspace(0, 1, groups + 1))
    band = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, groups - 1)
    rows = []
    for g in range(groups):
        m = band == g
        rows.append({"predicted_rent_band": f"€{edges[g]:,.0f}-€{edges[g + 1]:,.0f}", "n": int(m.sum()), "coverage_%": round(covered[m].mean() * 100, 1)})
    return pd.DataFrame(rows)


def score_from_gap(gap_log: np.ndarray, halfwidth: float) -> np.ndarray:
    """Gap in units of the interval half-width: |score| > 1 means outside the interval."""
    return gap_log / halfwidth


def label_from_score(score: np.ndarray) -> np.ndarray:
    out = np.full(score.shape, "fair", dtype=object)
    out[score > 1] = "overpriced"
    out[score < -1] = "below_typical"
    return out


def main() -> None:
    df = load_df()
    X, y = prepare_xy(df)
    y_log = np.log(y.to_numpy())
    actual = y.to_numpy()
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    results = {}
    oof_preds = {}
    for name, model in make_models().items():
        oof_log = cross_val_predict(model, X, y_log, cv=cv)
        pred_eur = np.exp(oof_log)
        oof_preds[name] = pred_eur
        results[name] = euro_metrics(actual, pred_eur)

    report = pd.DataFrame(results).T.round(3)
    print(f"Target: {TARGET} (n={len(y)}), {N_FOLDS}-fold out-of-fold predictions\n")
    print(report.to_string())

    best = min((n for n in results if n != "baseline_median"), key=lambda n: results[n]["MAE_eur"])
    print(f"\nBest model by MAE (euros): {best}")

    # Three-way split: train (fit point and scale models), calibration (set q), test (check coverage).
    idx = np.arange(len(y))
    idx_train, idx_rest = train_test_split(idx, test_size=CAL_FRAC + TEST_FRAC, random_state=SEED)
    idx_cal, idx_test = train_test_split(idx_rest, test_size=TEST_FRAC / (CAL_FRAC + TEST_FRAC), random_state=SEED)
    X_tr, X_cal, X_te = X.iloc[idx_train], X.iloc[idx_cal], X.iloc[idx_test]
    y_tr, y_cal, y_te = y_log[idx_train], y_log[idx_cal], y_log[idx_test]

    # Point model on the training split.
    point_model = make_models()[best].fit(X_tr, y_tr)
    cal_pred, te_pred = point_model.predict(X_cal), point_model.predict(X_te)

    # Scale model: learn each flat's typical error from OUT-OF-FOLD residuals inside the training split,
    # so the scale is not fitted to residuals the point model has already memorised.
    oof_tr = cross_val_predict(make_models()[best], X_tr, y_tr, cv=KFold(N_FOLDS, shuffle=True, random_state=SEED))
    scale_model = make_scale_model().fit(X_tr, np.log(np.abs(y_tr - oof_tr) + SCALE_EPS))
    sigma_all = np.exp(scale_model.predict(X))          # per-flat error scale, for every flat
    sigma_cal, sigma_te = sigma_all[idx_cal], sigma_all[idx_test]

    # Homoscedastic baseline (constant width), kept for comparison.
    q_const = conformal_halfwidth(np.abs(y_cal - cal_pred))
    # Heteroscedastic interval: conformal on normalised scores |r| / sigma.
    q_het = conformal_halfwidth(np.abs(y_cal - cal_pred) / sigma_cal)

    def summarise(name, half_te, covered):
        width_pct = np.median((np.exp(2 * half_te)) - 1) * 100
        return {"method": name, "test_coverage_%": round(covered.mean() * 100, 1),
                "median_width_%_of_pred": round(width_pct, 1), "n_test": len(covered)}

    cov_const = (y_te >= te_pred - q_const) & (y_te <= te_pred + q_const)
    half_het_te = q_het * sigma_te
    cov_het = (y_te >= te_pred - half_het_te) & (y_te <= te_pred + half_het_te)
    comparison = pd.DataFrame([
        summarise("constant width", np.full(len(y_te), q_const), cov_const),
        summarise("heteroscedastic", half_het_te, cov_het),
    ])
    print("\nInterval comparison on held-out test flats (target coverage {:.0f}%):".format(100 * (1 - ALPHA)))
    print(comparison.to_string(index=False))
    print("\nHeteroscedastic coverage by predicted-rent band (checks for under-covered price ranges):")
    print(coverage_by_group(cov_het, np.exp(te_pred)).to_string(index=False))

    # Flags for every flat use out-of-fold point predictions and the per-flat half-width.
    pred = oof_preds[best]
    gap = np.log(actual / pred)
    halfwidth = q_het * sigma_all
    score = gap / halfwidth
    labels = label_from_score(score)

    out = pd.DataFrame(
        {
            "Id": df["Id"].to_numpy(),
            "actual_eur": actual,
            "predicted_eur": pred.round(2),
            "gap_pct": ((np.exp(gap) - 1) * 100).round(2),
            "interval_low_eur": (pred * np.exp(-halfwidth)).round(2),
            "interval_high_eur": (pred * np.exp(halfwidth)).round(2),
            "score": score.round(3),
            "label": labels,
            "halfwidth_log": halfwidth.round(4),
            "sigma_log": sigma_all.round(4),
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    print(f"\nFlag distribution (outside the {100 * (1 - ALPHA):.0f}% heteroscedastic interval):")
    print(out["label"].value_counts().to_string())
    print(f"\nSaved labels to {OUTPUT}")


if __name__ == "__main__":
    main()
