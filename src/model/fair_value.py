"""Fair-value model: estimate what a Madrid flat should rent for, then label listings by how far they sit from that estimate.

Run from the repo root:  python -m src.model.fair_value

Labels are relative to comparable listings in this dataset, not to true market value.
Evaluation of the 3-way labels is circular (they come from the same model), so the
honest metric is the euro error of the out-of-fold price predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from src.data.data_acquire import load_df
    from src.data.preprocess import TARGET, build_preprocessor, prepare_xy
except ModuleNotFoundError:
    from data.data_acquire import load_df
    from data.preprocess import TARGET, build_preprocessor, prepare_xy


SEED = 0
N_FOLDS = 5
# Relative gap between actual and predicted price that counts as over/under priced.
BAND = 0.10
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
                    min_samples_leaf=20,
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


def label_from_gap(gap: np.ndarray) -> np.ndarray:
    out = np.full(gap.shape, "fair", dtype=object)
    out[gap > BAND] = "overpriced"
    out[gap < -BAND] = "below_typical"
    return out


def main() -> None:
    df = load_df()
    X, y = prepare_xy(df)
    y_log = np.log(y.to_numpy())
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    results = {}
    oof_preds = {}
    for name, model in make_models().items():
        oof_log = cross_val_predict(model, X, y_log, cv=cv)
        pred_eur = np.exp(oof_log)
        oof_preds[name] = pred_eur
        results[name] = euro_metrics(y.to_numpy(), pred_eur)

    report = pd.DataFrame(results).T.round(3)
    print(f"Target: {TARGET} (n={len(y)}), {N_FOLDS}-fold out-of-fold predictions\n")
    print(report.to_string())

    best = min(
        (n for n in results if n != "baseline_median"),
        key=lambda n: results[n]["MAE_eur"],
    )
    print(f"\nBest model by MAE (euros): {best}")

    pred = oof_preds[best]
    actual = y.to_numpy()
    gap = actual / pred - 1
    labels = label_from_gap(gap)

    out = pd.DataFrame(
        {
            "Id": df["Id"].to_numpy(),
            "actual_eur": actual,
            "predicted_eur": pred.round(2),
            "gap_pct": (gap * 100).round(2),
            "label": labels,
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    print(f"\nLabel distribution (band ±{int(BAND * 100)}%):")
    print(out["label"].value_counts().to_string())
    print(f"\nSaved labels to {OUTPUT}")


if __name__ == "__main__":
    main()
