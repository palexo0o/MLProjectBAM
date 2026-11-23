"""Build every number and figure cited in the C.A.S.A. report from this repository.

Run from the repo root:

    python -m src.model.report_assets --output /path/to/latex/project

It writes into --output:
    figures/*.pdf          the report figures (Okabe-Ito colour key from src/model/style.py)
    report_results.json    every number quoted in the report text and tables
    flag_scores.csv        per-flat percentage gap, interval and interval score

Nothing in the repository is modified. The flags are recomputed with the same code path as
src/model/fair_value.py and checked against the committed labels file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.data_acquire import DATA_PATH, SHEET, load_df
from src.data.preprocess import CATEGORICAL_COLS, MIN_FREQ, NUMERIC_COLS, prepare_xy
from src.model.fair_value import ALPHA, N_FOLDS, OUTPUT, SEED, euro_metrics, fit_intervals, label_from_score, make_models
from src.model.style import LABEL_COL

CENTRE = (40.4168, -3.7038)  # Puerta del Sol
GEO = REPO / "src" / "data" / "geo"
# Nominatim result types that only locate a neighbourhood or a landmark, not a street address.
APPROX_TYPES = {"suburb", "neighbourhood", "quarter", "city", "town", "village", "locality", "administrative",
                "station", "bus_stop", "platform"}
OVER, FAIR, BELOW = LABEL_COL["overpriced"], LABEL_COL["fair"], LABEL_COL["below_typical"]


def haversine_km(lat, lon, lat0, lon0):
    phi, phi0 = np.radians(lat), np.radians(lat0)
    dphi, dlam = phi - phi0, np.radians(lon - lon0)
    h = np.sin(dphi / 2) ** 2 + np.cos(phi) * np.cos(phi0) * np.sin(dlam / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def gb_pipeline(num_cols, cat_cols) -> Pipeline:
    """Gradient boosting with the repository's settings, on a chosen set of columns."""
    params = make_models()["gradient_boosting"].named_steps["model"].get_params()
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_FREQ,
                                                   sparse_output=False))]), cat_cols),
    ])
    return Pipeline([("pre", pre), ("model", HistGradientBoostingRegressor(**params))])


def euro_ticks(ax, ticks, both=False):
    """Plain euro tick labels on log axes (no scientific notation)."""
    import matplotlib.ticker as mtick
    axes = [ax.xaxis, ax.yaxis] if both else [ax.yaxis]
    for axis in axes:
        axis.set_major_locator(mtick.FixedLocator(ticks))
        axis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        axis.set_minor_formatter(mtick.NullFormatter())


def load_geocodes(df: pd.DataFrame):
    """Geocoded coordinates per flat; points outside the municipal barrio polygons are treated as missing."""
    import geopandas as gpd
    from src.data.geo.geocode import make_query

    query = [make_query(a, n) for a, n in zip(df["Address"], df["Number"])]
    geo = pd.read_csv(GEO / "geocoded_addresses.csv").drop_duplicates("query")
    m = pd.DataFrame({"query": query}).merge(geo[["query", "lat", "lon", "osm_type"]], on="query", how="left")
    barrios = gpd.read_file(GEO / "barrios" / "BARRIOS.shp").to_crs(4326)
    pts = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m["lon"], m["lat"]), crs=4326)
    joined = gpd.sjoin(pts, barrios[["NOMBRE", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].sort_index()
    inside = joined["NOMBRE"].notna().to_numpy()
    lat = np.where(inside, m["lat"], np.nan)
    lon = np.where(inside, m["lon"], np.nan)
    quality = np.where(~inside, "missing", np.where(m["osm_type"].isin(APPROX_TYPES), "area-level", "street-level"))
    return lat, lon, joined["NOMBRE"].where(inside).to_numpy(), quality, barrios


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, required=True, help="LaTeX project folder to write figures and results into")
    args = ap.parse_args()
    out = args.output.resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                         "font.family": "DejaVu Sans", "text.color": "#1f1f1e", "axes.labelcolor": "#1f1f1e",
                         "xtick.color": "#1f1f1e", "ytick.color": "#1f1f1e", "legend.frameon": False})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    S: dict = {}

    # ---------------- Data -------------------------------------------------------------
    raw = pd.read_excel(DATA_PATH, sheet_name=SHEET)
    n_dedup = len(raw.drop_duplicates(subset=[c for c in raw.columns if c != "Id"]))
    df = load_df()
    X, y = prepare_xy(df)
    a = y.to_numpy()
    ylog = np.log(a)
    S["data"] = {"n_raw": len(raw), "n_after_dedup": n_dedup, "n_duplicates": len(raw) - n_dedup,
                 "n_trimmed": n_dedup - len(df), "n": len(df), "rent_min": float(a.min()), "rent_max": float(a.max()),
                 "rent_median": float(np.median(a)), "sqm_range": [int(df["Sq.Mt"].min()), int(df["Sq.Mt"].max())],
                 "districts": int(df["District"].nunique()), "areas": int(df["Area"].nunique()),
                 "area_missing": int(df["Area"].isna().sum()),
                 "missing_pct": {c: round(float(X[c].isna().mean() * 100), 2) for c in NUMERIC_COLS + CATEGORICAL_COLS}}

    # ---------------- Model comparison (5-fold out-of-fold) ---------------------------------
    cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    preds, metrics, folds = {}, {}, {}
    for name, model in make_models().items():
        p = np.exp(cross_val_predict(model, X, ylog, cv=cv))
        preds[name] = p
        metrics[name] = euro_metrics(a, p)
        folds[name] = [float(np.mean(np.abs(a[te] - p[te]))) for _, te in cv.split(X)]
    gb = preds["gradient_boosting"]
    S["metrics"] = metrics
    S["fold_mae"] = folds
    S["fold_mae_sd"] = {k: float(np.std(v, ddof=1)) for k, v in folds.items()}
    S["gb_params"] = {k: v for k, v in make_models()["gradient_boosting"].named_steps["model"].get_params().items()
                      if k in ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")}
    d = np.abs(a - preds["ridge"]) - np.abs(a - gb)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(2000)])
    S["bootstrap_ridge_minus_gb"] = {"mean": float(d.mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist()}
    S["gb_wins_pct"] = float(np.mean(np.abs(a - gb) < np.abs(a - preds["ridge"])) * 100)
    S["gb_vs_baseline_reduction_pct"] = float((1 - metrics["gradient_boosting"]["MAE_eur"] / metrics["baseline_median"]["MAE_eur"]) * 100)
    S["gb_vs_ridge_reduction_pct"] = float((1 - metrics["gradient_boosting"]["MAE_eur"] / metrics["ridge"]["MAE_eur"]) * 100)

    # ---------------- Percentage gap (the report's score S = 100 g) and the fixed band ------
    g = a / gb - 1
    S["gap"] = {"median_pct": float(np.median(g) * 100),
                "band10": {"above": int((g > 0.10).sum()), "within": int((np.abs(g) <= 0.10).sum()), "below": int((g < -0.10).sum())},
                "thresholds": {str(t): float(np.mean(np.abs(g) <= t / 100) * 100) for t in range(5, 45, 5)}}

    # ---------------- Calibrated intervals (same code path as fair_value.py) -----------------
    iv = fit_intervals(X, ylog, "gradient_boosting")
    half = iv["q_het"] * iv["sigma_all"]
    score = np.log(a / gb) / half
    labels = label_from_score(score)
    committed = pd.read_csv(OUTPUT)
    S["consistency_with_committed_labels"] = {
        "labels_identical": bool((committed["label"].to_numpy() == labels).all()),
        "max_score_difference": float(np.max(np.abs(committed["score"].to_numpy() - np.round(score, 3)))),
    }
    te_pred, y_te = iv["test_pred"], iv["y_test"]
    rent_te = np.exp(te_pred)
    qtile = np.clip(np.searchsorted(np.quantile(rent_te, [0, .25, .5, .75, 1]), rent_te, side="right") - 1, 0, 3)
    half_const = np.full(len(te_pred), iv["q_const"])
    by_q = []
    for k in range(4):
        m = qtile == k
        by_q.append({"quartile": k + 1, "n": int(m.sum()),
                     "pred_range": [float(rent_te[m].min()), float(rent_te[m].max())],
                     "coverage_const_pct": float(iv["covered_const"][m].mean() * 100),
                     "coverage_het_pct": float(iv["covered_het"][m].mean() * 100),
                     "median_halfwidth_const_eur": float(np.median(rent_te[m] * (np.exp(half_const[m]) - 1))),
                     "median_halfwidth_het_eur": float(np.median(rent_te[m] * (np.exp(iv["half_het_test"][m]) - 1)))})
    width_het = rent_te * (np.exp(iv["half_het_test"]) - np.exp(-iv["half_het_test"]))
    width_const = rent_te * (np.exp(iv["q_const"]) - np.exp(-iv["q_const"]))
    S["intervals"] = {
        "nominal_pct": int(round(100 * (1 - ALPHA))),
        "n_train": len(iv["idx_train"]), "n_cal": len(iv["idx_cal"]), "n_test": len(iv["idx_test"]),
        "q_const_log": iv["q_const"], "const_multipliers": [float(np.exp(-iv["q_const"])), float(np.exp(iv["q_const"]))],
        "q_het": iv["q_het"],
        "coverage_const_pct": float(iv["covered_const"].mean() * 100),
        "coverage_het_pct": float(iv["covered_het"].mean() * 100),
        "covered_het_n": int(iv["covered_het"].sum()),
        "median_width_const_eur": float(np.median(width_const)), "median_width_het_eur": float(np.median(width_het)),
        "by_predicted_quartile": by_q,
        "halfwidth_pct_of_pred": {"min": float((np.exp(half.min()) - 1) * 100), "median": float((np.exp(np.median(half)) - 1) * 100),
                                  "max": float((np.exp(half.max()) - 1) * 100)},
        "sigma_pct_5_50_95": [float(v) for v in np.percentile((np.exp(iv["sigma_all"]) - 1) * 100, [5, 50, 95])],
        "test_mae_eur": float(np.mean(np.abs(np.exp(y_te) - rent_te))),
    }
    S["flags"] = {"overpriced": int((labels == "overpriced").sum()), "fair": int((labels == "fair").sum()),
                  "below_typical": int((labels == "below_typical").sum()),
                  "median_rent": {k: float(np.median(a[labels == k])) for k in ("overpriced", "fair", "below_typical")},
                  "outside_band10_but_inside_interval": int(((np.abs(g) > 0.10) & (labels == "fair")).sum())}

    pd.DataFrame({"Id": df["Id"], "actual_eur": a, "predicted_eur": gb.round(2), "signed_gap_pct": (100 * g).round(2),
                  "magnitude_pct": (100 * np.abs(g)).round(2), "interval_low_eur": (gb * np.exp(-half)).round(2),
                  "interval_high_eur": (gb * np.exp(half)).round(2), "interval_score": score.round(3),
                  "label": labels}).to_csv(out / "flag_scores.csv", index=False)

    # ---------------- Observed-rent strata (interval flags) -----------------------------------
    strata = []
    for lo, hi in [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3501)]:
        m = (a >= lo) & (a < hi)
        strata.append({"band": f"{lo}-{hi}", "n": int(m.sum()), "mae": float(np.abs(a[m] - gb[m]).mean()),
                       "mape": float((np.abs(a[m] - gb[m]) / a[m]).mean() * 100), "median_gap_pct": float(np.median(g[m]) * 100),
                       "above_pct": float((labels[m] == "overpriced").mean() * 100),
                       "below_pct": float((labels[m] == "below_typical").mean() * 100)})
    S["strata"] = strata

    # ---------------- Permutation importance (80/20, seed 0) ------------------------------------
    tr, te = train_test_split(np.arange(len(a)), test_size=0.2, random_state=SEED)
    gbm = make_models()["gradient_boosting"].fit(X.iloc[tr], ylog[tr])
    pi = permutation_importance(gbm, X.iloc[te], ylog[te], scoring="neg_mean_absolute_error", n_repeats=10, random_state=SEED)
    S["importance"] = {c: float(v) for c, v in zip(X.columns, pi.importances_mean)}
    S["importance_test_n"] = len(te)

    # ---------------- Neighbourhoods (15 largest labels) -----------------------------------------
    nb = pd.DataFrame({"Area": df["Area"], "S": 100 * g, "flag": labels})
    largest = nb["Area"].value_counts().head(15).index
    nb_sum = (nb[nb["Area"].isin(largest)].groupby("Area")
              .agg(n=("S", "size"), median_gap_pct=("S", "median"), flagged=("flag", lambda s: int((s != "fair").sum())))
              .sort_values("median_gap_pct"))
    S["neighbourhoods"] = nb_sum.reset_index().to_dict(orient="records")

    # ---------------- Distance to the centre (geocoded addresses) --------------------------------
    lat, lon, barrio, quality, barrios = load_geocodes(df)
    ok = ~np.isnan(lat)
    dist = haversine_km(lat, lon, *CENTRE)
    Xd = X.loc[ok].reset_index(drop=True).copy()
    Xd["distance_km"] = dist[ok]
    ad = a[ok]
    variants = {"District only": (NUMERIC_COLS, ["District"]),
                "District + neighbourhood": (NUMERIC_COLS, ["District", "Area"]),
                "District + distance": (NUMERIC_COLS + ["distance_km"], ["District"]),
                "District + neighbourhood + distance": (NUMERIC_COLS + ["distance_km"], ["District", "Area"])}
    dres = {}
    for label, (nums, cats) in variants.items():
        p = np.exp(cross_val_predict(gb_pipeline(nums, cats), Xd, np.log(ad), cv=KFold(N_FOLDS, shuffle=True, random_state=SEED)))
        dres[label] = euro_metrics(ad, p)
    S["distance"] = {"n": int(ok.sum()), "n_excluded": int((~ok).sum()),
                     "street_level": int((quality == "street-level").sum()), "area_level": int((quality == "area-level").sum()),
                     "centre_latlon": list(CENTRE), "median_distance_km": float(np.median(dist[ok])), "metrics": dres}

    # ---------------- Figures ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(a, bins=30, color=FAIR)
    ax.axvline(np.median(a), color="black", ls="--", lw=1.5)
    ax.set(xlabel="Monthly asking rent (EUR)", ylabel="Listings")
    save(fig, "rents")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.scatter(df["Sq.Mt"], a, s=9, alpha=0.3, color=FAIR)
    ax.set(xscale="log", yscale="log", xlabel="Floor area (m², log scale)", ylabel="Monthly rent (EUR, log scale)")
    euro_ticks(ax, [500, 1000, 2000, 3000])
    save(fig, "size")

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    lo_all, hi_all = gb * np.exp(-half), gb * np.exp(half)
    edges = np.quantile(gb, np.linspace(0, 1, 31))
    which = np.clip(np.searchsorted(edges, gb, side="right") - 1, 0, 29)
    xs = [np.median(gb[which == k]) for k in range(30) if (which == k).any()]
    ls = [np.median(lo_all[which == k]) for k in range(30) if (which == k).any()]
    hs = [np.median(hi_all[which == k]) for k in range(30) if (which == k).any()]
    ax.fill_between(xs, ls, hs, color="#bdbdbd", alpha=0.45, lw=0, label="90% interval (median per bin)")
    for lab, col, name in [("overpriced", OVER, "Overpriced"), ("fair", FAIR, "Fair"), ("below_typical", BELOW, "Below typical")]:
        m = labels == lab
        ax.scatter(gb[m], a[m], s=10 if lab == "fair" else 18, alpha=0.35 if lab == "fair" else 0.9, color=col, label=name, lw=0)
    x = np.array([400, 4000])
    ax.plot(x, x, "k--", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="Out-of-fold predicted rent (EUR)", ylabel="Observed asking rent (EUR)")
    euro_ticks(ax, [500, 1000, 2000, 3000], both=True)
    ax.legend(fontsize=11, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, "predicted")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(np.clip(100 * g, -60, 60), bins=np.arange(-60, 62.5, 2.5), color=FAIR)
    ax.axvspan(-10, 10, color="#bdbdbd", alpha=0.5, label="±10% band")
    med_half = np.median(half)
    for v in [(np.exp(-med_half) - 1) * 100, (np.exp(med_half) - 1) * 100]:
        ax.axvline(v, color="black", ls="--", lw=1.5)
    ax.plot([], [], "k--", label="Median 90% interval")
    ax.set(xlabel="Observed / predicted rent − 1 (%)", ylabel="Listings")
    ax.legend(fontsize=12, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, "gaps")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bands = np.arange(5, 45, 5)
    ax.plot(bands, [S["gap"]["thresholds"][str(t)] for t in bands], "o-", color=FAIR)
    mape = metrics["gradient_boosting"]["MAPE_pct"]
    ax.axvline(mape, color=OVER, ls="--", lw=1.5, label=f"MAPE ({mape:.1f}%)")
    ax.axvline(10, color="black", ls=":", lw=1.5, label="10% band")
    ax.set(xlabel="Symmetric band around prediction (%)", ylabel="Listings within band (%)", ylim=(0, 100))
    ax.legend(fontsize=12, loc="lower right")
    save(fig, "thresholds")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ix = np.arange(len(strata))
    ax.bar(ix - 0.18, [s["above_pct"] for s in strata], 0.36, color=OVER, label="Overpriced")
    ax.bar(ix + 0.18, [s["below_pct"] for s in strata], 0.36, color=BELOW, label="Below typical")
    ax.set_xticks(ix, ["<1000", "1000–1499", "1500–1999", "2000–2499", "2500–3500"], rotation=15)
    ax.set(xlabel="Observed monthly rent (EUR)", ylabel="Outside 90% interval (%)")
    ax.legend(fontsize=12)
    save(fig, "strata")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    order = np.argsort(pi.importances_mean)
    ax.barh(X.columns[order], pi.importances_mean[order], xerr=pi.importances_std[order], color=FAIR)
    ax.set(xlabel="Increase in mean absolute log-rent error")
    save(fig, "importance")

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    colors = [OVER if v > 0 else BELOW for v in nb_sum["median_gap_pct"]]
    names = [f"{a} (n={n})" for a, n in zip(nb_sum.index, nb_sum["n"])]
    ax.barh(names, nb_sum["median_gap_pct"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set(xlabel="Median percentage gap, observed vs predicted (%)")
    save(fig, "neighborhoods")

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    pick = np.argsort(rent_te)[np.linspace(0, len(rent_te) - 1, 50, dtype=int)]
    lo_te, hi_te = rent_te * np.exp(-iv["half_het_test"]), rent_te * np.exp(iv["half_het_test"])
    obs = np.exp(y_te)
    pos = np.arange(len(pick))
    ax.vlines(pos, lo_te[pick], hi_te[pick], color="#929292", alpha=0.8, lw=1.4)
    ax.scatter(pos, rent_te[pick], c="#333333", s=10, label="Prediction")
    col = np.where(obs[pick] > hi_te[pick], OVER, np.where(obs[pick] < lo_te[pick], BELOW, FAIR))
    ax.scatter(pos, obs[pick], c=col, s=22, label="Observed")
    ax.set(xlabel="50 test listings, sorted by prediction", ylabel="Monthly rent (EUR)")
    ax.legend(fontsize=12, loc="upper left")
    save(fig, "intervals")

    # Barrio maps from geocoded points (barrios with no flats are omitted; grey = fewer than 5 flats).
    plt.rcParams.update({"font.size": 13})
    per = (pd.DataFrame({"barrio": barrio, "pred": gb, "S": 100 * g}).dropna(subset=["barrio"])
           .groupby("barrio").agg(n=("pred", "size"), pred=("pred", "median"), S=("S", "median")))
    shown = barrios[barrios["NOMBRE"].isin(per.index)].merge(per, left_on="NOMBRE", right_index=True)
    S["maps"] = {"flats_placed": int(ok.sum()), "barrios_with_flats": int(len(per)), "barrios_at_least5": int((per["n"] >= 5).sum())}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    diverging = LinearSegmentedColormap.from_list("gap", [BELOW, "#f7f7f7", OVER])
    for axm, colname, cmap, title in [(axes[0], "pred", "Blues", "Median predicted rent (EUR/month)"),
                                      (axes[1], "S", diverging, "Median percentage gap (%)")]:
        shown.plot(ax=axm, color="#dddddd", edgecolor="white", linewidth=0.3)
        big = shown[shown["n"] >= 5]
        if colname == "S":
            lim = float(np.nanmax(np.abs(big["S"])))
            big.plot(ax=axm, column="S", cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), edgecolor="white", linewidth=0.3,
                     legend=True, legend_kwds={"shrink": 0.7})
        else:
            big.plot(ax=axm, column="pred", cmap=cmap, edgecolor="white", linewidth=0.3, legend=True, legend_kwds={"shrink": 0.7})
        axm.scatter([CENTRE[1]], [CENTRE[0]], marker="*", s=70, c="black", zorder=4)
        axm.set_title(title)
        axm.set_aspect(1 / np.cos(np.radians(40.42)))
        axm.axis("off")
    save(fig, "barrio_maps")

    # ---------------- Computing setup ------------------------------------------------------
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    import geopandas
    mem = sysctl("hw.memsize")
    S["computing"] = {"platform": platform.platform(), "machine": platform.machine(), "cpu": sysctl("machdep.cpu.brand_string"),
                      "logical_cpus": os.cpu_count(), "memory_gb": round(int(mem) / 2 ** 30) if mem else None,
                      "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                      "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__, "geopandas": geopandas.__version__,
                      "wall_seconds": round(time.perf_counter() - t0, 1)}

    with open(out / "report_results.json", "w") as f:
        json.dump(S, f, indent=2, default=float)
    print(json.dumps(S, indent=2, default=float))


if __name__ == "__main__":
    main()
