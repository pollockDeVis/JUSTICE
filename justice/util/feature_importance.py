import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold

from justice.util.model_time import TimeHorizon
from justice.util.data_loader import DataLoader
from justice.util.visualizer import justice_region_aggregator


# -------------------------------
# Aggregation helper (returning years, inclusive slicing)
# -------------------------------
def aggregate_abated_emissions(
    input_data_arrays,
    region_mapping_path,
    rice_region_dict_path,  # kept for API compatibility
    start_year,
    end_year,
    splice_start_year,
    splice_end_year,
    data_timestep=5,
    timestep=1,
):
    """
    Aggregate abated emissions to 12 regions and slice to [splice_start_year, splice_end_year] inclusively.
    Returns: (aggregated_array, region_list, years_vector)
      - aggregated_array shape: (12, T, S) regions x years x samples from FaIR ensemble members
      - years_vector: numpy array of years of length T
    """
    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    data_loader = DataLoader()

    with open(region_mapping_path, "r") as f:
        region_mapping_json = json.load(f)

    # Often used internally by your stack; load to ensure availability
    with open(rice_region_dict_path, "r") as f:
        _ = json.load(f)

    last_agg = None
    region_list = None

    for economic_data in input_data_arrays:
        # Aggregate to 12 regions
        region_list, economic_data_aggregated = justice_region_aggregator(
            data_loader, region_mapping_json, economic_data
        )

        # Inclusive end-year slice (ensure 2100 is included if requested)
        start_idx = time_horizon.year_to_timestep(splice_start_year, timestep=timestep)
        end_idx = (
            time_horizon.year_to_timestep(splice_end_year, timestep=timestep) + 1
        )  # inclusive

        economic_data_aggregated = np.asarray(economic_data_aggregated)[
            :, start_idx:end_idx, :
        ]
        last_agg = economic_data_aggregated

    years = np.arange(splice_start_year, splice_end_year + 1, timestep)
    return last_agg, region_list, years


# -------------------------------
# Build long DataFrame (regional + global, keep samples)
# -------------------------------
def build_long_dataframe(
    base_path="data/temporary/NU_DATA/mmBorg/",
    model_output_dir=None,
    region_mapping_path="data/input/12_regions.json",
    rice_region_dict_path="data/input/rice50_regions_dict.json",
    years_of_interest=(2030, 2050, 2070, 2100),
):
    """
    Build a long/tidy DataFrame with columns:
    Optimization, Regret, Scenario, Welfare, Region, Year, Sample, AbatedEmission, Scope
    Includes both Regional and Global (summed across regions) entries for selected years.

    The dataframe looks like:

    | Optimization | Regret | Scenario | Welfare     | Region   | Year | Sample | AbatedEmission | Scope    |
    |--------------|--------|----------|-------------|----------|------|--------|----------------|----------|
    | SSP126       | Min    | SSP126   | Utilitarian | Region A | 2030 | 0      | 123.45         | Regional |
    | SSP126       | Min    | SSP126   | Utilitarian | Region A | 2030 | 1      | 130.67         | Regional |
    | ...          | ...    | ...      | ...         | ...      | ...  | ...    | ...            | ...      |

    """
    scenario_list = ["SSP126", "SSP245", "SSP370", "SSP460", "SSP534"]

    # Reads the policy indices mapping. This dict maps (reference_scenario, welfare_type) → {regret_type: policy_index}
    with open(base_path + "min_regret_policy_indices.json", "r") as f:
        policy_indices = json.load(f)

    # ── resolve model_output_dir ───────────────────────────────────────────────
    model_output_path = Path(model_output_dir) if model_output_dir else Path(base_path)

    # Load the baseline emissions for all SSPs (shape expected: regions x years x SSP scenarios)
    baseline_path = Path(base_path) / "emissions_array_all_SSPs.npy"
    baseline_emissions = np.load(baseline_path)

    rows = []

    # Loop over policy combinations For each (reference_scenario, welfare_type, regret, scenario):
    for reference_scenario, welfare_map in policy_indices.items():
        for welfare_type, regret_map in welfare_map.items():
            for scenario_index, scenario in enumerate(scenario_list):
                for regret in regret_map.keys():
                    policy_idx = regret_map[regret]

                    # Construct the filename for the emissions data based on the current policy combination§
                    emissions_file = (
                        model_output_path
                        / f"{welfare_type}_{reference_scenario}_{regret}"
                        f"_emissions_idx{policy_idx}_{scenario}_emissions.npy"
                    )
                    # Load the policy emission array (shape expected: regions x years x samples from FaIR ensemble members)
                    emissions_data = np.load(emissions_file)

                    # Compute the Abated emissions relative to baseline
                    abated_emissions = (
                        baseline_emissions[:, :, scenario_index][:, :, np.newaxis]
                        - emissions_data
                    )

                    # Aggregate to 12 regions and slice years inclusively - Shape returned: (12, T, samples)
                    region_agg_arr, region_list, agg_years = aggregate_abated_emissions(
                        input_data_arrays=[abated_emissions],
                        region_mapping_path=region_mapping_path,
                        rice_region_dict_path=rice_region_dict_path,
                        start_year=2015,
                        end_year=2300,
                        splice_start_year=2025,
                        splice_end_year=2100,  # inclusive in helper
                        data_timestep=5,
                        timestep=1,
                    )

                    # Select only desired years - Shape: (12, len(years_of_interest), samples)
                    year_mask = np.isin(agg_years, years_of_interest)
                    if not np.any(year_mask):
                        continue
                    years_sel = agg_years[year_mask]
                    arr_sel = region_agg_arr[:, year_mask, :]

                    n_regions, n_years, n_samples = arr_sel.shape

                    # Regional rows. Each row corresponds to one region-year-sample combination.
                    vals = arr_sel.reshape(-1)  # flatten: region->year->sample.
                    region_rep = np.repeat(region_list, n_years * n_samples)
                    year_rep = np.tile(np.repeat(years_sel, n_samples), n_regions)
                    sample_rep = np.tile(np.arange(n_samples), n_regions * n_years)

                    rows.append(
                        pd.DataFrame(
                            {
                                "Optimization": reference_scenario,
                                "Regret": regret,
                                "Scenario": scenario,
                                "Welfare": welfare_type,
                                "Region": region_rep,
                                "Year": year_rep,
                                "Sample": sample_rep,
                                "AbatedEmission": vals.astype(np.float64),
                                "Scope": "Regional",
                            }
                        )
                    )

                    # Global rows (sum across regions, per year and sample). Each row corresponds to one year-sample combination.
                    global_arr = arr_sel.sum(axis=0)  # (n_years, n_samples)
                    g_vals = global_arr.reshape(-1)
                    g_year_rep = np.repeat(years_sel, n_samples)
                    g_sample_rep = np.tile(np.arange(n_samples), n_years)

                    rows.append(
                        pd.DataFrame(
                            {
                                "Optimization": reference_scenario,
                                "Regret": regret,
                                "Scenario": scenario,
                                "Welfare": welfare_type,
                                "Region": "Global",
                                "Year": g_year_rep,
                                "Sample": g_sample_rep,
                                "AbatedEmission": g_vals.astype(np.float64),
                                "Scope": "Global",
                            }
                        )
                    )

    long_df = pd.concat(rows, ignore_index=True)

    # Optimize dtypes
    for col in ["Optimization", "Regret", "Scenario", "Welfare", "Region", "Scope"]:
        long_df[col] = long_df[col].astype("category")
    long_df["Year"] = long_df["Year"].astype(np.int32)
    long_df["Sample"] = long_df["Sample"].astype(np.int32)

    return long_df


# -------------------------------
# Build cell-level datasets (raw/mean/median/P90 targets)
# -------------------------------
def build_cell_level_targets(
    long_df, years=(2030, 2050, 2070, 2100), target_stat="mean", scope="Global"
):
    """
    This method Aggregates the long_df into the structure the ML model will see.
    Aggregate across 1001 samples to get one scalar target per cell, unless target_stat='raw'
    in which case each sample is kept. target_stat = "raw", so no aggregation — each of the 1001 samples is a separate data point
    In case of 'raw', the ML model will be trained to predict the distribution of abated emissions across samples, rather than a single statistic.

    For scope="Global":

    | Column       | Type                     |
    | ------------ | ------------------------ |
    | Year         | int                      |
    | Optimization | category                 |
    | Regret       | category                 |
    | Scenario     | category                 |
    | Welfare      | category                 |
    | Sample       | category                 |
    | Y            | float (abated emissions) |

    For scope="Regional":

    | Column       | Type     |
    | ------------ | -------- |
    | Year         | int      |
    | Region       | category |
    | Optimization | category |
    | Regret       | category |
    | Scenario     | category |
    | Welfare      | category |
    | Sample       | category |
    | Y            | float    |

    Features for ML: ["Optimization", "Regret", "Scenario", "Welfare"] (+ "Sample" for raw).
    For Regional scope, ML models will be trained separately per region, so "Region" is not a feature but a grouping variable.

    """
    assert scope in ("Global", "Regional")
    stat = target_stat.lower()
    if stat == "p50":
        stat = "median"

    df = long_df[long_df["Year"].isin(years)].copy()

    if stat == "raw":
        if scope == "Global":
            df = df[df["Region"] == "Global"]
            keep_cols = [
                "Year",
                "Optimization",
                "Regret",
                "Scenario",
                "Welfare",
                "Sample",
                "AbatedEmission",
            ]
        else:
            df = df[df["Scope"] == "Regional"]
            keep_cols = [
                "Year",
                "Region",
                "Optimization",
                "Regret",
                "Scenario",
                "Welfare",
                "Sample",
                "AbatedEmission",
            ]
        df = df[keep_cols].rename(columns={"AbatedEmission": "Y"})
        for c in ["Optimization", "Regret", "Scenario", "Welfare"]:
            df[c] = df[c].astype("category")
        if scope == "Regional":
            df["Region"] = df["Region"].astype("category")
        df["Sample"] = df["Sample"].astype("category")
        df["Year"] = df["Year"].astype(np.int32)
        return df

    if scope == "Global":
        df = df[df["Region"] == "Global"]
        group_cols = ["Year", "Optimization", "Regret", "Scenario", "Welfare"]
    else:
        df = df[df["Scope"] == "Regional"]
        group_cols = ["Year", "Region", "Optimization", "Regret", "Scenario", "Welfare"]

    if stat == "mean":
        agg_df = (
            df.groupby(group_cols, observed=True)["AbatedEmission"].mean().reset_index()
        )
    elif stat == "median":
        agg_df = (
            df.groupby(group_cols, observed=True)["AbatedEmission"]
            .median()
            .reset_index()
        )
    elif stat == "p90":
        agg_df = (
            df.groupby(group_cols, observed=True)["AbatedEmission"]
            .quantile(0.90)
            .reset_index()
        )
    else:
        raise ValueError(
            "target_stat must be one of: 'raw', 'mean', 'median' (or 'p50'), 'p90'"
        )

    agg_df = agg_df.rename(columns={"AbatedEmission": "Y"})
    # Ensure categoricals
    for c in ["Optimization", "Regret", "Scenario", "Welfare"]:
        agg_df[c] = agg_df[c].astype("category")
    if scope == "Regional":
        agg_df["Region"] = agg_df["Region"].astype("category")
    agg_df["Year"] = agg_df["Year"].astype(np.int32)
    return agg_df


# -------------------------------
# CatBoost + SHAP with cross-validation and early stopping
# -------------------------------
def _loss_for_stat(target_stat):
    stat = target_stat.lower()
    if stat in ("mean", "raw"):
        return "RMSE"
    if stat in ("median", "p50"):
        return "Quantile:alpha=0.5"
    if stat == "p90":
        return "Quantile:alpha=0.9"
    raise ValueError("Unknown target_stat")


def _metrics_for_stat(y_true, y_pred, target_stat):
    """
    Compute evaluation metrics per fold.
    - For mean/raw (RMSE): return {'RMSE': ..., 'R2': ...}
    - For median/P90 (Quantile): return {'Pinball': ...} where alpha is 0.5 or 0.9
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    stat = target_stat.lower()

    def r2_score(y, yhat):
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot

    if stat in ("mean", "raw"):
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = float(r2_score(y_true, y_pred))
        return {"RMSE": rmse, "R2": r2}

    # Quantile pinball loss
    alpha = 0.5 if stat in ("median", "p50") else 0.9
    e = y_true - y_pred
    pinball = np.mean(np.maximum(alpha * e, (alpha - 1.0) * e))
    return {"Pinball": float(pinball)}


def _cat_indices(X, categorical_cols):
    return [X.columns.get_loc(c) for c in categorical_cols]


def shap_values_mean_abs(model, X, categorical_cols):
    """
    Mean absolute SHAP values per feature using CatBoost's SHAP.
    Measures influence strength, not positive/negative effect
    """
    pool = Pool(X, cat_features=_cat_indices(X, categorical_cols))
    shap_vals = model.get_feature_importance(data=pool, type="ShapValues")
    # shap_vals shape: (n_samples, n_features + 1); last column is expected value
    shap_main = np.abs(shap_vals[:, :-1])
    means = shap_main.mean(axis=0)
    return pd.Series(means, index=X.columns).sort_values(ascending=False)


def fit_catboost_cv_shap(
    X,
    y,
    categorical_cols,
    target_stat="mean",
    cv_folds=5,
    random_state=42,
    params=None,
):
    """
    Cross-validated CatBoost with early stopping.
    - Trains a model per fold with eval_set for early stopping
    - Collects SHAP importances on each validation fold, averages across folds
    - Returns:
        {
          'cv_metrics': list of dicts per fold,
          'cv_metrics_mean': dict of mean metrics,
          'best_iterations': list,
          'shap_cv_mean': Series,
          'model_full': fitted CatBoost on full data,
          'shap_full': Series
        }
    """
    if params is None:
        params = dict(
            depth=6,
            learning_rate=0.05,
            n_estimators=800,
            l2_leaf_reg=3.0,
            random_seed=random_state,
            loss_function=_loss_for_stat(target_stat),
            od_type="Iter",
            od_wait=50,  # early stopping patience
            use_best_model=True,  # keep best iteration wrt eval_set
            verbose=False,
            allow_writing_files=False,
        )

    k = min(cv_folds, max(2, len(X)))  # ensure at least 2, at most N

    # Cross-validation setup with shuffling for better fold randomness, fixed random state for reproducibility
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    cv_metrics = []
    best_iters = []
    shap_series_list = []

    cat_idx = _cat_indices(X, categorical_cols)

    # Splits data into folds (5 default), trains model on N-1 folds, validates on the held-out fold
    # Early stopping if val loss doesn't improve for 50 (default) iterations
    # Records metrics on validation fold, Validation metrics (RMSE + R² for mean/raw; Pinball loss for median/p90), Best iteration count (this means the number of trees that were actually used before early stopping), SHAP importances (mean absolute) on validation fold

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        valid_pool = Pool(X_va, y_va, cat_features=cat_idx)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool)

        # Evaluate on validation
        yhat_va = model.predict(valid_pool)
        fold_metrics = _metrics_for_stat(y_va.values, yhat_va, target_stat)
        cv_metrics.append(fold_metrics)

        # Record best iteration if available
        best_iter = getattr(model, "tree_count_", None)
        if best_iter is None:
            # Fallback: params n_estimators
            best_iter = int(params.get("n_estimators", 800))
        best_iters.append(best_iter)

        # SHAP on validation fold, mean absolute
        shap_series = shap_values_mean_abs(model, X_va, categorical_cols)
        shap_series_list.append(shap_series)

    # Aggregate metrics
    metric_keys = cv_metrics[0].keys()
    cv_metrics_mean = {
        k: float(np.mean([m[k] for m in cv_metrics])) for k in metric_keys
    }

    # Average SHAP across folds (align by feature name)
    shap_cv_df = pd.concat(shap_series_list, axis=1)
    shap_cv_mean = shap_cv_df.mean(axis=1).sort_values(ascending=False)

    # Fit final model on full data using average best_iters (rounded)
    avg_best = int(
        np.clip(np.round(np.mean(best_iters)), 50, params.get("n_estimators", 800))
    )
    final_params = dict(params)
    final_params["n_estimators"] = avg_best
    # disable use_best_model and overfitting detector for the final fit
    final_params["use_best_model"] = False
    final_params.pop("od_type", None)
    final_params.pop("od_wait", None)

    final_model = CatBoostRegressor(**final_params)
    final_model.fit(Pool(X, y, cat_features=cat_idx))  # no eval_set for final fit

    # SHAP on full data for reference
    shap_full = shap_values_mean_abs(final_model, X, categorical_cols)

    return {
        "cv_metrics": cv_metrics,
        "cv_metrics_mean": cv_metrics_mean,
        "best_iterations": best_iters,
        "shap_cv_mean": shap_cv_mean,
        "model_full": final_model,
        "shap_full": shap_full,
    }


# -------------------------------
# Plotting helpers
# -------------------------------
def _normalize(s):
    tot = float(s.sum())
    return s / tot if tot > 0 else s


def plot_bar_importance(
    series,
    title,
    outfile=None,
    normalized=True,
    figsize=(6, 4),
    rotate=0,
    color="#4C78A8",
):
    """
    If outfile ends with .csv/.json/.parquet, save the importance data to that file (no plot).
    Otherwise, create a bar plot and save/show as before.
    """
    s = series.copy()
    if normalized:
        s = _normalize(s)

    ext = str(outfile).lower() if outfile is not None else ""
    is_csv = ext.endswith(".csv")
    is_json = ext.endswith(".json")
    is_parquet = ext.endswith(".parquet")

    if outfile is not None and (is_csv or is_json or is_parquet):
        df = s.rename_axis("Feature").reset_index(name="Importance")
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving feature importance data to: {Path(outfile).resolve()}")
        if is_csv:
            df.to_csv(outfile, index=False)
        elif is_json:
            df.to_json(outfile, orient="records")
        elif is_parquet:
            df.to_parquet(outfile, index=False)
        return df

    plt.figure(figsize=figsize)
    plt.bar(s.index, s.values, color=color)
    plt.ylabel(
        "SHAP importance" + (" (normalized)" if normalized else " (mean |SHAP|)")
    )
    plt.title(title)
    if rotate:
        plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    if outfile is not None:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, dpi=200)
        plt.close()
    else:
        plt.show()
    return s


# -------------------------------
# Pipeline per scope/year/stat (SHAP only)
# -------------------------------
def run_ml_importance_for_scope(
    cell_df,
    scope="Global",
    years=(2030, 2050, 2070, 2100),
    target_stat="mean",
    output_dir="ml_importance_plots",
    cv_folds=5,  # Using 5-fold CV for more stable SHAP estimates. 5 balances computation and stability.
    random_state=0,
    model_params=None,
    normalized_plots=True,
    model_type="final",  # "final" or "cv-mean"
):
    """
    @args:
    cell_df: DataFrame with columns depending on scope (see build_cell_level_targets)
    scope: "Global" or "Regional"
    years: which years to run models for. Takes tuple of ints, e.g. (2030, 2050, 2070, 2100)
    target_stat: "raw", "mean", "median" (or "p50"), "p90" - for the FaIR ensemble members
    output_dir: base directory to save plots and importance data
    cv_folds: number of CV folds for CatBoost training and SHAP estimation. More folds = more stable SHAP but longer runtime.
    random_state: for reproducibility of CV splits and model training
    model_params: dict of CatBoost parameters to override defaults (e.g., depth, learning_rate)
    normalized_plots: whether to normalize SHAP importances to sum to 1 in the plots
    model_type: whether to plot SHAP from "final" model or "cv-mean" (average across CV folds).


    For the given scope (Global or Regional cell-level data), fit CatBoost models per year with CV.

    Filter to one year (e.g., 2030)

    For Global scope: one dataset

    For Regional scope: group by Region, fit one model per region

    Categorical features: CatBoost needs explicit cat feature indices:

    categorical_cols = ["Optimization", "Regret", "Scenario", "Welfare"]
    if target_stat == "raw":
        categorical_cols += ["Sample"]
    """
    results = {}
    target_name = f"Y_{target_stat.lower()}"
    df = cell_df.copy()
    df = df.rename(columns={"Y": target_name})

    # Core categorical features for all models
    feature_cols = ["Optimization", "Regret", "Scenario", "Welfare"]

    # For raw targets, we include "Sample" as a categorical feature to allow the model to learn sample-specific effects.
    # This shows the contribution of FaIR ensemble members to the abated emissions.
    # For aggregated targets, "Sample" is not included since the target is already a summary statistic across samples.

    if target_stat.lower() == "raw":
        feature_cols = feature_cols + ["Sample"]
    categorical_cols = feature_cols[:]

    outbase = Path(output_dir) / scope.lower() / target_stat.lower()
    outbase.mkdir(parents=True, exist_ok=True)

    # Per-year models
    for yr in years:
        d = df[df["Year"] == yr].copy()
        if d.empty:
            continue

        year_results = {}

        if scope == "Global":
            X = d[feature_cols].copy()
            y = d[target_name].copy()

            res = fit_catboost_cv_shap(
                X,
                y,
                categorical_cols,
                target_stat=target_stat,
                cv_folds=cv_folds,
                random_state=random_state,
                params=model_params,
            )

            year_results.update(res)

            if model_type == "cv-mean":
                plot_bar_importance(
                    res["shap_cv_mean"],
                    title=f"{scope} {yr} — SHAP importance (CV mean, {target_stat})",
                    outfile=outbase / f"{scope.lower()}_{yr}_shap_cv.csv",
                    normalized=normalized_plots,
                    figsize=(6, 4),
                )
            elif model_type == "final":
                plot_bar_importance(
                    res["shap_full"],
                    title=f"{scope} {yr} — SHAP importance (final model, {target_stat})",
                    outfile=outbase / f"{scope.lower()}_{yr}_shap_full.csv",
                    normalized=normalized_plots,
                    figsize=(6, 4),
                )

        else:
            year_results["regions"] = {}
            for region, d_r in d.groupby("Region", observed=True):
                X = d_r[feature_cols].copy()
                y = d_r[target_name].copy()
                if len(X) < 4:
                    continue

                res = fit_catboost_cv_shap(
                    X,
                    y,
                    categorical_cols,
                    target_stat=target_stat,
                    cv_folds=cv_folds,
                    random_state=random_state,
                    params=model_params,
                )
                year_results["regions"][str(region)] = res

                safe_region = str(region).replace(" ", "_").replace("/", "_")
                if model_type == "cv-mean":
                    plot_bar_importance(
                        res["shap_cv_mean"],
                        title=f"{region} {yr} — SHAP importance (CV mean, {target_stat})",
                        outfile=outbase / f"{safe_region}_{yr}_shap_cv.csv",
                        normalized=normalized_plots,
                        figsize=(6, 4),
                    )
                elif model_type == "final":
                    plot_bar_importance(
                        res["shap_full"],
                        title=f"{region} {yr} — SHAP importance (final model, {target_stat})",
                        outfile=outbase / f"{safe_region}_{yr}_shap_full.csv",
                        normalized=normalized_plots,
                        figsize=(6, 4),
                    )

        results[yr] = year_results

    return results


# -------------------------------
# Convenience runner for all stats and either scope (SHAP only)
# -------------------------------
def run_all_ml_importance(
    long_df,
    years=(2030, 2050, 2070, 2100),
    target_stats=("mean", "median", "p90"),
    output_dir="ml_importance_plots",
    cv_folds=5,
    random_state=0,
    model_params=None,
    normalized_plots=True,
    model_type="final",  # "final" or "cv-mean"
    scope="global",  # or "regional"
):
    """
    Build cell-level targets for each requested statistic and run CatBoost + SHAP
    for either Global or Regional scope.
    """
    all_results = {}
    scope = scope.lower()
    for stat in target_stats:
        if scope == "regional":
            regional_cells = build_cell_level_targets(
                long_df, years=years, target_stat=stat, scope="Regional"
            )
            regional_results = run_ml_importance_for_scope(
                regional_cells,
                scope="Regional",
                years=years,
                target_stat=stat,
                output_dir=output_dir,
                cv_folds=cv_folds,
                random_state=random_state,
                model_params=model_params,
                normalized_plots=normalized_plots,
                model_type=model_type,
            )
            all_results[stat] = {"Regional": regional_results}
        elif scope == "global":
            global_cells = build_cell_level_targets(
                long_df, years=years, target_stat=stat, scope="Global"
            )
            global_results = run_ml_importance_for_scope(
                global_cells,
                scope="Global",
                years=years,
                target_stat=stat,
                output_dir=output_dir,
                cv_folds=cv_folds,
                random_state=random_state,
                model_params=model_params,
                normalized_plots=normalized_plots,
                model_type=model_type,
            )
            all_results[stat] = {"Global": global_results}
        else:
            raise ValueError("scope must be either 'global' or 'regional'")
    return all_results
