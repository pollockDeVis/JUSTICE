#!/usr/bin/env python
"""
Feature importance (CatBoost + SHAP) analysis for the normative uncertainty (NU) paper,
run outside the notebook. Equivalent to NU_reproduction.ipynb Section 4: builds the
abated-emissions dataset, fits the global and regional CatBoost models, saves the SHAP
CSVs, regenerates the SHAP figures, and compares the new results with the backup taken
before the region-mapping fix.

Run from the repository root with the venv active:

    python -u run_nu_sensitivity_analysis.py                  # full run (about 1-1.5 h)
    python -u run_nu_sensitivity_analysis.py --figures-only   # redraw figures from existing CSVs
    python -u run_nu_sensitivity_analysis.py --compare-only   # only the before/after comparison
"""

import argparse
import importlib.util
import os
import re
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
os.chdir(REPO)  # all data paths in JUSTICE are relative to the repository root
sys.path.insert(0, str(REPO))

# ── CONFIG (same values as NU_reproduction.ipynb) ─────────────────────────────
BASE_PATH = "data/temporary/NU_DATA/mmBorg/reproduce"
MODEL_OUTPUT_DIR = f"{BASE_PATH}/model_outputs"
SENSITIVITY_DIR = f"{BASE_PATH}/sensitivity_analysis"
BACKUP_DIR = f"{BASE_PATH}/sensitivity_analysis_before_region_fix"
PLOTS_DIR = f"{BASE_PATH}/plots"
REGION_MAPPING_PATH = "data/input/12_regions_nu.json"
RICE50_DICT_PATH = "data/input/rice50_regions_dict.json"

YEARS = (2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100)

ML_PARAMS = dict(
    depth=6,
    learning_rate=0.05,
    n_estimators=800,
    l2_leaf_reg=3.0,
    loss_function="RMSE",
    random_seed=42,
    od_type="Iter",
    od_wait=50,
    use_best_model=True,
    verbose=False,
    allow_writing_files=False,
)

GROUP_MAP = {
    "Deep Uncertainty": ["Scenario"],
    "Normative Uncertainty": ["Optimization", "Welfare", "Regret"],
    "Stochastic Uncertainty": ["Sample"],
}

N_REGIONS = 12
START = time.time()


def log(msg):
    elapsed = (time.time() - START) / 60
    print(f"[{time.strftime('%H:%M:%S')} | +{elapsed:6.1f} min] {msg}", flush=True)


def step(title):
    print("\n" + "=" * 70, flush=True)
    log(title)
    print("=" * 70, flush=True)


# ── STEP 0: region mapping tests ──────────────────────────────────────────────
def run_mapping_tests():
    step("Step 0/5: checking the 12 macro-region mapping")
    path = REPO / "tests" / "test_nu_region_mapping.py"
    spec = importlib.util.spec_from_file_location("test_nu_region_mapping", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in sorted(dir(module)):
        if name.startswith("test_"):
            getattr(module, name)()
            log(f"passed {name}")


# ── STEPS 1-3: dataset + CatBoost/SHAP ────────────────────────────────────────
def run_shap_analysis():
    from justice.util.feature_importance import (
        build_long_dataframe,
        run_all_ml_importance,
    )

    step("Step 1/5: building the long dataframe of abated emissions")
    long_df = build_long_dataframe(
        base_path=f"{BASE_PATH}/",
        model_output_dir=MODEL_OUTPUT_DIR,
        region_mapping_path=REGION_MAPPING_PATH,
        rice_region_dict_path=RICE50_DICT_PATH,
        years_of_interest=YEARS,
    )
    log(f"long dataframe shape: {long_df.shape}")
    n_regions = long_df.loc[long_df["Scope"] == "Regional", "Region"].nunique()
    if n_regions != N_REGIONS:
        raise RuntimeError(f"expected {N_REGIONS} macro-regions, found {n_regions}")

    for i, scope in enumerate(("global", "regional"), start=2):
        step(f"Step {i}/5: CatBoost + SHAP, {scope} scope")
        run_all_ml_importance(
            long_df=long_df,
            years=YEARS,
            target_stats=("raw",),
            output_dir=SENSITIVITY_DIR,
            cv_folds=5,
            random_state=42,
            model_params=ML_PARAMS,
            normalized_plots=True,
            model_type="final",
            scope=scope,
        )
        log(f"{scope} CSVs saved to {SENSITIVITY_DIR}/{scope}/raw/")

    n_global = len(list(Path(SENSITIVITY_DIR, "global", "raw").glob("*_shap_full.csv")))
    n_regional = len(list(Path(SENSITIVITY_DIR, "regional", "raw").glob("*_shap_full.csv")))
    expected_regional = N_REGIONS * len(YEARS)
    log(f"SHAP CSVs: {n_global} global (expected {len(YEARS)}), "
        f"{n_regional} regional (expected {expected_regional})")
    if n_global != len(YEARS) or n_regional != expected_regional:
        log("WARNING: unexpected number of SHAP CSVs, check the output folders")


# ── STEP 4: figures ───────────────────────────────────────────────────────────
def make_figures():
    import matplotlib

    matplotlib.use("Agg")  # no display needed in a terminal run
    from justice.util.visualizer import (
        generate_uncertainty_visualizations,
        render_all_grouped_stacked_charts,
    )

    step("Step 4/5: regenerating the SHAP figures")
    for scope in ("global", "regional"):
        out = f"{PLOTS_DIR}/shap_{scope}"
        render_all_grouped_stacked_charts(
            base_dir=SENSITIVITY_DIR,
            scope=scope,
            stat="raw",
            model_type="final",
            years=YEARS,
            group_map=GROUP_MAP,
            output_dir=out,
        )
        log(f"stacked bar charts ({scope}) saved to {out}/")

    out = Path(PLOTS_DIR) / "uncertainty_viz"
    out.mkdir(parents=True, exist_ok=True)
    _, saved = generate_uncertainty_visualizations(
        base_dir=SENSITIVITY_DIR,
        region_mapping_path=REGION_MAPPING_PATH,
        stat="raw",
        model_type="final",
        years=YEARS,
        ternary_scale=8,
        quantize=True,
        annotate_points=True,
        marker_size=40,
        jitter_strength=0.02,
        random_state=0,
        output_dir=str(out),
    )
    log(f"ternary and choropleth figures saved to {out}/ ({len(saved)} files)")


# ── STEP 5: before/after comparison ───────────────────────────────────────────
def _grouped_shares(folder):
    import pandas as pd

    pattern = re.compile(r"^(?P<scope>.+)_(?P<year>\d{4})_shap_full\.csv$")
    rows = []
    for scope in ("global", "regional"):
        for f in sorted(Path(folder, scope, "raw").glob("*_shap_full.csv")):
            m = pattern.match(f.name)
            if not m:
                continue
            imp = pd.read_csv(f).set_index("Feature")["Importance"]
            total = imp.sum()
            row = {"Region": m["scope"], "Year": int(m["year"])}
            for group, feats in GROUP_MAP.items():
                row[group] = float(imp.reindex(feats).fillna(0).sum() / total)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["Region", "Year"]).sort_index()


def compare_before_after():
    import pandas as pd

    step("Step 5/5: comparing with the results before the region-mapping fix")
    if not Path(BACKUP_DIR).exists():
        log(f"no backup found at {BACKUP_DIR}, skipping the comparison")
        return
    old = _grouped_shares(BACKUP_DIR)
    new = _grouped_shares(SENSITIVITY_DIR)
    table = old.join(new, lsuffix=" (before)", rsuffix=" (after)", how="outer")
    for group in GROUP_MAP:
        table[f"{group} (change)"] = table[f"{group} (after)"] - table[f"{group} (before)"]
    out = Path(SENSITIVITY_DIR) / "comparison_before_after_region_fix.csv"
    table.round(4).to_csv(out)
    log(f"full comparison saved to {out}")

    short = {"Deep Uncertainty": "Deep", "Normative Uncertainty": "Norm",
             "Stochastic Uncertainty": "Stoch"}
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for region in ["global", "Southeast_Asia", "Rest_of_the_World"]:
            if region not in table.index.get_level_values(0):
                continue
            view = table.loc[region]
            cols = {f"{g} ({w})": f"{short[g]} {w}" for g in GROUP_MAP for w in ("before", "after")}
            print(f"\nShares for {region}:")
            print(view[list(cols)].rename(columns=cols).round(3).to_string())
    biggest = table[[f"{g} (change)" for g in GROUP_MAP]].abs().max(axis=1).sort_values(ascending=False)
    print("\nLargest changes in any uncertainty share (region, year):")
    print(biggest.head(10).round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figures-only", action="store_true",
                        help="skip the SHAP run and only redraw the figures")
    parser.add_argument("--compare-only", action="store_true",
                        help="only run the before/after comparison")
    args = parser.parse_args()

    log(f"working directory: {REPO}")
    try:
        if args.compare_only:
            compare_before_after()
        else:
            run_mapping_tests()
            if not args.figures_only:
                run_shap_analysis()
            make_figures()
            compare_before_after()
    except Exception:
        traceback.print_exc()
        log("FAILED, see the traceback above")
        sys.exit(1)
    log("DONE")


if __name__ == "__main__":
    main()
