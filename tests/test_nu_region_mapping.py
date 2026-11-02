"""
Tests for the 12 macro-region mapping used in the normative uncertainty (NU) SHAP analysis.

Background: with the original data/input/12_regions.json and a Jaccard similarity
threshold of 0.01, eleven single-country JUSTICE regions (Mexico, Indonesia, South Africa,
Turkey, Thailand, Vietnam, Malaysia, Argentina, Chile, Egypt, Ukraine) were not mapped to any
macro-region and were silently dropped from both the regional and the global aggregates.
data/input/12_regions_nu.json fixes the mapping; these tests guard against a repeat.

Run with pytest from the repository root, or directly: python tests/test_nu_region_mapping.py
"""

import json
import os
from pathlib import Path

import numpy as np

from justice.util.regional_configuration import (
    get_region_mapping,
    justice_region_aggregator,
)

REPO = Path(__file__).resolve().parents[1]
NU_MAPPING_PATH = REPO / "data/input/12_regions_nu.json"
RICE50_DICT_PATH = REPO / "data/input/rice50_regions_dict.json"

# Intended assignment of the 57 JUSTICE (RICE50+) regions to the 12 macro-regions.
NU_MACRO_REGIONS = {
    "United States": ["usa"],
    "India": ["nde"],
    "China": ["chn"],
    "Russia": ["rus"],
    "Brazil": ["bra"],
    "Other High Income": ["aus", "jpn", "cor", "can"],
    "Europe": ["aut", "bel", "rfa", "dnk", "esp", "fin", "fra", "gbr", "grc", "irl",
               "ita", "nld", "prt", "swe", "bgr", "rcz", "hun", "pol", "rom", "rsl",
               "slo", "cro", "nor", "sui", "blt", "oeu"],
    "South Asia": ["rsas"],
    "Southeast Asia": ["osea", "idn", "tha", "vnm", "mys"],
    "Gulf Countries": ["golf57"],
    "Sub-Saharan Africa": ["rsaf"],
    "Rest of the World": ["rcam", "noan", "noap", "meme", "rjan57", "rsam", "ris",
                          "mex", "arg", "chl", "zaf", "egy", "tur", "ukr"],
}


def _load(path):
    with open(path, "r") as f:
        return json.load(f)


class _StubDataLoader:
    """justice_region_aggregator only needs REGION_LIST from the DataLoader."""

    def __init__(self, region_list):
        self.REGION_LIST = list(region_list)


def test_intended_mapping_covers_all_57_regions_once():
    rice50 = _load(RICE50_DICT_PATH)
    flat = [r for regions in NU_MACRO_REGIONS.values() for r in regions]
    assert len(flat) == 57
    assert set(flat) == set(rice50)


def test_country_lists_match_rice50_regions():
    """Each macro-region's countries are exactly the countries of its JUSTICE regions,
    so map colouring (by country) and aggregation (by region) agree."""
    rice50 = _load(RICE50_DICT_PATH)
    mapping = _load(NU_MAPPING_PATH)
    assert list(mapping) == list(NU_MACRO_REGIONS)
    for macro, regions in NU_MACRO_REGIONS.items():
        expected = {c for r in regions for c in rice50[r]}
        assert set(mapping[macro]) == expected, macro
    countries = [c for v in mapping.values() for c in v]
    assert len(countries) == len(set(countries)), "a country is in two macro-regions"


def test_every_region_maps_to_intended_macro_region():
    rice50 = _load(RICE50_DICT_PATH)
    mapping = _load(NU_MAPPING_PATH)
    result = get_region_mapping(mapping, rice50, similarity_threshold=0.0)
    assert {k: set(v) for k, v in result.items()} == {
        k: set(v) for k, v in NU_MACRO_REGIONS.items()
    }


def test_aggregation_conserves_the_57_region_total():
    rice50 = _load(RICE50_DICT_PATH)
    mapping = _load(NU_MAPPING_PATH)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(57, 12, 7))
    cwd = os.getcwd()
    os.chdir(REPO)  # justice_region_aggregator reads data/input/* relative to the repo root
    try:
        region_list, aggregated = justice_region_aggregator(
            _StubDataLoader(rice50), mapping, data, similarity_threshold=0.0
        )
    finally:
        os.chdir(cwd)
    assert len(region_list) == 12
    np.testing.assert_allclose(aggregated.sum(axis=0), data.sum(axis=0))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
