import pandas as pd

from joulegrad.build_energy_lookup_table import build_lookup_table


def test_builder_keeps_ok_and_review_current_summary_rows(tmp_path):
    base = {
        "model_path": "Models/CPU/Linear/Linear_2_4.pt",
        "status": "COMPLETE",
        "quality_status": "OK",
        "energy_mean_mJ": 3.0,
        "detected_regions": 100,
        "expected_cycles": 100,
        "clock_alignment_status": "COMPLETE",
        "clock_alignment_uncertainty_s": 0.001,
        "clock_alignment_threshold_s": 0.005,
    }
    rows = [
        base,
        {**base, "quality_status": "REVIEW", "energy_mean_mJ": 99.0},
        {**base, "energy_mean_mJ": -1.0},
        {**base, "detected_regions": 99},
        {**base, "clock_alignment_uncertainty_s": 0.01},
    ]
    path = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    lookup = build_lookup_table([path])

    assert len(lookup) == 1
    assert lookup.iloc[0]["measurement_count"] == 2
    assert lookup.iloc[0]["energy_mean_mJ"] == 51.0