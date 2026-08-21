from itertools import product
import warnings

import pandas as pd
import pytest
import torch

from joulegrad import EnergyLookup


def _write_linear_lookup(path, *, omit=None):
    rows = []
    for input_features, output_features in product((2, 4), repeat=2):
        if (input_features, output_features) == omit:
            continue
        rows.append(
            {
                "layer_type": "linear",
                "input_features": input_features,
                "output_features": output_features,
                "energy_mean_mJ": input_features + 2 * output_features,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_linear_interpolation_keeps_mask_gradient(tmp_path):
    path = tmp_path / "lookup.csv"
    _write_linear_lookup(path)
    lookup = EnergyLookup(path)
    output_mask = torch.full((4,), 0.75, requires_grad=True)

    energy = lookup.linear(2, output_mask.sum())
    energy.backward()

    assert energy.item() == pytest.approx(8.0)
    assert torch.allclose(output_mask.grad, torch.full((4,), 2.0))


def test_missing_corner_and_out_of_range_fail_closed(tmp_path):
    path = tmp_path / "lookup.csv"
    _write_linear_lookup(path, omit=(4, 4))
    lookup = EnergyLookup(path)

    with pytest.raises(ValueError, match="missing measurements"):
        lookup.linear(3, 3)
    with pytest.raises(ValueError, match="outside"):
        lookup.linear(1, 2)


def test_missing_zero_weight_corner_does_not_block_exact_measurement(tmp_path):
    path = tmp_path / "lookup.csv"
    _write_linear_lookup(path, omit=(4, 4))
    lookup = EnergyLookup(path)

    assert lookup.linear(2, 2).item() == pytest.approx(6.0)


def test_warn_policy_clamps_out_of_range_coordinate(tmp_path):
    path = tmp_path / "lookup.csv"
    _write_linear_lookup(path)
    lookup = EnergyLookup(path, out_of_range="warn")
    input_features = torch.tensor(1.0, requires_grad=True)

    with pytest.warns(RuntimeWarning, match="outside.*clamping"):
        energy = lookup.linear(input_features, 2)
    energy.backward()

    assert energy.item() == pytest.approx(6.0)
    assert input_features.grad.item() == pytest.approx(0.0)


def test_warn_policy_fills_missing_grid_points_from_nearest_measurement(tmp_path):
    path = tmp_path / "lookup.csv"
    _write_linear_lookup(path, omit=(4, 4))

    with pytest.warns(RuntimeWarning, match="missing.*nearest measured"):
        lookup = EnergyLookup(path, out_of_range="warn")
    input_features = torch.tensor(3.0, requires_grad=True)
    energy = lookup.linear(input_features, 3)
    energy.backward()

    assert torch.isfinite(energy)
    assert torch.isfinite(input_features.grad)


def test_fallback_policy_clamps_and_fills_without_warnings(tmp_path):
    path = tmp_path / "lookup.csv"
    _write_linear_lookup(path, omit=(4, 4))
    input_features = torch.tensor(1.0, requires_grad=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lookup = EnergyLookup(path, out_of_range="fallback")
        energy = lookup.linear(input_features, 3)
    energy.backward()

    assert torch.isfinite(energy)
    assert input_features.grad.item() == pytest.approx(0.0)


def test_non_positive_measurement_is_rejected(tmp_path):
    path = tmp_path / "lookup.csv"
    pd.DataFrame(
        [{
            "layer_type": "linear",
            "input_features": 2,
            "output_features": 2,
            "energy_mean_mJ": -1.0,
        }]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-positive"):
        EnergyLookup(path)