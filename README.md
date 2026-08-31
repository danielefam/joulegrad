# JouleGrad

JouleGrad is an independent Python library for differentiable energy
estimation. It consumes a measured energy lookup CSV and returns PyTorch
tensors in millijoules per inference.

It does not collect measurements, create lookup tables, provide a command-line
interface, or depend on a specific pruning or optimization project.

## Installation

```bash
python -m pip install -e .
```

Runtime dependencies are only PyTorch and pandas.

## Layer API

```python
import torch
from joulegrad import EnergyEstimator

estimator = EnergyEstimator("energy_lookup_table.csv", out_of_range="error")

active_outputs = torch.tensor(96.0, requires_grad=True)
energy_mj = estimator.linear(128, active_outputs)
energy_mj.backward()
print(float(energy_mj), active_outputs.grad)
```

`linear`, `conv2d`, and `attention` queries support differentiable scalar
coordinates and vectorized `*_batch` forms.

## Model API

```python
prepared = estimator.prepare_model(
    model,
    input_shapes={"features.0": (1, 3, 32, 32)},
    module_names=("features.0", "classifier"),
)

energy_mj = prepared(masks)
details = prepared.estimate(masks)
```

For one-off diagnostics, use `estimator.estimate_model(...)`.

## Lookup input

JouleGrad reads the current JouleQuest `energy_lookup_table.csv` schema. A
lookup represents one hardware and software configuration and contains
accepted aggregate layer measurements. JouleGrad validates and interpolates
the table but does not know how it was produced.

See:

- [Python API](docs/API.md)
- [Lookup schema](docs/LOOKUP_SCHEMA.md)
- [Interpolation](docs/INTERPOLATION.md)
- [Performance architecture](docs/OPTIMIZATION.md)
- [Out-of-range policy evolution](docs/OUT_OF_RANGE_POLICIES.md)
- [Migration from energy_estimator](docs/MIGRATION_FROM_ENERGY_ESTIMATOR.md)

## Compatibility

`EnergyLookup` is an identity alias for `EnergyEstimator`. The
`ModelEnergyRegularizer`, `estimate_model_energy`, and
`multilinear_interpolate` APIs remain available.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -W error -p no:cacheprovider
```
