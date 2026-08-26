# Python API

JouleGrad is an API-only library. It has no command-line interface and does
not create lookup tables.

## `EnergyEstimator`

```python
from joulegrad import EnergyEstimator

estimator = EnergyEstimator("energy_lookup_table.csv", out_of_range="error")
```

The constructor reads and prepares one CSV. Prepared grids are cached on each
PyTorch device when first used.

Layer queries return scalar differentiable tensors in mJ per inference:

```python
linear_mj = estimator.linear(input_features, output_features)
conv_mj = estimator.conv2d(
    input_channels,
    output_channels,
    input_height,
    input_width,
    kernel_size=3,
    stride=1,
    padding=1,
)
attention_mj = estimator.attention(
    sequence_length,
    embed_dim,
    head_dim,
    rotary=False,
)
```

Each query also has a vectorized `*_batch` form.

## Prepared model API

Model preparation scans selected `nn.Linear` and `nn.Conv2d` modules once:

```python
prepared = estimator.prepare_model(
    model,
    input_shapes={"features.0": (1, 3, 32, 32)},
    module_names=("features.0", "classifier"),
)

energy_mj = prepared(masks)
details = prepared.estimate(masks)
```

`masks` maps module names to either an output-mask tensor or a dictionary with
`input` and/or `output` effective dimensions. A mask tensor contributes its
sum, preserving gradients.

For one-off diagnostics:

```python
details = estimator.estimate_model(
    model,
    masks=masks,
    input_shapes=input_shapes,
    module_names=module_names,
)
```

The result contains `layers` and the scalar `total_energy_mJ` tensor.

## Compatibility API

`EnergyLookup` is an identity alias for `EnergyEstimator`. The existing
`ModelEnergyRegularizer`, `estimate_model_energy`, and
`multilinear_interpolate` exports remain available.

## Out-of-range policies

The original estimator's implicit behavior and the rationale for each current
policy are documented in
[Out-of-range policy evolution](OUT_OF_RANGE_POLICIES.md).

- `error`: reject unsupported coordinates or missing required corners.
- `warn`: warn, clamp coordinates, fill missing points from the nearest
  measurement, and use the nearest measured convolution configuration.
- `fallback`: apply the same fallbacks silently.
- `clamp`: clamp coordinates but do not fill missing grids.
- `extrapolate`: extrapolate from the nearest cell.
- `extrapolate_fallback`: extrapolate and fill/substitute missing grids.