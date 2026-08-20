# joulegrad performance optimization

This document records the optimization work inherited from energyBANERA and
present in this estimator. The interpolation mathematics and units remain
unchanged: the returned value is mJ per inference.

## Prepared training path

Create the lookup and regularizer once after model placement, then reuse them
for every training step:

```python
from joulegrad import EnergyLookup, ModelEnergyRegularizer

lookup = EnergyLookup("measurements/BOARD_energy_lookup.csv", out_of_range="error")
regularizer = ModelEnergyRegularizer(
    model,
    lookup,
    input_shapes={"features.0": (batch_size, channels, height, width)},
)

energy_mj = regularizer(current_masks)
loss = task_loss + energy_weight * energy_mj
```

Use `regularizer.estimate(current_masks)` for diagnostics only. It constructs
per-layer detail dictionaries and is not the minimal loss-path API.

## Implemented improvements

### Prepared lookup grids

`EnergyLookup` parses and validates CSV data once. It creates immutable grids
for Linear, Conv2d, Attention, and RotaryAttention queries, keyed by operator
type and the Conv2d kernel, stride, and padding configuration. Pandas is not
used after lookup construction.

### Device-local grid cache

Axes and values move to a coordinate device only on first use. Repeated
queries reuse the cached device tensors.

### Vectorized interpolation

Interpolation corner bit patterns are cached by rank and device. Corner
indices, interpolation weights, and weighted sums are evaluated as PyTorch
tensor operations, retaining gradients with respect to soft effective
dimensions.

### Batched layer evaluation

`linear_batch`, `conv2d_batch`, and `attention_batch` evaluate compatible
coordinates together. `ModelEnergyRegularizer` groups operators by lookup grid
and reduces their returned energy vectors in one tensor operation.

### Prepared model topology

`ModelEnergyRegularizer` scans `named_modules()` during construction, storing
only the supported operator metadata it needs. Its forward path consumes live
masks without another model traversal. The one-shot
`estimate_model_energy(...)` helper remains available for diagnostics.

## Local safety additions

Ecological NAS retains these stricter rules on top of the optimized path:

- energy values must be positive;
- incomplete interpolation cells fail closed;
- training defaults to `out_of_range="error"`;
- grouped and dilated Conv2d operations are rejected without matching
  measurements; and
- generic topology inference is not used for the transformer bridges, which
  instead declare their operator mappings explicitly.

For the full energy-aware training contract, see
[differentiable_energy_estimator.md](differentiable_energy_estimator.md).