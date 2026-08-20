# joulegrad

This package provides differentiable, measured energy estimates for Linear,
Conv2d, Attention, and RotaryAttention operations. It turns one processed
hardware-measurement summary into a lookup table and evaluates that table in a
training loss.

The initial idea was inspired by
[aissa0803/energy_estimator](https://github.com/aissa0803/energy_estimator). The
implementation was substantially rewritten for JouleQuest and is maintained
here with additional strict validation for Ecological NAS.

## Performance implementation

The current implementation includes the JouleQuest optimization work:

- lookup grids are prepared once from the CSV rather than rebuilt per query;
- prepared tensors are cached once per device;
- interpolation corners are computed with vectorized PyTorch operations;
- Linear, Conv2d, and attention queries have batched APIs; and
- `ModelEnergyRegularizer` scans the model once and batches compatible
  operators on each training step.

These optimizations are additive to this project's stricter measurement
policy: lookup rows must have positive energy, and the builder accepts only
complete measurements whose quality status is `OK` or `REVIEW`, with matching
cycle and clock checks when those fields are present.

## Workflow

JouleQuest, a private companion project, captures and processes the hardware
measurements used by this estimator:
[github.com/danielefam/joulequest](https://github.com/danielefam/joulequest).

Place one processed summary CSV per board and runtime configuration in a
directory of your choice, then build a separate lookup for each summary:

```bash
python -m joulegrad.build_energy_lookup_table \
  summaries/pi5_summary.csv \
  --output pi5_energy_lookup.csv
```

Load it in Python with strict out-of-range behavior:

```python
from joulegrad import EnergyLookup

lookup = EnergyLookup("measurements/pi5_energy_lookup.csv", out_of_range="error")
energy_mj = lookup.linear(64, 128)
```

Coordinates may be differentiable scalar tensors, so the result can be added
to a loss. Missing interpolation corners and unsupported coordinates raise an
error rather than yielding an invented value.

See [differentiable_energy_estimator.md](differentiable_energy_estimator.md)
for the integration contract and [optimization.md](optimization.md) for the
performance design and constraints.