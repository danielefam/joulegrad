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
  measurements/summaries/pi5_summary.csv \
  --output measurements/pi5_energy_lookup.csv
```

Load it in Python with strict out-of-range behavior:

```python
from joulegrad import EnergyLookup

lookup = EnergyLookup("measurements/pi5_energy_lookup.csv", out_of_range="error")
energy_mj = lookup.linear(64, 128)
```

You can also query a lookup table directly from the terminal. Run the command
in an environment where JouleGrad's dependencies, including PyTorch, are
installed:

```bash
conda run -n banera_pt python -m joulegrad \
  measurements/pi5_energy_lookup.csv \
  linear 130 162
```

This prints the estimated energy per inference:

```text
0.128964165565 mJ/inference
```

The command supports `linear`, `conv`, and `attention` queries. Use
`python -m joulegrad --help` to see all available arguments.

Coordinates may be differentiable scalar tensors, so the result can be added
to a loss. `out_of_range="error"` rejects missing interpolation corners and
unsupported coordinates. `out_of_range="warn"` warns, clamps coordinates to
the measured range, and fills missing grid points from the nearest measured
configuration before interpolation. `out_of_range="fallback"` applies the same
fallbacks without warnings. Silent `"clamp"` and unchecked `"extrapolate"`
modes are also available; both still reject required missing corners.
For Conv2d, `"warn"` and `"fallback"` also select the nearest measured
`(kernel_size, stride, padding)` configuration when the requested discrete
configuration is absent. Warning mode reports the substitution explicitly.

See [differentiable_energy_estimator.md](differentiable_energy_estimator.md)
for the integration contract and [optimization.md](optimization.md) for the
performance design and constraints.