# Evolution of out-of-range handling

This document explains how the original `energy_estimator` handled coordinates
outside its measured tables, why that behavior was insufficient, and how
JouleGrad introduced explicit policies.

## Short answer

The original estimator had **no selectable out-of-range policy**. It did have
an implicit behavior:

1. clamp the coordinate only for choosing the edge interpolation cell;
2. compute the interpolation fraction from the original, unclamped coordinate;
3. therefore linearly extrapolate beyond the table edge;
4. do this silently, with no warning or strict failure mode.

JouleGrad made this behavior explicit and added safer alternatives. The default
is now `error`, which rejects unsupported coordinates rather than silently
inventing an estimate.

## 1. Original `energy_estimator` behavior

The original implementation used `clamp_index(grid, x)`. Its index lookup was:

```python
x_lookup = x.detach().clamp(grid[0], grid[-1])
hi = torch.searchsorted(grid, x_lookup)
lo = hi - 1
```

This selected the first or last measured cell for an out-of-range coordinate.
However, the interpolation fraction was calculated with the original `x`:

```python
t = (x - grid[lo]) / (grid[hi] - grid[lo])
```

For an in-range coordinate, $t\in[0,1]$. Outside the table:

- $t<0$ below the minimum;
- $t>1$ above the maximum.

The bilinear formula then extended the edge cell as a line. In other words,
although the helper was called `clamp_index`, the resulting energy estimate
was **linear extrapolation**.

### Example

For one measured axis with points $x_0=2$ and $x_1=4$, querying $x=5$ selected
the edge interval $[2,4]`, then calculated

$$
t=\frac{5-2}{4-2}=1.5.
$$

The estimate continued beyond the value measured at $x=4$. No exception or
warning indicated that this happened.

## 2. The original zero anchor

When loading each bundled Excel table, the old package inserted a synthetic
zero row and column:

$$
E(0,d_{out})=0,
\qquad
E(d_{in},0)=0.
$$

This made zero-width masks easy to query and supplied a low-end anchor. It was
a modeling assumption, not a hardware measurement.

JouleGrad does not automatically add this anchor to JouleQuest data. A zero
input or output width may still involve framework, launch, memory, or fixed
operator overhead, and a physically absent layer is different from executing a
zero-width layer. The lookup producer should contain an actual low-width
measurement if that region must be modeled.

Consequences in JouleGrad:

- `error` rejects a zero width when zero is outside the measured axis;
- `clamp` and `warn` use the smallest measured width;
- `extrapolate` extends the first measured cell toward zero, which may be
   inaccurate or even non-physical.

## 3. Other limitations of the original behavior

### No caller choice

Every query used the same silent extrapolation. A validation run could not fail
closed, and an exploratory run could not request warnings.

### No observability

The result did not report whether a coordinate was measured, interpolated, or
extrapolated.

### Missing corners were coarse-grained

If any of the four bilinear cell corners was `NaN`, the old implementation
raised `ValueError`, even when a missing corner had zero interpolation weight
for an exact measured coordinate.

### Discrete convolution configurations

Kernel, stride, padding, and dilation selected separate bundled files. If the
exact layer key did not exist, loading failed. There was no explicit nearest
configuration policy.

### Only two-dimensional interpolation

The old tables used input and output dimensions. The current JouleQuest schema
also models convolution output spatial area and attention coordinates, which
requires general multilinear interpolation.

## 4. First step forward: make strictness explicit

JouleGrad introduced an `out_of_range` constructor argument:

```python
from joulegrad import EnergyEstimator

estimator = EnergyEstimator(
    "energy_lookup_table.csv",
    out_of_range="error",
)
```

The first design rule was: production or scientific validation should not
silently use an unmeasured region. Therefore `error` became the default.

With `error`:

- a coordinate below or above a measured axis raises `ValueError`;
- an unavailable discrete convolution configuration raises `ValueError`;
- a required missing interpolation corner raises `ValueError`.

Cell indices are still selected from detached coordinates, while interpolation
weights use live tensors. Strict validation therefore does not remove gradients
for valid in-range queries.

## 5. Separate clamping from extrapolation

The old behavior mixed edge-cell selection and extrapolation. JouleGrad split
them into named policies.

### `clamp`

The coordinate itself is clamped before calculating interpolation weights:

$$
x'=\operatorname{clip}(x,x_{min},x_{max}).
$$

Energy equals the boundary estimate outside the measured interval. Because
PyTorch's clamp derivative is zero outside the interval, the energy gradient
with respect to that coordinate is also zero there.

Use this when a bounded estimate is preferable to extrapolation and silent
boundary saturation is acceptable.

### `extrapolate`

The edge cell is selected using a detached, bounded lookup coordinate, but the
weight uses the original live coordinate. This intentionally preserves the
original estimator's edge extrapolation and its gradient:

$$
t=\frac{x-x_{lo}}{x_{hi}-x_{lo}}.
$$

Use this only when extending the measured edge trend is a defensible model.
There is no guarantee that extrapolated energy remains positive or monotonic.

## 6. Add observability: `warn`

`warn` uses clamping but emits `RuntimeWarning` when a coordinate is outside a
measured axis. It also reports when JouleGrad must fill an incomplete grid or
substitute a discrete convolution configuration.

This policy was added for experiment development: a run can continue, but its
approximations remain visible in logs.

## 7. Handle incomplete current-format grids

JouleQuest lookup tables can be sparse because not every combination is
measured. JouleGrad distinguishes two problems:

1. **missing coordinate point inside an existing grid**;
2. **missing discrete Conv2d configuration** `(kernel_size, stride, padding)`.

### Missing grid points

For `warn`, `clamp`, and `extrapolate`, missing values are filled once when the
estimator is constructed. Each missing tensor cell takes the value of the
nearest measured cell using Euclidean distance in grid-index space.

This makes interpolation possible but does not create a real measurement.
`warn` reports how many points were filled.

For `error`, grids remain incomplete. A query fails only when a missing corner
has nonzero interpolation weight. Therefore an exact measured point can
succeed even if an adjacent, zero-weight corner is absent. This is more precise
than the old all-four-corners check.

### Missing Conv2d configuration

For `warn`, `clamp`, and `extrapolate`, JouleGrad selects the available
configuration minimizing squared distance over numeric `(kernel_size, stride,
padding)` values. `warn` reports the substitution.

`error` requires the exact configuration.

## 8. Four policies

The public policy set is intentionally limited to four names:

- `error` fails on unsupported coordinates, required missing corners, and
   unavailable convolution configurations;
- `warn` clamps coordinates, fills or substitutes sparse coverage, and emits
   `RuntimeWarning` messages;
- `clamp` performs the same bounded approximations silently;
- `extrapolate` preserves the original estimator's edge-cell extrapolation and
   gradients while silently filling or substituting sparse coverage.

The sparse-grid support is folded into the three continuing policies because
it is required to apply their coordinate behavior to current JouleQuest data.
The former `fallback` and `extrapolate_fallback` names are no longer accepted.

## 9. Policy matrix

| Policy | Outside measured axis | Warning | Missing grid point | Missing Conv config | Outside gradient |
| --- | --- | --- | --- | --- | --- |
| `error` | fail | no | fail if required | fail | n/a |
| `warn` | boundary value | yes | nearest fill | nearest substitute | zero |
| `clamp` | boundary value | no | nearest fill | nearest substitute | zero |
| `extrapolate` | linear edge extension | no | nearest fill | nearest substitute | retained |

For an axis containing only one measured value, extrapolation is undefined.
JouleGrad allows exact use of that coordinate, allows silent clamping under `clamp`,
warns and clamps under `warn` and `extrapolate`, and rejects incompatible queries
only under strict `error` mode.

## 10. How to choose a policy

### Reproducible final experiment

Use `error` and improve the measurement campaign until every required
coordinate and configuration is covered.

### Development with visible approximations

Use `warn`. Save warnings with experiment logs and treat them as part of the
result provenance.

### Bounded approximation without warnings

Use `clamp`, but document that coordinate gradients become zero beyond measured
boundaries and that sparse points may be substituted.

### Deliberate extrapolation study

Use `extrapolate`, and validate extrapolated and substituted values against
later hardware measurements.

## 11. Evolution summary

The progression was:

1. **Original estimator:** bundled Excel grids, synthetic zero anchor, silent
   edge extrapolation, no policy choice.
2. **Strict JouleGrad default:** explicit failure for unsupported coordinates
   and missing required data.
3. **Named clamp/extrapolate behavior:** separate bounded estimates from linear
   edge continuation.
4. **Warning mode:** make approximations visible without stopping experiments.
5. **Sparse-grid support:** let warning, clamping, and extrapolation modes work
   with incomplete JouleQuest grids and unavailable discrete configurations.
6. **Vectorized multilinear implementation:** extend the same semantics to
   Linear, Conv2d, Attention, and batched differentiable queries.

The central improvement is not merely adding options. It is making an
unmeasured estimate an explicit caller decision rather than an invisible side
effect of interpolation.
