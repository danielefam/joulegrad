# Differentiable interpolation

JouleGrad performs multilinear interpolation on rectilinear measured grids.
For one coordinate in a $D$-dimensional cell,

$$
\widehat{E}(x)=
\sum_{b\in\{0,1\}^{D}} E_b
\prod_{j=1}^{D}
\left[b_j a_j+(1-b_j)(1-a_j)\right],
$$

where $a_j$ is the normalized coordinate between the lower and upper measured
axis values and $E_b$ is one corner measurement.

Cell selection uses a detached coordinate because grid indices are discrete.
The interpolation fraction uses the live tensor, so gradients propagate to
effective dimensions inside the selected cell.

Lookup grids are built once and cached per device. Batched query methods group
compatible coordinates into one vectorized interpolation call.

Clamping can produce zero gradient outside the measured interval. Missing
corners fail under `error`; `warn`, `clamp`, and `extrapolate` fill missing
points with the nearest measured grid value. These substitutions are
approximations and should be chosen explicitly by the caller.

See [Out-of-range policy evolution](OUT_OF_RANGE_POLICIES.md) for the exact
behavior matrix and the difference from the original estimator's silent edge
extrapolation.