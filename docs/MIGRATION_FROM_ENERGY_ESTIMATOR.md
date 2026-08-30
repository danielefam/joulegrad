# Migration from `energy_estimator`

The original package selected bundled Excel tables by board name and exposed
one `estimate_model_energy(model, board, masks)` function. JouleGrad keeps the
same API idea while making the data source explicit and supporting the current
CSV schema.

## Before

```python
import energy_estimator as ee

result = ee.estimate_model_energy(
    model,
    board="JetsonNano",
    masks={"fc1": mask},
)
```

## Now

```python
from joulegrad import EnergyEstimator

estimator = EnergyEstimator("jetson_nano_energy_lookup_table.csv")
result = estimator.estimate_model(
    model,
    masks={"fc1": mask},
    module_names=("fc1", "fc2"),
)
```

The caller supplies the lookup path instead of a board label. Lookup files are
not bundled with the library. Results use `total_energy_mJ`, and layer entries
also report mJ. Conv2d modules require explicit input shapes.

`EnergyLookup` remains an alias for source compatibility, but new code should
use `EnergyEstimator`.

## Out-of-range behavior changed

The original `clamp_index` selected an edge cell with a clamped lookup value
but computed its interpolation fraction from the original coordinate. It
therefore silently extrapolated beyond measured axes. It also inserted a
synthetic zero row/column into bundled tables.

JouleGrad does not make either behavior implicit. Its default `error` policy
fails outside measured coverage, while named clamp, warning, and extrapolation
policies let the caller choose the approximation deliberately. The old
estimator also rejected `NaN` interpolation corners; its bundled,
two-dimensional, layer-specific tables simply did not expose the sparse
spatial-axis gaps found in current JouleQuest convolution grids.
See [Out-of-range policy evolution](OUT_OF_RANGE_POLICIES.md).