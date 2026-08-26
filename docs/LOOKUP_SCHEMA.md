# Lookup CSV schema

JouleGrad consumes one lookup table representing one hardware, runtime, batch,
dtype, power-mode, and acquisition configuration. It does not combine or
validate measurement campaigns; the producer owns those responsibilities.

JouleQuest emits this format through
`python -m processing_report.build_energy_lookup_table`.

## Columns

| Column | Meaning |
| --- | --- |
| `layer_type` | `linear`, `conv`, `attention`, or `rotaryattention` |
| `input_features` | Linear input width |
| `output_features` | Linear output width |
| `input_channels` | Conv2d input channels |
| `output_channels` | Conv2d output channels |
| `input_image_size` | Square Conv2d input side |
| `kernel_size` | Conv2d kernel side |
| `stride` | Conv2d stride |
| `padding` | Conv2d padding |
| `sequence_length` | Attention sequence length |
| `embed_dim` | Attention embedding width |
| `num_heads` | Measured attention head count |
| `head_dim` | Attention head dimension |
| `measurement_count` | Accepted repeats aggregated into the row |
| `energy_mean_mJ` | Mean energy per input sample in millijoules |
| `energy_stddev_mJ` | Standard deviation across accepted repeats |

Unused coordinates are empty for a row. `layer_type` and positive
`energy_mean_mJ` are always required. Coordinates forming one interpolation
grid must be unique.

Linear grids use `(input_features, output_features)`. Conv2d grids are split by
the discrete `(kernel_size, stride, padding)` configuration and interpolate
`(input_channels, output_channels, output_spatial_area)`. Attention grids use
`(sequence_length, embed_dim, head_dim)`.