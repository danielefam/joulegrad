# Differentiable energy integration

## Contract

Every lookup represents one board, batch size, runtime/software stack,
acquisition setup, and inference definition. `EnergyLookup` loads the CSV once,
caches device grids, reports mJ per inference, and uses
`out_of_range="error"` for training. Missing interpolation corners and absent
operator grids raise `ValueError`.

JouleQuest, a private companion project, captures and processes the hardware
measurements used to produce the summary CSVs:
[github.com/danielefam/joulequest](https://github.com/danielefam/joulequest).
Place one processed summary per board/configuration in
[`../measurements/summaries`](../measurements/summaries). Each board snapshot
is built separately:

```bash
conda run -n banera_pt ecological-nas build-lookup \
  measurements/summaries/pi5_summary.csv \
  --output measurements/pi5_energy_lookup.csv
```

The copied builder accepts only `COMPLETE`, quality-`OK` or quality-`REVIEW`, positive-energy rows
with matching detected/expected cycles and verified clock-alignment uncertainty.
It never merges boards.

## Visibility and pruning

Causal, padding, and framework attention masks remain visibility masks. This
package does not read or modify them. Structural energy coordinates come only
from live pruning probabilities owned by `MaskedTransformerLayer._compute_probs()`.
No logits or exported-mask checkpoint schema is duplicated.

## Transformer mapping

`TransformerEnergyBridge` is constructed once after final model placement from
an explicit ordered list of `llama_block(...)` or `dinov2_block(...)` specs. It
does not scan the model in its hot path.

The supported `per_linear` mapping for MHA is:

| Operator | Effective input | Effective output | Gates |
| --- | ---: | ---: | --- |
| Q | residual width | `sum(p_head * p_qk)` | head, QK channel |
| K | residual width | `sum(p_head * p_qk)` | head, QK channel |
| V | residual width | `sum(p_head * p_vo)` | head, VO channel |
| O | `sum(p_head * p_vo)` | residual width | head, VO channel |

RoPE-paired QK probabilities are consumed after the existing wrapper expands
each logical pair. Vicuna/LLaMA SwiGLU maps `p_mlp.sum()` to both gate/up output
widths and the down-projection input width. DINOv2 plain MLP maps it to `fc1`
output and `fc2` input. DINOv2 SwiGLU maps the fused input projection to
`2 * p_mlp.sum()` and the output projection input to `p_mlp.sum()`.

GQA/per-Q-head layouts fail closed because the available Linear campaign does
not describe grouped shared K/V projections. Layer-drop also fails because no
whole-block measurement exists.

The measured aggregate attention module is square: changing its `embed_dim`
changes residual input/output and all four projections. Vicuna and DINOv2
materialization keep residual width fixed, so their aggregate mode is rejected.
`AggregateAttentionOperatorSpec` is only for another topology that explicitly
matches that complete measured operation; its Q/K/V/O components cannot also
appear as Linear terms.

## Explicit Linear and Conv2d topology

`ExplicitTopologyEnergyBridge` accepts qualified operator declarations and
externally owned live floating masks. A producer's output mask must be named
explicitly as the consumer's input mask. Residual producers must declare one
shared output-mask key. Conv input height/width, kernel, stride, and padding are
fixed in the spec. Grouped and dilated convolution are rejected.

This is an accounting bridge, not a generic masking wrapper or materializer.
No CNN architecture is claimed supported until a real trainer and an
architecture-preserving physical materializer declare the same topology.

## Training use

The standalone Vicuna and DINO drivers already expose the opt-in arguments. A
positive weight requires a lookup path and an explicit board/configuration
label. Each driver builds and preflights its bridge once after wrapping and
device placement, then composes each step with:

```python
loss, energy_mj = compose_energy_loss(
    task_loss,
    sparsity_loss,
    bridge=energy_bridge,
    energy_weight=energy_config.weight,
)
```

`energy_mj.item()` is logged only after `loss` is constructed. Persisted energy
metadata is limited to `energy_checkpoint_metadata(...)`; the target
model/optimizer state remains the authority for mask logits.

`ecological-nas train-vicuna --help` and
`ecological-nas train-dino --help` list all driver options. The default
`--energy_weight 0` neither loads a CSV nor evaluates an energy bridge.

## Export validation

The target repository's thresholding, plan construction, and physical
materializers remain unchanged. After materialization, a separate
`ExplicitTopologyEnergyBridge` query using the same lookup identity declares
the resulting actual Linear/Conv dimensions. That value is an estimate, not a
hardware measurement; a hardware result requires re-measurement of the
materialized model.

## Current measurement snapshot

Applying the strict builder separately to the current summaries yields:

| Board summary | Linear rows | Conv rows | Attention rows |
| --- | ---: | ---: | ---: |
| Pi 5 | 60 | 164 | 0 |
| Jetson Nano base | 51 | 133 | 0 |
| AGX Orin | 1 | 11 | 0 |

The Linear axes begin at 64 features and have missing cells. Vicuna's dense MLP
width also exceeds the measured maximum of 8192. These snapshots are not a
complete transformer-training lookup and are not copied as production tables.
Experiments still in progress must be processed and validated before rebuilding
a board-specific table.