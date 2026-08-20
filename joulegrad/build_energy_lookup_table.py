"""Build one board/configuration lookup from strictly accepted summaries."""

import argparse
import re
from pathlib import Path

import pandas as pd


LINEAR_PATTERN = re.compile(r"^Linear_(\d+)_(\d+)$", re.IGNORECASE)
CONV_PATTERN = re.compile(
    r"^Conv_(\d+)_(\d+)_(\d+)_(\d+)(?:_(\d+))?$", re.IGNORECASE
)
RESNET_CONV_PATTERN = re.compile(
    r"^ResNetConv_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$", re.IGNORECASE
)
ATTENTION_PATTERN = re.compile(
    r"^(RotaryAttention|Attention)_(\d+)_(\d+)_(\d+)$", re.IGNORECASE
)


def _dimensions(layer_type, **values):
    dimensions = {
        "layer_type": layer_type,
        "input_features": None,
        "output_features": None,
        "input_channels": None,
        "output_channels": None,
        "input_image_size": None,
        "kernel_size": None,
        "stride": None,
        "padding": None,
        "sequence_length": None,
        "embed_dim": None,
        "num_heads": None,
        "head_dim": None,
    }
    dimensions.update(values)
    return dimensions


def _parse_layer(model_path):
    stem = Path(str(model_path)).stem
    match = LINEAR_PATTERN.fullmatch(stem)
    if match:
        input_features, output_features = map(int, match.groups())
        return _dimensions(
            "linear",
            input_features=input_features,
            output_features=output_features,
        )

    match = CONV_PATTERN.fullmatch(stem)
    if match:
        input_channels, image_size, kernel_size, padding, output_channels = (
            match.groups()
        )
        return _dimensions(
            "conv",
            input_channels=int(input_channels),
            output_channels=int(output_channels or 1),
            input_image_size=int(image_size),
            kernel_size=int(kernel_size),
            stride=1,
            padding=int(padding),
        )

    match = RESNET_CONV_PATTERN.fullmatch(stem)
    if match:
        values = map(int, match.groups())
        input_channels, output_channels, image_size, kernel_size, stride, padding = values
        return _dimensions(
            "conv",
            input_channels=input_channels,
            output_channels=output_channels,
            input_image_size=image_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    match = ATTENTION_PATTERN.fullmatch(stem)
    if match:
        layer_name, sequence_length, embed_dim, num_heads = match.groups()
        sequence_length, embed_dim, num_heads = map(
            int, (sequence_length, embed_dim, num_heads)
        )
        if embed_dim % num_heads != 0:
            return None
        layer_type = (
            "rotaryattention"
            if layer_name.lower().startswith("rotary")
            else "attention"
        )
        return _dimensions(
            layer_type,
            sequence_length=sequence_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
        )
    return None


def _accepted_rows(data):
    required = {"model_path", "energy_mean_mJ", "status", "quality_status"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Summary is missing strict acceptance columns: "
            + ", ".join(sorted(missing))
        )
    accepted = data[
        data["status"].eq("COMPLETE")
        & data["quality_status"].isin(("OK", "REVIEW"))
    ].copy()
    accepted["energy_mean_mJ"] = pd.to_numeric(
        accepted["energy_mean_mJ"], errors="coerce"
    )
    accepted = accepted[
        accepted["energy_mean_mJ"].notna() & (accepted["energy_mean_mJ"] > 0)
    ]
    if {"detected_regions", "expected_cycles"}.issubset(accepted.columns):
        accepted = accepted[
            accepted["detected_regions"].eq(accepted["expected_cycles"])
        ]
    if "clock_alignment_status" in accepted.columns:
        accepted = accepted[
            accepted["clock_alignment_status"].isin(("COMPLETE", "ALIGNED"))
        ]
    alignment_columns = {
        "clock_alignment_uncertainty_s",
        "clock_alignment_threshold_s",
    }
    if alignment_columns.issubset(accepted.columns):
        uncertainty = pd.to_numeric(
            accepted["clock_alignment_uncertainty_s"], errors="coerce"
        )
        threshold = pd.to_numeric(
            accepted["clock_alignment_threshold_s"], errors="coerce"
        )
        accepted = accepted[
            uncertainty.notna()
            & threshold.notna()
            & uncertainty.le(threshold)
        ]
    return accepted


def build_lookup_table(summary_paths):
    """Aggregate only strictly accepted measurements from one configuration."""

    measurements = []
    for summary_path in summary_paths:
        data = _accepted_rows(pd.read_csv(summary_path))
        for row in data.to_dict(orient="records"):
            dimensions = _parse_layer(row["model_path"])
            if dimensions is not None:
                measurements.append(
                    {**dimensions, "energy_mean_mJ": row["energy_mean_mJ"]}
                )

    columns = [
        "layer_type",
        "input_features",
        "output_features",
        "input_channels",
        "output_channels",
        "input_image_size",
        "kernel_size",
        "stride",
        "padding",
        "sequence_length",
        "embed_dim",
        "num_heads",
        "head_dim",
        "measurement_count",
        "energy_mean_mJ",
        "energy_stddev_mJ",
    ]
    if not measurements:
        return pd.DataFrame(columns=columns)

    dimensions = columns[:13]
    lookup = (
        pd.DataFrame(measurements)
        .groupby(dimensions, dropna=False, as_index=False)["energy_mean_mJ"]
        .agg(
            measurement_count="count",
            energy_mean_mJ="mean",
            energy_stddev_mJ="std",
        )
        .sort_values(dimensions, kind="stable")
        .reset_index(drop=True)
    )
    return lookup[columns]


def main():
    parser = argparse.ArgumentParser(
        description="Create one board/configuration energy lookup table."
    )
    parser.add_argument("summary_paths", type=Path, nargs="+")
    parser.add_argument(
        "--output", type=Path, default=Path("measurements/energy_lookup_table.csv")
    )
    args = parser.parse_args()
    lookup = build_lookup_table(args.summary_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(args.output, index=False)
    print(f"Wrote {len(lookup)} lookup rows to {args.output}")


if __name__ == "__main__":
    main()
