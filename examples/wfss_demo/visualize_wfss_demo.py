"""Visualize the prepared WFSS demo inputs.

This script does not require the large Mirage reference data. It reads the demo
catalog, SED HDF5 file, and WFSS YAML file, then creates a quick-look PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"


def read_catalog(catalog_path: Path) -> dict[str, np.ndarray]:
    with catalog_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]

    header = lines[0].split()
    values = np.array([[float(item) for item in row.split()] for row in lines[1:]])
    return {name: values[:, i] for i, name in enumerate(header)}


def plot_demo(output_dir: Path) -> Path:
    catalog_path = output_dir / "catalogs" / "demo_point_sources.cat"
    sed_path = output_dir / "seds" / "demo_source_seds.hdf5"
    yaml_path = output_dir / "yaml" / "demo_nircam_f250m_grismr.yaml"
    figure_path = output_dir / "wfss_demo_quicklook.png"

    catalog = read_catalog(catalog_path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 0.9])
    ax_field = fig.add_subplot(grid[:, 0])
    ax_sed = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, 1])

    mag = catalog["nircam_f250m_magnitude"]
    marker_size = 55 + 125 * (mag.max() - mag + 0.15) / (mag.max() - mag.min() + 0.15)
    scatter = ax_field.scatter(
        catalog["x_or_RA"],
        catalog["y_or_Dec"],
        s=marker_size,
        c=mag,
        cmap="viridis_r",
        edgecolor="black",
        linewidth=0.7,
    )
    for source_id, ra, dec in zip(catalog["index"], catalog["x_or_RA"], catalog["y_or_Dec"]):
        ax_field.text(ra, dec, f"  {int(source_id)}", va="center", fontsize=10)
    ax_field.set_title("Input source positions")
    ax_field.set_xlabel("RA [deg]")
    ax_field.set_ylabel("Dec [deg]")
    ax_field.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax_field, label="NIRCam F250M AB mag")

    with h5py.File(sed_path, "r") as hdf:
        for key in sorted(hdf.keys(), key=int):
            data = hdf[key][:]
            ax_sed.plot(data[0], data[1], label=f"source {key}")
    ax_sed.set_title("Input SEDs")
    ax_sed.set_xlabel("Wavelength [micron]")
    ax_sed.set_ylabel("F_lambda [cgs]")
    ax_sed.legend(frameon=False, fontsize=9)
    ax_sed.grid(alpha=0.25)

    ax_text.axis("off")
    summary = [
        "WFSS YAML summary",
        f"instrument: {params['Inst']['instrument']}",
        f"mode: {params['Inst']['mode']}",
        f"array: {params['Readout']['array_name']}",
        f"filter: {params['Readout']['filter']}",
        f"pupil/grism: {params['Readout']['pupil']}",
        f"readpatt: {params['Readout']['readpatt']}",
        f"ngroup/nint: {params['Readout']['ngroup']} / {params['Readout']['nint']}",
        f"output: {params['Output']['file']}",
    ]
    ax_text.text(0.0, 1.0, "\n".join(summary), va="top", family="monospace", fontsize=10)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a quick-look plot for the WFSS demo inputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_path = plot_demo(args.output_dir.resolve())
    print(f"Quick-look figure written to: {figure_path}")


if __name__ == "__main__":
    main()
