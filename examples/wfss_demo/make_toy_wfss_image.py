"""Create a toy WFSS dispersed image from the demo inputs.

This is a pedagogical visualization, not a replacement for Mirage's physical
WFSS simulator. It uses a simple linear dispersion relation plus Gaussian trace
widths so the data flow is visible without requiring large reference files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
SOURCE_COLORS = {1: "#4cc9f0", 2: "#f72585", 3: "#f9c74f"}


def read_catalog(catalog_path: Path) -> dict[str, np.ndarray]:
    with catalog_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]

    header = lines[0].split()
    values = np.array([[float(item) for item in row.split()] for row in lines[1:]])
    return {name: values[:, i] for i, name in enumerate(header)}


def source_pixels(catalog: dict[str, np.ndarray], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Map the tiny RA/Dec demo field to a detector-like pixel area."""
    height, _ = shape
    ra = catalog["x_or_RA"]
    dec = catalog["y_or_Dec"]

    ra0 = np.mean(ra)
    dec0 = np.mean(dec)
    x_arcsec = (ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
    y_arcsec = (dec - dec0) * 3600.0

    pix_scale = 0.065  # NIRCam-like arcsec/pixel, just for display
    x0 = 95 + x_arcsec / pix_scale
    y0 = height / 2 + y_arcsec / pix_scale
    return x0, y0


def add_gaussian_stamp(image: np.ndarray, x: float, y: float, amplitude: float, sigma: float = 1.5) -> None:
    radius = int(np.ceil(4 * sigma))
    height, width = image.shape
    xmin = max(0, int(x) - radius)
    xmax = min(width, int(x) + radius + 1)
    ymin = max(0, int(y) - radius)
    ymax = min(height, int(y) + radius + 1)
    if xmin >= xmax or ymin >= ymax:
        return

    yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
    stamp = amplitude * np.exp(-0.5 * (((xx - x) / sigma) ** 2 + ((yy - y) / sigma) ** 2))
    image[ymin:ymax, xmin:xmax] += stamp


def add_trace_segment(
    image: np.ndarray,
    x1: float,
    y1: float,
    f1: float,
    x2: float,
    y2: float,
    f2: float,
    source_scale: float,
    sigma: float,
) -> None:
    """Deposit flux continuously between two wavelength samples."""
    distance = np.hypot(x2 - x1, y2 - y1)
    n_steps = max(2, int(np.ceil(distance)))
    for frac in np.linspace(0.0, 1.0, n_steps, endpoint=False):
        x = x1 + frac * (x2 - x1)
        y = y1 + frac * (y2 - y1)
        flux = f1 + frac * (f2 - f1)
        add_gaussian_stamp(image, x, y, 9.0 * source_scale * flux, sigma=sigma)


def select_source(catalog: dict[str, np.ndarray], source_id: int | None) -> dict[str, np.ndarray]:
    if source_id is None:
        return catalog

    keep = catalog["index"].astype(int) == source_id
    if not np.any(keep):
        raise ValueError(f"Source {source_id} is not present in the demo catalog.")
    return {name: values[keep] for name, values in catalog.items()}


def make_toy_images(
    output_dir: Path,
    selected_source_id: int | None = None,
    overlay: bool = False,
) -> tuple[Path, Path]:
    catalog_path = output_dir / "catalogs" / "demo_point_sources.cat"
    sed_path = output_dir / "seds" / "demo_source_seds.hdf5"
    yaml_path = output_dir / "yaml" / "demo_nircam_f250m_grismr.yaml"
    suffix = "" if selected_source_id is None else f"_source{selected_source_id}"
    if overlay:
        suffix += "_overlay"
    png_path = output_dir / f"toy_wfss_simulation{suffix}.png"
    npy_path = output_dir / f"toy_wfss_simulation{suffix}.npy"

    catalog = select_source(read_catalog(catalog_path), selected_source_id)
    with yaml_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    shape = (420, 980)
    direct = np.zeros(shape, dtype=float)
    dispersed = np.zeros(shape, dtype=float)
    x0, y0 = source_pixels(catalog, shape)

    mag = catalog["nircam_f250m_magnitude"]
    source_scale = 10 ** (-0.4 * (mag - mag.min()))

    for src_id, scale, x, y in zip(catalog["index"].astype(int), source_scale, x0, y0):
        add_gaussian_stamp(direct, x, y, 2500 * scale, sigma=2.2)

    # F250M-like crossing-filter window. This is intentionally simplified.
    wave_min = 2.35
    wave_max = 2.72
    trace_length = 760.0
    cross_sigma = 2.0

    with h5py.File(sed_path, "r") as hdf:
        trace_examples = {}
        for idx, x_start, y_start, scale in zip(catalog["index"].astype(int), x0, y0, source_scale):
            sed = hdf[str(idx)][:]
            wave = sed[0]
            flux = sed[1]
            keep = (wave >= wave_min) & (wave <= wave_max)
            wave = wave[keep]
            flux = flux[keep]
            if wave.size == 0:
                continue

            flux = flux / np.nanmax(flux)
            order_tilt = 0.035 * (wave - wave_min) / (wave_max - wave_min) * trace_length
            x_trace = x_start + 45 + (wave - wave_min) / (wave_max - wave_min) * trace_length
            y_trace = y_start + order_tilt
            trace_examples[idx] = (wave.copy(), x_trace.copy(), y_trace.copy())

            for i in range(wave.size - 1):
                add_trace_segment(
                    dispersed,
                    x_trace[i],
                    y_trace[i],
                    flux[i],
                    x_trace[i + 1],
                    y_trace[i + 1],
                    flux[i + 1],
                    scale,
                    cross_sigma,
                )

    rng = np.random.default_rng(42)
    background = 1.0
    noisy = rng.poisson(np.clip(dispersed + background, 0, None)).astype(float)

    np.save(npy_path, noisy)

    if overlay:
        make_overlay_figure(
            png_path,
            direct,
            dispersed,
            catalog,
            x0,
            y0,
            trace_examples,
            wave_min,
            wave_max,
            selected_source_id,
        )
        return png_path, npy_path

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    direct_display = np.sqrt(direct)
    wfss_image_for_display = dispersed if selected_source_id is not None else noisy
    wfss_display = np.sqrt(wfss_image_for_display)

    axes[0].imshow(direct_display, origin="lower", cmap="gray")
    for src_id, x, y in zip(catalog["index"].astype(int), x0, y0):
        axes[0].text(
            x + 14,
            y + 10,
            f"S{src_id}",
            color=SOURCE_COLORS[src_id],
            fontsize=10,
            weight="bold",
        )
    axes[0].set_title("1. Input source positions / cutouts")
    axes[0].set_xlabel("Detector x [pixel]")
    axes[0].set_ylabel("Detector y [pixel]")

    with h5py.File(sed_path, "r") as hdf:
        for key in sorted(hdf.keys(), key=int):
            if selected_source_id is not None and int(key) != selected_source_id:
                continue
            sed = hdf[key][:]
            keep = (sed[0] >= wave_min) & (sed[0] <= wave_max)
            axes[1].plot(
                sed[0][keep],
                sed[1][keep] / np.nanmax(sed[1][keep]),
                color=SOURCE_COLORS[int(key)],
                linewidth=2.0,
                label=f"Source {key}",
            )
            if int(key) == 2:
                axes[1].axvline(2.55, color=SOURCE_COLORS[int(key)], linestyle=":", linewidth=1.3)
                axes[1].text(2.552, 0.92, "emission", color=SOURCE_COLORS[int(key)], fontsize=8)
    axes[1].set_title("2. Input 1D spectra / SEDs")
    axes[1].set_xlabel("Wavelength [micron]")
    axes[1].set_ylabel("Relative flux")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].imshow(wfss_display, origin="lower", cmap="magma")
    view_label = "single-source noiseless view" if selected_source_id is not None else "all sources"
    axes[2].set_title(
        f"3. Toy WFSS dispersed image ({view_label})"
    )
    axes[2].set_xlabel("Detector x [pixel]")
    axes[2].set_ylabel("Detector y [pixel]")

    for src_id, (wave_for_trace, x_for_trace, y_for_trace) in trace_examples.items():
        color = SOURCE_COLORS[src_id]
        axes[2].plot(x_for_trace[::8], y_for_trace[::8], color=color, linewidth=1.0, alpha=0.9)
        axes[2].text(
            x_for_trace[0] - 28,
            y_for_trace[0] + 8,
            f"S{src_id}",
            color=color,
            fontsize=10,
            weight="bold",
            ha="right",
        )

    # Label wavelength-to-position mapping on the brightest source trace.
    label_source = int(catalog["index"][np.argmin(mag)])
    label_wave, label_x, label_y = trace_examples[label_source]
    for wavelength in [2.35, 2.45, 2.55, 2.65, 2.72]:
        x_label = np.interp(wavelength, label_wave, label_x)
        y_label = np.interp(wavelength, label_wave, label_y)
        axes[2].plot(x_label, y_label, marker="|", color="white", markersize=11, markeredgewidth=1.5)
        axes[2].text(
            x_label,
            y_label + 17,
            f"{wavelength:.2f}",
            color="white",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        if selected_source_id == 2 and np.isclose(wavelength, 2.55):
            axes[2].text(
                x_label,
                y_label - 27,
                "emission peak",
                color=SOURCE_COLORS[2],
                ha="center",
                va="top",
                fontsize=8,
            )
    axes[2].text(
        0.03,
        0.96,
        "wavelength [micron]",
        color="white",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    if selected_source_id is None:
        for ax in (axes[0], axes[2]):
            ax.set_xlim(0, shape[1] - 1)
            ax.set_ylim(0, shape[0] - 1)
    else:
        wave_for_trace, x_for_trace, y_for_trace = trace_examples[selected_source_id]
        axes[0].set_xlim(max(0, x0[0] - 80), min(shape[1] - 1, x0[0] + 120))
        axes[0].set_ylim(max(0, y0[0] - 80), min(shape[0] - 1, y0[0] + 80))
        axes[2].set_xlim(max(0, x_for_trace.min() - 45), min(shape[1] - 1, x_for_trace.max() + 45))
        axes[2].set_ylim(max(0, y_for_trace.min() - 70), min(shape[0] - 1, y_for_trace.max() + 70))

    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    return png_path, npy_path


def make_overlay_figure(
    png_path: Path,
    direct: np.ndarray,
    dispersed: np.ndarray,
    catalog: dict[str, np.ndarray],
    x0: np.ndarray,
    y0: np.ndarray,
    trace_examples: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    wave_min: float,
    wave_max: float,
    selected_source_id: int | None,
) -> None:
    """Plot direct cutout and dispersed trace in one detector coordinate frame."""
    if selected_source_id is None:
        selected_source_id = int(catalog["index"][0])

    src_id = selected_source_id
    wave, x_trace, y_trace = trace_examples[src_id]
    color = SOURCE_COLORS[src_id]
    src_x = float(x0[0])
    src_y = float(y0[0])

    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)

    # Show the dispersed image as the main detector-coordinate map.
    display = np.sqrt(dispersed)
    ax.imshow(display, origin="lower", cmap="magma", alpha=0.92)

    # Overlay the original direct cutout as compact cyan contours at its true
    # direct-image detector position.
    direct_norm = direct / np.nanmax(direct) if np.nanmax(direct) > 0 else direct
    ax.contour(
        direct_norm,
        levels=[0.12, 0.35, 0.65],
        colors=["#7df9ff"],
        linewidths=[0.8, 1.0, 1.2],
        alpha=0.95,
        origin="lower",
    )
    ax.scatter([src_x], [src_y], s=34, facecolor="#7df9ff", edgecolor="black", linewidth=0.5, zorder=5)
    ax.text(src_x - 10, src_y + 18, "direct cutout", color="#7df9ff", ha="right", fontsize=9)

    # Overlay the wavelength trace.
    ax.plot(x_trace, y_trace, color=color, linewidth=1.6, alpha=0.95)
    ax.text(x_trace[0] - 12, y_trace[0] - 18, f"S{src_id} dispersed trace", color=color, ha="right", fontsize=9)

    for wavelength in [wave_min, 2.45, 2.55, 2.65, wave_max]:
        x_label = np.interp(wavelength, wave, x_trace)
        y_label = np.interp(wavelength, wave, y_trace)
        ax.plot(x_label, y_label, marker="|", color="white", markersize=12, markeredgewidth=1.5)
        ax.text(x_label, y_label + 14, f"{wavelength:.2f}", color="white", ha="center", va="bottom", fontsize=8)

    x_emit = np.interp(2.55, wave, x_trace)
    y_emit = np.interp(2.55, wave, y_trace)
    ax.scatter([x_emit], [y_emit], s=70, facecolor="none", edgecolor="#ffffff", linewidth=1.2)
    ax.text(x_emit, y_emit - 28, "emission line", color="white", ha="center", va="top", fontsize=9)

    ax.annotate(
        "",
        xy=(x_trace[0], y_trace[0]),
        xytext=(src_x, src_y),
        arrowprops={"arrowstyle": "->", "color": "#d8f3dc", "lw": 1.0, "alpha": 0.85},
    )

    ax.set_title("Source 2: direct cutout and WFSS emission trace in one detector frame")
    ax.set_xlabel("Detector x [pixel]")
    ax.set_ylabel("Detector y [pixel]")
    ax.text(
        0.02,
        0.96,
        "white labels: wavelength [micron]",
        color="white",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    height, width = direct.shape
    ax.set_xlim(max(0, min(src_x, x_trace.min()) - 70), min(width - 1, x_trace.max() + 60))
    ax.set_ylim(max(0, min(src_y, y_trace.min()) - 85), min(height - 1, max(src_y, y_trace.max()) + 85))

    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a toy WFSS dispersed image from demo inputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-id", type=int, default=None, help="Only render one source, e.g. 2.")
    parser.add_argument("--overlay", action="store_true", help="Plot direct cutout and dispersed trace together.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png_path, npy_path = make_toy_images(
        args.output_dir.resolve(),
        selected_source_id=args.source_id,
        overlay=args.overlay,
    )
    print(f"Toy WFSS PNG written to: {png_path}")
    print(f"Toy WFSS array written to: {npy_path}")


if __name__ == "__main__":
    main()
