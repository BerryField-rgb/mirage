"""Small WFSS demo for Mirage.

The default run creates the inputs needed for a WFSS simulation:
an ASCII source catalog, an HDF5 SED file, and a WFSS YAML file. Use --from-apt
to generate YAML files from the example APT XML/pointing files. Use --run-sim to
continue into WFSSSim when the large Mirage reference data are available locally.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = REPO_ROOT / "examples" / "wfss_example_data" / "niriss_wfss_example.xml"
DEFAULT_POINTING = REPO_ROOT / "examples" / "wfss_example_data" / "niriss_wfss_example.pointing"
DEFAULT_TEMPLATE = REPO_ROOT / "examples" / "wfss_example_data" / "wfss_f250m_test.yaml"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"


def ensure_demo_environment(output_dir: Path) -> None:
    """Set harmless defaults needed for offline YAML generation."""
    placeholder = output_dir / "mirage_data_placeholder"
    placeholder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MIRAGE_DATA", str(placeholder))
    os.environ.setdefault("CRDS_PATH", str(output_dir / "crds_cache"))
    os.environ.setdefault("CRDS_SERVER_URL", "https://jwst-crds.stsci.edu")


def make_source_catalog(catalog_dir: Path) -> Path:
    """Create a tiny point-source catalog with three NIRISS magnitude columns."""
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / "demo_point_sources.cat"

    rows = [
        # index, RA, Dec, NIRISS F090W/F150W/F200W, NIRCam F250M/F300M
        (1, 12.200144142, 12.1996128, 18.0, 18.2, 18.4, 18.5, 18.7),
        (2, 12.198338444, 12.1996128, 19.1, 18.7, 18.2, 18.0, 17.9),
        (3, 12.201050000, 12.1989000, 20.0, 19.5, 19.1, 18.9, 18.7),
    ]

    with catalog_path.open("w", encoding="utf-8") as handle:
        handle.write("# position_RA_Dec\n")
        handle.write("# abmag\n")
        handle.write("#\n")
        handle.write("#\n")
        handle.write(
            "index x_or_RA y_or_Dec "
            "niriss_f090w_magnitude niriss_f150w_magnitude niriss_f200w_magnitude "
            "nircam_f250m_magnitude nircam_f300m_magnitude\n"
        )
        for row in rows:
            handle.write(
                f"{row[0]} {row[1]:.9f} {row[2]:.9f} "
                f"{row[3]:.3f} {row[4]:.3f} {row[5]:.3f} {row[6]:.3f} {row[7]:.3f}\n"
            )

    return catalog_path


def make_sed_file(sed_dir: Path) -> Path:
    """Create source spectra, including one emission-line example."""
    sed_dir.mkdir(parents=True, exist_ok=True)
    sed_path = sed_dir / "demo_source_seds.hdf5"

    wavelength = np.arange(0.8, 5.25, 0.005, dtype=np.float32)
    continuum = np.full_like(wavelength, 1.0e-18)

    spectra = {
        # Source 1: nearly flat continuum.
        1: continuum,
        # Source 2: continuum plus a deliberately strong emission feature.
        2: continuum * (1.0 + 3.0 * np.exp(-0.5 * ((wavelength - 2.55) / 0.025) ** 2)),
        # Source 3: a sloped continuum, so the dispersed trace fades with wavelength.
        3: continuum * np.linspace(2.0, 0.35, wavelength.size, dtype=np.float32),
    }

    with h5py.File(sed_path, "w") as hdf:
        for source_index, flux in spectra.items():
            dataset = hdf.create_dataset(
                str(source_index),
                data=np.vstack([wavelength, flux.astype(np.float32)]),
                dtype="f",
                compression="gzip",
                compression_opts=9,
            )
            dataset.attrs["wavelength_units"] = "micron"
            dataset.attrs["flux_units"] = "flam_cgs"

    return sed_path


def create_yaml_from_template(template_yaml: Path, catalog_path: Path, output_dir: Path) -> list[Path]:
    """Create a self-contained demo WFSS YAML from a local Mirage template."""
    yaml_dir = output_dir / "yaml"
    sim_dir = output_dir / "sim_data"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    with template_yaml.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    params["Inst"]["instrument"] = "NIRCam"
    params["Inst"]["mode"] = "wfss"
    params["Readout"]["array_name"] = "NRCB5_FULL"
    params["Readout"]["filter"] = "F250M"
    params["Readout"]["pupil"] = "GRISMR"
    params["simSignals"]["pointsource"] = str(catalog_path)
    params["simSignals"]["galaxyListFile"] = "None"
    params["simSignals"]["extended"] = "None"
    params["simSignals"]["bkgdrate"] = "medium"
    params["Output"]["file"] = "demo_nircam_f250m_grismr_uncal.fits"
    params["Output"]["directory"] = str(sim_dir)
    params["Output"]["datatype"] = "raw"
    params["Output"]["grism_source_image"] = True
    params["Output"]["date_obs"] = "2026-05-21"

    yaml_path = yaml_dir / "demo_nircam_f250m_grismr.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(params, handle, sort_keys=False)

    return [yaml_path]


def generate_yamls(xml_file: Path, pointing_file: Path, catalog_path: Path, output_dir: Path) -> list[Path]:
    """Generate Mirage YAML files from the example APT XML and pointing files."""
    from mirage.yaml import yaml_generator

    yaml_dir = output_dir / "yaml"
    sim_dir = output_dir / "sim_data"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    catalogs = {"point_source": str(catalog_path)}
    generator = yaml_generator.SimInput(
        input_xml=str(xml_file),
        pointing_file=str(pointing_file),
        catalogs=catalogs,
        cosmic_rays={"library": "SUNMAX", "scale": 1.0},
        background="medium",
        roll_angle=12.5,
        dates="2026-05-21",
        reffile_defaults="crds",
        output_dir=str(yaml_dir),
        simdata_output_dir=str(sim_dir),
        datatype="raw",
        add_ghosts=False,
        verbose=True,
        offline=True,
    )
    generator.use_linearized_darks = True
    generator.create_inputs()
    return sorted(yaml_dir.glob("jw*.yaml"))


def find_wfss_yamls(yaml_files: list[Path]) -> list[Path]:
    wfss_files = []
    for yaml_file in yaml_files:
        with yaml_file.open("r", encoding="utf-8") as handle:
            params = yaml.safe_load(handle)
        if params["Inst"]["mode"].lower() == "wfss":
            wfss_files.append(yaml_file)
    return wfss_files


def grism_reference_ready() -> bool:
    mirage_data = os.environ.get("MIRAGE_DATA")
    if mirage_data is None:
        return False
    niriss_grism = Path(mirage_data) / "niriss" / "GRISM_NIRISS" / "current"
    nircam_grism = Path(mirage_data) / "nircam" / "GRISM_NIRCAM" / "current"
    return niriss_grism.is_dir() or nircam_grism.is_dir()


def normalizing_column_for_yaml(yaml_file: Path) -> str:
    with yaml_file.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    instrument = params["Inst"]["instrument"].lower()
    filt = params["Readout"]["filter"].lower()
    if instrument == "nircam":
        return f"nircam_{filt}_magnitude"
    return f"niriss_{filt[:-1].lower()}_magnitude" if filt.startswith("GR150") else f"niriss_{filt}_magnitude"


def run_wfss_simulation(wfss_yamls: list[Path], sed_path: Path) -> None:
    """Run WFSSSim for each WFSS YAML."""
    if not grism_reference_ready():
        raise RuntimeError(
            "MIRAGE_DATA is not pointing to complete WFSS reference data. "
            "Add GRISM_NIRISS/current or GRISM_NIRCAM/current, then rerun with --run-sim."
        )

    from mirage import wfss_simulator

    for yaml_file in wfss_yamls:
        sim = wfss_simulator.WFSSSim(
            str(yaml_file),
            SED_file=str(sed_path),
            SED_normalizing_catalog_column=normalizing_column_for_yaml(yaml_file),
            save_dispersed_seed=True,
            extrapolate_SED=True,
            create_continuum_seds=True,
        )
        sim.create()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small Mirage NIRISS WFSS demo.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="APT XML file exported from APT.")
    parser.add_argument("--pointing", type=Path, default=DEFAULT_POINTING, help="APT pointing file.")
    parser.add_argument("--template-yaml", type=Path, default=DEFAULT_TEMPLATE, help="Template YAML for default mode.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for demo outputs.")
    parser.add_argument("--from-apt", action="store_true", help="Generate YAML files from APT XML/pointing files.")
    parser.add_argument("--run-sim", action="store_true", help="Run WFSSSim after generating inputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_demo_environment(output_dir)

    catalog_path = make_source_catalog(output_dir / "catalogs")
    sed_path = make_sed_file(output_dir / "seds")
    if args.from_apt:
        yaml_files = generate_yamls(args.xml.resolve(), args.pointing.resolve(), catalog_path, output_dir)
    else:
        yaml_files = create_yaml_from_template(args.template_yaml.resolve(), catalog_path, output_dir)
    wfss_yamls = find_wfss_yamls(yaml_files)

    print("\nWFSS demo inputs created")
    print(f"  source catalog: {catalog_path}")
    print(f"  SED file:       {sed_path}")
    print(f"  YAML files:     {len(yaml_files)} total, {len(wfss_yamls)} WFSS")
    for yaml_file in wfss_yamls:
        print(f"    - {yaml_file}")

    if args.run_sim:
        run_wfss_simulation(wfss_yamls, sed_path)
        print("\nFull WFSS simulation finished.")
    else:
        print("\nPrepared-only run complete. Add --run-sim after MIRAGE_DATA is configured.")


if __name__ == "__main__":
    main()
