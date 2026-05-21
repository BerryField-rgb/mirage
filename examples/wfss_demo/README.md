# WFSS demo

This demo builds a small WFSS simulation setup. By default it creates the
catalog, SED file, and a NIRCam WFSS YAML from a local Mirage template, so it
can run on a machine that has not installed all Mirage runtime dependencies or
downloaded the large Mirage reference data.

Run from the repository root:

```powershell
python examples\wfss_demo\wfss_demo.py
```

If your active Python environment does not already have `numpy`, `h5py`, and
`pyyaml`, this lightweight command will run the prepared-only demo without
installing the full Mirage dependency tree:

```powershell
uv run --no-project --with numpy --with h5py --with pyyaml python examples\wfss_demo\wfss_demo.py
```

The script creates:

- `output/catalogs/demo_point_sources.cat`
- `output/seds/demo_source_seds.hdf5`
- `output/yaml/*.yaml`
- `output/sim_data/`

To visualize the prepared inputs:

```powershell
uv run --no-project --with numpy --with h5py --with pyyaml --with matplotlib python examples\wfss_demo\visualize_wfss_demo.py
```

This writes `output/wfss_demo_quicklook.png`, showing the input source
positions, SED curves, and the key WFSS YAML settings.

To create a simple toy dispersed WFSS image without the full Mirage reference
data:

```powershell
uv run --no-project --with numpy --with h5py --with pyyaml --with matplotlib python examples\wfss_demo\make_toy_wfss_image.py
```

This writes `output/toy_wfss_simulation.png` and
`output/toy_wfss_simulation.npy`. The PNG is a teaching visualization with a
linear dispersion model; use `wfss_demo.py --run-sim` for real Mirage WFSS data.

If you want to exercise Mirage's APT-to-YAML generator as part of the demo,
add `--from-apt`. That path imports Mirage and therefore needs the package
dependencies installed:

```powershell
python examples\wfss_demo\wfss_demo.py --from-apt
```

To run the full WFSS simulator after the inputs are created, configure the
reference data first and pass `--run-sim`:

```powershell
$env:MIRAGE_DATA="D:\path\to\mirage_data"
$env:CRDS_PATH="D:\path\to\crds_cache"
$env:CRDS_SERVER_URL="https://jwst-crds.stsci.edu"
python examples\wfss_demo\wfss_demo.py --run-sim
```

For WFSS, `MIRAGE_DATA` must include the usual Mirage reference data plus one of
these grism configuration directories:

- `$MIRAGE_DATA/niriss/GRISM_NIRISS/current`
- `$MIRAGE_DATA/nircam/GRISM_NIRCAM/current`
