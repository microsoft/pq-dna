# Contributing to pq-dna

Thanks for your interest in contributing.

## Contributor License Agreement

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Development setup

Create a virtual environment in the repository root, activate it, and install the package with the development extras. Every command below assumes the environment is active and the working directory is the repository root.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Linux / macOS (bash or zsh):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Running the checks

CI runs the linter and the test suite on Python 3.9 through 3.12 on Linux and Windows. Run the same two commands locally:

```bash
python -m ruff check src tests
python -m pytest tests -q
```

Both must pass before a pull request is merged. CI also reports coverage (`--cov=pqdna --cov-report=term-missing`), which you can add to the second command. The fixed-point export tests also compile the generated C header and compare it with the Python integer model when a C compiler is on the `PATH` (or named in the `CC` environment variable); without one, those three tests are skipped.

## Reproducing the published results

```bash
python -m pqdna.demo
```

This regenerates every metric quoted in [README.md](README.md) and [docs/METHOD.md](docs/METHOD.md) into `out/` under the current working directory, and takes a few minutes. If a change alters those numbers, update both documents in the same pull request and say so explicitly in the PR description.

## Coding conventions

- Python 3.9+ with `from __future__ import annotations`.
- Type hints on all public functions.
- NumPy-style docstrings on public classes and functions.
- Keep the on-device code path (`features.py`, `pqdna.py`, `chs.py`, `edge_export.py`) free of SciPy, pandas, and scikit-learn. Those are permitted only in the simulation and cloud-side modules (`synth.py`, `rul.py`, `demo.py`), because the on-device path must remain translatable to fixed-point C.
- Every source file carries the MIT copyright header.

## What we are especially looking for

- **Validation against real measurement data.** The current evaluation is entirely synthetic (see [Limitations](README.md#limitations) in the README and Section 6 of [docs/METHOD.md](docs/METHOD.md)). Contributions that evaluate the method against real power-quality recordings and real component-failure records are the single most valuable addition to this project.
- Additional physics-of-failure damage kernels with citations to the reliability literature.
- A C implementation of the rest of the on-device path (feature extraction and the damage kernels `g`) to sit alongside the update, CHS and RUL code in the generated fixed-point header.

## Reporting issues and getting support

See [SUPPORT.md](SUPPORT.md). For security issues, do **not** open a public issue — follow [SECURITY.md](SECURITY.md).
