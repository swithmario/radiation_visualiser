# Radiation visualiser handover

## Purpose and structure

Python studies of a prescribed accelerated point charge and the surrounding
electric-field geometry. Each script is a standalone Matplotlib experiment.

- `accelerated_charge_visualiser.py`: primary interactive view and snapshot CLI.
  It blends two uniform-velocity fields across a causal transition region.
- `field_line_kink_geometry.py`: relativistic angular field-line construction.
- `charge_worldline_diagram.py`: trajectory and acceleration light cones.
- `lienard_wiechert_field.py`: field slice with numerical retarded-time solution.
- `lienard_wiechert_shell_comparison.py`: field and causal-shell comparisons.
- `docs/radiation_visualiser.png`: primary documentation output.

## Run and check

Install `requirements.txt` in an isolated Python environment. Run
`python accelerated_charge_visualiser.py` for the primary interactive view.
For a headless check:

```sh
MPLBACKEND=Agg python accelerated_charge_visualiser.py \
  --output /path/outside/git/radiation_check.png --no-show
python -m py_compile *.py
```

There is no numerical physics regression suite. A successful plot verifies
execution, not agreement with a calibrated measurement or an independent solver.

## Dependencies, vendors, and boundaries

The runtime uses NumPy and Matplotlib. No package source is vendored. A PyCharge
reference is retained in the shell-comparison script; that reference alone is
not a complete provenance or numerical-conformance audit.

This repository is public. Keep private notes, captures, environments, and
temporary outputs outside Git. No licence change is part of the naming repair.
Use lowercase_snake_case for first-party filenames and DDMMMYYYY for dated reports.

## Constraints and current state

Use `c = 1` as the documented normalisation. Preserve the distinction between
the primary field blend, geometric sketches, and retarded-time field studies.
The primary blend is not a complete Maxwell solver. Two-dimensional slices,
singularity cutoffs, numerical root finding, and prescribed motion limit claims.

The 15SEP2026 rename maps `rad.py`, `bruhemstrahlung.py`, `cold.py`, `pasta.py`,
and `rice.py` to the five descriptive names above, in the same order. README
commands use the new names. Git retains the earlier names and source history.
The naming change also makes four Matplotlib math strings explicit raw strings.

Next, add independent analytical checks before making quantitative radiation
claims. Keep qualitative illustrations labelled as such. State Mac Mini M4 in
commits made on this machine.

## Documentation and Git identity

Use first person for personal decisions and experience in README prose. Use
direct technical language for software behaviour and instructions. Do not
describe the maintainer as "the owner". Preserve quoted source wording and
technical ownership terms.

Local commits must use `swithmario` and
`28229111+swithmario@users.noreply.github.com`. Verify both author and committer
before pushing. Histories were corrected on 15SEP2026; compare an older checkout
with the corrected remote before merging or pushing it. Record Mac Mini M4 in
commits made on this machine.
