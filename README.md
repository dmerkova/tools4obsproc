# tools4obsproc

Python tools for reading GSI-style conventional observation diag files
(`diag_conv_{t,uv,q}_{ges,anl}.YYYYMMDDHH.nc4`, also `.nc` and `.gz`).

## Install

```bash
pip install -e ".[test]"
```

## Usage

```python
from tools4obsproc import read_diag, DiagReader, uv_components

# all files in a directory
data = read_diag("/data/diag", variables=("t", "uv", "q"), dates="all")

# a single analysis time (YYYYMMDDHH, YYYYMMDD, ISO string, date or datetime)
data = read_diag("/data/diag", "t", dates="2020010100")

# a range (inclusive; a date-only end covers the whole day), optionally every 6 h
data = read_diag("/data/diag", ["t", "q"], dates=("20200101", "20200105"), freq="6h")
data = read_diag("/data/diag", "t", start="2020010100", end="2020010218")

# path template or glob; analysis stage "anl" instead of "ges"
data = read_diag("/data/{YYYYMMDD}/", "uv", dates=("20200101", "20200103"))
data = read_diag("/data/2020*/", "t", dates="all", stage="anl")

uv = uv_components(data["uv"])      # columns u, v (and u_omf, v_omf)

reader = DiagReader("/data/diag", ["t"], dates="all")
reader.files()   # {"t": [(datetime, path), ...]}
reader.read()    # {"t": DataFrame}
```

Template keys: `{YYYYMMDDHH}`, `{YYYYMMDD}`, `{YYYY}`, `{MM}`, `{DD}`, `{HH}`.

Each returned `pandas.DataFrame` has a `date` column (analysis time from the filename) and
decoded string fields `station_id`, `obs_class`, plus `obs_type`, `lat`, `lon`, `pressure`,
`height`, `time`, `obs`, `omf` (obs-minus-forecast), `errinv_*` (inverse error), `qc_flag`
(analysis use flag), `prep_qc`, ... (those present in the file; `uv` also has `u_obs`, `v_obs`,
`u_omf`, `v_omf`). `NoDiagFilesError` (a `FileNotFoundError`) is raised if nothing matches.

## Tests

```bash
pytest
```
