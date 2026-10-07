"""Reader for GSI conventional diag files (``diag_conv_{t,uv,q}_{ges,anl}.YYYYMMDDHH.nc4``)."""
from __future__ import annotations

import glob
import gzip
import logging
import os
import re
import shutil
import tempfile
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

VALID_VARIABLES = ("t", "uv", "q")
VALID_STAGES = ("ges", "anl")

DateLike = Union[datetime, date, str]
DatesSpec = Union[None, str, DateLike, Sequence[DateLike]]

_FILE_RE = re.compile(
    r"^diag_conv_(?P<var>t|uv|q)_(?P<stage>ges|anl)\.(?P<date>\d{10})(?:\.nc4?)?(?:\.gz)?$"
)
_TEMPLATE_KEYS = {"YYYYMMDDHH": "%Y%m%d%H", "YYYYMMDD": "%Y%m%d", "YYYY": "%Y",
                  "MM": "%m", "DD": "%d", "HH": "%H"}
_GLOB_CHARS = set("*?[")

# Fields kept (when present) -> output column names.
_COMMON_FIELDS = {
    "Station_ID": "station_id",
    "Observation_Class": "obs_class",
    "Observation_Type": "obs_type",
    "Observation_Subtype": "obs_subtype",
    "Latitude": "lat",
    "Longitude": "lon",
    "Pressure": "pressure",
    "Height": "height",
    "Time": "time",
    "Prep_QC_Mark": "prep_qc",
    "Setup_QC_Mark": "setup_qc",
    "Prep_Use_Flag": "prep_use_flag",
    "Analysis_Use_Flag": "qc_flag",
    "Errinv_Input": "errinv_input",
    "Errinv_Adjust": "errinv_adjust",
    "Errinv_Final": "errinv_final",
    "Observation": "obs",
    "Obs_Minus_Forecast_adjusted": "omf",
    "Obs_Minus_Forecast_unadjusted": "omf_unadjusted",
}
_UV_FIELDS = {
    "u_Observation": "u_obs",
    "v_Observation": "v_obs",
    "u_Obs_Minus_Forecast_adjusted": "u_omf",
    "v_Obs_Minus_Forecast_adjusted": "v_omf",
    "u_Obs_Minus_Forecast_unadjusted": "u_omf_unadjusted",
    "v_Obs_Minus_Forecast_unadjusted": "v_omf_unadjusted",
}
_STRING_COLUMNS = ("station_id", "obs_class")


class NoDiagFilesError(FileNotFoundError):
    """Raised when no diag file matches the requested path/variables/dates."""


def _parse_date(value: DateLike, end: bool = False) -> tuple[datetime, bool]:
    """Parse a date-like value. Returns ``(datetime, has_hour)``.

    Accepts ``datetime``, ``date``, and strings ``YYYYMMDDHH``, ``YYYYMMDD`` or ISO.
    """
    if isinstance(value, datetime):
        return value.replace(tzinfo=None), True
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day), False
    if isinstance(value, str):
        s = value.strip()
        if re.fullmatch(r"\d{10}", s):
            return datetime.strptime(s, "%Y%m%d%H"), True
        if re.fullmatch(r"\d{8}", s):
            return datetime.strptime(s, "%Y%m%d"), False
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            raise ValueError(f"Cannot parse date {value!r}") from None
        return dt.replace(tzinfo=None), "T" in s or " " in s
    raise TypeError(f"Unsupported date type: {type(value).__name__}")


def _validate_variables(variables: Union[str, Iterable[str]]) -> tuple[str, ...]:
    if isinstance(variables, str):
        variables = [variables]
    out = []
    for v in variables:
        key = str(v).lower()
        if key not in VALID_VARIABLES:
            raise ValueError(f"Invalid variable {v!r}; choose from {VALID_VARIABLES}")
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("variables must not be empty")
    return tuple(out)


def _decode(series: pd.Series) -> pd.Series:
    """Decode byte strings and strip padding/NULs."""
    def conv(x):
        if isinstance(x, (bytes, np.bytes_)):
            return bytes(x).decode("utf-8", errors="replace").replace("\x00", "").strip()
        return x
    return series.map(conv)


def _char_to_str(arr: np.ndarray) -> np.ndarray:
    """Collapse an (nobs, nchar) char array into an array of bytes strings."""
    if arr.ndim == 2:
        return np.array([b"".join(row) for row in arr], dtype=object)
    return arr


def uv_components(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with ``u`` and ``v`` columns (plus ``u_omf``/``v_omf`` if present)."""
    cols = {"u_obs": "u", "v_obs": "v", "u_omf": "u_omf", "v_omf": "v_omf"}
    missing = [c for c in ("u_obs", "v_obs") if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame has no wind components ({missing}); is it a 'uv' table?")
    return df[[c for c in cols if c in df.columns]].rename(columns=cols)


class DiagReader:
    """Locate and read GSI conventional diag files.

    Parameters
    ----------
    path
        Directory, glob pattern, or template containing ``{YYYYMMDD}``, ``{YYYYMMDDHH}``,
        ``{YYYY}``, ``{MM}``, ``{DD}``, ``{HH}`` (e.g. ``/data/{YYYYMMDD}/``).
    variables
        Subset of ``"t"``, ``"uv"``, ``"q"``.
    dates
        ``None``/``"all"`` for every file; a single date/datetime; or a ``(start, end)``
        pair (inclusive). ``start``/``end`` keywords may be used instead.
        Date-only ``end`` covers the whole day.
    stage
        ``"ges"`` or ``"anl"``.
    freq
        Optional pandas frequency (e.g. ``"6h"``) to keep only files on the grid
        ``start, start+freq, ...`` for ranges.
    """

    def __init__(
        self,
        path: Union[str, "os.PathLike[str]"],
        variables: Union[str, Iterable[str]] = VALID_VARIABLES,
        dates: DatesSpec = None,
        stage: str = "ges",
        *,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        freq: Optional[str] = None,
    ) -> None:
        self.path = os.fspath(path)
        self.variables = _validate_variables(variables)
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage {stage!r}; choose from {VALID_STAGES}")
        self.stage = stage
        self.freq = freq
        self._start, self._end = self._resolve_dates(dates, start, end)

    def _resolve_dates(self, dates, start, end):
        if (start is not None or end is not None):
            if dates is not None and not (isinstance(dates, str) and dates.lower() == "all"):
                raise ValueError("Use either `dates` or `start`/`end`, not both")
            dates = (start, end)
        if dates is None or (isinstance(dates, str) and dates.lower() == "all"):
            return None, None
        if isinstance(dates, (tuple, list)):
            if len(dates) != 2:
                raise ValueError("A date range must be (start, end)")
            s, e = dates
            bounds = []
            for val, is_end in ((s, False), (e, True)):
                if val is None:
                    bounds.append(None)
                    continue
                dt, has_hour = _parse_date(val)
                if is_end and not has_hour:
                    dt += timedelta(hours=23, minutes=59, seconds=59)
                bounds.append(dt)
            if bounds[0] and bounds[1] and bounds[0] > bounds[1]:
                raise ValueError("start date is after end date")
            return bounds[0], bounds[1]
        dt, has_hour = _parse_date(dates)
        if has_hour:
            return dt, dt
        return dt, dt + timedelta(hours=23, minutes=59, seconds=59)

    # ---- file discovery -------------------------------------------------
    def _patterns(self) -> list[str]:
        p = self.path
        if "{" in p:
            if self._start is not None and self._end is not None:
                times = pd.date_range(self._start.replace(hour=0, minute=0, second=0),
                                      self._end, freq="D")
                out = []
                for t in times:
                    q = p
                    for key, fmt in _TEMPLATE_KEYS.items():
                        q = q.replace("{" + key + "}", t.strftime(fmt))
                    out.append(q)
                return list(dict.fromkeys(out))
            q = p
            for key in _TEMPLATE_KEYS:
                q = q.replace("{" + key + "}", "*")
            return [q]
        return [p]

    def _candidates(self) -> list[str]:
        found: list[str] = []
        for pat in self._patterns():
            if _GLOB_CHARS & set(pat):
                for m in glob.glob(pat):
                    if os.path.isdir(m):
                        found.extend(os.path.join(m, f) for f in os.listdir(m))
                    else:
                        found.append(m)
            elif os.path.isdir(pat):
                found.extend(os.path.join(pat, f) for f in os.listdir(pat))
            elif os.path.isfile(pat):
                found.append(pat)
        return found

    def files(self) -> dict[str, list[tuple[datetime, str]]]:
        """Return ``{variable: [(analysis_datetime, filepath), ...]}`` sorted by date.

        Raises
        ------
        NoDiagFilesError
            If no file matches for any of the requested variables.
        """
        candidates = self._candidates()
        grid = None
        if self.freq and self._start is not None:
            grid = self._start
            step = pd.Timedelta(self.freq)
        result: dict[str, list[tuple[datetime, str]]] = {v: [] for v in self.variables}
        for fp in sorted(set(candidates)):
            m = _FILE_RE.match(os.path.basename(fp))
            if not m or m["var"] not in self.variables or m["stage"] != self.stage:
                continue
            when = datetime.strptime(m["date"], "%Y%m%d%H")
            if self._start is not None and when < self._start:
                continue
            if self._end is not None and when > self._end:
                continue
            if grid is not None and (pd.Timestamp(when) - pd.Timestamp(grid)) % step != pd.Timedelta(0):
                continue
            result[m["var"]].append((when, fp))
        for v in result:
            result[v].sort()
        if not any(result.values()):
            raise NoDiagFilesError(
                f"No diag files found for variables={self.variables}, stage={self.stage!r}, "
                f"path={self.path!r}, dates=({self._start}, {self._end})"
            )
        for v, items in result.items():
            if not items:
                logger.warning("No files found for variable %r", v)
            else:
                logger.info("Found %d file(s) for variable %r", len(items), v)
        return result

    # ---- reading --------------------------------------------------------
    @staticmethod
    def _read_file(fp: str, var: str, when: datetime) -> pd.DataFrame:
        tmpdir = None
        try:
            open_path = fp
            if fp.endswith(".gz"):
                tmpdir = tempfile.mkdtemp(prefix="tools4obsproc_")
                open_path = os.path.join(tmpdir, "diag.nc4")
                with gzip.open(fp, "rb") as src, open(open_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            wanted = dict(_COMMON_FIELDS)
            if var == "uv":
                wanted.update(_UV_FIELDS)
            data = {}
            with xr.open_dataset(open_path, decode_times=False) as ds:
                for src_name, out_name in wanted.items():
                    if src_name in ds.variables:
                        data[out_name] = _char_to_str(ds[src_name].values)
            df = pd.DataFrame(data)
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
        for col in _STRING_COLUMNS:
            if col in df.columns:
                df[col] = _decode(df[col])
        df.insert(0, "date", pd.Timestamp(when))
        return df

    def read(self) -> dict[str, pd.DataFrame]:
        """Read and concatenate files per variable.

        Returns ``{variable: DataFrame}`` with a ``date`` column holding the analysis time.
        Variables without matching files are omitted.
        """
        out: dict[str, pd.DataFrame] = {}
        for var, items in self.files().items():
            if not items:
                continue
            frames = []
            for when, fp in items:
                logger.debug("Reading %s", fp)
                frames.append(self._read_file(fp, var, when))
            out[var] = pd.concat(frames, ignore_index=True)
        return out


def read_diag(
    path: Union[str, "os.PathLike[str]"],
    variables: Union[str, Iterable[str]] = VALID_VARIABLES,
    dates: DatesSpec = None,
    stage: str = "ges",
    *,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    freq: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Read diag files; see :class:`DiagReader` for parameters."""
    return DiagReader(path, variables, dates, stage, start=start, end=end, freq=freq).read()
