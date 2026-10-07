import gzip
import shutil

import netCDF4
import numpy as np
import pytest

N = 4


def _chars(text, width=8):
    return [bytes([c]) for c in text.ljust(width).encode()]


def write_diag(path, var, seed=0):
    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("nobs", N)
        nc.createDimension("Station_ID_maxstrlen", 8)
        sid = nc.createVariable("Station_ID", "S1", ("nobs", "Station_ID_maxstrlen"))
        ids = np.array([_chars(s) for s in ["AB123", "KXYZ", "7", "SHIP"]], dtype="S1")
        sid[:] = ids
        cls = nc.createVariable("Observation_Class", "S1", ("nobs", "Station_ID_maxstrlen"))
        cls[:] = np.array([_chars("  " + var) for _ in range(N)], dtype="S1")
        for name, vals in {
            "Observation_Type": [120, 120, 180, 180],
            "Prep_QC_Mark": [2, 2, 3, 9],
            "Analysis_Use_Flag": [1, 1, -1, 1],
        }.items():
            nc.createVariable(name, "i4", ("nobs",))[:] = vals
        for name in ["Latitude", "Longitude", "Pressure", "Height", "Time",
                     "Errinv_Final", "Observation", "Obs_Minus_Forecast_adjusted"]:
            nc.createVariable(name, "f4", ("nobs",))[:] = rng.random(N)
        if var == "uv":
            for name in ["u_Observation", "v_Observation",
                         "u_Obs_Minus_Forecast_adjusted", "v_Obs_Minus_Forecast_adjusted"]:
                nc.createVariable(name, "f4", ("nobs",))[:] = rng.random(N)


@pytest.fixture
def diagdir(tmp_path):
    for day in ("20200101", "20200102"):
        d = tmp_path / day
        d.mkdir()
        for hh in ("00", "06"):
            for var in ("t", "uv", "q"):
                write_diag(d / f"diag_conv_{var}_ges.{day}{hh}.nc4", var)
    write_diag(tmp_path / "20200101" / "diag_conv_t_anl.2020010100.nc4", "t")
    return tmp_path
