import gzip
import shutil
from datetime import date, datetime

import pytest

from tools4obsproc import DiagReader, NoDiagFilesError, read_diag, uv_components
from conftest import N, write_diag


def test_all(diagdir):
    out = read_diag(str(diagdir / "*"), dates="all")
    assert set(out) == {"t", "uv", "q"}
    assert len(out["t"]) == 4 * N
    assert out["t"]["date"].nunique() == 4


def test_decode_and_fields(diagdir):
    df = read_diag(diagdir / "20200101", "t", "2020010100")["t"]
    assert list(df["station_id"]) == ["AB123", "KXYZ", "7", "SHIP"]
    assert df["obs_class"].iloc[0] == "t"
    for c in ("lat", "lon", "pressure", "height", "time", "obs", "omf", "errinv_final",
              "qc_flag", "obs_type"):
        assert c in df.columns


def test_single_date_variants(diagdir):
    p = str(diagdir / "*")
    assert len(read_diag(p, "t", "2020010106")["t"]) == N
    assert len(read_diag(p, "t", datetime(2020, 1, 1, 6))["t"]) == N
    assert len(read_diag(p, "t", date(2020, 1, 2))["t"]) == 2 * N
    assert len(read_diag(p, "t", "2020-01-02T06:00")["t"]) == N


def test_range_and_kwargs_and_freq(diagdir):
    p = str(diagdir / "*")
    assert len(read_diag(p, "t", ("20200101", "20200101"))["t"]) == 2 * N
    assert len(read_diag(p, "t", start="2020010106", end="2020010200")["t"]) == 2 * N
    assert len(read_diag(p, "t", start="20200101")["t"]) == 4 * N
    out = read_diag(p, "t", ("2020010100", "2020010206"), freq="12h")["t"]
    assert out["date"].nunique() == 2


def test_template(diagdir):
    tpl = str(diagdir / "{YYYYMMDD}") + "/"
    assert len(read_diag(tpl, "q", ("20200101", "20200102"))["q"]) == 4 * N
    assert len(read_diag(tpl, "q", "all")["q"]) == 4 * N
    assert len(read_diag(tpl, "q", "2020010200")["q"]) == N


def test_uv_and_stage(diagdir):
    df = read_diag(diagdir / "20200101", "uv", "2020010100")["uv"]
    uv = uv_components(df)
    assert {"u", "v"} <= set(uv.columns)
    anl = read_diag(diagdir / "20200101", "t", stage="anl")["t"]
    assert len(anl) == N
    with pytest.raises(KeyError):
        uv_components(read_diag(diagdir / "20200101", "t")["t"])


def test_gzip(tmp_path):
    src = tmp_path / "x.nc"
    write_diag(src, "t")
    with open(src, "rb") as f, gzip.open(tmp_path / "diag_conv_t_ges.2020010100.nc4.gz", "wb") as g:
        shutil.copyfileobj(f, g)
    assert len(read_diag(tmp_path, "t")["t"]) == N


def test_files_listing(diagdir):
    r = DiagReader(diagdir / "20200101", ["t", "uv"], "all")
    f = r.files()
    assert [d for d, _ in f["t"]] == [datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 6)]


def test_errors(diagdir):
    with pytest.raises(NoDiagFilesError):
        read_diag(diagdir, "t", "2030010100")
    with pytest.raises(NoDiagFilesError):
        read_diag(diagdir / "nothing", "t")
    with pytest.raises(ValueError):
        read_diag(diagdir, "x")
    with pytest.raises(ValueError):
        read_diag(diagdir, "t", stage="foo")
    with pytest.raises(ValueError):
        read_diag(diagdir, "t", ("20200102", "20200101"))
    with pytest.raises(ValueError):
        read_diag(diagdir, "t", "notadate")
