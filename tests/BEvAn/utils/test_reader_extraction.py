import pytest
from fake_megalib import iter_sim_events

from utils.reader_extraction import ChunkScopedIds, load_geometry, open_sim_reader


def test_chunk_scoped_ids_detect_resets():
    ids = ChunkScopedIds()
    keys = [ids.key(i) for i in (1, 2, 3, 1, 2, 2)]
    assert keys == [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 2)]


def test_chunk_scoped_ids_single_chunk_passthrough():
    ids = ChunkScopedIds()
    assert [ids.key(i) for i in (1, 5, 9)] == [(0, 1), (0, 5), (0, 9)]


def test_reader_stream_matches_direct_parse(geo_path, sim_path):
    geometry = load_geometry(str(geo_path))
    reader = open_sim_reader(geometry, str(sim_path))
    n = 0
    while reader.GetNextEvent() is not None:
        n += 1
    reader.Close()
    assert n == sum(1 for _ in iter_sim_events(sim_path))
    assert reader.GetNextEvent() is None  # closed reader stays at EOF


def test_load_geometry_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="geometry"):
        load_geometry(str(tmp_path / "missing.geo.setup"))


def test_open_sim_reader_missing_file_raises(geo_path, tmp_path):
    geometry = load_geometry(str(geo_path))
    with pytest.raises(RuntimeError, match="simulation"):
        open_sim_reader(geometry, str(tmp_path / "missing.sim"))
