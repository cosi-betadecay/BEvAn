import math

import pytest
import torch
from fake_megalib import MPhysicalEvent, PhysicalEvent, direction_to_pixel, write_tra_file

from physics.compton_cone_reconstruction import FarFieldImager


@pytest.fixture()
def imager(geo_path) -> FarFieldImager:
    return FarFieldImager(geometry_file=geo_path, n_phi=8, n_theta=4)


def compton(event_id: int, direction: tuple[float, float, float]) -> PhysicalEvent:
    return PhysicalEvent(event_id, MPhysicalEvent.c_Compton, direction)


def test_init_rejects_missing_geometry_and_bad_coordinates(tmp_path, geo_path):
    with pytest.raises(RuntimeError, match="geometry"):
        FarFieldImager(geometry_file=tmp_path / "missing.geo.setup")
    with pytest.raises(ValueError, match="coordinate_system"):
        FarFieldImager(geometry_file=geo_path, coordinate_system="cubic")


def test_peak_direction_bin_centers(imager):
    imager.accumulated[2 * 8 + 5] = 1.0  # theta bin 2, phi bin 5
    theta, phi = imager.peak_direction()
    assert theta == pytest.approx(2.5 * math.pi / 4)
    assert phi == pytest.approx(5.5 * 2.0 * math.pi / 8)

    unit = imager.peak_direction_cartesian()
    expected = torch.tensor(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)],
        dtype=torch.float64,
    )
    assert torch.allclose(unit, expected)
    assert unit.norm().item() == pytest.approx(1.0)


def test_image_2d_layout_matches_peak_indexing(imager):
    imager.accumulated[2 * 8 + 5] = 3.0
    image = imager.image_2d()
    assert image.shape == (4, 8)
    assert image[2, 5].item() == pytest.approx(3.0)
    assert image.sum().item() == pytest.approx(3.0)


def test_project_event_compton_only(imager):
    event = compton(1, (0.0, 0.0, 1.0))
    projected = imager.project_event(event)
    assert projected is not None
    bins, values = projected
    assert bins.tolist() == [direction_to_pixel(imager.bp, event.direction)]
    assert values.tolist() == [1.0]

    photo = PhysicalEvent(2, MPhysicalEvent.c_Photo)
    assert imager.project_event(photo) is None


def test_project_event_scratch_buffers_are_reused(imager):
    _, values = imager.project_event(compton(1, (0.0, 0.0, 1.0)))
    first = values.clone()
    imager.project_event(compton(2, (1.0, 0.0, 0.0)))
    assert torch.equal(values, first)  # toy weight is 1.0 either way; the view stayed alive


def test_backproject_event_accumulates(imager):
    event = compton(1, (1.0, 0.0, 0.0))
    assert imager.backproject_event(event)
    assert imager.backproject_event(event)
    assert not imager.backproject_event(PhysicalEvent(2, MPhysicalEvent.c_Photo))
    pixel = direction_to_pixel(imager.bp, event.direction)
    assert imager.accumulated[pixel].item() == pytest.approx(2.0)
    assert imager.accumulated.sum().item() == pytest.approx(2.0)


def test_backproject_file_counts_compton_events(tmp_path, imager):
    tra = write_tra_file(
        tmp_path / "synthetic.tra",
        [
            compton(1, (0.0, 0.0, 1.0)),
            PhysicalEvent(2, MPhysicalEvent.c_Photo),
            compton(3, (1.0, 0.0, 0.0)),
            compton(4, (0.0, 1.0, 0.0)),
        ],
    )
    assert imager.backproject_file(tra) == 3
    assert imager.accumulated.sum().item() == pytest.approx(3.0)


def test_backproject_file_max_events_cap(tmp_path, imager):
    tra = write_tra_file(tmp_path / "capped.tra", [compton(i, (0.0, 0.0, 1.0)) for i in range(1, 6)])
    assert imager.backproject_file(tra, max_events=2) == 2


def test_backproject_file_missing_raises(tmp_path, imager):
    with pytest.raises(RuntimeError, match="Failed to open"):
        imager.backproject_file(tmp_path / "missing.tra")


def test_backproject_fixture_tra(tra_path, imager):
    # All 1851 Compton events of the fixture succeed under the toy backprojection.
    assert imager.backproject_file(tra_path) == 1851
    assert imager.accumulated.sum().item() == pytest.approx(1851.0)


def test_peak_recovers_injected_source_direction(geo_path, tmp_path):
    imager = FarFieldImager(geometry_file=geo_path, n_phi=36, n_theta=18)
    direction = (1.0, 0.0, 0.0)
    tra = write_tra_file(tmp_path / "source.tra", [compton(i, direction) for i in range(1, 21)])
    assert imager.backproject_file(tra) == 20
    recovered = imager.peak_direction_cartesian()
    dot = float(recovered @ torch.tensor(direction, dtype=torch.float64))
    assert dot > 0.98  # within one 10-degree sky bin


def test_backproject_file_grouped(tmp_path, imager):
    tra = write_tra_file(
        tmp_path / "grouped.tra",
        [
            compton(1, (1.0, 0.0, 0.0)),
            compton(2, (0.0, 1.0, 0.0)),
            PhysicalEvent(3, MPhysicalEvent.c_Photo),
            compton(1, (0.0, 0.0, 1.0)),  # id reset -> chunk 1
        ],
    )
    groups = {"a": {(0, 1)}, "b": {(1, 1)}, "unmatched": set()}
    images, counts = imager.backproject_file_grouped(tra, groups)

    assert counts == {"a": 1, "b": 1, "unmatched": 0}
    assert images["a"].shape == (4, 8)
    assert images["a"].sum().item() == pytest.approx(1.0)
    assert images["b"].sum().item() == pytest.approx(1.0)
    assert images["unmatched"].sum().item() == 0.0
    pixel_b = direction_to_pixel(imager.bp, (0.0, 0.0, 1.0))
    assert images["b"].flatten()[pixel_b].item() == pytest.approx(1.0)
    # The instance accumulator is untouched by grouped projection.
    assert imager.accumulated.sum().item() == 0.0


def test_backproject_file_grouped_broken_join_raises(tmp_path, imager):
    tra = write_tra_file(tmp_path / "nomatch.tra", [compton(1, (0.0, 0.0, 1.0))])
    with pytest.raises(RuntimeError, match="ID join"):
        imager.backproject_file_grouped(tra, {"a": {(9, 9)}})
