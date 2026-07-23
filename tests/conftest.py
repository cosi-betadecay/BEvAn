from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fake_megalib import SimEvent

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

# Install the fake ROOT module before any pipeline module runs `import ROOT as M`,
# so the suite needs no MEGAlib/ROOT installation (e.g. in CI).
if "ROOT" not in sys.modules:
    import fake_megalib

    sys.modules["ROOT"] = fake_megalib


@pytest.fixture(scope="session")
def sim_path() -> Path:
    """Provide the path to the checked-in ``.sim`` reference fixture."""
    return TESTS_DIR / "test.sim"


@pytest.fixture(scope="session")
def tra_path() -> Path:
    """Provide the path to the checked-in ``.tra`` reference fixture."""
    return TESTS_DIR / "test.tra"


@pytest.fixture()
def geo_path(tmp_path: Path) -> Path:
    """A geometry file that exists (the fake only checks existence, never content)."""
    path = tmp_path / "dummy.geo.setup"
    path.write_text("// fake geometry for unit tests\n")
    return path


def synthetic_sim_events() -> list[SimEvent]:
    """Small event mix covering every (class, bucket) cell.

    Per bucket: β⁺ events carry an ANNI whose secondary (IA id 2) hits sum into
    the 511 keV window; background events have no ANNI. Hit counts 1/2/3 land in
    buckets 1/2/3. One trailing zero-hit event exercises the skip path.
    """
    from fake_megalib import SimEvent, SimHit, SimIA

    def beta_ias() -> list[SimIA]:
        return [SimIA("ANNI", 1, 0), SimIA("COMP", 2, 1)]

    events = []
    for i in range(3):
        events.append(
            SimEvent(
                len(events) + 1, hits=[SimHit((0.0, 0.0, 1.0), 509.5 + 0.4 * i, origins=(2,))], ias=beta_ias()
            )
        )
        events.append(SimEvent(len(events) + 1, hits=[SimHit((0.0, 0.0, 1.0), 300.0 + 10.0 * i)]))
    for i in range(6):
        events.append(
            SimEvent(
                len(events) + 1,
                hits=[
                    SimHit((0.0, 0.0, 0.0), 255.0, origins=(2,)),
                    SimHit((0.1 * i, 1.0, 0.0), 253.2 + 0.4 * i, origins=(2,)),
                ],
                ias=beta_ias(),
            )
        )
        events.append(
            SimEvent(
                len(events) + 1,
                hits=[SimHit((0.0, 0.0, 0.0), 350.0), SimHit((0.1 * i, 1.0, 0.0), 355.0 + 5.0 * i)],
            )
        )
    for i in range(6):
        events.append(
            SimEvent(
                len(events) + 1,
                hits=[
                    SimHit((0.0, 0.0, 0.0), 10.0),
                    SimHit((1.0, 0.0, 0.0), 511.0, origins=(2,)),
                    SimHit((-1.0, 0.0, 0.0), 513.0 + i),
                ],
                ias=beta_ias(),
            )
        )
        events.append(
            SimEvent(
                len(events) + 1,
                hits=[
                    SimHit((0.0, 0.0, 0.0), 10.0),
                    SimHit((1.0, 0.0, 0.0), 505.0 + i),
                    SimHit((-1.0, 0.0, 0.0), 514.0 + i),
                ],
            )
        )
    events.append(SimEvent(len(events) + 1))  # zero hits
    return events


@pytest.fixture()
def synthetic_dataset(tmp_path: Path, geo_path: Path) -> dict[str, str | int]:
    """Self-consistent tiny ``.sim``/``.tra``/prior files for entry-script tests."""
    from fake_megalib import MPhysicalEvent, PhysicalEvent, write_sim_file, write_tra_file

    events = synthetic_sim_events()
    tra_events = [PhysicalEvent(2, MPhysicalEvent.c_Photo)] + [
        PhysicalEvent(i, MPhysicalEvent.c_Compton, (0.0, 0.0, 1.0)) for i in (3, 7, 11, 15, 19, 23)
    ]
    return {
        "geo": str(geo_path),
        "sim": str(write_sim_file(tmp_path / "tiny.sim", events)),
        "tra": str(write_tra_file(tmp_path / "tiny.tra", tra_events)),
        "prior": str(write_sim_file(tmp_path / "tiny_P1.sim", events)),
        "n_nonempty": len(events) - 1,
    }
