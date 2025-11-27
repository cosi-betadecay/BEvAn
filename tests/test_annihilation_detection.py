import importlib
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class FakeInteraction:
    def __init__(self, process: str, origin_id: int):
        self._process = process
        self._origin_id = origin_id

    def GetProcess(self):
        return self._process

    def GetOriginID(self):
        return self._origin_id


class FakeHit:
    def __init__(self, origin_id: int, energy: float, position=(0.0, 0.0, 0.0)):
        self._origin_id = origin_id
        self._energy = energy
        self._position = position

    def IsOrigin(self, origin_id: int) -> bool:
        return self._origin_id == origin_id

    def GetEnergy(self) -> float:
        return self._energy

    def GetPosition(self):
        return self._position


class FakeEvent:
    def __init__(self, interactions=None, hits=None):
        self._interactions = interactions or []
        self._hits = hits or []

    def GetNIAs(self):
        return len(self._interactions)

    def GetIAAt(self, index: int):
        return self._interactions[index]

    def GetNHTs(self):
        return len(self._hits)

    def GetHTAt(self, index: int):
        return self._hits[index]


@pytest.fixture
def annihilation_detection(monkeypatch):
    fake_root = types.SimpleNamespace()

    class MString(str):
        pass

    fake_root.MString = MString
    fake_root.SetOwnership = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "ROOT", fake_root)

    module_name = "beta_decay.physics.annihilation_detection"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    return importlib.reload(module)


def test_process_counts_annihilation_events_with_matching_energy(annihilation_detection):
    interactions = [
        FakeInteraction("ANNI", 0),
        FakeInteraction("COMP", 1),
    ]
    hits = [
        FakeHit(origin_id=2, energy=250.0),
        FakeHit(origin_id=2, energy=261.0),
        FakeHit(origin_id=99, energy=120.0),
    ]
    event = FakeEvent(interactions=interactions, hits=hits)

    count = annihilation_detection.process(event, ref_energy=511.0, tolerance=1.5)

    assert count == 1


@pytest.mark.parametrize(
    ("energies", "expected"),
    [
        ([200.0, 311.0], True),
        ([], False),
        ([100.0, 120.0, 130.0], False),
    ],
)
def test_detected_511_event_uses_energy_combinations(annihilation_detection, energies, expected):
    hits = [FakeHit(origin_id=i + 1, energy=e) for i, e in enumerate(energies)]
    event = FakeEvent(hits=hits)

    detected = annihilation_detection.detected_511_event(
        ref_energy=511.0, Event=event, tolerance=0.5
    )

    assert detected is expected


def test_annihilation_extractor_computes_metrics_and_logs(monkeypatch, annihilation_detection):
    events = [
        types.SimpleNamespace(process_result=1, detection_result=True),
        types.SimpleNamespace(process_result=1, detection_result=False),
        types.SimpleNamespace(process_result=0, detection_result=True),
        types.SimpleNamespace(process_result=0, detection_result=False),
    ]

    class FakeReader:
        def __init__(self, queued_events):
            self._events = iter(queued_events)

        def GetNextEvent(self):
            return next(self._events, None)

    monkeypatch.setattr(
        annihilation_detection, "get_reader", lambda *_args, **_kwargs: FakeReader(events)
    )
    monkeypatch.setattr(
        annihilation_detection, "process", lambda event, *_args: event.process_result
    )
    monkeypatch.setattr(
        annihilation_detection,
        "detected_511_event",
        lambda ref_energy, event, tolerance: event.detection_result,
    )

    logged = {}

    class FakeWandb:
        def log(self, data):
            logged.update(data)

    monkeypatch.setattr(annihilation_detection, "wandb", FakeWandb())

    metrics = annihilation_detection.annihilation_extractor(
        geometry_file="geometry.xml", sim_file="simulation.sim", ref_energy=511.0, tolerance=1.0
    )

    assert metrics == (1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5)
    assert logged == {
        "TP": 1,
        "FP": 1,
        "FN": 1,
        "TN": 1,
        "precision": 0.5,
        "recall": 0.5,
        "false_positive_rate": 0.5,
        "f1_score": 0.5,
    }
