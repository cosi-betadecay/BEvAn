import fake_megalib
import pytest

from utils import megalib_types


def protocol_methods(protocol: type) -> list[str]:
    """Public methods a protocol declares (the surface the pipeline relies on)."""
    return [name for name, value in vars(protocol).items() if not name.startswith("_") and callable(value)]


@pytest.mark.parametrize(
    ("protocol", "implementation"),
    [
        (megalib_types.MSimEvent, fake_megalib.SimEvent),
        (megalib_types.MFileEventsSim, fake_megalib.MFileEventsSim),
        (megalib_types.MPhysicalEvent, fake_megalib.MPhysicalEvent),
        (megalib_types.MDGeometryQuest, fake_megalib.MDGeometryQuest),
    ],
)
def test_fake_classes_cover_the_protocol_surface(protocol, implementation):
    for name in protocol_methods(protocol):
        assert callable(getattr(implementation, name, None)), f"{implementation.__name__} lacks {name}"


def test_protocols_declare_the_accessors_the_pipeline_calls():
    assert set(protocol_methods(megalib_types.MSimEvent)) == {"GetNIAs", "GetIAAt", "GetNHTs", "GetHTAt"}
    assert set(protocol_methods(megalib_types.MFileEventsSim)) == {"GetNextEvent", "Close"}
    assert set(protocol_methods(megalib_types.MPhysicalEvent)) == {"GetType", "GetId"}
