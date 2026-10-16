from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _quiet_stderr(request: pytest.FixtureRequest, capsys: pytest.CaptureFixture[str]) -> Generator[None]:
    """Fail any test that writes to stderr unless marked `expects_stderr`."""
    yield
    if request.node.get_closest_marker("expects_stderr") is None:
        assert capsys.readouterr().err == ""
