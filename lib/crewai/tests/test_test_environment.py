"""The suite must never send telemetry, however its tests are ordered.

`.env.test` turns telemetry off once, at import. These two tests run on the same
worker (`--dist=loadfile`), so whichever runs second proves the setting survived
the first test and the first test's own change to it.
"""

import os

import pytest


@pytest.mark.parametrize("leaked_value", ["false", None])
def test_telemetry_is_off_in_every_test(leaked_value: str | None) -> None:
    assert os.environ.get("CREWAI_DISABLE_TELEMETRY") == "true"
    assert os.environ.get("OTEL_SDK_DISABLED") == "true"

    if leaked_value is None:
        os.environ.pop("CREWAI_DISABLE_TELEMETRY", None)
        os.environ.pop("OTEL_SDK_DISABLED", None)
    else:
        os.environ["CREWAI_DISABLE_TELEMETRY"] = leaked_value
        os.environ["OTEL_SDK_DISABLED"] = leaked_value
