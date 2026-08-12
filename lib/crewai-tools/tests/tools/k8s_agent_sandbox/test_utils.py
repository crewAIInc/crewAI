from types import ModuleType
from unittest.mock import patch

import pytest

from crewai_tools.tools.k8s_agent_sandbox.utils import (
    lazy_import_k8s_agent_sandbox,
)


@pytest.fixture
def import_module():
    # Keep the module cache out of it, in both directions.
    with patch.dict(
        "crewai_tools.tools.k8s_agent_sandbox.utils._MODULE_CACHE", clear=True
    ):
        with patch(
            "crewai_tools.tools.k8s_agent_sandbox.utils.importlib.import_module"
        ) as import_module:
            yield import_module


def test_imported_module_is_cached(import_module):
    module = ModuleType("k8s_agent_sandbox.sandbox_client")
    import_module.return_value = module

    assert lazy_import_k8s_agent_sandbox("sandbox_client") is module
    assert lazy_import_k8s_agent_sandbox("sandbox_client") is module

    import_module.assert_called_once_with("k8s_agent_sandbox.sandbox_client")


def test_missing_sdk_is_reported_as_a_missing_extra(import_module):
    import_module.side_effect = ModuleNotFoundError(
        "No module named 'k8s_agent_sandbox'", name="k8s_agent_sandbox"
    )

    with pytest.raises(ImportError, match="k8s-agent-sandbox"):
        lazy_import_k8s_agent_sandbox("sandbox_client")


def test_missing_sdk_dependency_is_left_alone(import_module):
    import_module.side_effect = ModuleNotFoundError(
        "No module named 'httpx'", name="httpx"
    )

    with pytest.raises(ModuleNotFoundError, match="httpx"):
        lazy_import_k8s_agent_sandbox("sandbox_client")
