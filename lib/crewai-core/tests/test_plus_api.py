from unittest.mock import MagicMock

from crewai_core.plus_api import PlusAPI
import pytest


def test_get_integration_token_returns_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.json.return_value = {"api_key": "integration-token"}
    api = PlusAPI("login-token", organization_id="organization-id")
    make_request = MagicMock(return_value=response)
    monkeypatch.setattr(api, "_make_request", make_request)

    assert api.get_integration_token() == "integration-token"

    make_request.assert_called_once_with(
        "GET", "/crewai_plus/api/v1/me/integration-token", timeout=30
    )
    response.raise_for_status.assert_called_once_with()


@pytest.mark.parametrize("payload", [{}, {"api_key": ""}, {"api_key": 1}, []])
def test_get_integration_token_rejects_missing_or_invalid_api_key(
    monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    response = MagicMock()
    response.json.return_value = payload
    api = PlusAPI("login-token", organization_id="organization-id")
    monkeypatch.setattr(api, "_make_request", MagicMock(return_value=response))

    with pytest.raises(RuntimeError, match="did not return an integration token"):
        api.get_integration_token()
