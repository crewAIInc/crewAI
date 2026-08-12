import pytest

from crewai_tools.tools.lexmex_tool.lexmex_tool import LexMexTool


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("LEXMEX_API_KEY", raising=False)
    tool = LexMexTool()
    with pytest.raises(ValueError):
        tool._resolved_key()


def test_happy_path(monkeypatch):
    tool = LexMexTool(api_key="lmx_live_test")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "respuesta": "El despido justificado requiere...",
                "fuentes": [{"cita": "LFT, art. 47"}],
                "confianza": "alta",
            }

    monkeypatch.setattr(
        "crewai_tools.tools.lexmex_tool.lexmex_tool.requests.post",
        lambda *a, **kw: FakeResponse(),
    )

    resultado = tool.run(pregunta="¿Causales de despido justificado?")
    assert "despido justificado" in resultado
    assert "LFT, art. 47" in resultado
