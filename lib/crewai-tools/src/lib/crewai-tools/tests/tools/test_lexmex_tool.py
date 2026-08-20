import pytest

from crewai_tools.tools.lexmex_tool.lexmex_tool import LexMexTool


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("LEXMEX_API_KEY", raising=False)
    tool = LexMexTool()
    with pytest.raises(ValueError):
        tool._resolved_key()


def test_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("LEXMEX_API_KEY", "lmx_live_env")
    tool = LexMexTool()
    assert tool._resolved_key() == "lmx_live_env"


def test_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("LEXMEX_API_KEY", "lmx_live_env")
    tool = LexMexTool(api_key="lmx_live_explicit")
    assert tool._resolved_key() == "lmx_live_explicit"


def test_api_key_excluded_from_serialization():
    """CodeRabbit: la key nunca debe aparecer en model_dump/repr."""
    tool = LexMexTool(api_key="lmx_live_secret")
    dumped = tool.model_dump()
    assert "api_key" not in dumped
    assert "lmx_live_secret" not in repr(tool)


def test_happy_path(monkeypatch):
    """CodeRabbit: se verifica el contrato real de la llamada saliente
    (URL fija, header X-API-Key, header X-LexMex-Client), no solo el
    resultado final formateado."""
    tool = LexMexTool(api_key="lmx_live_test")
    captured = {}

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

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "crewai_tools.tools.lexmex_tool.lexmex_tool.requests.post",
        fake_post,
    )

    resultado = tool.run(pregunta="¿Causales de despido justificado?")

    assert captured["url"] == "https://lex-mex.xyz/api/v1/consulta"
    assert captured["json"] == {"pregunta": "¿Causales de despido justificado?"}
    assert captured["headers"]["X-API-Key"] == "lmx_live_test"
    assert captured["headers"]["X-LexMex-Client"] == "crewai-lexmex-tool/1.1.0"
    assert captured["timeout"] == 30

    assert "despido justificado" in resultado
    assert "LFT, art. 47" in resultado
    assert "alta" in resultado


def test_unauthorized(monkeypatch):
    tool = LexMexTool(api_key="lmx_live_bad")

    class FakeResponse:
        status_code = 401

    monkeypatch.setattr(
        "crewai_tools.tools.lexmex_tool.lexmex_tool.requests.post",
        lambda *a, **kw: FakeResponse(),
    )

    resultado = tool.run(pregunta="cualquier pregunta")
    assert "inválida" in resultado.lower()


def test_insufficient_balance(monkeypatch):
    tool = LexMexTool(api_key="lmx_live_test")

    class FakeResponse:
        status_code = 402

    monkeypatch.setattr(
        "crewai_tools.tools.lexmex_tool.lexmex_tool.requests.post",
        lambda *a, **kw: FakeResponse(),
    )

    resultado = tool.run(pregunta="cualquier pregunta")
    assert "saldo" in resultado.lower()


def test_rate_limited(monkeypatch):
    tool = LexMexTool(api_key="lmx_live_test")

    class FakeResponse:
        status_code = 429

    monkeypatch.setattr(
        "crewai_tools.tools.lexmex_tool.lexmex_tool.requests.post",
        lambda *a, **kw: FakeResponse(),
    )

    resultado = tool.run(pregunta="cualquier pregunta")
    assert "límite" in resultado.lower()
