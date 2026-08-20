"""
lexmex_tool.py — LEX-MEX v1.1
═══════════════════════════════════════════════════════════════════════
Tool de CrewAI para consultar LEX-MEX (asesor jurídico de leyes
federales mexicanas) desde cualquier Agent de un Crew.

Requiere una API key real de LEX-MEX (plan VIP o pay-as-you-go),
obtenida por un humano en https://lex-mex.xyz tras registrarse
(Google OAuth). Esta tool NO emite ni gestiona la key, solo la usa.

Uso:
    from crewai_tools.tools.lexmex_tool.lexmex_tool import LexMexTool
    from crewai import Agent

    abogado = Agent(
        role="Asesor legal",
        goal="Responder dudas de derecho federal mexicano con cita exacta",
        backstory="Experto en legislación federal mexicana.",
        tools=[LexMexTool(api_key="lmx_live_...")],
    )

    # o vía variable de entorno LEXMEX_API_KEY
    import os
    os.environ["LEXMEX_API_KEY"] = "lmx_live_..."
    tool = LexMexTool()

─── Changelog v1.1 ────────────────────────────────────────────────────
- Seguridad: api_key ahora se excluye de la serialización del modelo
  (Field(exclude=True, repr=False)) para que nunca quede persistida en
  logs ni en estados de Crew guardados.
- Seguridad: se eliminó el campo configurable `api_base`; el host
  autenticado ahora es siempre la constante fija LEXMEX_API_BASE, para
  que ningún caller pueda redirigir la API key a un host arbitrario
  (hallazgo de CodeRabbit: API-key exfiltration risk).
- Se agregó el header X-LexMex-Client para que el backend de LexMex
  pueda distinguir tráfico proveniente de esta tool.
"""

from __future__ import annotations

import os
from typing import Optional, Type

import requests
from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "LexMexTool requiere crewai. Instala con: pip install crewai"
    ) from e


LEXMEX_API_BASE = "https://lex-mex.xyz"
LEXMEX_CLIENT_ID = "crewai-lexmex-tool/1.1.0"


class LexMexInput(BaseModel):
    pregunta: str = Field(
        ...,
        description=(
            "Consulta jurídica en español sobre derecho federal mexicano "
            "(316 leyes federales indexadas, sincronizadas con "
            "diputados.gob.mx). Ej: '¿Cuáles son las causales de despido "
            "justificado según la LFT?'"
        ),
        min_length=1,
        max_length=2000,
    )


class LexMexTool(BaseTool):
    """Consulta leyes federales mexicanas con respuestas citadas al DOF.

    Motor RAG+LLM real de lex-mex.xyz — no alucina artículos, cita la
    ley y el artículo exacto de donde saca cada respuesta. Requiere
    API key de plan VIP o de créditos pay-as-you-go.
    """

    name: str = "Consulta legal LEX-MEX"
    description: str = (
        "Útil para responder preguntas sobre leyes federales mexicanas "
        "(laboral, civil, fiscal, penal, mercantil, etc.). Devuelve la "
        "respuesta junto con las fuentes legales citadas (ley y "
        "artículo) y un nivel de confianza. Input: una pregunta legal "
        "en español, texto plano."
    )
    args_schema: Type[BaseModel] = LexMexInput

    # Se excluye de la serialización (model_dump/repr) para que la key
    # nunca quede persistida en logs, estados de Crew guardados, etc.
    api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    timeout: int = Field(default=30, gt=0, description="Request timeout in seconds.")

    def _resolved_key(self) -> str:
        key = self.api_key or os.getenv("LEXMEX_API_KEY")
        if not key:
            raise ValueError(
                "Falta la API key de LEX-MEX. Pásala como LexMexTool(api_key=...) "
                "o define la variable de entorno LEXMEX_API_KEY. Genera una en "
                "https://lex-mex.xyz tras registrarte (plan VIP o pay-as-you-go)."
            )
        return key

    def _run(self, pregunta: str) -> str:
        resp = requests.post(
            f"{LEXMEX_API_BASE}/api/v1/consulta",
            json={"pregunta": pregunta},
            headers={
                "X-API-Key": self._resolved_key(),
                "X-LexMex-Client": LEXMEX_CLIENT_ID,
            },
            timeout=self.timeout,
        )

        if resp.status_code == 401:
            return "Error: API key de LEX-MEX inválida o revocada."
        if resp.status_code == 402:
            return "Error: sin saldo/plan suficiente en LEX-MEX para esta consulta."
        if resp.status_code == 429:
            return "Error: límite diario de consultas de LEX-MEX alcanzado."
        resp.raise_for_status()
        data = resp.json()

        fuentes = data.get("fuentes") or []
        fuentes_txt = "; ".join(
            f.get("cita", str(f)) if isinstance(f, dict) else str(f)
            for f in fuentes
        ) or "sin fuentes citadas"

        return (
            f"{data.get('respuesta', '')}\n\n"
            f"Fuentes: {fuentes_txt}\n"
            f"Confianza: {data.get('confianza', 'n/d')}"
        )
