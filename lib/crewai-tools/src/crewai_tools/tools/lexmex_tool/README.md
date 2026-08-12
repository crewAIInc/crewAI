# LexMexTool

Consulta [LEX-MEX](https://lex-mex.xyz) — asesor jurídico con IA sobre
las 316 leyes federales mexicanas vigentes, sincronizadas con
`diputados.gob.mx`. Cada respuesta cita la ley y el artículo exacto
(sin alucinar fuentes).

## Instalación

Esta tool solo necesita `requests` y `pydantic`, ambos ya son
dependencias de `crewai-tools`. No requiere paquete extra.

## Variables de entorno

| Variable | Requerida | Descripción |
| --- | --- | --- |
| `LEXMEX_API_KEY` | Sí (o pásala como `api_key=` al instanciar) | API key de LEX-MEX. Se genera en `POST /api/v1/keys` tras registrarte en https://lex-mex.xyz (Google OAuth). Plan VIP ilimitado o pay-as-you-go por créditos. |

## Uso

```python
from crewai import Agent
from crewai_tools import LexMexTool

abogado = Agent(
    role="Asesor legal",
    goal="Responder dudas de derecho federal mexicano con cita exacta",
    backstory="Experto en legislación federal mexicana.",
    tools=[LexMexTool()],  # toma LEXMEX_API_KEY del entorno
)
```

## Errores comunes

- `401` → API key inválida o revocada.
- `402` → sin saldo/plan suficiente para la consulta.
- `429` → límite diario de consultas alcanzado.
