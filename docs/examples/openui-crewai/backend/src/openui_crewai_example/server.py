"""FastAPI entry point for the CrewAI + OpenUI example."""

from ag_ui_crewai.endpoint import add_crewai_flow_fastapi_endpoint
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openui_crewai_example.flow import OpenUIFlow

load_dotenv()

app = FastAPI(title="CrewAI + OpenUI")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a lightweight readiness signal."""

    return {"status": "ok"}


add_crewai_flow_fastapi_endpoint(
    app=app,
    flow=OpenUIFlow(),
    path="/openui",
)
