"""CrewAI CLI API client extensions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import quote, urljoin

from crewai_core.plus_api import PlusAPI as _CorePlusAPI
import httpx


HttpMethod = Literal["GET", "POST", "PATCH", "DELETE"]


class PlusAPI(_CorePlusAPI):
    """CLI API client.

    The ZIP deployment and integration-token methods live here as well as in
    newer crewai-core versions so editable CLI installs still work when an
    older crewai-core is present in the runtime environment. The evaluation
    methods live here because only the CLI calls them.
    """

    EVALUATIONS_RESOURCE = f"{_CorePlusAPI.TRACING_RESOURCE}/evaluations"
    # AMP reads the run's spans from Wharf inside the POST; a large run takes a while.
    EVALUATION_START_TIMEOUT = 120.0
    EVALUATION_POLL_TIMEOUT = 30.0
    # The export carries every setup's outputs, checks and traces — tens of MB
    # for a large comparison — and AMP gathers them before it answers.
    EVALUATION_EXPORT_TIMEOUT = httpx.Timeout(EVALUATION_POLL_TIMEOUT, read=120.0)
    CLI_INTEGRATION_TOKEN_RESOURCE = "/crewai_plus/api/v1/me/integration-token"  # noqa: S105 - API path, not a secret
    CREW_RUN_RESOURCE = "/crewai_plus/api/v1/crew"
    # AMP gives a sleeping deployment 60 s to wake before it answers.
    CREW_RUN_TIMEOUT = httpx.Timeout(5.0, read=90.0)
    # A status read answers from the runtime's store; keep it short so one hung
    # read cannot carry `crewai deploy test` far past its --timeout.
    CREW_STATUS_TIMEOUT = httpx.Timeout(5.0, read=30.0)

    def _crew_run_path(self, uuid: str | None, name: str | None) -> str:
        if uuid:
            return f"{self.CREW_RUN_RESOURCE}/{quote(uuid, safe='')}"
        if name:
            return f"{self.CREW_RUN_RESOURCE}/by-name/{quote(name, safe='')}"
        raise ValueError("Pass the deployment's uuid or name")

    def kickoff_crew(
        self,
        inputs: dict[str, Any],
        *,
        uuid: str | None = None,
        name: str | None = None,
    ) -> httpx.Response:
        """Start one run of the deployment UUID (or the one named NAME) with INPUTS.

        AMP answers `{"kickoff_id": ...}`; the run counts as an execution.
        """
        return self._make_request(
            "POST",
            f"{self._crew_run_path(uuid, name)}/kickoff",
            json={"inputs": inputs},
            timeout=self.CREW_RUN_TIMEOUT,
        )

    def crew_kickoff_status(
        self,
        kickoff_id: str,
        *,
        uuid: str | None = None,
        name: str | None = None,
    ) -> httpx.Response:
        """The runtime's status of run KICKOFF_ID: `{state, status, result}`."""
        run = quote(kickoff_id, safe="")
        return self._make_request(
            "GET",
            f"{self._crew_run_path(uuid, name)}/tasks/{run}/status",
            timeout=self.CREW_STATUS_TIMEOUT,
        )

    def get_integration_token(self) -> str:
        """Return the Enterprise Action Auth Token for the active organization."""
        response = self._make_request(
            "GET", self.CLI_INTEGRATION_TOKEN_RESOURCE, timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("api_key") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise RuntimeError("AMP did not return an integration token")
        return token

    def create_evaluation(
        self,
        execution_id: str,
        *,
        eval_config: str | None = None,
        project_id: str | None = None,
    ) -> httpx.Response:
        """Ask AMP to evaluate the traced run EXECUTION_ID (crewai eval).

        EVAL_CONFIG is the project's own `eval.jsonc` when it has one: what
        good means for this crew, in its own words. Sent as it was written,
        comments and all, and read by the grader rather than here. PROJECT_ID
        is `[tool.crewai].project_id`, so the evaluation is filed under the
        project it is about.
        """
        body: dict[str, str] = {"execution_id": execution_id}
        if eval_config:
            body["eval_config"] = eval_config
        if project_id:
            body["project_id"] = project_id
        return self._make_request(
            "POST",
            self.EVALUATIONS_RESOURCE,
            json=body,
            timeout=self.EVALUATION_START_TIMEOUT,
        )

    def create_models_evaluation(
        self,
        models: list[str],
        *,
        project_id: str,
        eval_config: str | None = None,
        deployment_id: str | None = None,
    ) -> httpx.Response:
        """Ask AMP to run the project's deployment once per model and compare them.

        AMP finds the deployment by PROJECT_ID (`[tool.crewai].project_id`)
        unless DEPLOYMENT_ID names one. MODELS are `provider/model` strings;
        the deployment's own models are the baseline, so they are not sent.
        EVAL_CONFIG is the project's `eval.jsonc`, as `create_evaluation` sends it.
        """
        body: dict[str, Any] = {
            "kind": "models",
            "project_id": project_id,
            "models": list(models),
        }
        if eval_config:
            body["eval_config"] = eval_config
        if deployment_id:
            body["deployment_id"] = deployment_id
        return self._make_request(
            "POST",
            self.EVALUATIONS_RESOURCE,
            json=body,
            timeout=self.EVALUATION_START_TIMEOUT,
        )

    def get_evaluation(self, evaluation_id: str) -> httpx.Response:
        """The evaluation's status and, once done, its verdict.

        `can_export=1` says this CLI has `crewai eval export`, so the brief may
        point at it. AMP passes it on to the evaluation service; an AMP from
        before exports drops it, so the brief names the command only when the
        CLI, AMP and the service all serve it.
        """
        return self._make_request(
            "GET",
            f"{self.EVALUATIONS_RESOURCE}/{evaluation_id}",
            params={"can_export": "1"},
            timeout=self.EVALUATION_POLL_TIMEOUT,
        )

    def export_evaluation(
        self, evaluation_id: str, *, timeout: httpx.Timeout | None = None
    ) -> httpx.Response:
        """The evaluation's full data as files (crewai eval export):
        `{"id", "kind", "files": {relative path: text}}`. TIMEOUT replaces the
        default, which allows the two minutes a large comparison may take."""
        return self._make_request(
            "GET",
            f"{self.EVALUATIONS_RESOURCE}/{evaluation_id}/export",
            timeout=timeout or self.EVALUATION_EXPORT_TIMEOUT,
        )

    def _make_multipart_request(
        self,
        method: HttpMethod,
        endpoint: str,
        *,
        zip_file_path: str | Path,
        data: dict[str, str] | None = None,
        timeout: float | None = None,
        verify: bool = True,
    ) -> httpx.Response:
        """Send an authenticated multipart request containing a project ZIP."""
        url = urljoin(self.base_url, endpoint)
        headers = dict(cast(dict[str, str], self.headers))
        headers.pop("Content-Type", None)
        path = Path(zip_file_path)
        request_kwargs: dict[str, Any] = {"headers": headers}
        if data is not None:
            request_kwargs["data"] = data
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        # An older crewai-core has no _http_client; its clients ignore proxies.
        http_client = getattr(self, "_http_client", None)

        with (
            path.open("rb") as file_handle,
            (
                http_client(verify)
                if http_client
                else httpx.Client(verify=verify, trust_env=False)
            ) as client,
        ):
            files = {
                "zip_file": (path.name, file_handle, "application/zip"),
            }
            return client.request(method, url, files=files, **request_kwargs)

    def create_crew_from_zip(
        self,
        zip_file_path: str | Path,
        *,
        name: str | None = None,
        env: dict[str, str] | None = None,
        project_id: str | None = None,
    ) -> httpx.Response:
        """Create a crew deployment from a local project ZIP archive.

        PROJECT_ID is `[tool.crewai].project_id`, so AMP knows which project the
        deployment runs; sent only when the project has one.
        """
        data: dict[str, str] = {}
        if name:
            data["name"] = name
        if project_id:
            data["project_id"] = project_id
        if env:
            data.update({f"env[{key}]": value for key, value in env.items()})
        return self._make_multipart_request(
            "POST",
            f"{self.CREWS_RESOURCE}/zip",
            zip_file_path=zip_file_path,
            data=data or None,
            timeout=300,
        )

    def update_crew_from_zip(
        self,
        uuid: str,
        zip_file_path: str | Path,
        *,
        env: dict[str, str] | None = None,
        project_id: str | None = None,
        force: bool = False,
    ) -> httpx.Response:
        """Update an existing crew deployment from a local project ZIP archive.

        PROJECT_ID as for `create_crew_from_zip`; FORCE redeploys even while a
        models comparison runs on the deployment (AMP answers 409
        ``comparison_running`` otherwise; an older AMP ignores it).
        """
        data: dict[str, str] = {}
        if project_id:
            data["project_id"] = project_id
        if force:
            data["force"] = "true"
        if env:
            data.update({f"env[{key}]": value for key, value in env.items()})
        return self._make_multipart_request(
            "POST",
            f"{self.CREWS_RESOURCE}/{uuid}/zip_update",
            zip_file_path=zip_file_path,
            data=data or None,
            timeout=300,
        )


__all__ = ["PlusAPI"]
