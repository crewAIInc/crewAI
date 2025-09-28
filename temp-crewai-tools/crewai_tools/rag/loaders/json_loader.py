import json

from crewai_tools.rag.source_content import SourceContent
from crewai_tools.rag.base_loader import BaseLoader, LoaderResult


class JSONLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs) -> LoaderResult:
        source_ref = source_content.source_ref
        content = source_content.source

        if source_content.is_url():
            content = self._load_from_url(source_ref, kwargs)
        elif source_content.path_exists():
            content = self._load_from_file(source_ref)

        return self._parse_json(content, source_ref)

    def _load_from_url(self, url: str, kwargs: dict) -> str:
        import requests

        headers = kwargs.get("headers", {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; crewai-tools JSONLoader)"
        })

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text if not self._is_json_response(response) else json.dumps(response.json(), indent=2)
        except Exception as e:
            raise ValueError(f"Error fetching JSON from URL {url}: {str(e)}")

    def _is_json_response(self, response) -> bool:
        try:
            response.json()
            return True
        except ValueError:
            return False

    def _load_from_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def _parse_json(self, content: str, source_ref: str) -> LoaderResult:
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                text = "\n".join(f"{k}: {json.dumps(v, indent=0)}" for k, v in data.items())
            elif isinstance(data, list):
                text = "\n".join(json.dumps(item, indent=0) for item in data)
            else:
                text = json.dumps(data, indent=0)

            metadata = {
                "format": "json",
                "type": type(data).__name__,
                "size": len(data) if isinstance(data, (list, dict)) else 1
            }
        except json.JSONDecodeError as e:
            text = content
            metadata = {"format": "json", "parse_error": str(e)}

        return LoaderResult(
            content=text,
            source=source_ref,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=source_ref, content=text)
        )
