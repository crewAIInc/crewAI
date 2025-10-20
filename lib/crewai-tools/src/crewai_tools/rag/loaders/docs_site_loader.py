"""Documentation site loader."""

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import requests

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class DocsSiteLoader(BaseLoader):
    """Loader for documentation websites."""

    def load(self, source: SourceContent, **kwargs) -> LoaderResult:  # type: ignore[override]
        """Load content from a documentation site.

        Args:
            source: Documentation site URL
            **kwargs: Additional arguments

        Returns:
            LoaderResult with documentation content
        """
        docs_url = source.source

        try:
            response = requests.get(docs_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(
                f"Unable to fetch documentation from {docs_url}: {e}"
            ) from e

        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else "Documentation"

        for selector in [
            "main",
            "article",
            '[role="main"]',
            ".content",
            "#content",
            ".documentation",
        ]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find("body")

        if not main_content:
            raise ValueError(
                f"Unable to extract content from documentation site: {docs_url}"
            )

        text_parts = [f"Title: {title_text}", ""]

        headings = main_content.find_all(["h1", "h2", "h3"])
        if headings:
            text_parts.append("Table of Contents:")
            for heading in headings[:15]:
                level = int(heading.name[1])
                indent = "  " * (level - 1)
                text_parts.append(f"{indent}- {heading.get_text(strip=True)}")
            text_parts.append("")

        text = main_content.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text_parts.extend(lines)

        nav_links = []
        for nav_selector in ["nav", ".sidebar", ".toc", ".navigation"]:
            nav = soup.select_one(nav_selector)
            if nav:
                links = nav.find_all("a", href=True)
                for link in links[:20]:
                    href = link.get("href", "")
                    if isinstance(href, str) and not href.startswith(
                        ("http://", "https://", "mailto:", "#")
                    ):
                        full_url = urljoin(docs_url, href)
                        nav_links.append(f"- {link.get_text(strip=True)}: {full_url}")

        if nav_links:
            text_parts.append("")
            text_parts.append("Related documentation pages:")
            text_parts.extend(nav_links[:10])

        content = "\n".join(text_parts)

        if len(content) > 100000:
            content = content[:100000] + "\n\n[Content truncated...]"

        return LoaderResult(
            content=content,
            metadata={
                "source": docs_url,
                "title": title_text,
                "domain": urlparse(docs_url).netloc,
            },
            doc_id=self.generate_doc_id(source_ref=docs_url, content=content),
        )
