import logging
from pathlib import Path
import re
import time
from typing import Any, ClassVar
import urllib.parse
import xml.etree.ElementTree as ET

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field
import requests

from crewai_tools.security.safe_requests import safe_download, safe_get


logger = logging.getLogger(__file__)


class ArxivToolInput(BaseModel):
    search_query: str = Field(
        ..., description="Search query for Arxiv, e.g., 'transformer neural network'"
    )
    max_results: int = Field(
        5, ge=1, le=100, description="Max results to fetch; must be between 1 and 100"
    )


class ArxivPaperTool(BaseTool):
    BASE_API_URL: ClassVar[str] = "https://export.arxiv.org/api/query"
    SLEEP_DURATION: ClassVar[int] = 1
    SUMMARY_TRUNCATE_LENGTH: ClassVar[int] = 300
    ATOM_NAMESPACE: ClassVar[str] = "{http://www.w3.org/2005/Atom}"
    REQUEST_TIMEOUT: ClassVar[int] = 10
    name: str = "Arxiv Paper Fetcher and Downloader"
    description: str = "Fetches metadata from Arxiv based on a search query and optionally downloads PDFs."
    args_schema: type[BaseModel] = ArxivToolInput
    model_config = ConfigDict(extra="allow")
    package_dependencies: list[str] = Field(default_factory=lambda: ["pydantic"])
    env_vars: list[EnvVar] = Field(default_factory=list)
    download_pdfs: bool = False
    save_dir: str = "./arxiv_pdfs"
    use_title_as_filename: bool = False

    def _run(self, search_query: str, max_results: int = 5) -> str:
        try:
            args = ArxivToolInput(search_query=search_query, max_results=max_results)
            logger.info(
                f"Running Arxiv tool: query='{args.search_query}', max_results={args.max_results}, "
                f"download_pdfs={self.download_pdfs}, save_dir='{self.save_dir}', "
                f"use_title_as_filename={self.use_title_as_filename}"
            )

            papers = self.fetch_arxiv_data(args.search_query, args.max_results)

            if self.download_pdfs:
                save_dir = self._validate_save_path(self.save_dir)
                for paper in papers:
                    if paper["pdf_url"]:
                        if self.use_title_as_filename:
                            safe_title = re.sub(
                                r'[\\/*?:"<>|]', "_", paper["title"]
                            ).strip()
                            filename_base = safe_title or paper["arxiv_id"]
                        else:
                            filename_base = paper["arxiv_id"]
                        filename = f"{filename_base[:500]}.pdf"
                        save_path = Path(save_dir) / filename

                        self.download_pdf(paper["pdf_url"], save_path)
                        time.sleep(self.SLEEP_DURATION)

            results = [self._format_paper_result(p) for p in papers]
            return "\n\n" + "-" * 80 + "\n\n".join(results)

        except Exception as e:
            logger.error(f"ArxivTool Error: {e!s}")
            return f"Failed to fetch or download Arxiv papers: {e!s}"

    def fetch_arxiv_data(
        self, search_query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Query the Arxiv API and parse the resulting Atom feed into a list
        of paper records.

        Args:
            search_query: Free-text Arxiv search query.
            max_results: Maximum number of entries to request from the API.

        Returns:
            A list of dicts with keys `arxiv_id`, `title`, `summary`,
            `authors`, `published_date`, `pdf_url` (the last is `None` if no
            PDF link was found in the entry).

        Raises:
            requests.RequestException: If the request to Arxiv fails (network
                error, timeout, or a non-2xx response).
            ValueError: If `api_url` (or a redirect it follows) fails
                `safe_get`'s SSRF validation.
            xml.etree.ElementTree.ParseError: If the response body isn't
                valid XML.
        """
        api_url = f"{self.BASE_API_URL}?search_query={urllib.parse.quote(search_query)}&start=0&max_results={max_results}"
        logger.info(f"Fetching data from Arxiv API: {api_url}")

        try:
            response = safe_get(api_url, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.text
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Error fetching data from Arxiv: {e}")
            raise

        root = ET.fromstring(data)  # noqa: S314
        papers = []

        for entry in root.findall(self.ATOM_NAMESPACE + "entry"):
            raw_id = self._get_element_text(entry, "id")
            arxiv_id = raw_id.split("/")[-1].replace(".", "_") if raw_id else "unknown"

            title = self._get_element_text(entry, "title") or "No Title"
            summary = self._get_element_text(entry, "summary") or "No Summary"
            published = self._get_element_text(entry, "published") or "No Publish Date"
            authors = [
                self._get_element_text(author, "name") or "Unknown"
                for author in entry.findall(self.ATOM_NAMESPACE + "author")
            ]

            pdf_url = self._extract_pdf_url(entry)

            papers.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "summary": summary,
                    "authors": authors,
                    "published_date": published,
                    "pdf_url": pdf_url,
                }
            )

        return papers

    @staticmethod
    def _get_element_text(entry: ET.Element, element_name: str) -> str | None:
        elem = entry.find(f"{ArxivPaperTool.ATOM_NAMESPACE}{element_name}")
        return elem.text.strip() if elem is not None and elem.text else None

    def _extract_pdf_url(self, entry: ET.Element) -> str | None:
        for link in entry.findall(self.ATOM_NAMESPACE + "link"):
            if link.attrib.get("title", "").lower() == "pdf":
                return link.attrib.get("href")
        for link in entry.findall(self.ATOM_NAMESPACE + "link"):
            href = link.attrib.get("href")
            if href and "pdf" in href:
                return href
        return None

    def _format_paper_result(self, paper: dict[str, Any]) -> str:
        summary = (
            (paper["summary"][: self.SUMMARY_TRUNCATE_LENGTH] + "...")
            if len(paper["summary"]) > self.SUMMARY_TRUNCATE_LENGTH
            else paper["summary"]
        )
        authors_str = ", ".join(paper["authors"])
        return (
            f"Title: {paper['title']}\n"
            f"Authors: {authors_str}\n"
            f"Published: {paper['published_date']}\n"
            f"PDF: {paper['pdf_url'] or 'N/A'}\n"
            f"Summary: {summary}"
        )

    @staticmethod
    def _validate_save_path(path: str) -> Path:
        save_path = Path(path).resolve()
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def download_pdf(self, pdf_url: str, save_path: str | Path) -> None:
        """Download a single PDF to `save_path` via `safe_download`.

        Args:
            pdf_url: The PDF URL, as extracted from an Arxiv API entry.
            save_path: Destination file path. `safe_download` writes to a
                temp file alongside it and renames into place on success, so
                a failed download never leaves a truncated file here.

        Raises:
            requests.RequestException: If the request fails (network error,
                timeout, or a non-2xx response).
            ValueError: If `pdf_url` (or a redirect it follows) fails
                `safe_download`'s SSRF validation.
            OSError: If `save_path` can't be written (permissions, missing
                parent directory, etc.).
        """
        try:
            logger.info(f"Downloading PDF from {pdf_url} to {save_path}")
            # pdf_url comes from the Arxiv API's XML response, not directly from
            # tool input -- but it's still an untrusted, remotely-supplied URL
            # (e.g. a network MITM tampering with the plain-HTTP-era API response,
            # or a malformed/malicious link ever indexed upstream). safe_download
            # validates the URL and every redirect target, and pins each
            # connection's DNS resolution to the address that was actually
            # validated, closing the SSRF exposure a raw urlretrieve() call has.
            safe_download(pdf_url, str(save_path), timeout=self.REQUEST_TIMEOUT)
            logger.info(f"PDF saved: {save_path}")
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Network error occurred while downloading {pdf_url}: {e}")
            raise
        except OSError as e:
            logger.error(f"File save error for {save_path}: {e}")
            raise
