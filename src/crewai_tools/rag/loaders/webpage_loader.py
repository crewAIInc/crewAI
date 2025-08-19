import re
import requests
from bs4 import BeautifulSoup

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent

class WebPageLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs) -> LoaderResult:
        url = source_content.source
        headers = kwargs.get("headers", {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
        })

        try:
            response = requests.get(url, timeout=15, headers=headers)
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, "html.parser")

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(" ")
            text = re.sub("[ \t]+", " ", text)
            text = re.sub("\\s+\n\\s+", "\n", text)
            text = text.strip()

            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            metadata = {
                "url": url,
                "title": title,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", "")
            }

            return LoaderResult(
                content=text,
                source=url,
                metadata=metadata,
                doc_id=self.generate_doc_id(source_ref=url, content=text)
            )

        except Exception as e:
            raise ValueError(f"Error loading webpage {url}: {str(e)}")
