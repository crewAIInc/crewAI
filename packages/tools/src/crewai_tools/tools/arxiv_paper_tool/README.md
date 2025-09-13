# ArxivPaperTool


# 📚 ArxivPaperTool

The **ArxivPaperTool** is a utility for fetching metadata and optionally downloading PDFs of academic papers from the [arXiv](https://arxiv.org) platform using its public API. It supports configurable queries, batch retrieval, PDF downloading, and clean formatting for summaries and metadata. This tool is particularly useful for researchers, students, academic agents, and AI tools performing automated literature reviews.

---

## Description

This tool:

* Accepts a **search query** and retrieves a list of papers from arXiv.
* Allows configuration of the **maximum number of results** to fetch.
* Optionally downloads the **PDFs** of the matched papers.
* Lets you specify whether to name PDF files using the **arXiv ID** or **paper title**.
* Saves downloaded files into a **custom or default directory**.
* Returns structured summaries of all fetched papers including metadata.

---

## Arguments

| Argument                | Type   | Required | Description                                                                       |
| ----------------------- | ------ | -------- | --------------------------------------------------------------------------------- |
| `search_query`          | `str`  | ✅        | Search query string (e.g., `"transformer neural network"`).                       |
| `max_results`           | `int`  | ✅        | Number of results to fetch (between 1 and 100).                                   |
| `download_pdfs`         | `bool` | ❌        | Whether to download the corresponding PDFs. Defaults to `False`.                  |
| `save_dir`              | `str`  | ❌        | Directory to save PDFs (created if it doesn’t exist). Defaults to `./arxiv_pdfs`. |
| `use_title_as_filename` | `bool` | ❌        | Use the paper title as the filename (sanitized). Defaults to `False`.             |

---

## 📄 `ArxivPaperTool` Usage Examples

This document shows how to use the `ArxivPaperTool` to fetch research paper metadata from arXiv and optionally download PDFs.

### 🔧 Tool Initialization

```python
from crewai_tools import ArxivPaperTool 
```

---

### Example 1: Fetch Metadata Only (No Downloads)

```python
tool = ArxivPaperTool()
result = tool._run(
    search_query="deep learning",
    max_results=1
)
print(result)
```

---

### Example 2: Fetch and Download PDFs (arXiv ID as Filename)

```python
tool = ArxivPaperTool(download_pdfs=True)
result = tool._run(
    search_query="transformer models",
    max_results=2
)
print(result)
```

---

### Example 3: Download PDFs into a Custom Directory

```python
tool = ArxivPaperTool(
    download_pdfs=True,
    save_dir="./my_papers"
)
result = tool._run(
    search_query="graph neural networks",
    max_results=2
)
print(result)
```

---

### Example 4: Use Paper Titles as Filenames

```python
tool = ArxivPaperTool(
    download_pdfs=True,
    use_title_as_filename=True
)
result = tool._run(
    search_query="vision transformers",
    max_results=1
)
print(result)
```

---

### Example 5: All Options Combined

```python
tool = ArxivPaperTool(
    download_pdfs=True,
    save_dir="./downloads",
    use_title_as_filename=True
)
result = tool._run(
    search_query="stable diffusion",
    max_results=3
)
print(result)
```

---

### Run via `__main__`

Your file can also include:

```python
if __name__ == "__main__":
    tool = ArxivPaperTool(
        download_pdfs=True,
        save_dir="./downloads2",
        use_title_as_filename=False
    )
    result = tool._run(
        search_query="deep learning",
        max_results=1
    )
    print(result)
```

---


