# ContextualAIParseTool

## Description
This tool is designed to integrate Contextual AI's enterprise-grade document parsing capabilities with CrewAI, enabling you to leverage advanced AI-powered document understanding for complex layouts, tables, and figures. Use this tool to extract structured content from your documents using Contextual AI's powerful document parser.

## Installation
To incorporate this tool into your project, follow the installation instructions below:

```
pip install 'crewai[tools]' contextual-client
```

**Note**: You'll need a Contextual AI API key. Sign up at [app.contextual.ai](https://app.contextual.ai) to get your free API key.

## Example

```python
from crewai_tools import ContextualAIParseTool

tool = ContextualAIParseTool(api_key="your_api_key_here")

result = tool._run(
    file_path="/path/to/document.pdf",
    parse_mode="standard",
    page_range="0-5",
    output_types=["markdown-per-page"]
)
print(result)
```

The result will show the parsed contents of your document. For example: 
```
{
  "file_name": "attention_is_all_you_need.pdf",
  "status": "completed",
  "pages": [
    {
      "index": 0,
      "markdown": "Provided proper attribution ...
    },
    {
      "index": 1,
      "markdown": "## 1 Introduction ...
    },
    ...
  ] 
}
```
## Parameters
- `api_key`: Your Contextual AI API key
- `file_path`: Path to document to parse
- `parse_mode`: Parsing mode (default: "standard")
- `figure_caption_mode`: Figure caption handling (default: "concise")
- `enable_document_hierarchy`: Enable hierarchy detection (default: True)
- `page_range`: Pages to parse (e.g., "0-5", None for all)
- `output_types`: Output formats (default: ["markdown-per-page"])

## Key Features
- **Advanced Document Understanding**: Handles complex PDF layouts, tables, and multi-column documents
- **Figure and Table Extraction**: Intelligent extraction of figures, charts, and tabular data
- **Page Range Selection**: Parse specific pages or entire documents

## Use Cases
- Extract structured content from complex PDFs and research papers
- Parse financial reports, legal documents, and technical manuals
- Convert documents to markdown for further processing in RAG pipelines

For more detailed information about Contextual AI's capabilities, visit the [official documentation](https://docs.contextual.ai).