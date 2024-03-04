# CodeDocsSearchTool

## Description
The CodeDocsSearchTool is a powerful RAG (Retrieval-Augmented Generation) tool designed for semantic searches within code documentation. It enables users to efficiently find specific information or topics within code documentation. By providing a `docs_url` during initialization, the tool narrows down the search to that particular documentation site. Alternatively, without a specific `docs_url`, it searches across a wide array of code documentation known or discovered throughout its execution, making it versatile for various documentation search needs.

## Installation
To start using the CodeDocsSearchTool, first, install the crewai_tools package via pip:
```shell
pip install 'crewai[tools]'
```

## Example
Utilize the CodeDocsSearchTool as follows to conduct searches within code documentation:
```python
from crewai_tools import CodeDocsSearchTool

# To search any code documentation content if the URL is known or discovered during its execution:
tool = CodeDocsSearchTool()

# OR

# To specifically focus your search on a given documentation site by providing its URL:
tool = CodeDocsSearchTool(docs_url='https://docs.example.com/reference')
```
Note: Substitute 'https://docs.example.com/reference' with your target documentation URL and 'How to use search tool' with the search query relevant to your needs.

## Arguments
- `docs_url`: Optional. Specifies the URL of the code documentation to be searched. Providing this during the tool's initialization focuses the search on the specified documentation content.