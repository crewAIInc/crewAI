# XMLSearchTool

## Description
The XMLSearchTool is a cutting-edge RAG tool engineered for conducting semantic searches within XML files. Ideal for users needing to parse and extract information from XML content efficiently, this tool supports inputting a search query and an optional XML file path. By specifying an XML path, users can target their search more precisely to the content of that file, thereby obtaining more relevant search outcomes.

## Installation
To start using the XMLSearchTool, you must first install the crewai_tools package. This can be easily done with the following command:

```shell
pip install 'crewai[tools]'
```

## Example
Here are two examples demonstrating how to use the XMLSearchTool. The first example shows searching within a specific XML file, while the second example illustrates initiating a search without predefining an XML path, providing flexibility in search scope.

```python
from crewai_tools.tools.xml_search_tool import XMLSearchTool

# Allow agents to search within any XML file's content as it learns about their paths during execution
tool = XMLSearchTool()

# OR

# Initialize the tool with a specific XML file path for exclusive search within that document
tool = XMLSearchTool(xml='path/to/your/xmlfile.xml')
```

## Arguments
- `xml`: This is the path to the XML file you wish to search. It is an optional parameter during the tool's initialization but must be provided either at initialization or as part of the `run` method's arguments to execute a search.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = XMLSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```
