# RagTool: A Dynamic Knowledge Base Tool

RagTool is designed to answer questions by leveraging the power of RAG by leveraging (EmbedChain). It integrates seamlessly with the CrewAI ecosystem, offering a versatile and powerful solution for information retrieval.

## **Overview**

RagTool enables users to dynamically query a knowledge base, making it an ideal tool for applications requiring access to a vast array of information. Its flexible design allows for integration with various data sources, including files, directories, web pages, yoututbe videos and custom configurations.

## **Usage**

RagTool can be instantiated with data from different sources, including:

- 📰 PDF file
- 📊 CSV file
- 📃 JSON file
- 📝 Text
- 📁 Directory/ Folder
- 🌐 HTML Web page
- 📽️ Youtube Channel
- 📺 Youtube Video
- 📚 Docs website
- 📝 MDX file
- 📄 DOCX file
- 🧾 XML file
- 📬 Gmail
- 📝 Github
- 🐘 Postgres
- 🐬 MySQL
- 🤖 Slack
- 💬 Discord
- 🗨️ Discourse
- 📝 Substack
- 🐝 Beehiiv
- 💾 Dropbox
- 🖼️ Image
- ⚙️ Custom

#### **Creating an Instance**

```python
from crewai_tools import RagTool

# Example: Loading from a file
rag_tool = RagTool().from_file('path/to/your/file.txt')

# Example: Loading from a directory
rag_tool = RagTool().from_directory('path/to/your/directory')

# Example: Loading from a web page
rag_tool = RagTool().from_web_page('https://example.com')
```

## **Contribution**

Contributions to RagTool and the broader CrewAI tools ecosystem are welcome. To contribute, please follow the standard GitHub workflow for forking the repository, making changes, and submitting a pull request.

## **License**

RagTool is open-source and available under the MIT license.

Thank you for considering RagTool for your knowledge base needs. Your contributions and feedback are invaluable to making RagTool even better.
