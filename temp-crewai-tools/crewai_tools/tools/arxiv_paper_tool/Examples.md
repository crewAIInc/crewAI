### Example 1: Fetching Research Papers from arXiv with CrewAI

This example demonstrates how to build a simple CrewAI workflow that automatically searches for and downloads academic papers from [arXiv.org](https://arxiv.org). The setup uses:

* A custom `ArxivPaperTool` to fetch metadata and download PDFs
* A single `Agent` tasked with locating relevant papers based on a given research topic
* A `Task` to define the data retrieval and download process
* A sequential `Crew` to orchestrate execution

The downloaded PDFs are saved to a local directory (`./DOWNLOADS`). Filenames are optionally based on sanitized paper titles, ensuring compatibility with your operating system.

> The saved PDFs can be further used in **downstream tasks**, such as:
>
> * **RAG (Retrieval-Augmented Generation)**
> * **Summarization**
> * **Citation extraction**
> * **Embedding-based search or analysis**

---


```
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ArxivPaperTool 



llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
    temperature=0.1
)


topic = "Crew AI"
max_results = 3
save_dir = "./DOWNLOADS"
use_title_as_filename = True

tool = ArxivPaperTool(
    download_pdfs=True,
    save_dir=save_dir,
    use_title_as_filename=True
)
tool.result_as_answer = True #Required,otherwise


arxiv_paper_fetch = Agent(
    role="Arxiv Data Fetcher",
    goal=f"Retrieve relevant papers from arXiv based on a research topic {topic} and maximum number of papers to be downloaded is{max_results},try to use title as filename {use_title_as_filename} and download PDFs to {save_dir},",
    backstory="An expert in scientific data retrieval, skilled in extracting academic content from arXiv.",
    # tools=[ArxivPaperTool()],
    llm=llm,
    verbose=True,
    allow_delegation=False
)
fetch_task = Task(
    description=(
        f"Search arXiv for the topic '{topic}' and fetch up to {max_results} papers. "
        f"Download PDFs for analysis and store them at {save_dir}."
    ),
    expected_output="PDFs saved to disk for downstream agents.",
    agent=arxiv_paper_fetch,
    tools=[tool],  # Use the actual tool instance here
    
)


pdf_qa_crew = Crew(
    agents=[arxiv_paper_fetch],
    tasks=[fetch_task],
    process=Process.sequential,
    verbose=True,
)


result = pdf_qa_crew.kickoff()

print(f"\n🤖 Answer:\n\n{result.raw}\n")
```
