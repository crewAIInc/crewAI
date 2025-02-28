# ApifyActorsTool
Integrate [Apify Actors](https://apify.com/) into your CrewAI workflows with Ease.

## Description
The `ApifyActorsTool` seamlessly integrates [Apify Actors](https://apify.com/) - cloud-based web scraping and automation programs—into your CrewAI workflows. Whether you need to extract data, crawl websites, or automate tasks, this tool simplifies the process without requiring infrastructure management.

Key features:
- **Run Actors Directly**: Execute Actors like the [RAG Web Browser](https://apify.com/apify/rag-web-browser) within CrewAI agents.
- **Real-Time Data**: Ideal for tasks requiring up-to-date web data or automation.
- **Explore More**: Discover additional Actors in the [Apify Store](https://apify.com/store).

For detailed integration guidance, see the [Apify CrewAI documentation](https://docs.apify.com/platform/integrations/crewai).

## Installation
To use `ApifyActorsTool`, install the required packages and configure your Apify API token. You’ll need an API token from Apify - see the [Apify API documentation](https://docs.apify.com/platform/integrations/api) for instructions.

### Steps
1. **Install Dependencies**
   Use pip to install `crewai[tools]` and `langchain-apify`:
   ```bash
   pip install 'crewai[tools]' langchain-apify
   ```
   Alternatively, with `uv`:
   ```bash
   uv pip install 'crewai[tools]' langchain-apify
   ```

2. **Set Your API Token**
   Export the token as an environment variable:
   - On Linux/macOS:
     ```bash
     export APIFY_API_TOKEN='your-api-token-here'
     ```
   - On Windows (Command Prompt):
     ```cmd
     set APIFY_API_TOKEN=your-api-token-here
     ```
   - Or add it to your `.env` file and load it with a library like `python-dotenv`.

3. **Verify Installation**
   Run `python -c "import langchain_apify; print('Setup complete')"` to ensure dependencies are installed.

## Usage example
Here’s how to use `ApifyActorsTool` to run the [RAG Web Browser Actor](https://apify.com/apify/rag-web-browser) for web searching within a CrewAI workflow:

```python
from crewai_tools import ApifyActorsTool

# Initialize the tool with an Apify Actor
tool = ApifyActorsTool(actor_name="apify/rag-web-browser")

# Run the tool with input parameters
results = tool.run(run_input={"query": "What is CrewAI?", "maxResults": 5})

# Process the results
for result in results:
    print(f"URL: {result['metadata']['url']}")
    print(f"Content: {result['markdown'][:100]}...")  # Snippet of markdown content
```

### Expected output
```
URL: https://www.example.com/crewai-intro
Content: CrewAI is a framework for building AI-powered workflows...
URL: https://docs.crewai.com/
Content: Official documentation for CrewAI...
```

Try other Actors from the [Apify Store](https://apify.com/store) by changing `actor_name` and adjusting `run_input` per the Actor's input schema.

## Configuration
The `ApifyActorsTool` requires specific inputs to operate:

- **`actor_name` (str, required)**
  The ID of the Apify Actor to run (e.g., `"apify/rag-web-browser"`). Find Actors in the [Apify Store](https://apify.com/store).
- **`run_input` (dict, required at runtime)**
  A dictionary of input parameters for the Actor. Examples:
  - For `apify/rag-web-browser`: `{"query": "search term", "maxResults": 5}`
  - Check each Actor’s [input schema](https://apify.com/apify/rag-web-browser/input-schema) for details.

The tool adapts dynamically to the chosen Actor.

## Resources
- **[Apify Platform](https://apify.com/)**: Explore the Apify ecosystem.
- **[RAG Web Browser Actor](https://apify.com/apify/rag-web-browser)**: Try this popular Actor for web data retrieval.
- **[Apify Actors Documentation](https://docs.apify.com/platform/actors)**: Learn how to use and create Actors.
- **[CrewAI Integration Guide](https://docs.apify.com/platform/integrations/crewai)**: Official guide for Apify and CrewAI.

---

Streamline your CrewAI workflows with `ApifyActorsTool` - combine the power of Apify’s web scraping and automation with agent-based intelligence.
