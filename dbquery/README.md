# 🤖 CrewAI: Secure Database Analysis Crew

This feature introduces a robust, production-ready **Database Querying Crew**. It bridges the gap between natural language and SQL databases while implementing two critical production guardrails: **Read-Only Security Enforcement** and **Token Management**.

---

## 🌟 Key Features

### 1. Safe NL2SQL Tooling (`SafeNL2SQLTool`)
Prevents accidental or malicious database modification by wrapping the standard NL2SQLTool.

- **Command Filtering:** Automatically blocks `DROP`, `DELETE`, `TRUNCATE`, `UPDATE`, `INSERT`, `ALTER`, and `GRANT`.
- **Select-Only Policy:** Only permits queries starting with `SELECT`.

### 2. The Token Gatekeeper
A specialized agent and task callback that prevents "Context Overflow" errors.

- **Token Truncation:** If data exceeds `MAX_TOKEN_LIMIT`, it truncates the context and injects a **System Notice** header so the subsequent Analyst Agent knows they are working with a partial dataset.

### 3. Structured Output 
The crew returns a validated **Pydantic object**:

- `summary`: A human-readable breakdown.
- `sql_query_used`: The exact SQL code generated.
- `row_count`: Total rows processed.
- `data`: The raw data list.
- `recommendation`: AI-driven business advice (can be grounded with external or domain specific knowledge).


## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
uv pip install crewai-tools[sqlalchemy]
uv pip install psycopg2-binary
```

### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

**Add your `MAX_TOKEN_LIMIT` into the `.env` file**

**Add your `DATABASE_URL=postgresql://user:pass@localhost:5432/temp_db` into the `.env` file**

- Modify `src/dbquery/config/agents.yaml` to define your agents
- Modify `src/dbquery/config/tasks.yaml` to define your tasks
- Modify `src/dbquery/crew.py` to add your own logic, tools and specific args
- Modify `src/dbquery/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the dbquery Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

---

## Support

For support, questions, or feedback regarding the Dbquery Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
