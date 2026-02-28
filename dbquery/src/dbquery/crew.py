import os
import tiktoken
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import NL2SQLTool
from pydantic import BaseModel, Field
from typing import List, Optional

class DatabaseResponse(BaseModel):
    summary: str = Field(..., description="A brief human-readable summary of the findings.")
    sql_query_used: str = Field(..., description="The actual SQL query that was executed.")
    row_count: int = Field(..., description="Number of rows returned from the database.")
    data: List[dict] = Field(..., description="The actual data rows found.")
    recommendation: Optional[str] = Field(None, description="Any business advice based on the data.")

class SafeNL2SQLTool(NL2SQLTool):
    def _run(self, **kwargs) -> str:
        raw_query = kwargs.get('sql_query') or kwargs.get('query') or ""
        query = raw_query.strip().upper()
        forbidden = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "ALTER", "GRANT"]
        
        if not query.startswith("SELECT") or any(word in query for word in forbidden):
            return "Error: Read Only Access. Only SELECT queries are permitted."

        return super()._run(sql_query=raw_query)

@CrewBase
class DbQueryCrew():
    """Database Querying Crew"""
    
    # Tool initialization
    sql_tool = SafeNL2SQLTool(db_uri=os.getenv("DATABASE_URL"))

    @agent
    def db_querier(self) -> Agent:
        return Agent(
            config=self.agents_config['db_querier'], 
            tools=[self.sql_tool], 
            verbose=True
        )

    @agent
    def token_gatekeeper(self) -> Agent:
        return Agent(
            config=self.agents_config['token_gatekeeper'],
            verbose=True
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'], 
            verbose=True
        )

    @task
    def extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['extraction_task'],
            agent=self.db_querier()  
        )

    @task
    def token_validation_task(self) -> Task:
        # Pull limit from env or default to 1000
        token_limit = int(os.getenv("MAX_TOKEN_LIMIT"))

        def truncate_callback(output):
            """
            Checks token count and truncates if it exceeds the limit.
            Modifies the output in-place so the next agent receives the truncated version.
            """
            raw_content = str(output.raw)
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            
            # 1. Encode to check actual token length
            tokens = encoding.encode(raw_content)
            actual_count = len(tokens)

            if actual_count > token_limit:
                # 2. Truncate tokens to the allowed limit
                truncated_tokens = tokens[:token_limit]
                truncated_text = encoding.decode(truncated_tokens)

                # 3. Add a clear system header so the Analyst knows data is missing
                # This prevents the Analyst from giving "false" complete summaries.
                final_output = (
                    f"--- [SYSTEM NOTICE: DATA TRUNCATED FOR CONTEXT LIMITS] ---\n"
                    f"Original Token Count: {actual_count}\n"
                    f"Allowed Limit: {token_limit}\n"
                    f"Showing first {token_limit} tokens below:\n"
                    f"----------------------------------------------------------\n\n"
                    f"{truncated_text}\n"
                    f"\n... [END OF DATA SEGMENT] ..."
                )
                
                # Update the task output so the next agent sees THIS version
                output.raw = final_output
                print(f"--- [TRUNCATED {actual_count} -> {token_limit}] ---")
            else:
                print(f"--- [PASSED ({actual_count} tokens)] ---")

        return Task(
            config=self.tasks_config['token_validation_task'],
            agent=self.token_gatekeeper(),
            context=[self.extraction_task()],
            callback=truncate_callback 
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'],
            agent=self.data_analyst(),
            output_pydantic=DatabaseResponse,
            context=[self.token_validation_task()] 
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )