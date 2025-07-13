import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import Field, field_validator

from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.agents.agent_builder.base_agent import BaseAgent


class ChunkBasedTask(Task):
    """Task that processes large files by breaking them into chunks and analyzing sequentially."""
    
    file_path: Union[str, Path] = Field(
        description="Path to the file to be processed in chunks"
    )
    chunk_size: int = Field(
        default=4000,
        description="Size of each chunk in characters"
    )
    chunk_overlap: int = Field(
        default=200, 
        description="Number of characters to overlap between chunks"
    )
    aggregation_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for aggregating chunk results"
    )
    chunk_results: List[TaskOutput] = Field(
        default_factory=list,
        description="Results from processing each chunk"
    )
    
    @field_validator("file_path")
    def validate_file_path(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path

    def _read_file(self) -> str:
        """Read the content of the file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def _create_chunk_task(self, chunk: str, chunk_index: int, total_chunks: int) -> str:
        """Create a task description for processing a single chunk."""
        chunk_context = f"Processing chunk {chunk_index + 1} of {total_chunks}"
        if self.chunk_results:
            chunk_context += f"\n\nPrevious chunk insights from memory will be available."
        
        return f"{chunk_context}\n\n{self.description}\n\nChunk content:\n{chunk}"

    def _execute_core(
        self,
        agent: Optional[BaseAgent],
        context: Optional[str],
        tools: Optional[List[Any]],
    ) -> TaskOutput:
        """Execute chunk-based processing."""
        agent = agent or self.agent
        if not agent:
            raise Exception(f"The chunk-based task '{self.description}' has no agent assigned")
        
        self.start_time = datetime.datetime.now()
        
        file_content = self._read_file()
        chunks = self._chunk_text(file_content)
        
        if not chunks:
            raise ValueError("No valid chunks found in the file")
        
        self.chunk_results = []
        
        for i, chunk in enumerate(chunks):
            chunk_description = self._create_chunk_task(chunk, i, len(chunks))
            
            chunk_task = Task(
                description=chunk_description,
                expected_output=self.expected_output,
                agent=agent,
                tools=tools or self.tools or []
            )
            
            chunk_result = chunk_task._execute_core(agent, context, tools)
            self.chunk_results.append(chunk_result)
            
            if hasattr(agent, 'crew') and agent.crew and hasattr(agent.crew, '_short_term_memory'):
                try:
                    agent.crew._short_term_memory.save(
                        value=f"Chunk {i+1} analysis: {chunk_result.raw}",
                        metadata={
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "task_description": self.description
                        },
                        agent=agent.role
                    )
                except Exception as e:
                    print(f"Failed to save chunk result to memory: {e}")
        
        return self._aggregate_results(agent, context, tools)

    def _aggregate_results(
        self,
        agent: BaseAgent,
        context: Optional[str],
        tools: Optional[List[Any]]
    ) -> TaskOutput:
        """Aggregate results from all chunks."""
        if not self.chunk_results:
            raise ValueError("No chunk results to aggregate")
        
        chunk_summaries = []
        for i, result in enumerate(self.chunk_results):
            chunk_summaries.append(f"Chunk {i+1} result: {result.raw}")
        
        aggregation_description = self.aggregation_prompt or f"""
        Analyze and synthesize the following chunk analysis results into a comprehensive summary.
        
        Original task: {self.description}
        Expected output: {self.expected_output}
        
        Chunk results:
        {chr(10).join(chunk_summaries)}
        
        Provide a comprehensive analysis that synthesizes insights from all chunks.
        """
        
        aggregation_task = Task(
            description=aggregation_description,
            expected_output=self.expected_output,
            agent=agent,
            tools=tools or self.tools or []
        )
        
        final_result = aggregation_task._execute_core(agent, context, tools)
        
        self.output = final_result
        self.end_time = datetime.datetime.now()
        return final_result
    
    def get_chunk_results(self) -> List[TaskOutput]:
        """Get results from individual chunk processing."""
        return self.chunk_results
