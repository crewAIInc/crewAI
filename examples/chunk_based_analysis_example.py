"""
Example: Sequential Chunk-Based File Analysis with CrewAI

This example demonstrates how to use ChunkBasedTask to analyze large files
by processing them in chunks with agent memory aggregation.
"""

from crewai import Agent, Crew
from crewai.tasks.chunk_based_task import ChunkBasedTask


def main():
    document_analyzer = Agent(
        role="Document Analyzer",
        goal="Analyze documents thoroughly and extract key insights",
        backstory="""You are an expert document analyst with years of experience 
        in processing and understanding complex documents. You excel at identifying 
        patterns, themes, and important information across large texts."""
    )
    
    analysis_task = ChunkBasedTask(
        description="""Analyze the provided document and identify:
        1. Main themes and topics
        2. Key arguments or points made
        3. Important facts or data mentioned
        4. Overall structure and organization""",
        expected_output="""A comprehensive analysis report containing:
        - Summary of main themes
        - List of key points
        - Notable facts and data
        - Assessment of document structure""",
        file_path="path/to/your/large_document.txt",
        chunk_size=4000,
        chunk_overlap=200,
        aggregation_prompt="""Synthesize the analysis from all document chunks into 
        a cohesive report that captures the document's essence while highlighting 
        the most important insights discovered."""
    )
    
    crew = Crew(
        agents=[document_analyzer],
        tasks=[analysis_task],
        memory=True,
        verbose=True
    )
    
    result = crew.kickoff()
    
    print("Analysis Complete!")
    print("Final Result:", result)
    
    chunk_results = analysis_task.get_chunk_results()
    print(f"Processed {len(chunk_results)} chunks")


if __name__ == "__main__":
    main()
