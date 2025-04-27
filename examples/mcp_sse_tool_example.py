import logging
import os
import signal
import sys
import time

from crewai.tools import MCPToolConnector, Tool


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def handle_exit(signum, frame):
    """Handle exit signals gracefully."""
    print("\nExiting...")
    sys.exit(0)


def main():
    """Main function to demonstrate MCP SSE tool connection."""
    setup_logging()
    signal.signal(signal.SIGINT, handle_exit)
    
    print("CrewAI MCP SSE Tool Connection Example")
    print("--------------------------------------")
    print("This example connects tools to the MCP SSE server.")
    print("Make sure you're logged in with 'crewai login' first.")
    print("Press Ctrl+C to exit.")
    print()
    
    def search(query: str) -> str:
        """Search for information."""
        return f"Searching for: {query}"
    
    search_tool = Tool(
        name="search",
        description="Search for information",
        func=search
    )
    
    tools = [search_tool]
    
    connector = MCPToolConnector(tools=tools)
    
    try:
        print("Connecting to MCP SSE server...")
        connector.connect()
        print("Connected! Listening for tool events...")
        
        connector.listen()
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        connector.close()


if __name__ == "__main__":
    main()
