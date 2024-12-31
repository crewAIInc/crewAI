# Building a Multi-Agent Support Crew UI with Gradio

​
## Introduction
This guide demonstrates how to create an interactive web interface for CrewAI multi-agent systems using Gradio. We'll build a customer support application where multiple agents collaborate to provide comprehensive responses to user inquiries.

​
## Overview
The application features:
- A support representative agent that handles initial customer inquiries
- A QA specialist agent that reviews and refines responses
- Real-time message updates showing agent interactions
- Document context support through web scraping
- A clean, user-friendly interface built with Gradio

![Overview of the gradio chatbot UI](/images/gradio-overview.png)

​
## Prerequisites
Before starting, ensure you have the following installed:

```bash
pip install crewai gradio crewai-tools
```

## Implementation

### 1. Set Up Message Queue System
First, import all the required packages. Next, we'll create a message queue to manage the flow of communications between agents:

```python
import gradio as gr
import asyncio
import threading
import os
import queue
from typing import List, Dict, Generator

class SupportMessageQueue:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.last_agent = None

    def add_message(self, message: Dict):
        self.message_queue.put(message)

    def get_messages(self) -> List[Dict]:
        messages = []
        while not self.message_queue.empty():
            messages.append(self.message_queue.get())
        return messages
```

### 2. Create the Support Crew Class
Next, we'll implement the main support crew class that handles agent initialization and task creation:

```python
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool

class SupportCrew:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.message_queue = SupportMessageQueue()
        self.support_agent = None
        self.qa_agent = None
        self.current_agent = None
        self.scrape_tool = None

    def initialize_agents(self, website_url: str):
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        os.environ["OPENAI_API_KEY"] = self.api_key
        self.scrape_tool = ScrapeWebsiteTool(website_url=website_url)

        self.support_agent = Agent(
            role="Senior Support Representative",
            goal="Be the most friendly and helpful support representative in your team",
            backstory=(
                "You work at crewAI and are now working on providing support to customers. "
                "You need to make sure that you provide the best support! "
                "Make sure to provide full complete answers, and make no assumptions."
            ),
            allow_delegation=False,
            verbose=True
        )

        self.qa_agent = Agent(
            role="Support Quality Assurance Specialist",
            goal="Get recognition for providing the best support quality assurance in your team",
            backstory=(
                "You work at crewAI and are now working with your team on customer requests "
                "ensuring that the support representative is providing the best support possible. "
                "You need to make sure that the support representative is providing full "
                "complete answers, and make no assumptions."
            ),
            verbose=True
        )

    # Implement Task Creation for the agents
    def create_tasks(self, inquiry: str) -> List[Task]:
        inquiry_resolution = Task(
            description=(
                f"A customer just reached out with a super important ask:\n{inquiry}\n\n"
                "Make sure to use everything you know to provide the best support possible. "
                "You must strive to provide a complete and accurate response to the customer's inquiry."
            ),
            expected_output=(
                "A detailed, informative response to the customer's inquiry that addresses "
                "all aspects of their question.\n"
                "The response should include references to everything you used to find the answer."
            ),
            tools=[self.scrape_tool],
            agent=self.support_agent
        )

        quality_assurance_review = Task(
            description=(
                "Review the response drafted by the Senior Support Representative for the customer's inquiry. "
                "Ensure that the answer is comprehensive, accurate, and adheres to the "
                "high-quality standards expected for customer support."
            ),
            expected_output=(
                "A final, detailed, and informative response ready to be sent to the customer.\n"
                "This response should fully address the customer's inquiry, incorporating all "
                "relevant feedback and improvements."
            ),
            agent=self.qa_agent
        )

        return [inquiry_resolution, quality_assurance_review] 

    # main processing function
    async def process_support(self, inquiry: str, website_url: str) -> Generator[List[Dict], None, None]:
        def add_agent_messages(agent_name: str, tasks: str, emoji: str = "🤖"):
            self.message_queue.add_message({
                "role": "assistant",
                "content": agent_name,
                "metadata": {"title": f"{emoji} {agent_name}"}
            })
            
            self.message_queue.add_message({
                "role": "assistant",
                "content": tasks,
                "metadata": {"title": f"📋 Task for {agent_name}"}
            })

        # Manages transition between agents
        def setup_next_agent(current_agent: str) -> None:
            if current_agent == "Senior Support Representative":
                self.current_agent = "Support Quality Assurance Specialist"
                add_agent_messages(
                    "Support Quality Assurance Specialist",
                    "Review and improve the support representative's response"
                )

        def task_callback(task_output) -> None:
            print(f"Task callback received: {task_output}")
            
            raw_output = task_output.raw
            if "## Final Answer:" in raw_output:
                content = raw_output.split("## Final Answer:")[1].strip()
            else:
                content = raw_output.strip()
            
            if self.current_agent == "Support Quality Assurance Specialist":
                self.message_queue.add_message({
                    "role": "assistant",
                    "content": "Final response is ready!",
                    "metadata": {"title": "✅ Final Response"}
                })
                
                formatted_content = content
                formatted_content = formatted_content.replace("\n#", "\n\n#")
                formatted_content = formatted_content.replace("\n-", "\n\n-")
                formatted_content = formatted_content.replace("\n*", "\n\n*")
                formatted_content = formatted_content.replace("\n1.", "\n\n1.")
                formatted_content = formatted_content.replace("\n\n\n", "\n\n")
                
                self.message_queue.add_message({
                    "role": "assistant",
                    "content": formatted_content
                })
            else:
                self.message_queue.add_message({
                    "role": "assistant",
                    "content": content,
                    "metadata": {"title": f"✨ Output from {self.current_agent}"}
                })
                setup_next_agent(self.current_agent)

        try:
            self.initialize_agents(website_url)
            self.current_agent = "Senior Support Representative"

            yield [{
                "role": "assistant",
                "content": "Starting to process your inquiry...",
                "metadata": {"title": "🚀 Process Started"}
            }]

            add_agent_messages(
                "Senior Support Representative",
                "Analyze customer inquiry and provide comprehensive support"
            )

            crew = Crew(
                agents=[self.support_agent, self.qa_agent],
                tasks=self.create_tasks(inquiry),
                verbose=True,
                task_callback=task_callback
            )

            def run_crew():
                try:
                    crew.kickoff()
                except Exception as e:
                    print(f"Error in crew execution: {str(e)}")
                    self.message_queue.add_message({
                        "role": "assistant",
                        "content": f"An error occurred: {str(e)}",
                        "metadata": {"title": "❌ Error"}
                    })

            thread = threading.Thread(target=run_crew)
            thread.start()

            while thread.is_alive() or not self.message_queue.message_queue.empty():
                messages = self.message_queue.get_messages()
                if messages:
                    print(f"Yielding messages: {messages}")
                    yield messages
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error in process_support: {str(e)}")
            yield [{
                "role": "assistant",
                "content": f"An error occurred: {str(e)}",
                "metadata": {"title": "❌ Error"}
            }]
```


### 3. Create the Gradio Interface
Finally, implement the Gradio interface that ties everything together:

```python
import gradio as gr

def create_demo():
    support_crew = None

    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        gr.Markdown("# 🎯 AI Customer Support Crew")
        gr.Markdown("This is a friendly, high-performing multi-agent application built with Gradio and CrewAI. Enter a webpage URL and your questions from that webpage.")
        openai_api_key = gr.Textbox(
            label='OpenAI API Key',
            type='password',
            placeholder='Type your OpenAI API Key and press Enter to access the app...',
            interactive=True
        )

        chatbot = gr.Chatbot(
            label="Support Process",
            height=700,
            type="messages",
            show_label=True,
            visible=False,
            avatar_images=(None, "https://avatars.githubusercontent.com/u/170677839?v=4"),
            render_markdown=True
        )

        with gr.Row(equal_height=True):
            inquiry = gr.Textbox(
                label="Your Inquiry",
                placeholder="Enter your question...",
                scale=4,
                visible=False
            )
            website_url = gr.Textbox(
                label="Documentation URL",
                placeholder="Enter documentation URL to search...",
                scale=4,
                visible=False
            )
            btn = gr.Button("Get Support", variant="primary", scale=1, visible=False)

        async def process_input(inquiry_text, website_url_text, history, api_key):
            nonlocal support_crew
            if not api_key:
                history = history or []
                history.append({
                    "role": "assistant",
                    "content": "Please provide an OpenAI API key.",
                    "metadata": {"title": "❌ Error"}
                })
                yield history
                return

            if support_crew is None:
                support_crew = SupportCrew(api_key=api_key)

            history = history or []
            history.append({
                "role": "user", 
                "content": f"Question: {inquiry_text}\nDocumentation: {website_url_text}"
            })
            yield history

            try:
                async for messages in support_crew.process_support(inquiry_text, website_url_text):
                    history.extend(messages)
                    yield history
            except Exception as e:
                history.append({
                    "role": "assistant",
                    "content": f"An error occurred: {str(e)}",
                    "metadata": {"title": "❌ Error"}
                })
                yield history

        def show_interface():
            return {
                openai_api_key: gr.Textbox(visible=False),
                chatbot: gr.Chatbot(visible=True),
                inquiry: gr.Textbox(visible=True),
                website_url: gr.Textbox(visible=True),
                btn: gr.Button(visible=True)
            }

        openai_api_key.submit(show_interface, None, [openai_api_key, chatbot, inquiry, website_url, btn])
        btn.click(process_input, [inquiry, website_url, chatbot, openai_api_key], [chatbot])
        inquiry.submit(process_input, [inquiry, website_url, chatbot, openai_api_key], [chatbot])

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.queue()
    demo.launch(debug=True)
```

## Running the Application
To run the application:

1. Save all the code in a single Python file (e.g., `support_crew_app.py`)
2. Install the required dependencies:
```bash
pip install crewai gradio crewai-tools
```
3. Run the application:
```bash
python support_crew_app.py
```
4. Open your browser and navigate to the URL shown in the console (typically `http://127.0.0.1:7860`)

## How It Works

1. **Initial Setup**: When the application starts, it creates a clean interface with an API key input field.

2. **Agent Initialization**: After entering the OpenAI API key, the interface reveals the main components:
   - An inquiry input field for user questions
   - A URL input field for documentation reference
   - A chat interface showing the conversation and agent interactions

3. **Agent Interaction Flow**:
   - The Support Representative agent receives the initial inquiry and documentation URL
   - It processes the request and provides a detailed response
   - The QA Specialist agent reviews and refines the response
   - The final answer is displayed in the chat interface

4. **Real-time Updates**: The interface shows real-time updates of the agents' work, including:
   - Task assignments
   - Agent responses
   - Final refined answers

## Customization

You can customize the application by:
- Modifying agent roles and backstories
- Adding additional agents to the crew
- Customizing the interface theme and layout
- Adding new tools for the agents to use

## Conclusion

This implementation demonstrates how to create a user-friendly interface for CrewAI multi-agent systems using Gradio. The combination of CrewAI's powerful agent capabilities with Gradio's simple interface creation makes it easy to build interactive AI applications that showcase agent collaboration and task processing.
