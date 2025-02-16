import streamlit as st
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Data Models
@dataclass
class Agent:
    id: int
    name: str
    role: str

@dataclass
class Crew:
    id: int
    name: str
    agents: List[int]
    status: str

@dataclass
class Task:
    id: int
    crew_id: int
    description: str
    status: str

# Initialize Mock Data in Session State if not already present
if 'mock_agents' not in st.session_state:
    st.session_state.mock_agents = {
        2: Agent(2, "Researcher Agent", "Market Research"),
        3: Agent(3, "Analyst Agent", "Data Analysis"),
        5: Agent(5, "Writer Agent", "Content Creation"),
        6: Agent(6, "Editor Agent", "Content Editing"),
        7: Agent(7, "Publisher Agent", "Content Publishing")
    }
if 'mock_crews' not in st.session_state:
    st.session_state.mock_crews = [
        Crew(1, "Research Crew", [2, 3], "Active"),
        Crew(4, "Content Creation Crew", [5, 6, 7], "Idle")
    ]
if 'mock_tasks' not in st.session_state:
    st.session_state.mock_tasks = [
        Task(1, 1, "Research Competitors", "Completed"),
        Task(2, 1, "Analyze Market Trends", "In Progress"),
        Task(3, 4, "Write Blog Post", "Pending"),
        Task(4, 4, "Edit Content", "In Progress")
    ]

mock_agents = st.session_state.mock_agents
mock_crews = st.session_state.mock_crews
mock_tasks = st.session_state.mock_tasks


# Helper Functions
def get_agent_names(agent_ids: List[int]) -> str:
    return ", ".join([mock_agents[aid].name for aid in agent_ids if aid in mock_agents])

def get_next_id(data_dict_or_list):
    if isinstance(data_dict_or_list, dict):
        if not data_dict_or_list:
            return 1
        return max(data_dict_or_list.keys()) + 1
    elif isinstance(data_dict_or_list, list):
        if not data_dict_or_list:
            return 1
        return max([item.id for item in data_dict_or_list]) + 1
    return 1

# Main App
st.set_page_config(
    page_title="CrewAI Frontend Prototype",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 CrewAI Frontend Prototype")
st.markdown("---")

## Agent Management Section
st.subheader("Agent Management")

# Display Agents
agents_data = []
for agent_id, agent in mock_agents.items():
    agents_data.append({
        "ID": agent.id,
        "Name": agent.name,
        "Role": agent.role
    })
st.dataframe(
    agents_data,
    use_container_width=True,
    hide_index=True
)

# Add New Agent Form
with st.expander("Add New Agent"):
    with st.form("add_agent_form"):
        new_agent_name = st.text_input("Agent Name")
        new_agent_role = st.text_input("Agent Role")
        submit_agent = st.form_submit_button("Add Agent")

        if submit_agent:
            if new_agent_name and new_agent_role:
                new_agent_id = get_next_id(mock_agents)
                new_agent = Agent(new_agent_id, new_agent_name, new_agent_role)
                mock_agents[new_agent_id] = new_agent
                st.session_state.mock_agents = mock_agents # Update session state
                st.success(f"Agent '{new_agent_name}' added successfully!")
                st.rerun() # Rerun to update the displayed table
            else:
                st.error("Agent Name and Role are required.")

st.markdown("---")

## Crew Management Section
st.subheader("Crew Management")

# Display Crews
crews_data = []
for crew in mock_crews:
    crews_data.append({
        "ID": crew.id,
        "Name": crew.name,
        "Agents": get_agent_names(crew.agents),
        "Status": crew.status
    })

st.dataframe(
    crews_data,
    use_container_width=True,
    hide_index=True
)

# Add New Crew Form
with st.expander("Add New Crew"):
    with st.form("add_crew_form"):
        new_crew_name = st.text_input("Crew Name")
        new_crew_status = st.selectbox("Status", ["Active", "Idle", "Planning"])
        available_agents = [agent for agent in mock_agents.values()]
        agent_options = {agent.name: agent.id for agent in available_agents}
        selected_agent_names = st.multiselect("Agents", options=agent_options.keys())
        selected_agent_ids = [agent_options[name] for name in selected_agent_names]

        submit_crew = st.form_submit_button("Add Crew")

        if submit_crew:
            if new_crew_name:
                new_crew_id = get_next_id(mock_crews)
                new_crew = Crew(new_crew_id, new_crew_name, selected_agent_ids, new_crew_status)
                mock_crews.append(new_crew)
                st.session_state.mock_crews = mock_crews # Update session state
                st.success(f"Crew '{new_crew_name}' added successfully!")
                st.rerun() # Rerun to update the displayed table
            else:
                st.error("Crew Name is required.")


st.markdown("---")

## Task Management Section
st.subheader("Task Management")

# Crew Filter for Tasks Overview (moved here for task management section)
crew_options_task_filter = ["All Crews"] + [crew.name for crew in mock_crews]
selected_crew_task_filter = st.selectbox("Filter Tasks by Crew:", crew_options_task_filter, key="task_filter_selectbox") # Added key

# Display Tasks (Filtered)
filtered_tasks = mock_tasks
if selected_crew_task_filter != "All Crews":
    selected_crew_id_filter = next(crew.id for crew in mock_crews if crew.name == selected_crew_task_filter)
    filtered_tasks = [task for task in mock_tasks if task.crew_id == selected_crew_id_filter]

tasks_data = []
for task in filtered_tasks:
    crew_name = next((crew.name for crew in mock_crews if crew.id == task.crew_id), "Unknown Crew") #Handle case where crew might be deleted?
    tasks_data.append({
        "ID": task.id,
        "Crew": crew_name,
        "Description": task.description,
        "Status": task.status
    })

st.dataframe(
    tasks_data,
    use_container_width=True,
    hide_index=True
)


# Add New Task Form
with st.expander("Add New Task"):
    with st.form("add_task_form"):
        new_task_description = st.text_area("Task Description")
        new_task_status = st.selectbox("Status", ["Pending", "In Progress", "Completed", "Blocked"])
        crew_options = {crew.name: crew.id for crew in mock_crews}
        selected_crew_name = st.selectbox("Crew", options=crew_options.keys())
        selected_crew_id = crew_options[selected_crew_name]

        submit_task = st.form_submit_button("Add Task")

        if submit_task:
            if new_task_description and selected_crew_id:
                new_task_id = get_next_id(mock_tasks)
                new_task = Task(new_task_id, selected_crew_id, new_task_description, new_task_status)
                mock_tasks.append(new_task)
                st.session_state.mock_tasks = mock_tasks # Update session state
                st.success(f"Task '{new_task_description}' added successfully to '{selected_crew_name}'!")
                st.rerun() # Rerun to update the displayed table
            else:
                st.error("Task Description and Crew are required.")


st.markdown("---")

# Dashboard Layout (Simplified - keeping only Status Overview from original)
st.subheader("Dashboard Overview")
st.subheader("Status Overview")

# Calculate statistics
total_crews = len(mock_crews)
active_crews = sum(1 for crew in mock_crews if crew.status == "Active")
total_agents = len(mock_agents)
total_tasks = len(mock_tasks)
completed_tasks = sum(1 for task in mock_tasks if task.status == "Completed")
pending_tasks = sum(1 for task in mock_tasks if task.status == "Pending")
in_progress_tasks = sum(1 for task in mock_tasks if task.status == "In Progress")


# Display statistics in a grid
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Total Crews", total_crews)
with col2:
    st.metric("Active Crews", active_crews)
with col3:
    st.metric("Total Agents", total_agents)
with col4:
    st.metric("Total Tasks", total_tasks)
with col5:
    st.metric("Completed Tasks", completed_tasks)
with col6:
    st.metric("Pending Tasks", pending_tasks)


# Error Handling Demo (Keep as is)
st.markdown("---")
st.subheader("Error Handling Demo")
if st.button("Simulate API Error"):
    st.error("Failed to connect to the CrewAI backend. Please check your connection and try again.")