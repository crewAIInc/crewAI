---
name: streamlit
description: |
  Guide for building Streamlit web applications in Python. Use this skill when:
  - Creating interactive data apps, dashboards, or web UIs with Python
  - Working with st.* functions for widgets, layouts, charts, or data display
  - Implementing caching (@st.cache_data, @st.cache_resource) or session state
  - Building multipage Streamlit apps with st.navigation
  - Configuring Streamlit themes, secrets, or database connections
---

# Streamlit Development Guide

Streamlit turns Python scripts into interactive web apps. The framework reruns your entire script from top to bottom whenever users interact with widgets or source code changes.

## Core Data Flow

```python
import streamlit as st

# Script runs top-to-bottom on every interaction
x = st.slider('Value', 0, 100, 50)  # Widget interaction triggers rerun
st.write(f"Result: {x * 2}")
```

**Key Principle**: Every widget interaction causes a full script rerun. Use caching and session state to preserve expensive computations and user data.

## Essential Patterns

### Display Data

```python
import streamlit as st
import pandas as pd

# Magic: standalone variables auto-render
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df  # Automatically displayed

# Explicit display
st.write("Swiss Army knife for any data type")
st.dataframe(df.style.highlight_max(axis=0))  # Interactive
st.table(df)  # Static
```

### Widgets with Keys

```python
import streamlit as st

# Access widget values via session_state
st.text_input("Name", key="user_name")
st.number_input("Age", key="user_age")

# Use values anywhere
if st.session_state.user_name:
    st.write(f"Hello {st.session_state.user_name}!")
```

### Layout

```python
import streamlit as st

# Sidebar
with st.sidebar:
    option = st.selectbox("Choose", ["A", "B", "C"])
    threshold = st.slider("Threshold", 0, 100)

# Columns
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Metric 1", 42)
with col2:
    st.metric("Metric 2", 84)
with col3:
    st.metric("Metric 3", 126)

# Tabs
tab1, tab2 = st.tabs(["Chart", "Data"])
with tab1:
    st.line_chart(data)
with tab2:
    st.dataframe(data)
```

### Caching (Critical for Performance)

```python
import streamlit as st

# Cache data computations (returns copy each call)
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# Cache resources (returns same object, shared across sessions)
@st.cache_resource
def load_model(name: str):
    return SomeLargeModel(name)

# Usage
df = load_data("data.csv")  # Cached after first call
model = load_model("gpt")   # Shared across all users
```

**Rule of Thumb**:
- `@st.cache_data` → DataFrames, dicts, lists, strings (serializable data)
- `@st.cache_resource` → ML models, database connections (global resources)

### Session State

```python
import streamlit as st

# Initialize state (only runs once per session)
if "counter" not in st.session_state:
    st.session_state.counter = 0
    st.session_state.history = []

# Modify state
if st.button("Increment"):
    st.session_state.counter += 1
    st.session_state.history.append(st.session_state.counter)

st.write(f"Count: {st.session_state.counter}")
```

### Database Connections

```python
import streamlit as st

# Automatic connection management with caching
conn = st.connection("my_database")
df = conn.query("SELECT * FROM users WHERE active = true")
st.dataframe(df)
```

Configure in `.streamlit/secrets.toml`:
```toml
[connections.my_database]
type = "sql"
dialect = "postgresql"
host = "localhost"
port = 5432
database = "mydb"
username = "user"
password = "pass"
```

### Multipage Apps

```python
# streamlit_app.py (entry point)
import streamlit as st

# Define pages
home = st.Page("pages/home.py", title="Home", icon="🏠")
dashboard = st.Page("pages/dashboard.py", title="Dashboard", icon="📊")
settings = st.Page("pages/settings.py", title="Settings", icon="⚙️")

# Create navigation
pg = st.navigation([home, dashboard, settings])
pg.run()
```

Project structure:
```
my_app/
├── streamlit_app.py      # Entry point with navigation
├── pages/
│   ├── home.py
│   ├── dashboard.py
│   └── settings.py
└── .streamlit/
    ├── config.toml       # Theme configuration
    └── secrets.toml      # Database credentials (gitignored)
```

## Charts and Visualization

```python
import streamlit as st
import pandas as pd
import numpy as np

# Built-in charts
st.line_chart(df)
st.bar_chart(df)
st.area_chart(df)
st.scatter_chart(df, x="col1", y="col2", color="category")

# Maps
st.map(df)  # Requires 'lat' and 'lon' columns

# External libraries
import plotly.express as px
fig = px.scatter(df, x="x", y="y", color="category")
st.plotly_chart(fig)

import altair as alt
chart = alt.Chart(df).mark_circle().encode(x='x', y='y')
st.altair_chart(chart)
```

## Forms and Callbacks

```python
import streamlit as st

# Forms batch inputs (single rerun on submit)
with st.form("my_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    submitted = st.form_submit_button("Submit")

if submitted:
    st.success(f"Registered {name} with {email}")

# Callbacks execute before script reruns
def on_change():
    st.session_state.processed = process(st.session_state.input_value)

st.text_input("Input", key="input_value", on_change=on_change)
```

## Progress and Status

```python
import streamlit as st
import time

# Progress bar
progress = st.progress(0)
for i in range(100):
    progress.progress(i + 1)
    time.sleep(0.01)

# Status messages
st.success("Operation completed!")
st.error("Something went wrong")
st.warning("Check your input")
st.info("Processing...")

# Spinner
with st.spinner("Loading..."):
    time.sleep(2)
st.success("Done!")

# Empty placeholder for dynamic updates
placeholder = st.empty()
placeholder.text("Waiting...")
# Later...
placeholder.text("Updated!")
```

## File Handling

```python
import streamlit as st

# File upload
uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df)

# File download
csv = df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="data.csv",
    mime="text/csv"
)
```

## Theming

Configure in `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## Static Files

For direct URL access to files:
```
my_app/
├── static/
│   └── logo.png     # Accessible at /app/static/logo.png
└── streamlit_app.py
```

Enable in config:
```toml
[server]
enableStaticServing = true
```

## Running Apps

```bash
# Development
streamlit run app.py

# With arguments
streamlit run app.py -- --data-path ./data

# Configuration
streamlit run app.py --server.port 8080 --server.headless true
```

## Best Practices

1. **Minimize reruns**: Use `@st.cache_data` for expensive operations
2. **Preserve state**: Use `st.session_state` for user data that should persist
3. **Batch inputs**: Use `st.form` when multiple inputs should submit together
4. **Structure large apps**: Use multipage navigation with `st.Page` and `st.navigation`
5. **Secure secrets**: Store credentials in `.streamlit/secrets.toml`, never in code
6. **Responsive layout**: Use `st.columns` and `st.sidebar` for organized UIs

## Reference Files

- [references/widgets-catalog.md](references/widgets-catalog.md) - Complete widget reference: text inputs, selection, numeric, date/time, buttons, callbacks
- [references/layout-patterns.md](references/layout-patterns.md) - Advanced layouts: columns, containers, tabs, dialogs, dashboard examples
- [references/state-patterns.md](references/state-patterns.md) - State patterns: forms, multi-step wizards, authentication, cache vs session state
