# Streamlit Layout Patterns

Advanced layout techniques for building professional Streamlit apps.

## Contents

- [Column Layouts](#column-layouts)
- [Container Patterns](#container-patterns)
- [Expander](#expander)
- [Tabs](#tabs)
- [Sidebar](#sidebar)
- [Empty Placeholders](#empty-placeholders)
- [Popover](#popover)
- [Dialog (Modal)](#dialog-modal)
- [Page Configuration](#page-configuration)
- [Dashboard Layout Example](#dashboard-layout-example)

## Column Layouts

```python
import streamlit as st

# Equal columns
col1, col2, col3 = st.columns(3)

# Weighted columns
left, right = st.columns([2, 1])  # 2:1 ratio

# With gap control
cols = st.columns(3, gap="large")  # small, medium, large

# Nested columns
outer1, outer2 = st.columns(2)
with outer1:
    inner1, inner2 = st.columns(2)
    with inner1:
        st.metric("A", 100)
    with inner2:
        st.metric("B", 200)
```

## Container Patterns

```python
import streamlit as st

# Basic container
with st.container():
    st.write("Grouped content")

# Container with border
with st.container(border=True):
    st.write("Boxed content")

# Fixed height with scrolling
with st.container(height=300):
    for i in range(50):
        st.write(f"Line {i}")
```

## Expander

```python
import streamlit as st

# Basic expander
with st.expander("See details"):
    st.write("Hidden content here")

# Start expanded
with st.expander("FAQ", expanded=True):
    st.write("Answer to question")
```

## Tabs

```python
import streamlit as st

tab1, tab2, tab3 = st.tabs(["Data", "Chart", "Settings"])

with tab1:
    st.dataframe(df)

with tab2:
    st.line_chart(df)

with tab3:
    st.slider("Option", 0, 100)
```

## Sidebar

```python
import streamlit as st

# Using 'with' syntax
with st.sidebar:
    st.title("Controls")
    option = st.selectbox("Filter", ["All", "Active", "Archived"])

# Using object notation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data", "About"])
```

## Empty Placeholders

```python
import streamlit as st
import time

# Dynamic content updates
placeholder = st.empty()

# Replace content
placeholder.text("Loading...")
time.sleep(1)
placeholder.text("Still loading...")
time.sleep(1)
placeholder.success("Done!")

# Clear content
placeholder.empty()

# Use as container
with placeholder.container():
    st.write("Multiple elements")
    st.write("In a placeholder")
```

## Popover

```python
import streamlit as st

with st.popover("Settings"):
    st.checkbox("Enable notifications")
    st.slider("Volume", 0, 100)
```

## Dialog (Modal)

```python
import streamlit as st

@st.dialog("Confirm Delete")
def confirm_delete(item_name):
    st.write(f"Are you sure you want to delete {item_name}?")
    col1, col2 = st.columns(2)
    if col1.button("Cancel"):
        st.rerun()
    if col2.button("Delete", type="primary"):
        # Perform deletion
        st.session_state.deleted = item_name
        st.rerun()

if st.button("Delete Item"):
    confirm_delete("My Document")
```

## Page Configuration

```python
import streamlit as st

# Must be first Streamlit command
st.set_page_config(
    page_title="My App",
    page_icon="🎯",
    layout="wide",          # or "centered"
    initial_sidebar_state="expanded",  # or "collapsed", "auto"
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': "https://example.com/bug",
        'About': "# My App\nBuilt with Streamlit"
    }
)
```

## Dashboard Layout Example

```python
import streamlit as st

st.set_page_config(layout="wide")

# Header row
st.title("Dashboard")

# Metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Users", "1,234", "+12%")
m2.metric("Revenue", "$12.3K", "+8%")
m3.metric("Orders", "456", "-3%")
m4.metric("Rating", "4.8", "+0.2")

st.divider()

# Main content with sidebar
with st.sidebar:
    date_range = st.date_input("Date Range", [])
    category = st.multiselect("Category", ["A", "B", "C"])

# Two-column content
chart_col, data_col = st.columns([2, 1])

with chart_col:
    st.subheader("Trend")
    st.line_chart(trend_data)

with data_col:
    st.subheader("Top Items")
    st.dataframe(top_items, hide_index=True)

# Tabbed details
tab1, tab2 = st.tabs(["Details", "Settings"])
with tab1:
    st.write("Detailed view")
with tab2:
    st.write("Configuration options")
```
