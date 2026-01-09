# Streamlit Widgets Catalog

Complete reference for all Streamlit input widgets and their usage patterns.

## Contents

- [Text Input Widgets](#text-input-widgets)
- [Selection Widgets](#selection-widgets)
- [Numeric Widgets](#numeric-widgets)
- [Date and Time](#date-and-time)
- [Media and Files](#media-and-files)
- [Buttons and Actions](#buttons-and-actions)
- [Widget Keys and Session State](#widget-keys-and-session-state)
- [Widget Callbacks](#widget-callbacks)
- [Disabled and Label Visibility](#disabled-and-label-visibility)
- [Help Text](#help-text)

## Text Input Widgets

```python
import streamlit as st

# Single-line text
name = st.text_input("Name", value="", placeholder="Enter name...")

# Multi-line text
bio = st.text_area("Bio", height=150)

# Number input
age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)

# Password (masked)
password = st.text_input("Password", type="password")
```

## Selection Widgets

```python
import streamlit as st

# Dropdown
option = st.selectbox("Choose one", ["A", "B", "C"], index=0)

# Multi-select
options = st.multiselect("Choose many", ["A", "B", "C"], default=["A"])

# Radio buttons
choice = st.radio("Pick one", ["Option 1", "Option 2"], horizontal=True)

# Checkbox
agree = st.checkbox("I agree", value=False)

# Toggle
enabled = st.toggle("Enable feature")
```

## Numeric Widgets

```python
import streamlit as st

# Slider (single value)
value = st.slider("Value", min_value=0, max_value=100, value=50)

# Range slider
low, high = st.slider("Range", 0, 100, (25, 75))

# Select slider (discrete values)
size = st.select_slider("Size", options=["S", "M", "L", "XL"])
```

## Date and Time

```python
import streamlit as st
from datetime import date, time, datetime

# Date picker
d = st.date_input("Date", value=date.today())

# Date range
start, end = st.date_input("Date range", value=(date(2024, 1, 1), date.today()))

# Time picker
t = st.time_input("Time", value=time(12, 0))
```

## Media and Files

```python
import streamlit as st

# File uploader
file = st.file_uploader("Upload", type=["csv", "xlsx", "pdf"])
files = st.file_uploader("Upload many", accept_multiple_files=True)

# Camera input
photo = st.camera_input("Take a photo")

# Color picker
color = st.color_picker("Pick color", "#FF0000")
```

## Buttons and Actions

```python
import streamlit as st

# Standard button
if st.button("Click me", type="primary"):
    st.write("Clicked!")

# Download button
st.download_button(
    label="Download",
    data=csv_data,
    file_name="data.csv",
    mime="text/csv"
)

# Link button
st.link_button("Go to docs", "https://docs.streamlit.io")

# Form submit button (only inside forms)
with st.form("form"):
    st.text_input("Name")
    st.form_submit_button("Submit")
```

## Widget Keys and Session State

Every widget can have a `key` parameter linking it to session state:

```python
import streamlit as st

# Widget with key
st.text_input("Name", key="user_name")

# Access via session_state
if st.session_state.user_name:
    st.write(f"Hello, {st.session_state.user_name}")

# Programmatically set widget value
if st.button("Reset"):
    st.session_state.user_name = ""
```

## Widget Callbacks

Execute code when widgets change:

```python
import streamlit as st

def on_name_change():
    # Runs BEFORE the rest of the script
    st.session_state.greeting = f"Hello, {st.session_state.name}!"

st.text_input("Name", key="name", on_change=on_name_change)

# For buttons
def on_click():
    st.session_state.counter += 1

st.button("Increment", on_click=on_click)
```

## Disabled and Label Visibility

```python
import streamlit as st

# Disable widget
st.text_input("Locked", disabled=True)

# Hide label (for custom layouts)
st.text_input("Hidden label", label_visibility="hidden")

# Collapse label
st.text_input("Collapsed", label_visibility="collapsed")
```

## Help Text

```python
import streamlit as st

st.text_input(
    "API Key",
    help="Find your API key in the settings page"
)
```
