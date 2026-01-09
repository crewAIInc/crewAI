# Streamlit State Management Patterns

Best practices for managing state in Streamlit applications.

## Contents

- [Session State Basics](#session-state-basics)
- [Widget-State Binding](#widget-state-binding)
- [Callback Pattern](#callback-pattern)
- [Form State Management](#form-state-management)
- [Multi-Step Wizard](#multi-step-wizard)
- [Data Loading with State](#data-loading-with-state)
- [Authentication State](#authentication-state)
- [Cache vs Session State](#cache-vs-session-state)

## Session State Basics

```python
import streamlit as st

# Check and initialize
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.user = None
    st.session_state.data = []
    st.session_state.settings = {"theme": "light"}

# Access patterns
value = st.session_state.key           # Attribute access
value = st.session_state["key"]        # Dict access
value = st.session_state.get("key", default)  # Safe access

# Update patterns
st.session_state.key = new_value
st.session_state["key"] = new_value
```

## Widget-State Binding

```python
import streamlit as st

# Widgets with keys auto-sync to session_state
st.text_input("Name", key="name")
st.number_input("Age", key="age")

# Values persist across reruns
st.write(f"Name: {st.session_state.name}, Age: {st.session_state.age}")

# Update widget value programmatically
if st.button("Clear"):
    st.session_state.name = ""
    st.session_state.age = 0
```

## Callback Pattern

```python
import streamlit as st

# Callbacks run BEFORE the main script
def update_total():
    st.session_state.total = (
        st.session_state.quantity * st.session_state.price
    )

st.number_input("Quantity", key="quantity", on_change=update_total)
st.number_input("Price", key="price", on_change=update_total)

# Total is always up-to-date
if "total" in st.session_state:
    st.metric("Total", f"${st.session_state.total:.2f}")
```

## Form State Management

```python
import streamlit as st

# Initialize form data
if "form_data" not in st.session_state:
    st.session_state.form_data = {
        "name": "",
        "email": "",
        "submitted": False
    }

with st.form("registration"):
    name = st.text_input("Name", value=st.session_state.form_data["name"])
    email = st.text_input("Email", value=st.session_state.form_data["email"])

    if st.form_submit_button("Submit"):
        st.session_state.form_data = {
            "name": name,
            "email": email,
            "submitted": True
        }

if st.session_state.form_data["submitted"]:
    st.success(f"Registered: {st.session_state.form_data['name']}")
```

## Multi-Step Wizard

```python
import streamlit as st

# Track wizard state
if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.wizard_data = {}

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

# Step 1
if st.session_state.step == 1:
    st.header("Step 1: Basic Info")
    name = st.text_input("Name", key="wizard_name")
    if st.button("Next", on_click=next_step):
        st.session_state.wizard_data["name"] = name

# Step 2
elif st.session_state.step == 2:
    st.header("Step 2: Details")
    details = st.text_area("Details", key="wizard_details")
    col1, col2 = st.columns(2)
    col1.button("Back", on_click=prev_step)
    if col2.button("Next", on_click=next_step):
        st.session_state.wizard_data["details"] = details

# Step 3
elif st.session_state.step == 3:
    st.header("Step 3: Confirm")
    st.write(st.session_state.wizard_data)
    st.button("Back", on_click=prev_step)
    if st.button("Submit"):
        # Process data
        st.session_state.step = 1
        st.session_state.wizard_data = {}
```

## Data Loading with State

```python
import streamlit as st
import pandas as pd

# Preserve loaded data across reruns
if "data" not in st.session_state:
    st.session_state.data = None
    st.session_state.data_loaded = False

uploaded = st.file_uploader("Upload CSV")

if uploaded and not st.session_state.data_loaded:
    st.session_state.data = pd.read_csv(uploaded)
    st.session_state.data_loaded = True

if st.session_state.data is not None:
    # Data persists even after file uploader clears
    st.dataframe(st.session_state.data)

    # Filter without reloading
    col = st.selectbox("Filter column", st.session_state.data.columns)
    value = st.text_input("Filter value")

    if value:
        filtered = st.session_state.data[
            st.session_state.data[col].astype(str).str.contains(value)
        ]
        st.dataframe(filtered)
```

## Authentication State

```python
import streamlit as st

# Auth state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None

def login(username, password):
    # Validate credentials
    if username == "admin" and password == "secret":
        st.session_state.authenticated = True
        st.session_state.user = {"username": username, "role": "admin"}
        return True
    return False

def logout():
    st.session_state.authenticated = False
    st.session_state.user = None

# Login form
if not st.session_state.authenticated:
    st.title("Login")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if not login(username, password):
                st.error("Invalid credentials")
else:
    # Protected content
    st.title(f"Welcome, {st.session_state.user['username']}")
    st.button("Logout", on_click=logout)
```

## Cache vs Session State

```python
import streamlit as st

# CACHE: Shared across all users, tied to function inputs
@st.cache_data
def load_global_config():
    """Same result for everyone"""
    return load_from_database()

# SESSION STATE: Per-user, per-session
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {}

# Use cache for:
# - Expensive computations
# - Data that doesn't change per user
# - API calls with same parameters

# Use session state for:
# - User-specific data
# - Form inputs
# - Navigation state
# - Shopping carts, selections
```
