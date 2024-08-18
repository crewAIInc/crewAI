import streamlit as st
from app import run_crew, run_postmortem

st.set_page_config(page_title="AI Analysis", page_icon="🤖", layout="wide")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'postmortem_complete' not in st.session_state:
    st.session_state.postmortem_complete = False
if 'postmortem_result' not in st.session_state:
    st.session_state.postmortem_result = None

st.title("AI Analysis")

# Text input for user's custom request
user_request = st.text_area("Enter your request for analysis:", 
                            "Analyze the latest advancements in AI in 2024. Identify key trends, breakthrough technologies, and potential industry impacts.")

if st.button("Run Analysis") or (user_request and not st.session_state.analysis_complete):
    if user_request.strip() == "":
        st.warning("Please enter a request for analysis.")
    else:
        with st.spinner("Running analysis... This may take a few minutes."):
            st.session_state.analysis_result = run_crew(user_request)
        st.session_state.analysis_complete = True

if st.session_state.analysis_complete:
    st.success("Analysis complete!")
    
    st.header("Analysis and Blog Post")
    st.markdown(str(st.session_state.analysis_result))

    # New section for postmortem analysis
    st.header("Postmortem Analysis")
    postmortem_request = st.text_area("Enter your request for postmortem analysis:", 
                                      "Conduct a postmortem on the team's performance. How did we do and what could we improve for next time?")

    if st.button("Run Postmortem") or (postmortem_request and not st.session_state.postmortem_complete):
        if postmortem_request.strip() == "":
            st.warning("Please enter a request for postmortem analysis.")
        else:
            with st.spinner("Running postmortem analysis..."):
                st.session_state.postmortem_result = run_postmortem(postmortem_request, str(st.session_state.analysis_result))
            st.session_state.postmortem_complete = True

    if st.session_state.postmortem_complete:
        st.success("Postmortem analysis complete!")
        st.markdown(str(st.session_state.postmortem_result))

st.sidebar.header("About")
st.sidebar.info(
    "This app uses CrewAI to analyze topics based on your custom request, generate a blog post about the findings, and conduct a postmortem analysis on the team's performance."
)

# Add a reset button
if st.button("Reset"):
    st.session_state.analysis_complete = False
    st.session_state.analysis_result = None
    st.session_state.postmortem_complete = False
    st.session_state.postmortem_result = None
    st.experimental_rerun()