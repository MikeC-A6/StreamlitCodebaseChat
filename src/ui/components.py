import streamlit as st
from typing import List, Optional
from ..services.base import RetrievalService
from ..utils.logging import setup_logger
import asyncio

logger = setup_logger(__name__)

def render_header():
    st.title("Pinecone Vector Search Interface")
    st.markdown("""
    <div class="main-title">
        Search through your vector database using natural language queries
    </div>
    """, unsafe_allow_html=True)

def render_search_form(retrieval_service: RetrievalService):
    st.markdown("### Search")
    with st.form("search_form"):
        query = st.text_input("Enter your search query")
        col1, col2 = st.columns(2)
        with col1:
            k = st.number_input("Number of results", min_value=1, max_value=10, value=3)
        with col2:
            namespaces = st.multiselect(
                "Select namespaces",
                options=["repo_githubcloner"],  # Add more as needed
                default=["repo_githubcloner"]
            )

        submitted = st.form_submit_button("Search")
        if submitted and query:
            try:
                results = asyncio.run(
                    retrieval_service.execute(
                        query=query,
                        k=k,
                        namespaces=namespaces
                    )
                )
                st.session_state.search_results = results
                st.session_state.error_message = None
            except Exception as e:
                st.session_state.error_message = str(e)
                st.session_state.search_results = None

def render_results():
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        return

    if st.session_state.search_results:
        st.markdown("### Results")
        documents = st.session_state.search_results.get("documents", [])

        if not documents:
            st.warning("No results found.")
            return

        for i, doc in enumerate(documents, 1):
            with st.container():
                st.markdown(f"#### Result {i}")
                st.markdown(f"**Content:**\n{doc['content']}")

                with st.expander("Metadata"):
                    st.json(doc['metadata'])

                st.markdown(f"**Namespace:** {doc['namespace']}")

                if "github_url" in doc:
                    st.markdown(f"[View on GitHub]({doc['github_url']})")

                st.markdown("---")
