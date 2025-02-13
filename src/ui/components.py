import streamlit as st
from typing import List, Optional
from ..services.base import RetrievalService
from ..utils.logging import setup_logger
import asyncio

logger = setup_logger(__name__)

def render_header():
    st.title("AI Chat Interface with Vector Search")
    st.markdown("""
    <div class="main-title">
        Chat with your codebase using AI and vector search
    </div>
    """, unsafe_allow_html=True)

def init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def render_chat_interface(retrieval_service: RetrievalService):
    st.markdown("### Chat")

    # Settings sidebar
    with st.sidebar:
        st.markdown("### Search Settings")
        k = st.number_input("Number of results to retrieve", min_value=1, max_value=10, value=3)
        namespaces = st.multiselect(
            "Select namespaces",
            options=["repo_githubcloner"],  # Add more as needed
            default=["repo_githubcloner"]
        )

    # Initialize chat history
    init_chat_state()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.markdown("**Sources:**")
                for source in message["sources"]:
                    if "github_url" in source:
                        st.markdown(f"- [{source['content'][:100]}...]({source['github_url']})")

    # Chat input
    if prompt := st.chat_input("Ask a question about your codebase"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(
                        retrieval_service.execute(
                            query=prompt,
                            k=k,
                            namespaces=namespaces
                        )
                    )

                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["documents"]
                    })

                    # Display response and sources
                    st.markdown(response["answer"])
                    if response["documents"]:
                        st.markdown("**Sources:**")
                        for doc in response["documents"]:
                            if "github_url" in doc:
                                st.markdown(f"- [{doc['content'][:100]}...]({doc['github_url']})")
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)