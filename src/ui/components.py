import streamlit as st
from typing import List, Optional
from src.services.base import RetrievalService
from src.services.openai_service import OpenAIService
from src.utils.logging import setup_logger
import asyncio

logger = setup_logger(__name__)

def render_header():
    st.title("AI Chat Interface with Vector Search")
    st.markdown("""
    Chat with your codebase using AI and vector search
    """, unsafe_allow_html=True)

def init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def render_chat_interface(openai_service: OpenAIService, retrieval_service: RetrievalService):
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
            if message.get("sources"):
                st.markdown("**Sources:**")
                for doc in message["sources"]:
                    if doc.get("github_url"):
                        score = doc.get("score", 0)
                        score_percentage = f"{score * 100:.1f}%" if score else "N/A"
                        st.markdown(
                            f"- [{doc['content'][:100]}...]({doc['github_url']}) "
                            f"(Relevance: {score_percentage})"
                        )

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
                    # Use OpenAI service with retrieval function
                    response = asyncio.run(
                        openai_service.get_response(
                            query=prompt,
                            retrieval_function=retrieval_service.execute,
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
                            if doc.get("github_url"):
                                score = doc.get("score", 0)
                                score_percentage = f"{score * 100:.1f}%" if score else "N/A"
                                st.markdown(
                                    f"- [{doc['content'][:100]}...]({doc['github_url']}) "
                                    f"(Relevance: {score_percentage})"
                                )
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)