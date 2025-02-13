import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
from src.services.pinecone_service import PineconeService
from src.services.retrieval_service import RetrievalToolService
from src.ui.components import render_header, render_chat_interface
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def init_session_state():
    """Initialize session state variables."""
    if 'pinecone_service' not in st.session_state:
        st.session_state.pinecone_service = None
    if 'retrieval_service' not in st.session_state:
        st.session_state.retrieval_service = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None

def initialize_services():
    """Initialize services if not already initialized."""
    try:
        if not st.session_state.pinecone_service:
            logger.info("Attempting to initialize PineconeService...")
            pinecone_service = PineconeService()
            st.session_state.pinecone_service = pinecone_service
            st.session_state.retrieval_service = RetrievalToolService(pinecone_service)
            logger.info("Services initialized successfully")
            return True
        else:
            logger.info("Services already initialized")
            return True
    except Exception as e:
        error_msg = f"Failed to initialize services: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)
        return False

def main():
    st.set_page_config(
        page_title="AI Chat with Vector Search",
        page_icon="🤖",
        layout="wide"
    )

    # Load custom CSS
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Initialize services
    services_initialized = initialize_services()

    if services_initialized:
        # Render UI components
        render_header()
        render_chat_interface(st.session_state.retrieval_service)
    else:
        st.error("Could not initialize services. Please check the logs for details.")

if __name__ == "__main__":
    main()