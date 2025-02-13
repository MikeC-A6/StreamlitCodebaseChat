import streamlit as st
import asyncio
import json
from typing import Optional, Dict, Any
import logging
import os

# Import the required libraries
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    if 'pinecone_service' not in st.session_state:
        st.session_state.pinecone_service = None
    if 'retrieval_tool' not in st.session_state:
        st.session_state.retrieval_tool = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None

# Custom exception for service errors
class PineconeServiceError(Exception):
    pass

# PineconeService class
class PineconeService:
    def __init__(self):
        try:
            logger.info("Initializing PineconeService")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.environ["OPENAI_API_KEY"]
            )

            self.dimension = 1536  # For 'text-embedding-3-small'
            self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

            try:
                self.index = self.pc.Index(os.environ["PINECONE_INDEX"])
                logger.info(f"Connected to existing Pinecone index: {os.environ['PINECONE_INDEX']}")
            except Exception as e:
                error_msg = f"Failed to connect to Pinecone index: {str(e)}"
                logger.error(error_msg)
                raise PineconeServiceError(error_msg)

            # Log index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Total vectors in index: {stats.total_vector_count}")

        except Exception as e:
            error_msg = f"Failed to initialize Pinecone service: {str(e)}"
            logger.error(error_msg)
            raise PineconeServiceError(error_msg)

    async def similarity_search(self, query: str, k: int = 2, namespaces: Optional[list] = None):
        try:
            query_embedding = self.embeddings.embed_query(query)
            all_results = []

            if namespaces:
                for namespace in namespaces:
                    try:
                        response = self.index.query(
                            vector=query_embedding,
                            top_k=k,
                            include_metadata=True,
                            namespace=namespace
                        )

                        for match in response.matches:
                            result = {
                                "score": match.score,
                                "metadata": match.metadata,
                                "namespace": namespace
                            }
                            all_results.append(result)

                    except Exception as e:
                        logger.error(f"Error searching namespace {namespace}: {str(e)}")
                        continue

                all_results.sort(key=lambda x: float(x["score"]), reverse=True)
                return all_results[:k]
            else:
                response = self.index.query(
                    vector=query_embedding,
                    top_k=k,
                    include_metadata=True
                )
                return [{
                    "score": match.score,
                    "metadata": match.metadata
                } for match in response.matches]

        except Exception as e:
            error_msg = f"Error in similarity search: {str(e)}"
            logger.error(error_msg)
            raise PineconeServiceError(error_msg)

class RetrievalTool:
    def __init__(self, pinecone_service: PineconeService):
        self.pinecone_service = pinecone_service

    async def execute(self, query: str, k: int, namespaces: list) -> Dict[str, Any]:
        results = await self.pinecone_service.similarity_search(
            query=query,
            k=k,
            namespaces=namespaces
        )

        documents = []
        formatted_content_parts = []

        for r in results:
            content = r["metadata"].get("text", "")
            doc = {
                "content": content,
                "metadata": r["metadata"],
                "namespace": r["namespace"]
            }
            if "github_url" in r["metadata"]:
                doc["github_url"] = r["metadata"]["github_url"]

            documents.append(doc)
            formatted_content_parts.append(content)

        formatted_content = "\n---\n".join(formatted_content_parts)
        return {
            "documents": documents,
            "formatted_content": formatted_content
        }

def initialize_services():
    """Initialize PineconeService and RetrievalTool if not already initialized"""
    try:
        if not st.session_state.pinecone_service:
            st.session_state.pinecone_service = PineconeService()
            st.session_state.retrieval_tool = RetrievalTool(st.session_state.pinecone_service)
            logger.info("Services initialized successfully")
            return True
    except Exception as e:
        st.error(f"Failed to initialize services: {str(e)}")
        logger.error(f"Service initialization error: {str(e)}")
        return False

# Streamlit UI components
def render_header():
    st.title("Pinecone Vector Search Interface")
    st.markdown("""
    <div class="main-title">
        Search through your vector database using natural language queries
    </div>
    """, unsafe_allow_html=True)

def render_search_form():
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
                    st.session_state.retrieval_tool.execute(
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

def main():
    st.set_page_config(
        page_title="Pinecone Vector Search",
        page_icon="🔍",
        layout="wide"
    )

    # Load custom CSS
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Initialize services
    if initialize_services():
        # Render UI components
        render_header()
        render_search_form()
        render_results()
    else:
        st.error("Failed to initialize services. Please check your API credentials.")

if __name__ == "__main__":
    main()