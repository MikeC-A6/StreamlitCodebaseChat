import streamlit as st
import asyncio
import json
from typing import Optional, Dict, Any
import logging

# Import the PineconeService and RetrievalTool from the provided code
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

# PineconeService class (from the provided code)
class PineconeService:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, pinecone_index: str):
        try:
            logger.info("Initializing PineconeService")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=openai_api_key
            )

            self.dimension = 1536  # For 'text-embedding-3-small'
            self.pc = Pinecone(api_key=pinecone_api_key)

            try:
                self.index = self.pc.Index(pinecone_index)
                logger.info(f"Connected to existing Pinecone index: {pinecone_index}")
            except Exception:
                logger.info(f"Index {pinecone_index} not found, creating new index...")
                self.pc.create_index(
                    name=pinecone_index,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                self.index = self.pc.Index(pinecone_index)
                logger.info(f"Created new Pinecone index: {pinecone_index}")

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

# Streamlit UI components
def render_header():
    st.title("Pinecone Vector Search Interface")
    st.markdown("""
    <div class="main-title">
        Search through your vector database using natural language queries
    </div>
    """, unsafe_allow_html=True)

def render_credentials_form():
    with st.expander("API Credentials", expanded=not st.session_state.pinecone_service):
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password")
        pinecone_index = st.text_input("Pinecone Index Name")
        
        if st.button("Initialize Service"):
            try:
                st.session_state.pinecone_service = PineconeService(
                    openai_api_key=openai_api_key,
                    pinecone_api_key=pinecone_api_key,
                    pinecone_index=pinecone_index
                )
                st.session_state.retrieval_tool = RetrievalTool(st.session_state.pinecone_service)
                st.success("Service initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize service: {str(e)}")
                st.session_state.pinecone_service = None
                st.session_state.retrieval_tool = None

def render_search_form():
    if not st.session_state.pinecone_service:
        st.warning("Please initialize the service with your API credentials first.")
        return

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

    # Render UI components
    render_header()
    render_credentials_form()
    render_search_form()
    render_results()

if __name__ == "__main__":
    main()
