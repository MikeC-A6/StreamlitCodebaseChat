from typing import Dict, Any, List
from openai import AsyncOpenAI
from src.services.base import RetrievalService, VectorService
from src.utils.logging import setup_logger
import os

logger = setup_logger(__name__)

class RetrievalToolService(RetrievalService):
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def execute(self, query: str, k: int, namespaces: List[str]) -> Dict[str, Any]:
        # Get relevant documents from vector store
        results = await self.vector_service.similarity_search(
            query=query,
            k=k,
            namespaces=namespaces
        )

        documents = []
        context_parts = []

        for r in results:
            content = r["metadata"].get("text", "")
            doc = {
                "content": content,
                "metadata": r["metadata"],
                "namespace": r["namespace"],
                "score": r["score"]
            }
            if "github_url" in r["metadata"]:
                doc["github_url"] = r["metadata"]["github_url"]

            documents.append(doc)
            context_parts.append(f"Content: {content}\nSource: {r['metadata'].get('github_url', 'Unknown')}")

        # Prepare context for the AI
        context = "\n\n".join(context_parts)

        # Construct the prompt
        system_prompt = """You are a helpful AI assistant that answers questions about code repositories. 
        Use the provided context to answer questions accurately and concisely. 
        If you're not sure about something, say so. Always reference the source files when possible."""

        user_prompt = f"""Context from codebase:
        {context}

        Question: {query}

        Please provide a clear and concise answer based on the context above."""

        # Get AI completion
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using the specified model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )

            answer = response.choices[0].message.content

            return {
                "documents": documents,
                "answer": answer
            }
        except Exception as e:
            logger.error(f"Error in AI completion: {str(e)}")
            raise