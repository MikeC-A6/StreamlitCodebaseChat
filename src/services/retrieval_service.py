from typing import Dict, Any, List
from .base import RetrievalService, VectorService
from ..models.types import SearchResponse
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class RetrievalToolService(RetrievalService):
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service

    async def execute(self, query: str, k: int, namespaces: List[str]) -> SearchResponse:
        results = await self.vector_service.similarity_search(
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
