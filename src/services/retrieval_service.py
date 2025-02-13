from typing import Dict, Any, List
from src.services.base import RetrievalService, VectorService
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class RetrievalToolService(RetrievalService):
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service

    async def execute(self, query: str, k: int, namespaces: List[str]) -> Dict[str, Any]:
        # Get relevant documents from vector store
        results = await self.vector_service.similarity_search(
            query=query,
            k=k,
            namespaces=namespaces
        )

        documents = []

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

        return documents