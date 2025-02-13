from typing import Dict, Any, List
from src.services.base import RetrievalService, VectorService
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class RetrievalToolService(RetrievalService):
    def __init__(self, vector_service: VectorService):
        logger.info("Initializing RetrievalToolService")
        self.vector_service = vector_service

    async def execute(self, query: str, k: int, namespaces: List[str]) -> Dict[str, Any]:
        logger.info(f"Executing retrieval query: '{query}' with k={k}, namespaces={namespaces}")

        # Get relevant documents from vector store
        try:
            results = await self.vector_service.similarity_search(
                query=query,
                k=k,
                namespaces=namespaces
            )
            logger.info(f"Vector search returned {len(results)} results")

            documents = []

            for idx, r in enumerate(results, 1):
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
                logger.debug(f"Processed result {idx}: score={r['score']}, namespace={r['namespace']}")

            logger.info(f"Successfully processed {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Error during retrieval execution: {str(e)}")
            raise