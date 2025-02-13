from typing import List, Dict, Any, Optional
import pinecone
from langchain_openai import OpenAIEmbeddings
from src.services.base import VectorService
from src.utils.logging import setup_logger
from src.config.settings import settings

logger = setup_logger(__name__)

class PineconeServiceError(Exception):
    pass

class PineconeService(VectorService):
    def __init__(self):
        try:
            logger.info("Initializing PineconeService")
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY
            )

            self.pc = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)

            try:
                self.index = self.pc.Index(settings.PINECONE_INDEX)
                logger.info(f"Connected to existing Pinecone index: {settings.PINECONE_INDEX}")
            except Exception as e:
                error_msg = f"Failed to connect to Pinecone index: {str(e)}"
                logger.error(error_msg)
                raise PineconeServiceError(error_msg)

            stats = self.index.describe_index_stats()
            logger.info(f"Total vectors in index: {stats.total_vector_count}")

        except Exception as e:
            error_msg = f"Failed to initialize Pinecone service: {str(e)}"
            logger.error(error_msg)
            raise PineconeServiceError(error_msg)

    async def similarity_search(self, query: str, k: int = 2, namespaces: Optional[List[str]] = None) -> List[Dict[str, Any]]:
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
                    "metadata": match.metadata,
                    "namespace": "default"
                } for match in response.matches]

        except Exception as e:
            error_msg = f"Error in similarity search: {str(e)}"
            logger.error(error_msg)
            raise PineconeServiceError(error_msg)