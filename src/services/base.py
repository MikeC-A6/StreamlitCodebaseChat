from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorService(ABC):
    @abstractmethod
    async def similarity_search(self, query: str, k: int = 2, namespaces: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Execute a similarity search query."""
        pass

class RetrievalService(ABC):
    @abstractmethod
    async def execute(self, query: str, k: int, namespaces: List[str]) -> Dict[str, Any]:
        """Execute a retrieval query and return formatted results."""
        pass
