from typing import TypedDict, List, Optional, Dict, Any

class SearchResult(TypedDict):
    score: float
    metadata: Dict[str, Any]
    namespace: str

class SearchResponse(TypedDict):
    documents: List[Dict[str, Any]]
    formatted_content: str
