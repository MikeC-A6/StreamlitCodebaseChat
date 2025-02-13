from typing import Dict, Any
from dataclasses import dataclass, asdict

from ..base import AITool

@dataclass
class SearchKnowledgeBaseTool(AITool):
    """Tool for searching the knowledge base."""
    name: str = "search_knowledge_base"
    description: str = "Search the knowledge base to find relevant documentation and code snippets that help answer the question."
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documentation"
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "num_results": {
                                "type": "number",
                                "description": "Number of results to return"
                            },
                            "namespaces": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of namespaces to search in"
                            }
                        },
                        "required": ["num_results", "namespaces"]
                    }
                },
                "required": ["query", "options"]
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
