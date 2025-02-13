from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

@runtime_checkable
class AITool(Protocol):
    """Protocol defining the interface for AI tools."""
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool definition to dictionary format for AI provider."""
        pass

@dataclass
class ToolCall:
    """Represents a tool call from the AI."""
    id: str
    name: str
    arguments: Dict[str, Any]

class AIResponse:
    """Encapsulates a response from an AI service."""
    def __init__(self, content: str, tool_calls: Optional[List[ToolCall]] = None):
        self.content = content
        self.tool_calls = tool_calls or []

class AIService(ABC):
    """Base class for AI service implementations."""
    
    @abstractmethod
    async def get_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[AITool]] = None
    ) -> AIResponse:
        """Get a response from the AI service."""
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for the given text."""
        pass
