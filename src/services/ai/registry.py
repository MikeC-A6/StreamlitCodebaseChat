from typing import Dict, Type, List
from .base import AITool

class ToolRegistry:
    """Registry for managing available AI tools."""
    
    _instance = None
    _tools: Dict[str, AITool] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, tool: AITool) -> None:
        """Register a new tool."""
        if not isinstance(tool, AITool):
            raise TypeError(f"Tool must implement AITool protocol, got {type(tool)}")
        cls._tools[tool.name] = tool

    @classmethod
    def get_tool(cls, name: str) -> AITool:
        """Get a tool by name."""
        return cls._tools.get(name)

    @classmethod
    def get_all_tools(cls) -> List[AITool]:
        """Get all registered tools."""
        return list(cls._tools.values())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools."""
        cls._tools.clear()
