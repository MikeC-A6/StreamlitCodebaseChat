from typing import Dict, Any, List, Optional
import json
from openai import AsyncOpenAI
import os

from ..base import AIService, AIResponse, AITool, ToolCall
from ..registry import ToolRegistry
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class OpenAIProvider(AIService):
    """OpenAI implementation of AIService."""

    def __init__(self):
        logger.info("Initializing OpenAIProvider")
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # the newest OpenAI model is "gpt-4o-mini" which was released January 1, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o-mini"
        logger.info(f"OpenAIProvider initialized with model: {self.model}")

    async def get_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[AITool]] = None
    ) -> AIResponse:
        """Get a response from OpenAI."""
        try:
            # Prepare messages
            messages = []

            # Add system message if provided
            if context and context.get("system_message"):
                messages.append({
                    "role": "system",
                    "content": context["system_message"]
                })

            # Add context if available
            if context and context.get("search_results"):
                messages.append({
                    "role": "system",
                    "content": context["search_results"]
                })

            # Add the user's query
            messages.append({"role": "user", "content": query})

            # Prepare tools if provided
            openai_tools = None
            if tools:
                openai_tools = [tool.to_dict() for tool in tools]

            logger.info(f"Making OpenAI API call with model {self.model}")
            logger.debug(f"Messages being sent to OpenAI: {json.dumps(messages, indent=2)}")

            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto" if openai_tools else None
            )

            response_message = completion.choices[0].message
            tool_calls = None

            if response_message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=call.id,
                        name=call.function.name,
                        arguments=json.loads(call.function.arguments)
                    )
                    for call in response_message.tool_calls
                ]

            return AIResponse(
                content=response_message.content,
                tool_calls=tool_calls
            )

        except Exception as e:
            error_msg = f"Error in OpenAI service: {str(e)}"
            logger.error(error_msg)
            raise

    async def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for the given text."""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            error_msg = f"Error getting embedding: {str(e)}"
            logger.error(error_msg)
            raise