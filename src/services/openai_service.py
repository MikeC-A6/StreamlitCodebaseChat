from typing import Dict, Any, List
from openai import AsyncOpenAI
import json
import os
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class OpenAIService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # Define the function for retrieving context
        self.tools = [{
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search the knowledge base to find relevant documentation and code snippets that help answer the question.",
                "parameters": {
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
            }
        }]

    async def get_response(self, query: str, retrieval_function, k: int, namespaces: List[str]) -> Dict[str, Any]:
        try:
            # First message to evaluate if search is needed
            completion = await self.client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model released May 13, 2024
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful AI assistant for answering questions about code repositories. 
                        For any user question, ALWAYS use the search_knowledge_base function first to gather relevant context 
                        before providing an answer. This is crucial for accurate responses."""
                    },
                    {"role": "user", "content": query}
                ],
                tools=self.tools,
                tool_choice={"type": "function", "function": {"name": "search_knowledge_base"}}  # Force function call
            )

            response_message = completion.choices[0].message
            logger.info(f"Initial model response: {response_message}")

            # The model should always make a tool call due to tool_choice
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]

                # Parse the function arguments
                args = json.loads(tool_call.function.arguments)
                logger.info(f"Function call arguments: {args}")

                # Call the retrieval function
                search_results = await retrieval_function(
                    query=args["query"],
                    k=args["options"]["num_results"],
                    namespaces=args["options"]["namespaces"]
                )

                # Send the results back to the model
                messages = [
                    {
                        "role": "system",
                        "content": """You are a helpful AI assistant for answering questions about code repositories. 
                        Analyze the search results and provide a clear, detailed response. Always reference specific 
                        files and code snippets when relevant."""
                    },
                    {"role": "user", "content": query},
                    response_message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(search_results)
                    }
                ]

                # Get final response from the model
                final_completion = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )

                return {
                    "answer": final_completion.choices[0].message.content,
                    "documents": search_results
                }
            else:
                # This shouldn't happen due to tool_choice, but handle it just in case
                logger.error("Model did not make expected function call")
                return {
                    "answer": "I apologize, but I was unable to search the codebase. Please try again.",
                    "documents": []
                }

        except Exception as e:
            logger.error(f"Error in OpenAI service: {str(e)}")
            raise