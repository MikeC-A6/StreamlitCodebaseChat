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
                "description": "Search the knowledge base to find relevant documentation and code snippets",
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
            # First, ask the model if it needs to search the knowledge base
            completion = await self.client.chat.completions.create(
                model="gpt-4o",  # The newest OpenAI model released May 13, 2024
                messages=[{
                    "role": "user",
                    "content": query
                }],
                tools=self.tools
            )

            response_message = completion.choices[0].message

            # If the model wants to search the knowledge base
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                
                # Parse the function arguments
                args = json.loads(tool_call.function.arguments)
                
                # Call the retrieval function
                search_results = await retrieval_function(
                    query=args["query"],
                    k=args["options"]["num_results"],
                    namespaces=args["options"]["namespaces"]
                )

                # Send the results back to the model
                messages = [
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
                # Model didn't need to search, return direct response
                return {
                    "answer": response_message.content,
                    "documents": []
                }

        except Exception as e:
            logger.error(f"Error in OpenAI service: {str(e)}")
            raise
