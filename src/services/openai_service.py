from typing import Dict, Any, List
from openai import AsyncOpenAI
import json
import os
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class OpenAIService:
    def __init__(self):
        logger.info("Initializing OpenAIService")
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
        logger.info("OpenAIService initialized with tools configuration")

    async def get_response(self, query: str, retrieval_function, k: int, namespaces: List[str]) -> Dict[str, Any]:
        try:
            logger.info(f"Processing query: '{query}' with k={k}, namespaces={namespaces}")

            # First message to evaluate if search is needed
            logger.info("Making initial OpenAI API call with gpt-4o-mini")
            completion = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini as specified
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
                tool_choice={"type": "function", "function": {"name": "search_knowledge_base"}}
            )

            response_message = completion.choices[0].message
            logger.info(f"Initial model response received: {response_message}")

            # The model should always make a tool call due to tool_choice
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                logger.info(f"Tool call received: {tool_call}")

                # Parse the function arguments
                args = json.loads(tool_call.function.arguments)
                logger.info(f"Function call arguments: {args}")

                # Call the retrieval function
                logger.info("Calling retrieval function with parsed arguments")
                search_results = await retrieval_function(
                    query=args["query"],
                    k=args["options"]["num_results"],
                    namespaces=args["options"]["namespaces"]
                )
                logger.info(f"Retrieved {len(search_results)} results from vector store")

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

                logger.info("Making final OpenAI API call to generate response")
                final_completion = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )

                return {
                    "answer": final_completion.choices[0].message.content,
                    "documents": search_results
                }
            else:
                logger.error("Model did not make expected function call")
                return {
                    "answer": "I apologize, but I was unable to search the codebase. Please try again.",
                    "documents": []
                }

        except Exception as e:
            logger.error(f"Error in OpenAI service: {str(e)}")
            raise