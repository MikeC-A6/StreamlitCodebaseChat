from typing import Dict, Any, List
import json
from src.utils.logging import setup_logger
from src.services.ai.providers.openai_provider import OpenAIProvider
from src.services.ai.tools.knowledge_base import SearchKnowledgeBaseTool
from src.services.ai.registry import ToolRegistry

logger = setup_logger(__name__)

class OpenAIService:
    def __init__(self):
        logger.info("Initializing OpenAIService")
        self.provider = OpenAIProvider()

        # Register tools
        self.registry = ToolRegistry()
        self.registry.register(SearchKnowledgeBaseTool())
        logger.info("OpenAIService initialized with tools configuration")

    async def get_response(self, query: str, retrieval_function, k: int, namespaces: List[str]) -> Dict[str, Any]:
        try:
            logger.info(f"Processing query: '{query}' with k={k}, namespaces={namespaces}")

            # Get AI response with knowledge base tool
            response = await self.provider.get_response(
                query=query,
                context={
                    "system_message": f"""You are a helpful AI assistant for answering questions about code repositories. 
                    Always use the search_knowledge_base function to gather relevant context before providing an answer.
                    You must use these exact namespaces for searching: {namespaces}
                    You must use this exact number of results: {k}"""
                },
                tools=self.registry.get_all_tools()
            )

            # Handle tool calls
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                logger.info(f"Tool call received: {tool_call}")

                # Parse the function arguments
                args = tool_call.arguments
                logger.info(f"Function call arguments: {args}")

                # Override the options to ensure correct values
                args["options"]["num_results"] = k
                args["options"]["namespaces"] = namespaces

                # Call the retrieval function
                logger.info("Calling retrieval function with parsed arguments")
                search_results = await retrieval_function(
                    query=args["query"],
                    k=k,
                    namespaces=namespaces
                )
                logger.info(f"Retrieved {len(search_results)} results from vector store")

                # Format search results for better context
                context_str = "Here are the relevant code snippets and documentation:\n\n"
                for idx, doc in enumerate(search_results, 1):
                    content = doc.get('metadata', {}).get('content', '')
                    score = doc.get('score', 0)
                    url = doc.get('metadata', {}).get('github_url', '')
                    context_str += f"Document {idx} (Relevance: {score:.2f}):\n"
                    context_str += f"Source: {url}\n"
                    context_str += f"Content: {content}\n\n"

                # Get final response with search results
                final_response = await self.provider.get_response(
                    query=query,
                    context={
                        "system_message": """You are a helpful AI assistant for answering questions about code repositories. 
                        Based on the provided search results, give a detailed and accurate answer. Reference specific 
                        files, code snippets, and documentation in your response. Include relevant GitHub URLs when available.""",
                        "search_results": context_str
                    },
                    tools=None  # No tools needed for final response
                )

                return {
                    "answer": final_response.content,
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