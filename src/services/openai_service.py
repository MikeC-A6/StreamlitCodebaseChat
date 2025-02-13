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

            # Initial system message for tool use
            system_message = f"""You are a helpful AI assistant for answering questions about code repositories. 
            Always use the search_knowledge_base function to gather relevant context before providing an answer.
            You must use these exact namespaces for searching: {namespaces}
            You must use this exact number of results: {k}
            After receiving search results, analyze them carefully and provide detailed answers referencing the specific code and documentation found."""

            # Get AI response with knowledge base tool
            response = await self.provider.get_response(
                query=query,
                context={"system_message": system_message},
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
                context_str = "Here are the relevant code snippets and documentation from the repository:\n\n"
                for idx, doc in enumerate(search_results, 1):
                    content = doc.get('metadata', {}).get('content', '')
                    score = doc.get('score', 0)
                    url = doc.get('metadata', {}).get('github_url', '')
                    namespace = doc.get('namespace', 'default')

                    context_str += f"Document {idx} [Relevance Score: {score:.2f}]\n"
                    context_str += f"Location: {url}\n"
                    context_str += f"Namespace: {namespace}\n"
                    context_str += f"Content:\n{content}\n"
                    context_str += "-" * 80 + "\n\n"

                # Get final response with search results
                final_response = await self.provider.get_response(
                    query=query,
                    context={
                        "system_message": """You are a helpful AI assistant for answering questions about code repositories.
                        Your task is to:
                        1. Carefully analyze the provided search results
                        2. Reference specific code snippets and documentation in your answer
                        3. Include relevant file locations when discussing code
                        4. Explain how the referenced code relates to the user's question
                        5. If a GitHub URL is available, mention it for further reference""",
                        "search_results": context_str
                    },
                    tools=None  # No tools needed for final response
                )

                logger.info("Generated final response with context from search results")
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