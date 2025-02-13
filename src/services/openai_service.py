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
                    "system_message": "You are a helpful AI assistant for answering questions about code repositories. "
                                    "Use the search_knowledge_base function to find relevant information before answering."
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

                # Format search results for AI consumption
                context_str = "Based on the search results from the codebase:\n\n"
                sources = []  # Keep track of sources for the end

                for idx, doc in enumerate(search_results):
                    logger.debug(f"Processing document {idx + 1}:")
                    logger.debug(f"Document metadata: {json.dumps(doc.get('metadata', {}), indent=2)}")
                    
                    content = doc.get('content', '')
                    github_url = doc.get('metadata', {}).get('file_github_url', '')
                    
                    if not content:
                        logger.warning(f"Document {idx + 1} has no content!")
                    else:
                        logger.debug(f"Document {idx + 1} content length: {len(content)}")
                    
                    context_str += f"{content}\n\n"
                    if github_url:
                        sources.append(github_url)

                # Add sources section at the end
                if sources:
                    context_str += "\nSources:\n"
                    for url in sources:
                        context_str += f"- {url}\n"

                logger.debug("Formatted context for AI consumption:")
                logger.debug(context_str)

                system_message = (
                    "You are a helpful AI assistant for answering questions about code repositories. "
                    "Here are relevant code snippets and documentation from the codebase:\n\n"
                    f"{context_str}\n\n"
                    "Use the code snippets and documentation provided above to give detailed answers. "
                    "Always reference specific code or documentation when explaining. "
                    "When referencing files, use the URLs from the Sources section."
                )

                logger.debug("System message being sent to OpenAI:")
                logger.debug(system_message)

                # Get final response with search results
                final_response = await self.provider.get_response(
                    query=query,
                    context={
                        "system_message": system_message
                    }
                )

                logger.info(f"Generated response for query: '{query}'")
                logger.debug(f"Full response content: {final_response.content}")
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