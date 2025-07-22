"""
AutoGen Deep Research Tools Module

This module provides tool implementations for the Deep Research system,
including search APIs and MCP server integration.
"""

from typing import List, Dict, Any, Optional, Callable
import os
import json
import asyncio
from datetime import datetime

# AutoGen tools
from autogen_core.tools import FunctionTool

# External APIs
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

try:
    import duckduckgo_search
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

# Local imports
from configuration import Configuration, SearchAPI, MCPConfig
from state import WebSearchResult, WebPageSummary


# ==============================================
# Search Tool Implementations
# ==============================================

async def web_search_tavily(query: str, max_results: int = 5) -> WebSearchResult:
    """
    Search the web using Tavily API
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        WebSearchResult object containing search results
    """
    if not TavilyClient:
        raise ImportError("Tavily client not installed. Install with: pip install tavily-python")
        
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")
        
    client = TavilyClient(api_key=api_key)
    
    try:
        # Perform search
        response = client.search(
            query=query,
            max_results=max_results,
            include_domains=[],
            exclude_domains=[]
        )
        
        # Format results
        results = []
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", ""),
                "score": result.get("score", 0.0)
            })
            
        return WebSearchResult(
            query=query,
            results=results,
            source="tavily",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise Exception(f"Tavily search error: {str(e)}")


async def web_search_duckduckgo(query: str, max_results: int = 5) -> WebSearchResult:
    """
    Search the web using DuckDuckGo
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        WebSearchResult object containing search results
    """
    if not DDGS:
        raise ImportError("DuckDuckGo search not installed. Install with: pip install duckduckgo-search")
        
    try:
        # Perform search
        with DDGS() as ddgs:
            search_results = list(ddgs.text(
                query,
                max_results=max_results
            ))
            
        # Format results
        results = []
        for result in search_results:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", ""),
                "score": 1.0  # DuckDuckGo doesn't provide scores
            })
            
        return WebSearchResult(
            query=query,
            results=results,
            source="duckduckgo",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise Exception(f"DuckDuckGo search error: {str(e)}")


async def summarize_web_page(url: str, model_client: Any) -> WebPageSummary:
    """
    Fetch and summarize a web page
    
    Args:
        url: URL of the web page to summarize
        model_client: AutoGen model client for summarization
        
    Returns:
        WebPageSummary object
    """
    # In a real implementation, this would:
    # 1. Fetch the web page content
    # 2. Extract main content (removing ads, navigation, etc.)
    # 3. Use the model to summarize
    
    # For now, returning a mock summary
    return WebPageSummary(
        url=url,
        summary="This would be a summary of the web page content",
        key_excerpts=["Key point 1", "Key point 2"],
        relevance_score=0.8
    )


# ==============================================
# Tool Factory Functions
# ==============================================

def get_search_tools(config: Configuration) -> List[FunctionTool]:
    """
    Create search tools based on configuration (supports multiple APIs)
    
    Args:
        config: Configuration object
        
    Returns:
        List of FunctionTool objects for web search
    """
    tools = []
    
    if not hasattr(config, 'search_apis') or not config.search_apis:
        return tools
        
    for search_api in config.search_apis:
        if search_api == SearchAPI.NONE:
            continue
            
        elif search_api == SearchAPI.TAVILY:
            tools.append(FunctionTool(
                func=web_search_tavily,
                description="Search the web using Tavily API for relevant information",
                name="web_search_tavily"
            ))
            
        elif search_api == SearchAPI.DUCKDUCKGO:
            tools.append(FunctionTool(
                func=web_search_duckduckgo,
                description="Search the web using DuckDuckGo for relevant information", 
                name="web_search_duckduckgo"
            ))
            
        # For future APIs (ANTHROPIC, OPENAI native search, etc.)
        # Add more cases here
        
    return tools


def get_search_tool(config: Configuration) -> Optional[FunctionTool]:
    """
    Legacy function for backward compatibility - returns first available search tool
    """
    tools = get_search_tools(config)
    return tools[0] if tools else None


def get_mcp_tools(mcp_config: MCPConfig) -> List[FunctionTool]:
    """
    Create MCP tools based on configuration
    
    Args:
        mcp_config: MCP configuration object
        
    Returns:
        List of FunctionTool objects for MCP operations
    """
    tools = []
    
    if not mcp_config or not mcp_config.tools:
        return tools
        
    # Create tools based on MCP configuration
    for tool_name in mcp_config.tools:
        if tool_name == "file_read":
            tools.append(create_file_read_tool())
        elif tool_name == "database_query":
            tools.append(create_database_query_tool())
        # Add more MCP tools as needed
        
    return tools


# ==============================================
# MCP Tool Implementations
# ==============================================

def create_file_read_tool() -> FunctionTool:
    """Create a file reading tool for MCP"""
    
    async def read_file(file_path: str) -> str:
        """
        Read contents of a file through MCP
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File contents as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
            
    return FunctionTool(
        func=read_file,
        description="Read contents of a file from the local filesystem",
        name="read_file"
    )


def create_database_query_tool() -> FunctionTool:
    """Create a database query tool for MCP"""
    
    async def query_database(query: str, database: str = "default") -> List[Dict[str, Any]]:
        """
        Execute a database query through MCP
        
        Args:
            query: SQL query to execute
            database: Database identifier
            
        Returns:
            Query results as list of dictionaries
        """
        # This would connect to actual database through MCP
        # For now, returning mock data
        return [
            {"id": 1, "name": "Sample Result 1"},
            {"id": 2, "name": "Sample Result 2"}
        ]
        
    return FunctionTool(
        func=query_database,
        description="Execute SQL queries against configured databases",
        name="query_database"
    )


# ==============================================
# Utility Functions
# ==============================================

def format_search_results_for_llm(search_result: WebSearchResult) -> str:
    """
    Format search results for LLM consumption
    
    Args:
        search_result: WebSearchResult object
        
    Returns:
        Formatted string for LLM processing
    """
    formatted = f"Search Query: {search_result.query}\n"
    formatted += f"Source: {search_result.source}\n"
    formatted += f"Results found: {len(search_result.results)}\n\n"
    
    for i, result in enumerate(search_result.results, 1):
        formatted += f"{i}. {result['title']}\n"
        formatted += f"   URL: {result['url']}\n"
        formatted += f"   Snippet: {result['snippet']}\n"
        formatted += f"   Relevance: {result.get('score', 'N/A')}\n\n"
        
    return formatted


async def multi_search(query: str, config: Configuration, max_results: int = 5) -> WebSearchResult:
    """
    Perform search using multiple APIs simultaneously and merge results
    
    Args:
        query: Search query string
        config: Configuration object with multiple search APIs
        max_results: Maximum results per API
        
    Returns:
        Merged WebSearchResult from all configured APIs
    """
    if not hasattr(config, 'search_apis') or not config.search_apis:
        raise ValueError("No search APIs configured")
        
    # Execute searches in parallel
    search_tasks = []
    
    for search_api in config.search_apis:
        if search_api == SearchAPI.NONE:
            continue
            
        elif search_api == SearchAPI.TAVILY:
            search_tasks.append(web_search_tavily(query, max_results))
            
        elif search_api == SearchAPI.DUCKDUCKGO:
            search_tasks.append(web_search_duckduckgo(query, max_results))
            
    if not search_tasks:
        return WebSearchResult(
            query=query,
            results=[],
            source="no_apis_available", 
            timestamp=datetime.now()
        )
        
    # Execute all searches concurrently
    try:
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Filter out failed searches
        valid_results = []
        for result in search_results:
            if isinstance(result, WebSearchResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                print(f"Search failed: {result}")
                
        return merge_search_results(valid_results)
        
    except Exception as e:
        raise Exception(f"Multi-search failed: {str(e)}")


def merge_search_results(results: List[WebSearchResult]) -> WebSearchResult:
    """
    Merge multiple search results into one
    
    Args:
        results: List of WebSearchResult objects
        
    Returns:
        Merged WebSearchResult
    """
    if not results:
        return WebSearchResult(
            query="",
            results=[],
            source="merged",
            timestamp=datetime.now()
        )
        
    # Combine all results
    all_results = []
    queries = []
    sources = []
    
    for result in results:
        all_results.extend(result.results)
        queries.append(result.query)
        sources.append(result.source)
        
    # Remove duplicates based on URL
    seen_urls = set()
    unique_results = []
    
    for result in all_results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
            
    # Sort by score if available
    unique_results.sort(
        key=lambda x: x.get("score", 0),
        reverse=True
    )
    
    return WebSearchResult(
        query=" | ".join(set(queries)),  # Remove duplicate queries
        results=unique_results,
        source=" + ".join(set(sources)),  # Show combined sources
        timestamp=datetime.now()
    )


# ==============================================
# Tool Validation and Testing
# ==============================================

async def validate_search_tools(config: Configuration) -> Dict[str, bool]:
    """
    Validate that search tools are properly configured
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary mapping search API names to validation status
    """
    validation_results = {}
    
    if not hasattr(config, 'search_apis') or not config.search_apis:
        return validation_results
        
    for search_api in config.search_apis:
        if search_api == SearchAPI.NONE:
            validation_results["none"] = True
            
        elif search_api == SearchAPI.TAVILY:
            validation_results["tavily"] = bool(os.getenv("TAVILY_API_KEY")) and TavilyClient is not None
            
        elif search_api == SearchAPI.DUCKDUCKGO:
            validation_results["duckduckgo"] = DDGS is not None
            
        # For native search APIs (ANTHROPIC, OPENAI)
        else:
            validation_results[str(search_api)] = True
            
    return validation_results


async def validate_search_tool(search_api: SearchAPI) -> bool:
    """Legacy function for backward compatibility"""
    if search_api == SearchAPI.NONE:
        return True
    elif search_api == SearchAPI.TAVILY:
        return bool(os.getenv("TAVILY_API_KEY")) and TavilyClient is not None
    elif search_api == SearchAPI.DUCKDUCKGO:
        return DDGS is not None
    return True


async def test_search_tools(config: Configuration) -> None:
    """
    Test all configured search tools
    
    Args:
        config: Configuration object
    """
    if not hasattr(config, 'search_apis') or not config.search_apis:
        print("No search APIs configured")
        return
        
    tools = get_search_tools(config)
    if not tools:
        print("No search tools created")
        return
        
    for i, search_api in enumerate(config.search_apis):
        if search_api == SearchAPI.NONE:
            continue
            
        print(f"\n=== Testing {search_api.value} ===")
        
        try:
            # Test with a simple query
            if search_api == SearchAPI.TAVILY:
                result = await web_search_tavily("test query", max_results=2)
            elif search_api == SearchAPI.DUCKDUCKGO:
                result = await web_search_duckduckgo("test query", max_results=2)
            else:
                print(f"Native search for {search_api} - no direct test available")
                continue
                
            print(f"✓ Search test successful: {len(result.results)} results found")
            print(f"Source: {result.source}")
            
        except Exception as e:
            print(f"❌ Search test failed: {str(e)}")


async def test_search_tool(config: Configuration) -> None:
    """Legacy function for backward compatibility"""
    await test_search_tools(config)


# Example usage
if __name__ == "__main__":
    async def main():
        from configuration import load_configuration
        
        # Load configuration
        config = load_configuration()
        
        # Validate search tool
        is_valid = await validate_search_tool(config.search_api)
        print(f"Search tool valid: {is_valid}")
        
        # Test search if valid
        if is_valid:
            await test_search_tool(config)
            
    asyncio.run(main())