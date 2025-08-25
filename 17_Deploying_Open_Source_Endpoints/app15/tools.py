"""Toolbelt assembly for agents.

Collects third-party tools and local tools (like RAG) into a single list that
graphs can bind to their language models.
"""
from __future__ import annotations

from typing import List
from langchain_core.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from app.rag import retrieve_information


@tool
def google_scholar_search(query: str) -> str:
    """Search Google Scholar for academic papers and research articles.
    
    Args:
        query: The search query for Google Scholar
        
    Returns:
        Search results from Google Scholar
    """
    # For now, we'll use Tavily with Google Scholar focus
    # In a production environment, you might want to use a dedicated Google Scholar API
    tavily_tool = TavilySearchResults(max_results=5, search_depth="advanced")
    results = tavily_tool.invoke(f"site:scholar.google.com {query}")
    return f"Google Scholar search results for '{query}':\n{results}"


@tool
def ijspt_search(query: str) -> str:
    """Search the International Journal of Sports Physical Therapy (IJSPT) for relevant articles.
    
    Args:
        query: The search query for IJSPT
        
    Returns:
        Search results from IJSPT
    """
    # Use Tavily to search specifically on ijspt.org
    tavily_tool = TavilySearchResults(max_results=5, search_depth="advanced")
    results = tavily_tool.invoke(f"site:ijspt.org {query}")
    return f"IJSPT search results for '{query}':\n{results}"


def get_tool_belt() -> List:
    """Return the list of tools available to agents (Tavily, Arxiv, RAG, Google Scholar, IJSPT)."""
    tavily_tool = TavilySearchResults(max_results=5)
    return [
        tavily_tool, 
        ArxivQueryRun(), 
        retrieve_information,
        google_scholar_search,
        ijspt_search
    ]
