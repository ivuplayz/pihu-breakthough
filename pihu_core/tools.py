"""Unified tool registry and execution engine for Pihu-BreakThough (Stage 5).

Provides schema generation, error-isolated execution, and integrations for
Google GenAI and Groq function calling.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry managing callable tools and their JSON schemas for LLM function calling."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters_schema: Optional[Dict[str, Any]] = None,
    ) -> Callable[..., Any]:
        """Register a Python callable as an LLM-invocable tool.

        Args:
            func: Target Python callable to register.
            name: Optional unique identifier. Defaults to func.__name__.
            description: Optional functional description. Defaults to func.__doc__.
            parameters_schema: Optional explicit JSON schema for function parameters.

        Returns:
            The registered callable.
        """
        tool_name = name or func.__name__
        tool_doc = (description or func.__doc__ or f"Execute {tool_name}").strip()

        if parameters_schema is None:
            sig = inspect.signature(func)
            properties: Dict[str, Any] = {}
            required: List[str] = []
            type_mapping = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
            }

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                ann = param.annotation
                ann_key = getattr(ann, "__name__", str(ann)).lower() if ann is not inspect.Parameter.empty else "string"
                param_type = type_mapping.get(ann, type_mapping.get(ann_key, "string"))
                prop: Dict[str, Any] = {"type": param_type}
                if param.default is not inspect.Parameter.empty:
                    prop["default"] = param.default
                else:
                    required.append(param_name)
                properties[param_name] = prop

            parameters_schema = {
                "type": "object",
                "properties": properties,
                "required": required,
            }

        self._tools[tool_name] = func
        self._schemas[tool_name] = {
            "name": tool_name,
            "description": tool_doc,
            "parameters": parameters_schema,
        }
        logger.debug("Registered tool '%s' in ToolRegistry.", tool_name)
        return func

    def get_tool(self, name: str) -> Optional[Callable[..., Any]]:
        """Retrieve a registered callable by name."""
        return self._tools.get(name)

    def get_tools(self) -> List[Callable[..., Any]]:
        """Return all registered callables as a list."""
        return list(self._tools.values())

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve tool schema by name."""
        return self._schemas.get(name)

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Safely invoke a registered tool by name with arguments.

        Args:
            name: Registered tool name.
            **kwargs: Keyword arguments matching the function parameters.

        Returns:
            Result of tool invocation, or error string if execution fails.
        """
        func = self.get_tool(name)
        if not func:
            msg = f"Tool '{name}' is not registered."
            logger.warning(msg)
            return f"Error: {msg}"

        try:
            return func(**kwargs)
        except Exception as exc:
            logger.error("Error executing tool '%s': %s", name, exc)
            return f"Error executing tool '{name}': {exc}"

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Format schemas for Groq / OpenAI compatible tool declarations."""
        declarations: List[Dict[str, Any]] = []
        for schema in self._schemas.values():
            declarations.append({
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                },
            })
        return declarations

    def to_gemini_format(self) -> List[Any]:
        """Format callables for Google GenAI SDK tool declarations."""
        return list(self._tools.values())


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web in real-time using DuckDuckGo and return formatted snippets.

    Args:
        query: Search query string.
        max_results: Maximum number of search results to return (default: 5).

    Returns:
        Formatted multi-line text containing numbered titles, summaries, and URLs.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        return "Search query cannot be empty."

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.error("duckduckgo_search is not installed.")
        return "Web search is currently unavailable: duckduckgo_search library not installed."

    try:
        ddgs = DDGS()
        results = list(ddgs.text(clean_query, max_results=max_results))
        if not results:
            return f"No web search results found for query: '{clean_query}'."

        formatted_items: List[str] = []
        for i, item in enumerate(results, start=1):
            title = item.get("title", "No Title")
            snippet = item.get("body", "")
            href = item.get("href", "")
            formatted_items.append(f"[{i}] {title}\nSummary: {snippet}\nSource: {href}")

        return "\n\n".join(formatted_items)
    except Exception as exc:
        logger.warning("DuckDuckGo web search failed for query '%s': %s", clean_query, exc)
        return f"Web search error for '{clean_query}': {exc}"


# Global singleton ToolRegistry instance
tool_registry = ToolRegistry()

# Register the standard search_web tool
tool_registry.register(
    search_web,
    name="search_web",
    description="Search the web in real-time using DuckDuckGo to obtain up-to-date facts, current events, and live information.",
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to retrieve (default: 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)


def save_core_memory(category: str, content: str) -> str:
    """Save an important user preference, fact, or project context to long-term memory.

    Args:
        category: Broad category classification (e.g., 'preference', 'fact', 'project').
        content: The core fact or context to remember about the user.

    Returns:
        Status message confirming the memory has been saved.
    """
    clean_category = (category or "fact").strip().lower()
    clean_content = (content or "").strip()
    if not clean_content:
        return "Error: Memory content cannot be empty."

    try:
        from pihu_core.repository import ConversationRepository
        repo = ConversationRepository()
        if not repo.is_available():
            return "Memory storage is currently unavailable (database unconfigured)."

        repo.save_memory(
            category=clean_category,
            content=clean_content,
        )
        return f"Successfully saved to long-term memory under category '{clean_category}': {clean_content}"
    except Exception as exc:
        logger.error("Failed to save core memory: %s", exc)
        return f"Error saving memory: {exc}"


tool_registry.register(
    save_core_memory,
    name="save_core_memory",
    description="Save an important user preference, personal fact, or project context to long-term memory for future conversations.",
    parameters_schema={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Category of the memory (e.g., 'preference', 'fact', 'project').",
            },
            "content": {
                "type": "string",
                "description": "The specific detail, fact, or preference to remember.",
            },
        },
        "required": ["category", "content"],
    },
)

