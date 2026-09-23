"""Unified tool registry and execution engine for Pihu-BreakThough (Stage 5).

Provides schema generation, error-isolated execution, and integrations for
Google GenAI and Groq function calling.
"""
from __future__ import annotations

import ast
from datetime import datetime
import inspect
import json
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Union
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

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


# ============================================================================
# Stage 8: Autonomous Tools Expansion (Calculate, Current Time, Live Weather)
# ============================================================================

ALLOWED_MATH_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _safe_factorial(n: Union[int, float]) -> int:
    """Safely calculate factorial with size bounds to prevent CPU exhaustion."""
    if not isinstance(n, int) and not (isinstance(n, float) and n.is_integer()):
        raise ValueError("factorial() argument must be an integer")
    val = int(n)
    if val < 0:
        raise ValueError("factorial() is not defined for negative numbers")
    if val > 100:
        raise ValueError("factorial() argument too large (maximum allowed is 100)")
    return math.factorial(val)


ALLOWED_MATH_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": _safe_factorial,
    "radians": math.radians,
    "degrees": math.degrees,
    "pow": pow,
}


def _eval_math_ast_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate an AST expression node within a strict mathematical whitelist.

    Args:
        node: The parsed ast.AST node.

    Returns:
        Evaluated numeric result.

    Raises:
        ValueError: For unauthorized operations, unknown functions, or invalid types.
        ZeroDivisionError: For division or modulo by zero.
    """
    if isinstance(node, ast.Expression):
        return _eval_math_ast_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: '{type(node.value).__name__}'")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_math_ast_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: '{type(node.op).__name__}'")

    if isinstance(node, ast.BinOp):
        left = _eval_math_ast_node(node.left)
        right = _eval_math_ast_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("integer division or modulo by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("integer division or modulo by zero")
            return left % right
        if isinstance(node.op, ast.Pow):
            if abs(right) > 1000:
                raise ValueError("Exponent too large (maximum allowed exponent is 1000)")
            return left ** right
        raise ValueError(f"Unsupported binary operator: '{type(node.op).__name__}'")

    if isinstance(node, ast.Name):
        if node.id in ALLOWED_MATH_CONSTANTS:
            return ALLOWED_MATH_CONSTANTS[node.id]
        raise ValueError(f"Unknown or unauthorized identifier: '{node.id}'")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Method and attribute calls are not permitted.")
        func_name = node.func.id
        if func_name not in ALLOWED_MATH_FUNCTIONS:
            raise ValueError(f"Unknown or unauthorized function: '{func_name}'")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in mathematical functions.")
        args = [_eval_math_ast_node(arg) for arg in node.args]
        return ALLOWED_MATH_FUNCTIONS[func_name](*args)

    raise ValueError(f"Unsupported syntax or unauthorized operation: '{type(node).__name__}'")


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression without raw eval.

    Supports basic arithmetic (+, -, *, /, //, %, **), parentheses,
    mathematical constants (pi, e, tau), and standard mathematical functions
    (sqrt, abs, round, sin, cos, tan, asin, acos, atan, log, log10, log2,
    exp, ceil, floor, factorial, radians, degrees).

    Args:
        expression: Mathematical expression string to evaluate (e.g. 'sqrt(144) + 10').

    Returns:
        String result of the calculation or an informative error message.
    """
    clean_expr = (expression or "").strip()
    if not clean_expr:
        return "Error: Mathematical expression cannot be empty."

    if len(clean_expr) > 500:
        return "Error: Mathematical expression exceeds maximum limit of 500 characters."

    try:
        parsed = ast.parse(clean_expr, mode="eval")
        raw_result = _eval_math_ast_node(parsed)

        if isinstance(raw_result, float):
            rounded = round(raw_result, 10)
            if rounded.is_integer() and abs(rounded) < 1e15:
                return str(int(rounded))
            return str(rounded)
        return str(raw_result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except OverflowError:
        return "Error: Result exceeded numerical limits (overflow)."
    except SyntaxError:
        return "Error: Invalid mathematical expression syntax."
    except ValueError as val_err:
        return f"Error: {val_err}"
    except TypeError as typ_err:
        return f"Error: {typ_err}"
    except Exception as exc:
        logger.warning("Error evaluating math expression '%s': %s", clean_expr, exc)
        return f"Error evaluating expression: {exc}"


tool_registry.register(
    calculate,
    name="calculate",
    description="Safely evaluate a mathematical expression (supporting +, -, *, /, //, %, **, parentheses, and functions like sqrt, abs, round, sin, cos, log).",
    parameters_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate, e.g. 'sqrt(144) + 12' or 'round(sin(pi/2), 2)'",
            }
        },
        "required": ["expression"],
    },
)


TIMEZONE_ALIASES: Dict[str, str] = {
    "utc": "UTC",
    "gmt": "UTC",
    "ist": "Asia/Kolkata",
    "est": "America/New_York",
    "edt": "America/New_York",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "bst": "Europe/London",
    "cet": "Europe/Paris",
    "cest": "Europe/Paris",
    "jst": "Asia/Tokyo",
    "kst": "Asia/Seoul",
    "sgt": "Asia/Singapore",
    "aest": "Australia/Sydney",
    "aedt": "Australia/Sydney",
}


def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date, time, weekday, and timezone.

    Args:
        timezone: Target timezone name or abbreviation (e.g. 'UTC', 'Asia/Kolkata', 'America/New_York', 'IST', 'PST'). Defaults to 'UTC'.

    Returns:
        Formatted string containing the current date, time, day of the week, timezone, and ISO 8601 timestamp.
    """
    clean_tz = (timezone or "UTC").strip()
    target_tz_str = TIMEZONE_ALIASES.get(clean_tz.lower(), clean_tz)

    fallback_note = ""
    try:
        tz = ZoneInfo(target_tz_str)
        resolved_name = target_tz_str
    except Exception:
        tz = ZoneInfo("UTC")
        resolved_name = "UTC"
        fallback_note = f" (Fallback: timezone '{timezone}' unrecognized)"

    now = datetime.now(tz)
    day_name = now.strftime("%A")
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    tz_abbr = now.strftime("%Z")
    iso_str = now.isoformat()

    return (
        f"Current Date: {date_str}\n"
        f"Current Time: {time_str} {tz_abbr}\n"
        f"Day of Week: {day_name}\n"
        f"Timezone: {resolved_name}{fallback_note}\n"
        f"ISO 8601: {iso_str}"
    )


tool_registry.register(
    get_current_time,
    name="get_current_time",
    description="Get the current date, time, weekday, and timezone (defaults to UTC, supports timezones like 'Asia/Kolkata', 'America/New_York', 'UTC', 'IST', 'PST').",
    parameters_schema={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Target timezone name or abbreviation (e.g. 'UTC', 'Asia/Kolkata', 'America/New_York', 'IST', 'PST'). Defaults to 'UTC'.",
                "default": "UTC",
            }
        },
        "required": [],
    },
)


WMO_WEATHER_CODES: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_weather(location: str) -> str:
    """Retrieve current live weather conditions for any location via Open-Meteo.

    Uses Open-Meteo's free public geocoding and weather forecast APIs. Does not require
    external API keys or external dependencies.

    Args:
        location: City, region, or place name (e.g. 'Tokyo', 'London', 'San Francisco', 'Mumbai').

    Returns:
        Formatted summary with condition, temperature (°C), relative humidity (%), and wind speed (km/h).
    """
    clean_loc = (location or "").strip()
    if not clean_loc:
        return "Error: Location cannot be empty."

    try:
        # Step 1: Geocoding lookup
        geo_url = (
            f"https://geocoding-api.open-meteo.com/v1/search?"
            f"name={urllib.parse.quote(clean_loc)}&count=1&language=en&format=json"
        )
        req_geo = urllib.request.Request(
            geo_url,
            headers={"User-Agent": "Pihu-BreakThough/1.0"},
        )
        with urllib.request.urlopen(req_geo, timeout=8.0) as resp:
            geo_data = json.loads(resp.read().decode("utf-8"))

        results = geo_data.get("results")
        if not results:
            return f"Error: Location '{clean_loc}' not found."

        loc_info = results[0]
        name = loc_info.get("name", clean_loc)
        country = loc_info.get("country", "")
        admin1 = loc_info.get("admin1", "")
        lat = loc_info.get("latitude")
        lon = loc_info.get("longitude")

        loc_parts = [name]
        if admin1 and admin1 != name:
            loc_parts.append(admin1)
        if country:
            loc_parts.append(country)
        resolved_name = ", ".join(loc_parts)

        # Step 2: Forecast conditions
        forecast_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
        )
        req_forecast = urllib.request.Request(
            forecast_url,
            headers={"User-Agent": "Pihu-BreakThough/1.0"},
        )
        with urllib.request.urlopen(req_forecast, timeout=8.0) as resp:
            weather_data = json.loads(resp.read().decode("utf-8"))

        current = weather_data.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        code = current.get("weather_code")
        wind = current.get("wind_speed_10m")
        condition = WMO_WEATHER_CODES.get(code, f"Weather code {code}")

        return (
            f"Weather for {resolved_name} (Lat: {lat}, Lon: {lon}):\n"
            f"- Condition: {condition} (WMO {code})\n"
            f"- Temperature: {temp}°C\n"
            f"- Relative Humidity: {humidity}%\n"
            f"- Wind Speed: {wind} km/h"
        )
    except Exception as exc:
        logger.warning("Weather lookup failed for '%s': %s", clean_loc, exc)
        return f"Error retrieving weather for '{clean_loc}': {exc}"


tool_registry.register(
    get_weather,
    name="get_weather",
    description="Get current live weather conditions (temperature, condition, humidity, wind speed) for any global city or location via Open-Meteo.",
    parameters_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, region, or address to retrieve weather for (e.g., 'London', 'Tokyo', 'San Francisco', 'Mumbai').",
            }
        },
        "required": ["location"],
    },
)


