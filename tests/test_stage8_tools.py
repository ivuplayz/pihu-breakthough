"""Unit and integration test suite for Stage 8: Autonomous Tools Expansion.

Tests:
- calculate: Safe AST math evaluation (arithmetic, functions, constants, divide by zero, syntax errors, dangerous input rejection).
- get_current_time: Timezone resolution (UTC, popular timezones, aliases, graceful invalid fallback).
- get_weather: Open-Meteo geocoding and weather forecast (mocked responses, missing location, network failures).
- ToolRegistry: Schema generation, OpenAI format, Gemini format, and error-isolated execution.
- Provider and Router: Verification that function calls for new tools execute cleanly.
"""

from __future__ import annotations

import io
import json
import math
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from pihu_core.tools import (
    ALLOWED_MATH_CONSTANTS,
    ALLOWED_MATH_FUNCTIONS,
    calculate,
    get_current_time,
    get_weather,
    tool_registry,
)


class TestCalculateTool(unittest.TestCase):
    """Test suite for safe AST-based calculate() tool."""

    def test_basic_arithmetic(self) -> None:
        """Verify standard arithmetic operations and operator precedence."""
        self.assertEqual(calculate("2 + 2"), "4")
        self.assertEqual(calculate("10 - 4"), "6")
        self.assertEqual(calculate("6 * 7"), "42")
        self.assertEqual(calculate("20 / 4"), "5")
        self.assertEqual(calculate("7 // 2"), "3")
        self.assertEqual(calculate("10 % 3"), "1")
        self.assertEqual(calculate("2 ** 8"), "256")
        self.assertEqual(calculate("2 + 3 * 4"), "14")
        self.assertEqual(calculate("(2 + 3) * 4"), "20")

    def test_floating_point_precision(self) -> None:
        """Verify float calculations eliminate IEEE floating point noise."""
        self.assertEqual(calculate("0.1 + 0.2"), "0.3")
        self.assertEqual(calculate("10 / 3"), str(round(10 / 3, 10)))
        self.assertEqual(calculate("1.5 * 3"), "4.5")

    def test_unary_operators(self) -> None:
        """Verify positive and negative unary operators."""
        self.assertEqual(calculate("+5"), "5")
        self.assertEqual(calculate("-10"), "-10")
        self.assertEqual(calculate("-(-7)"), "7")
        self.assertEqual(calculate("-5 * -5"), "25")

    def test_mathematical_constants(self) -> None:
        """Verify pi, e, and tau constants."""
        self.assertEqual(calculate("pi"), str(round(math.pi, 10)))
        self.assertEqual(calculate("e"), str(round(math.e, 10)))
        self.assertEqual(calculate("tau"), str(round(math.tau, 10)))
        self.assertEqual(calculate("pi * 2"), str(round(math.pi * 2, 10)))

    def test_mathematical_functions(self) -> None:
        """Verify supported mathematical functions."""
        self.assertEqual(calculate("sqrt(144)"), "12")
        self.assertEqual(calculate("abs(-42)"), "42")
        self.assertEqual(calculate("round(3.14159, 2)"), "3.14")
        self.assertEqual(calculate("sin(pi / 2)"), "1")
        self.assertEqual(calculate("cos(0)"), "1")
        self.assertEqual(calculate("tan(0)"), "0")
        self.assertEqual(calculate("log(e)"), "1")
        self.assertEqual(calculate("log10(100)"), "2")
        self.assertEqual(calculate("log2(8)"), "3")
        self.assertEqual(calculate("ceil(4.1)"), "5")
        self.assertEqual(calculate("floor(4.9)"), "4")
        self.assertEqual(calculate("factorial(5)"), "120")
        self.assertEqual(calculate("degrees(pi)"), "180")
        self.assertEqual(calculate("pow(2, 5)"), "32")

    def test_nested_complex_expressions(self) -> None:
        """Verify deeply nested but valid mathematical expressions."""
        expr = "sqrt((3 ** 2) + (4 ** 2))"
        self.assertEqual(calculate(expr), "5")

        expr2 = "round(sin(pi / 6) + cos(pi / 3), 2)"
        self.assertEqual(calculate(expr2), "1")

    def test_empty_and_whitespace(self) -> None:
        """Verify empty expressions return informative errors."""
        self.assertTrue(calculate("").startswith("Error:"))
        self.assertTrue(calculate("   ").startswith("Error:"))
        self.assertIn("cannot be empty", calculate(""))

    def test_division_by_zero(self) -> None:
        """Verify division and modulo by zero are safely caught."""
        res_div = calculate("10 / 0")
        self.assertTrue(res_div.startswith("Error:"))
        self.assertIn("zero", res_div.lower())

        res_floor = calculate("10 // 0")
        self.assertTrue(res_floor.startswith("Error:"))
        self.assertIn("zero", res_floor.lower())

        res_mod = calculate("10 % 0")
        self.assertTrue(res_mod.startswith("Error:"))
        self.assertIn("zero", res_mod.lower())

    def test_syntax_errors(self) -> None:
        """Verify invalid syntax returns informative errors without crashing."""
        self.assertTrue(calculate("2 + + +").startswith("Error:"))
        self.assertTrue(calculate("(5 + 2").startswith("Error:"))
        self.assertTrue(calculate("5 + * 2").startswith("Error:"))

    def test_dangerous_imports_and_statements_rejected(self) -> None:
        """Verify statements, imports, and arbitrary execution are safely rejected."""
        # Statement
        res_import = calculate("import os")
        self.assertTrue(res_import.startswith("Error:"))

        # Function call to builtins
        res_builtin = calculate("__import__('os').system('ls')")
        self.assertTrue(res_builtin.startswith("Error:"))

        # File access
        res_open = calculate("open('/etc/passwd')")
        self.assertTrue(res_open.startswith("Error:"))

        # Eval / exec
        res_eval = calculate("eval('2 + 2')")
        self.assertTrue(res_eval.startswith("Error:"))
        res_exec = calculate("exec('x = 1')")
        self.assertTrue(res_exec.startswith("Error:"))

    def test_attribute_and_method_access_rejected(self) -> None:
        """Verify object attribute traversal is strictly disallowed."""
        res_subclass = calculate("().__class__.__bases__[0].__subclasses__()")
        self.assertTrue(res_subclass.startswith("Error:"))

        res_attr = calculate("pi.__class__")
        self.assertTrue(res_attr.startswith("Error:"))

    def test_exponent_size_limit(self) -> None:
        """Verify oversized exponents are rejected to prevent DoS."""
        res = calculate("2 ** 10001")
        self.assertTrue(res.startswith("Error:"))
        self.assertIn("Exponent too large", res)

    def test_factorial_bounds(self) -> None:
        """Verify factorial input bounds prevent CPU exhaustion."""
        res_negative = calculate("factorial(-5)")
        self.assertTrue(res_negative.startswith("Error:"))

        res_oversized = calculate("factorial(150)")
        self.assertTrue(res_oversized.startswith("Error:"))

    def test_expression_length_limit(self) -> None:
        """Verify expressions exceeding 500 characters are rejected."""
        long_expr = "1 + " * 200 + "1"
        self.assertTrue(len(long_expr) > 500)
        res = calculate(long_expr)
        self.assertTrue(res.startswith("Error:"))
        self.assertIn("exceeds maximum limit", res)


class TestGetCurrentTimeTool(unittest.TestCase):
    """Test suite for get_current_time() tool."""

    def test_default_utc(self) -> None:
        """Verify default call resolves to UTC with required fields."""
        output = get_current_time()
        self.assertIn("Current Date:", output)
        self.assertIn("Current Time:", output)
        self.assertIn("Day of Week:", output)
        self.assertIn("Timezone: UTC", output)
        self.assertIn("ISO 8601:", output)

    def test_popular_timezones_and_aliases(self) -> None:
        """Verify timezone aliases and IANA timezone strings resolve correctly."""
        # IST
        output_ist = get_current_time("IST")
        self.assertIn("Timezone: Asia/Kolkata", output_ist)

        # Asia/Kolkata
        output_kolkata = get_current_time("Asia/Kolkata")
        self.assertIn("Timezone: Asia/Kolkata", output_kolkata)

        # EST / America/New_York
        output_est = get_current_time("EST")
        self.assertIn("Timezone: America/New_York", output_est)

        # PST / America/Los_Angeles
        output_pst = get_current_time("PST")
        self.assertIn("Timezone: America/Los_Angeles", output_pst)

        # JST / Asia/Tokyo
        output_jst = get_current_time("JST")
        self.assertIn("Timezone: Asia/Tokyo", output_jst)

        # Europe/London
        output_london = get_current_time("Europe/London")
        self.assertIn("Timezone: Europe/London", output_london)

    def test_invalid_timezone_fallback(self) -> None:
        """Verify invalid or unparseable timezones fall back to UTC gracefully."""
        output = get_current_time("Atlantis/Ocean")
        self.assertIn("Current Date:", output)
        self.assertIn("Timezone: UTC (Fallback: timezone 'Atlantis/Ocean' unrecognized)", output)
        self.assertIn("ISO 8601:", output)

    def test_empty_timezone_string(self) -> None:
        """Verify empty string falls back cleanly to UTC."""
        output = get_current_time("")
        self.assertIn("Timezone: UTC", output)


class TestGetWeatherTool(unittest.TestCase):
    """Test suite for get_weather() tool."""

    def test_empty_location(self) -> None:
        """Verify empty location parameter returns informative error."""
        res = get_weather("")
        self.assertTrue(res.startswith("Error:"))
        self.assertIn("Location cannot be empty", res)

        res_space = get_weather("   ")
        self.assertTrue(res_space.startswith("Error:"))

    @patch("urllib.request.urlopen")
    def test_mocked_weather_success(self, mock_urlopen: MagicMock) -> None:
        """Verify successful geocoding and forecast pipeline using mocked HTTP."""
        geo_payload = {
            "results": [
                {
                    "name": "London",
                    "country": "United Kingdom",
                    "admin1": "England",
                    "latitude": 51.5085,
                    "longitude": -0.1257,
                }
            ]
        }
        weather_payload = {
            "current": {
                "temperature_2m": 18.5,
                "relative_humidity_2m": 62,
                "weather_code": 2,
                "wind_speed_10m": 12.3,
            }
        }

        mock_geo_resp = MagicMock()
        mock_geo_resp.read.return_value = json.dumps(geo_payload).encode("utf-8")
        mock_geo_resp.__enter__.return_value = mock_geo_resp

        mock_weather_resp = MagicMock()
        mock_weather_resp.read.return_value = json.dumps(weather_payload).encode("utf-8")
        mock_weather_resp.__enter__.return_value = mock_weather_resp

        mock_urlopen.side_effect = [mock_geo_resp, mock_weather_resp]

        result = get_weather("London")
        self.assertIn("Weather for London, England, United Kingdom", result)
        self.assertIn("Partly cloudy (WMO 2)", result)
        self.assertIn("18.5°C", result)
        self.assertIn("62%", result)
        self.assertIn("12.3 km/h", result)

    @patch("urllib.request.urlopen")
    def test_location_not_found(self, mock_urlopen: MagicMock) -> None:
        """Verify nonexistent location returns clean not-found message."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"results": []}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        result = get_weather("ImaginaryCityXYZ999")
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("not found", result)

    @patch("urllib.request.urlopen")
    def test_network_failure_handling(self, mock_urlopen: MagicMock) -> None:
        """Verify network or HTTP exceptions are handled without crashing."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        result = get_weather("Berlin")
        self.assertTrue(result.startswith("Error retrieving weather for 'Berlin':"))
        self.assertIn("Connection refused", result)


class TestToolRegistryIntegration(unittest.TestCase):
    """Test suite verifying ToolRegistry schemas, registration, and execution for Stage 8 tools."""

    def test_tools_registered(self) -> None:
        """Verify calculate, get_current_time, and get_weather are in tool_registry."""
        tools = tool_registry.get_tools()
        names = [f.__name__ for f in tools]
        self.assertIn("calculate", names)
        self.assertIn("get_current_time", names)
        self.assertIn("get_weather", names)

    def test_schemas_present(self) -> None:
        """Verify registered schemas for new tools have complete definitions."""
        calc_schema = tool_registry.get_schema("calculate")
        self.assertIsNotNone(calc_schema)
        self.assertEqual(calc_schema["name"], "calculate")
        self.assertIn("expression", calc_schema["parameters"]["properties"])
        self.assertIn("expression", calc_schema["parameters"]["required"])

        time_schema = tool_registry.get_schema("get_current_time")
        self.assertIsNotNone(time_schema)
        self.assertEqual(time_schema["name"], "get_current_time")
        self.assertIn("timezone", time_schema["parameters"]["properties"])

        weather_schema = tool_registry.get_schema("get_weather")
        self.assertIsNotNone(weather_schema)
        self.assertEqual(weather_schema["name"], "get_weather")
        self.assertIn("location", weather_schema["parameters"]["properties"])
        self.assertIn("location", weather_schema["parameters"]["required"])

    def test_tool_registry_execute(self) -> None:
        """Verify executing new tools through tool_registry.execute()."""
        res_calc = tool_registry.execute("calculate", expression="25 * 4")
        self.assertEqual(res_calc, "100")

        res_time = tool_registry.execute("get_current_time", timezone="UTC")
        self.assertIn("Timezone: UTC", res_time)

    def test_openai_and_gemini_format(self) -> None:
        """Verify OpenAI schemas and Gemini callables include Stage 8 tools."""
        openai_tools = tool_registry.to_openai_format()
        gemini_tools = tool_registry.to_gemini_format()

        openai_names = [t["function"]["name"] for t in openai_tools]
        self.assertIn("calculate", openai_names)
        self.assertIn("get_current_time", openai_names)
        self.assertIn("get_weather", openai_names)

        gemini_names = [fn.__name__ for fn in gemini_tools]
        self.assertIn("calculate", gemini_names)
        self.assertIn("get_current_time", gemini_names)
        self.assertIn("get_weather", gemini_names)


if __name__ == "__main__":
    unittest.main()
