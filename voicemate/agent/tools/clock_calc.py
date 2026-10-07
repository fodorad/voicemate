"""Local utility tools: current time and a safe calculator."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from voicemate.agent.tools.base import ToolContext, ToolError, report

_BINARY: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

#: A number followed by ``%`` with no operand after it is a percentage.
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(.])")

#: Largest allowed exponent, so ``9**9**9`` cannot freeze the process.
MAX_EXPONENT: int = 1000


def safe_eval(expression: str) -> float | int:
    """Evaluate an arithmetic expression without ``eval``.

    Supports numbers, ``+ - * / // % **``, parentheses, ``pi``, ``e`` and
    ``abs round min max sqrt log exp``. A trailing ``%`` after a number means percent
    when no operand follows it (``17% * 340``); ``10 % 3`` stays modulo. ``×``, ``÷`` and
    ``^`` are accepted as ``*``, ``/`` and ``**``.

    Raises:
        ToolError: For anything else (names, attributes, huge exponents, syntax errors).
    """
    cleaned = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    cleaned = _PERCENT.sub(r"(\1/100)", cleaned)
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Cannot parse expression {expression!r}") from exc

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ToolError("Exponent too large")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](*(visit(arg) for arg in node.args))
        raise ToolError(f"Unsupported element in expression: {ast.dump(node)[:60]}")

    try:
        result = visit(tree)
    except (ZeroDivisionError, ValueError, OverflowError) as exc:
        raise ToolError(f"Math error: {exc}") from exc
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        return int(result)
    return round(result, 10) if isinstance(result, float) else result


def build(ctx: ToolContext) -> list[BaseTool]:
    """Create the time and calculator tools."""

    async def get_time(config: RunnableConfig) -> str:
        """Get the current local date, time and weekday."""
        now = ctx.now()
        report(config, "get_time", {}, "local")
        return now.strftime("%Y-%m-%d %H:%M, %A, time zone %Z (UTC%z)")

    async def calculator(expression: str, config: RunnableConfig) -> str:
        """Evaluate an arithmetic expression exactly. Use it for any non-trivial math.

        Args:
            expression: Arithmetic such as "23*42", "17% * 340", "sqrt(2)**2", "(3+4)/7".
        """
        result = safe_eval(expression)
        report(config, "calculator", {"expression": expression}, "local")
        return f"{expression} = {result}"

    return [
        StructuredTool.from_function(coroutine=get_time, name="get_time", parse_docstring=True),
        StructuredTool.from_function(coroutine=calculator, name="calculator", parse_docstring=True),
    ]
