"""
Tool registry and built-in tool implementations.

Define tools with the @tool decorator — the harness auto-generates
the JSON schema from the function signature and docstring.
"""

import inspect
import json
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class ToolRegistry:
    """Registry that maps tool names to their schema + implementation."""

    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, Callable] = {}

    def register(self, func: Callable, schema: dict | None = None):
        if schema is None:
            schema = _schema_from_func(func)
        self._tools[schema["name"]] = schema
        self._handlers[schema["name"]] = func

    @property
    def definitions(self) -> list[dict]:
        return list(self._tools.values())

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = handler(**arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, default=str)
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def has(self, name: str) -> bool:
        return name in self._handlers


def tool(func: Callable) -> Callable:
    """Decorator that marks a function as an agent tool.

    The function's name, docstring, and type annotations are used to
    generate the JSON schema Claude needs.
    """
    func._is_tool = True
    func._tool_schema = _schema_from_func(func)
    return func


# ---------------------------------------------------------------------------
# Schema generation from function signature
# ---------------------------------------------------------------------------

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_from_func(func: Callable) -> dict:
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    description, param_docs = _parse_docstring(doc)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        ann = param.annotation
        json_type = _PY_TO_JSON.get(ann, "string")
        prop: dict[str, Any] = {"type": json_type}

        if name in param_docs:
            prop["description"] = param_docs[name]

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": func.__name__,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Extract the summary line and Args section from a Google-style docstring."""
    lines = textwrap.dedent(doc).strip().splitlines()
    summary_parts: list[str] = []
    param_docs: dict[str, str] = {}
    in_args = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if stripped.startswith("returns:") or stripped.startswith("raises:"):
                break
            if ":" in stripped and not stripped.startswith(" "):
                pname, pdesc = stripped.split(":", 1)
                param_docs[pname.strip()] = pdesc.strip()
        else:
            if stripped:
                summary_parts.append(stripped)

    return " ".join(summary_parts), param_docs


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------


@tool
def read_file(path: str) -> str:
    """Read the contents of a file from disk.

    Args:
        path: Absolute or relative path to the file to read.
    """
    p = Path(path).resolve()
    if not p.is_file():
        return json.dumps({"error": f"File not found: {path}"})
    content = p.read_text(errors="replace")
    if len(content) > 50_000:
        content = content[:50_000] + "\n... [truncated at 50,000 chars]"
    return content


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating directories as needed.

    Args:
        path: Absolute or relative path for the file.
        content: The full content to write.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return json.dumps({"status": "ok", "path": str(p), "bytes": len(content)})


@tool
def list_directory(path: str = ".") -> str:
    """List files and directories at the given path.

    Args:
        path: Directory path to list. Defaults to current directory.
    """
    p = Path(path).resolve()
    if not p.is_dir():
        return json.dumps({"error": f"Not a directory: {path}"})
    entries = []
    for item in sorted(p.iterdir()):
        kind = "dir" if item.is_dir() else "file"
        size = item.stat().st_size if item.is_file() else None
        entries.append({"name": item.name, "type": kind, "size": size})
    return json.dumps(entries[:200], indent=2)


@tool
def run_shell(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return stdout/stderr.

    Args:
        command: The shell command to run.
        timeout: Max seconds to wait (default 30).
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = {
            "stdout": result.stdout[-10_000:] if result.stdout else "",
            "stderr": result.stderr[-5_000:] if result.stderr else "",
            "returncode": result.returncode,
        }
        return json.dumps(output)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout}s"})


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A Python math expression (e.g. '2**10 + 3*7').
    """
    allowed = set("0123456789+-*/.() %,eE")
    clean = expression.replace("^", "**")
    if not all(c in allowed or c.isspace() for c in clean):
        return json.dumps({"error": "Expression contains disallowed characters"})
    try:
        import math
        result = eval(clean, {"__builtins__": {}}, {"math": math})  # noqa: S307
        return json.dumps({"expression": expression, "result": result})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def current_datetime() -> str:
    """Return the current date, time, and timezone."""
    now = datetime.now(timezone.utc)
    return json.dumps({
        "utc": now.isoformat(),
        "unix": int(now.timestamp()),
    })


BUILTIN_TOOLS = [
    read_file,
    write_file,
    list_directory,
    run_shell,
    calculate,
    current_datetime,
]
