"""Sandboxed file tools: list, find, read and write inside the configured roots.

Safety rules (docs/sdlc/spec.md §7):

* every path is resolved (``~``, ``..`` and symlinks) *before* the root check, so neither
  traversal nor a symlink pointing outside can escape the sandbox;
* hidden files and directories are never listed, read or written;
* replacing an existing file always asks the user first (human-in-the-loop via
  :func:`voicemate.agent.hitl.ask_user`), so a prompt injection cannot overwrite documents;
* there is no delete tool.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from voicemate.agent.hitl import ask_user, is_affirmative
from voicemate.agent.tools.base import ToolContext, ToolError, display_path, report
from voicemate.agent.tools.documents import pdf_to_text

#: Largest content ``write_file`` accepts (characters).
MAX_WRITE_CHARS: int = 1_000_000

#: Maximum entries returned by listing and search.
MAX_ENTRIES: int = 200

#: Maximum files visited by one ``find_files`` call.
MAX_SCANNED: int = 50_000


class Sandbox:
    """Resolves user-supplied paths and enforces the allowed roots.

    Args:
        roots: Allowed directories; relative paths are interpreted inside the first one.
    """

    def __init__(self, roots: list[Path]) -> None:
        if not roots:
            raise ValueError("At least one sandbox root is required")
        self.roots = [root.expanduser().resolve() for root in roots]

    def resolve(self, path: str) -> Path:
        """Absolute, symlink-free path inside a root.

        Raises:
            ToolError: If the path escapes every root or touches a hidden component.
        """
        raw = Path(path.strip() or ".").expanduser()
        candidate = (raw if raw.is_absolute() else self.roots[0] / raw).resolve()
        for root in self.roots:
            if candidate == root or root in candidate.parents:
                relative = candidate.relative_to(root)
                if any(part.startswith(".") for part in relative.parts):
                    raise ToolError("Hidden files and folders are not accessible")
                return candidate
        allowed = ", ".join(display_path(r) for r in self.roots)
        raise ToolError(f"Access denied: only {allowed} are accessible")


class FileOperations:
    """The blocking file operations behind the tools (run in worker threads).

    Args:
        sandbox: Path policy.
        max_read: Maximum characters returned by one read.
    """

    def __init__(self, sandbox: Sandbox, max_read: int) -> None:
        self.sandbox = sandbox
        self.max_read = max_read

    def listing(self, path: str) -> str:
        """Folders first, then files; hidden entries are skipped."""
        folder = self.sandbox.resolve(path)
        if not folder.is_dir():
            raise ToolError(f"Not a folder: {display_path(folder)}")
        entries = sorted(
            (p for p in folder.iterdir() if not p.name.startswith(".")),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
        lines = [f"{display_path(folder)}:"]
        lines += [f"  {e.name}/" if e.is_dir() else f"  {e.name}" for e in entries[:MAX_ENTRIES]]
        if len(entries) > MAX_ENTRIES:
            lines.append(f"  … {len(entries) - MAX_ENTRIES} more")
        return "\n".join(lines)

    def find(self, pattern: str, path: str) -> str:
        """Case-insensitive name search; a plain fragment matches anywhere in the name."""
        roots = [self.sandbox.resolve(path)] if path else self.sandbox.roots
        glob = (pattern if any(c in pattern for c in "*?[") else f"*{pattern}*").lower()
        found: list[str] = []
        scanned = 0
        for root in roots:
            for current, dirs, names in os.walk(root):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                scanned += len(names)
                found += [
                    display_path(Path(current) / name)
                    for name in names
                    if not name.startswith(".") and fnmatch.fnmatch(name.lower(), glob)
                ]
                if len(found) >= MAX_ENTRIES or scanned >= MAX_SCANNED:
                    break
        return "\n".join(found[:MAX_ENTRIES]) if found else f"No files matching {pattern!r}."

    def read(self, path: str) -> str:
        """Text or PDF content, truncated to ``max_read`` characters."""
        file = self.sandbox.resolve(path)
        if not file.is_file():
            raise ToolError(f"No such file: {display_path(file)}")
        if file.suffix.lower() == ".pdf":
            text, pages = pdf_to_text(file.read_bytes())
            header = f"[{display_path(file)}, PDF, {pages} pages]"
        else:
            with file.open("rb") as handle:
                data = handle.read(self.max_read + 1)
            if b"\x00" in data[:8192]:
                raise ToolError("Binary file; only text and PDF files can be read")
            text = data.decode("utf-8", errors="replace")
            header = f"[{display_path(file)}]"
        suffix = "\n[… truncated]" if len(text) > self.max_read else ""
        return f"{header}\n{text[: self.max_read]}{suffix}"

    def target(self, path: str, content: str) -> tuple[Path, bool]:
        """Validate a write; returns the resolved file and whether it already exists."""
        file = self.sandbox.resolve(path)
        if len(content) > MAX_WRITE_CHARS:
            raise ToolError("Content too large")
        if file.is_dir():
            raise ToolError(f"{display_path(file)} is a folder")
        return file, file.exists()

    @staticmethod
    def write(file: Path, content: str) -> str:
        """Write ``content`` (creating parent folders)."""
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {display_path(file)}."


def build(ctx: ToolContext) -> list[BaseTool]:
    """Create ``list_files``, ``find_files``, ``read_file`` and ``write_file``."""
    ops = FileOperations(Sandbox(ctx.config.files.root_paths), ctx.config.files.max_read_bytes)

    async def list_files(config: RunnableConfig, path: str = "") -> str:
        """List a folder in the user's Documents or Downloads.

        Args:
            path: Folder such as "~/Downloads" or "~/Documents/cv"; empty for Documents.
        """
        result = await asyncio.to_thread(ops.listing, path)
        report(config, "list_files", {"path": path}, "local")
        return result

    async def find_files(pattern: str, config: RunnableConfig, path: str = "") -> str:
        """Find files by name in Documents and Downloads.

        Args:
            pattern: Name fragment or glob, e.g. "invoice" or "*.pdf".
            path: Optional folder to limit the search.
        """
        result = await asyncio.to_thread(ops.find, pattern, path)
        report(config, "find_files", {"pattern": pattern, "path": path}, "local")
        return result

    async def read_file(path: str, config: RunnableConfig) -> str:
        """Read a text or PDF file from Documents or Downloads.

        Args:
            path: File path, e.g. "~/Downloads/voicemate-papers/2401.01234-paper.pdf".
        """
        result = await asyncio.to_thread(ops.read, path)
        report(config, "read_file", {"path": path}, "local")
        return result

    async def write_file(path: str, content: str, config: RunnableConfig) -> str:
        """Create a text file (e.g. a summary or draft) in Documents or Downloads.

        If the file already exists, the user is asked before it is replaced.

        Args:
            path: Target path, e.g. "~/Documents/voicemate/summary.md".
            content: Full text to write.
        """
        file, exists = await asyncio.to_thread(ops.target, path, content)
        if exists:
            answer = ask_user(f"{display_path(file)} already exists. Replace it?")
            if not is_affirmative(answer):
                report(config, "write_file", {"path": path}, "local", "not replaced")
                return f"The user did not approve replacing the file. They said: {answer!r}"
        result = await asyncio.to_thread(ops.write, file, content)
        report(config, "write_file", {"path": path}, "local")
        return result

    return [
        StructuredTool.from_function(coroutine=fn, name=fn.__name__, parse_docstring=True)
        for fn in (list_files, find_files, read_file, write_file)
    ]
