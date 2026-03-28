import os
import re
import subprocess
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr
import requests


_EXCLUDE_DIRS = frozenset(
    "node_modules __pycache__ venv .venv dist build target coverage".split()
)
_EXCLUDE_EXT = frozenset(".min.js .min.css .bundle.js .map .pyc .lock".split())
_EXCLUDE_GLOBS: list[str] = [*_EXCLUDE_DIRS, *[f"*{ext}" for ext in _EXCLUDE_EXT], ".*"]

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([a-z_][a-z0-9_]*)>([\s\S]*?)</function>\s*</tool_call>",
    re.IGNORECASE,
)

_PARAM_RE = re.compile(
    r"<parameter=([a-z_][a-z0-9_]*)>([\s\S]*?)</parameter>",
    re.IGNORECASE,
)


class WarpGrepToolSchema(BaseModel):
    """Input schema for WarpGrepTool."""

    search_query: str = Field(
        ...,
        description="Natural language query describing what to search for in the codebase",
    )
    directory: str | None = Field(
        default=None,
        description="Directory to search in. Defaults to the tool's configured directory or cwd.",
    )


class WarpGrepTool(BaseTool):
    """API-based codebase search via Morph's WarpGrep agent (requires ripgrep)."""

    name: str = "WarpGrep Codebase Search"
    description: str = (
        "Searches a local codebase using Morph's WarpGrep agent. Provide a natural-language query "
        "and the tool returns relevant code spans with file paths and line numbers."
    )
    args_schema: type[BaseModel] = WarpGrepToolSchema
    api_url: str = "https://api.morphllm.com/v1/chat/completions"
    model: str = "morph-warp-grep-v2"
    max_turns: int = 4
    max_output_lines: int = 200
    max_read_lines: int = 800
    max_tree_entries: int = 200
    max_tree_depth: int = 2
    api_timeout: int = 60
    rg_timeout: int = 30
    directory: str | None = None
    api_key: str | None = Field(default_factory=lambda: os.getenv("MORPH_API_KEY"))
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(name="MORPH_API_KEY", description="Morph API key", required=True)
        ],
    )
    _rg_checked: bool = PrivateAttr(default=False)

    def _run(self, search_query: str, directory: str | None = None) -> str:
        """Orchestrate the multi-turn WarpGrep agent loop."""
        if not self.api_key:
            return "Error: MORPH_API_KEY is required (https://morphllm.com)"
        self._check_rg()
        try:
            return self._execute_search(search_query, directory)
        except requests.RequestException as e:
            return f"Error calling WarpGrep API: {e!s}"

    def _check_rg(self) -> None:
        if self._rg_checked:
            return
        try:
            subprocess.run(
                ["rg", "--version"],  # noqa: S607
                capture_output=True,
                check=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            msg = "ripgrep (rg) not found or broken. See https://github.com/BurntSushi/ripgrep#installation"
            raise RuntimeError(msg) from exc
        self._rg_checked = True

    def _execute_search(self, search_query: str, directory: str | None = None) -> str:
        repo_root = os.path.abspath(directory or self.directory or os.getcwd())
        if not os.path.isdir(repo_root):
            return f"Error: directory does not exist: {repo_root}"
        tree = self._build_file_tree(repo_root)
        repo_name = os.path.basename(repo_root)
        tree_text = f"{repo_name}/\n{tree}" if tree else f"{repo_name}/"
        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    f"<repo_structure>\n{tree_text}\n</repo_structure>\n\n"
                    f"<search_string>\n{search_query}\n</search_string>\n"
                    f"Turn 0/{self.max_turns}"
                ),
            },
        ]
        for turn in range(1, self.max_turns + 1):
            resp = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "messages": messages,
                },
                timeout=self.api_timeout,
            )
            resp.raise_for_status()
            choices = resp.json().get("choices")
            assistant_content: str = ""
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message")
                if isinstance(msg, dict):
                    assistant_content = str(msg.get("content", ""))
            if not assistant_content:
                return "Error: empty response from WarpGrep model"
            messages.append({"role": "assistant", "content": assistant_content})
            tool_calls = _parse_tool_calls(assistant_content)
            if not tool_calls:
                return assistant_content
            finish_calls = [tc for tc in tool_calls if tc["name"] == "finish"]
            results = [
                f"<tool_response>\n{self._execute_tool(tc, repo_root)}\n</tool_response>"
                for tc in tool_calls
                if tc["name"] != "finish"
            ]
            if finish_calls:
                return self._process_finish(finish_calls[0], repo_root)
            if results:
                left = self.max_turns - turn
                turn_msg = (
                    f"\nYou have used {turn} turns, you only have 1 turn remaining. "
                    "You have run out of turns to explore the code base and MUST call the finish tool now"
                    if left == 1
                    else f"\nYou have used {turn} turn{'s' if turn != 1 else ''} and have {left} remaining"
                )
                messages.append(
                    {"role": "user", "content": "\n".join(results) + turn_msg}
                )
        return "Error: WarpGrep exhausted all turns without producing a result"

    def _build_file_tree(self, root: str) -> str:
        lines: list[str] = []

        def walk(directory: str, depth: int) -> None:
            if depth > self.max_tree_depth or len(lines) >= self.max_tree_entries:
                return
            try:
                entries = sorted(os.listdir(directory))
            except PermissionError:
                return
            for name in entries:
                if len(lines) >= self.max_tree_entries:
                    return
                if name.startswith(".") or name in _EXCLUDE_DIRS:
                    continue
                if any(name.endswith(ext) for ext in _EXCLUDE_EXT):
                    continue
                full = os.path.join(directory, name)
                is_dir = os.path.isdir(full)
                lines.append(f"{'  ' * depth}{name}{'/' if is_dir else ''}")
                if is_dir:
                    walk(full, depth + 1)

        walk(root, 0)
        return "\n".join(lines)

    def _execute_tool(self, tool_call: dict[str, Any], repo_root: str) -> str:
        name, args = tool_call["name"], tool_call["arguments"]
        if name == "ripgrep":
            return self._execute_ripgrep(args, repo_root)
        if name == "read":
            fp = str(args.get("path", ""))
            ls = str(args.get("lines", ""))
            return self._read_lines(
                _resolve_path(repo_root, fp), fp, _parse_ranges(ls) if ls else None
            )
        if name == "list_directory":
            dp = str(args.get("path", "."))
            ap = _resolve_path(repo_root, dp)
            if ap is None:
                return f"[PATH ERROR] Path escapes repository root: {dp}"
            if not os.path.isdir(ap):
                return f'[NOT A DIRECTORY] "{dp}" is not a directory'
            return self._build_file_tree(ap)
        return f"Unknown tool: {name}"

    def _execute_ripgrep(self, args: dict[str, Any], repo_root: str) -> str:
        pattern, target_path = str(args.get("pattern", "")), str(args.get("path", "."))
        abs_path = _resolve_path(repo_root, target_path)
        if abs_path is None:
            return f"[PATH ERROR] Path escapes repository root: {target_path}"
        rel = os.path.relpath(abs_path, repo_root) if abs_path != repo_root else "."
        cmd = "rg --no-config --no-heading --with-filename --line-number --color=never --trim --max-columns=400".split()
        cmd.extend(["-C", str(args.get("context_lines", 1))])
        if args.get("case_sensitive") is False:
            cmd.append("--ignore-case")
        if glob_val := args.get("glob"):
            cmd.extend(["--glob", str(glob_val)])
        for excl in _EXCLUDE_GLOBS:
            cmd.extend(["-g", f"!{excl}"])
        cmd.extend([pattern, rel])
        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                cwd=repo_root,
                timeout=self.rg_timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[TIMEOUT] ripgrep timed out after {self.rg_timeout} seconds"
        if result.returncode not in (0, 1):
            err = result.stderr.strip()
            return f"[RIPGREP ERROR] exit code {result.returncode}{': ' + err if err else ''}"
        stdout = result.stdout.strip()
        if not stdout:
            return "no matches"
        out_lines = stdout.split("\n")
        if len(out_lines) > self.max_output_lines:
            truncated = out_lines[: self.max_output_lines]
            truncated.append(
                f"... (truncated at {self.max_output_lines} of {len(out_lines)} lines)"
            )
            return "\n".join(truncated)
        return stdout

    def _process_finish(self, finish_call: dict[str, Any], repo_root: str) -> str:
        args = finish_call["arguments"]
        if text_result := args.get("text_result"):
            return str(text_result)
        files_raw = str(args.get("files_raw", ""))
        if not files_raw:
            return "No relevant code found."
        file_specs = _parse_finish_files(files_raw)
        if not file_specs:
            return files_raw
        parts: list[str] = []
        for fp, ranges in file_specs:
            abs_path = _resolve_path(repo_root, fp)
            if abs_path is None or not os.path.isfile(abs_path):
                parts.append(f"--- {fp} ---\n[File not found]")
            else:
                parts.append(
                    f"--- {fp} ---\n{self._read_lines(abs_path, fp, ranges or None)}"
                )
        return "\n\n".join(parts) if parts else "No relevant code found."

    def _read_lines(
        self,
        abs_path: str | None,
        display_path: str,
        ranges: list[tuple[int, int]] | None = None,
    ) -> str:
        if abs_path is None:
            return f"[PATH ERROR] Path escapes repository root: {display_path}"
        if not os.path.isfile(abs_path):
            return f'[FILE NOT FOUND] "{display_path}" not found.'
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except OSError as e:
            return f'[READ ERROR] Failed to read "{display_path}": {e!s}'
        total = len(all_lines)
        if not all_lines:
            return "(empty file)"
        if ranges is None:
            ranges = [(1, total)]
        output: list[str] = []
        for idx, (start, end) in enumerate(ranges):
            s = max(1, start)
            end_line = min(end, total)
            if s > total:
                continue
            if idx > 0 or s > 1:
                output.append(f"// ... (lines {s}-{end_line}) ...")
            output.extend(
                f"{i}|{all_lines[i - 1].rstrip()}" for i in range(s, end_line + 1)
            )
        if not output:
            return "(empty file)"
        if len(output) > self.max_read_lines:
            output = output[: self.max_read_lines]
            output.append(f"... (output truncated at {self.max_read_lines} lines)")
        return "\n".join(output)


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    tool_calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(cleaned):
        func_name = match.group(1).lower()
        params: dict[str, str] = {
            m.group(1).lower(): m.group(2).strip()
            for m in _PARAM_RE.finditer(match.group(2))
        }
        tc = _build_tool_call(func_name, params)
        if tc is not None:
            tool_calls.append(tc)
    return tool_calls


def _build_tool_call(name: str, p: dict[str, str]) -> dict[str, Any] | None:
    if name == "ripgrep":
        pattern = p.get("pattern")
        if not pattern:
            return None
        args: dict[str, Any] = {"pattern": pattern, "path": p.get("path", ".")}
        if glob_val := p.get("glob"):
            args["glob"] = glob_val
        if ctx := p.get("context_lines"):
            try:
                args["context_lines"] = int(ctx)
            except ValueError:
                pass
        if cs := p.get("case_sensitive"):
            args["case_sensitive"] = cs.lower() == "true"
        return {"name": "ripgrep", "arguments": args}
    if name == "read":
        path = p.get("path")
        if not path:
            return None
        args = {"path": path}
        if lines_str := p.get("lines"):
            args["lines"] = lines_str
        return {"name": "read", "arguments": args}
    if name == "list_directory":
        dir_path = p.get("path", ".")
        return {"name": "list_directory", "arguments": {"path": dir_path}}
    if name == "finish":
        if p.get("result") and not p.get("files"):
            return {"name": "finish", "arguments": {"text_result": p["result"]}}
        fs = p.get("files", "")
        args = {"files_raw": fs} if fs else {"text_result": "No relevant code found."}
        return {"name": "finish", "arguments": args}
    return None


def _parse_ranges(ranges_str: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for part in ranges_str.split(","):
        trimmed = part.strip()
        if not trimmed or trimmed == "*":
            continue
        parts = trimmed.split("-")
        try:
            s = int(parts[0].strip())
            e = int(parts[1].strip()) if len(parts) > 1 else s
            if s > 0 and e >= s:
                ranges.append((s, e))
        except (ValueError, IndexError):
            continue
    return ranges


def _parse_finish_files(files_str: str) -> list[tuple[str, list[tuple[int, int]]]]:
    results: list[tuple[str, list[tuple[int, int]]]] = []
    for line in files_str.strip().split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue
        ci = trimmed.find(":")
        if ci == -1:
            results.append((trimmed, []))
        elif trimmed[ci + 1 :].strip() in ("*", ""):
            results.append((trimmed[:ci], []))
        else:
            results.append((trimmed[:ci], _parse_ranges(trimmed[ci + 1 :])))
    return results


def _resolve_path(repo_root: str, relative_path: str) -> str | None:
    abs_path = os.path.normpath(
        relative_path
        if os.path.isabs(relative_path)
        else os.path.join(repo_root, relative_path),
    )
    root = os.path.normpath(repo_root)
    return abs_path if abs_path == root or abs_path.startswith(root + os.sep) else None
