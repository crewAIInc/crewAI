"""
OnCell Tool — gives CrewAI agents a persistent sandboxed environment.

Each crew member can read/write files, run code, search content,
and store data in a persistent per-user environment powered by OnCell.

Usage:
    from crewai_tools import OnCellTool

    tool = OnCellTool(customer_id="user-1")
    agent = Agent(tools=[tool])
"""

import os
import json
from typing import Optional, Any
from crewai.tools import BaseTool
from pydantic import Field


class OnCellTool(BaseTool):
    name: str = "OnCell Sandbox"
    description: str = (
        "A persistent sandboxed environment with file storage, database, search, "
        "and shell access. Use this to read/write files, run code, search content, "
        "and store data. State persists across sessions.\n\n"
        "Commands:\n"
        "- write <path> <content> — save a file\n"
        "- read <path> — read a file\n"
        "- list [dir] — list files\n"
        "- search <query> — search across files\n"
        "- shell <command> — run a shell command\n"
        "- store <key> <value> — save to database\n"
        "- load <key> — load from database"
    )
    customer_id: str = Field(description="Unique ID for the user/session")
    cell_id: Optional[str] = Field(default=None, description="OnCell cell ID (auto-created if not set)")
    _oncell: Any = None

    def model_post_init(self, __context: Any) -> None:
        from oncell import OnCell
        self._oncell = OnCell(api_key=os.environ.get("ONCELL_API_KEY", ""))

        if not self.cell_id:
            agent_code = """
module.exports = {
  async run(ctx, params) {
    const { action } = params;
    if (action === 'write') {
      ctx.store.write(params.path, params.content);
      return { ok: true, path: params.path };
    } else if (action === 'read') {
      return { content: ctx.store.read(params.path) };
    } else if (action === 'list') {
      return { files: ctx.store.list(params.dir || '') };
    } else if (action === 'search') {
      return { results: ctx.search.query(params.query, 5) };
    } else if (action === 'shell') {
      return ctx.shell(params.command);
    } else if (action === 'db_get') {
      return { value: ctx.db.get(params.key) };
    } else if (action === 'db_set') {
      ctx.db.set(params.key, params.value);
      return { ok: true };
    }
    return { error: 'unknown action: ' + action };
  },
};
"""
            cell = self._oncell.cells.create(
                customer_id=self.customer_id,
                agent=agent_code,
                permanent=True,
            )
            self.cell_id = cell.id

    def _run(self, command: str) -> str:
        parts = command.strip().split(" ", 2)
        action = parts[0]

        params: dict = {"action": action}
        if action == "write" and len(parts) >= 3:
            params["path"] = parts[1]
            params["content"] = parts[2]
        elif action == "read" and len(parts) >= 2:
            params["path"] = parts[1]
        elif action == "list":
            params["dir"] = parts[1] if len(parts) > 1 else ""
        elif action == "search" and len(parts) >= 2:
            params["query"] = " ".join(parts[1:])
        elif action == "shell" and len(parts) >= 2:
            params["command"] = " ".join(parts[1:])
        elif action == "store" and len(parts) >= 3:
            params["action"] = "db_set"
            params["key"] = parts[1]
            params["value"] = parts[2]
        elif action == "load" and len(parts) >= 2:
            params["action"] = "db_get"
            params["key"] = parts[1]

        result = self._oncell.cells.request(self.cell_id, "run", params)
        return json.dumps(result)
