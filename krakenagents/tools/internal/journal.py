"""Trade Journal Tool for logging and reviewing trades."""

from datetime import datetime
from typing import Any

from crewai.tools import BaseTool


class TradeJournalTool(BaseTool):
    """Tool for maintaining a trade journal with entries and analysis.

    Supports logging trades, adding notes, and reviewing historical entries.
    """

    name: str = "trade_journal"
    description: str = (
        "Log trades with rationale, notes, and outcomes. Review past trades for "
        "pattern analysis and learning. Essential for trading discipline and improvement."
    )

    # In-memory journal storage
    _entries: list = []
    _entry_id_counter: int = 0

    def _run(
        self,
        action: str = "list",
        trade_id: str | None = None,
        entry_type: str | None = None,
        symbol: str | None = None,
        side: str | None = None,
        size: float | None = None,
        price: float | None = None,
        rationale: str | None = None,
        notes: str | None = None,
        outcome: str | None = None,
        pnl: float | None = None,
        desk: str | None = None,
        strategy: str | None = None,
        entry_id: int | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> str:
        """Execute trade journal action.

        Args:
            action: One of 'log_entry', 'log_exit', 'add_note', 'list', 'get', 'analyze'
            trade_id: External trade ID from exchange
            entry_type: Type of entry ('entry', 'exit', 'adjustment', 'note')
            symbol: Trading symbol
            side: Trade side ('buy', 'sell')
            size: Position size
            price: Execution price
            rationale: Reason for the trade
            notes: Additional notes
            outcome: Trade outcome description
            pnl: Profit/loss amount
            desk: Trading desk ('spot', 'futures')
            strategy: Strategy name
            entry_id: Journal entry ID for updates
            limit: Number of entries to return for list action
        """
        timestamp = datetime.now().isoformat()

        if action == "log_entry":
            return self._log_entry(
                trade_id, symbol, side, size, price, rationale, desk, strategy, timestamp
            )
        elif action == "log_exit":
            return self._log_exit(
                trade_id, symbol, side, size, price, outcome, pnl, notes, timestamp
            )
        elif action == "add_note":
            return self._add_note(entry_id, notes, timestamp)
        elif action == "list":
            return self._list_entries(desk, strategy, symbol, limit, timestamp)
        elif action == "get":
            return self._get_entry(entry_id, timestamp)
        elif action == "analyze":
            return self._analyze_entries(desk, strategy, timestamp)
        else:
            return f"Unknown action: {action}. Available: log_entry, log_exit, add_note, list, get, analyze"

    def _log_entry(
        self,
        trade_id: str | None,
        symbol: str | None,
        side: str | None,
        size: float | None,
        price: float | None,
        rationale: str | None,
        desk: str | None,
        strategy: str | None,
        timestamp: str,
    ) -> str:
        """Log a trade entry."""
        if not symbol:
            return "Error: symbol is required"

        self._entry_id_counter += 1
        entry = {
            "id": self._entry_id_counter,
            "trade_id": trade_id,
            "type": "entry",
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": price,
            "exit_price": None,
            "rationale": rationale,
            "notes": [],
            "outcome": None,
            "pnl": None,
            "desk": desk,
            "strategy": strategy,
            "status": "open",
            "entry_time": timestamp,
            "exit_time": None,
        }
        self._entries.append(entry)
        return f"Trade entry logged: {entry}"

    def _log_exit(
        self,
        trade_id: str | None,
        symbol: str | None,
        side: str | None,
        size: float | None,
        price: float | None,
        outcome: str | None,
        pnl: float | None,
        notes: str | None,
        timestamp: str,
    ) -> str:
        """Log a trade exit."""
        # Find matching open entry
        for entry in reversed(self._entries):
            if (
                entry["status"] == "open"
                and (trade_id is None or entry["trade_id"] == trade_id)
                and (symbol is None or entry["symbol"] == symbol)
            ):
                entry["exit_price"] = price
                entry["exit_time"] = timestamp
                entry["outcome"] = outcome
                entry["pnl"] = pnl
                entry["status"] = "closed"
                if notes:
                    entry["notes"].append({"timestamp": timestamp, "note": notes})
                return f"Trade exit logged: {entry}"

        # If no matching entry, create a standalone exit record
        self._entry_id_counter += 1
        entry = {
            "id": self._entry_id_counter,
            "trade_id": trade_id,
            "type": "exit",
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": None,
            "exit_price": price,
            "rationale": None,
            "notes": [{"timestamp": timestamp, "note": notes}] if notes else [],
            "outcome": outcome,
            "pnl": pnl,
            "desk": None,
            "strategy": None,
            "status": "closed",
            "entry_time": None,
            "exit_time": timestamp,
        }
        self._entries.append(entry)
        return f"Trade exit logged (no matching entry): {entry}"

    def _add_note(self, entry_id: int | None, notes: str | None, timestamp: str) -> str:
        """Add a note to an existing entry."""
        if not entry_id:
            return "Error: entry_id is required"
        if not notes:
            return "Error: notes is required"

        for entry in self._entries:
            if entry["id"] == entry_id:
                entry["notes"].append({"timestamp": timestamp, "note": notes})
                return f"Note added to entry {entry_id}"

        return f"Entry {entry_id} not found"

    def _list_entries(
        self,
        desk: str | None,
        strategy: str | None,
        symbol: str | None,
        limit: int,
        timestamp: str,
    ) -> str:
        """List journal entries with optional filters."""
        filtered = self._entries

        if desk:
            filtered = [e for e in filtered if e.get("desk") == desk]
        if strategy:
            filtered = [e for e in filtered if e.get("strategy") == strategy]
        if symbol:
            filtered = [e for e in filtered if e.get("symbol") == symbol]

        recent = list(reversed(filtered))[:limit]
        return {
            "timestamp": timestamp,
            "filters": {"desk": desk, "strategy": strategy, "symbol": symbol},
            "count": len(recent),
            "total": len(filtered),
            "entries": recent,
        }

    def _get_entry(self, entry_id: int | None, timestamp: str) -> str:
        """Get a specific journal entry."""
        if not entry_id:
            return "Error: entry_id is required"

        for entry in self._entries:
            if entry["id"] == entry_id:
                return {"timestamp": timestamp, "entry": entry}

        return f"Entry {entry_id} not found"

    def _analyze_entries(self, desk: str | None, strategy: str | None, timestamp: str) -> str:
        """Analyze journal entries for patterns and statistics."""
        filtered = self._entries

        if desk:
            filtered = [e for e in filtered if e.get("desk") == desk]
        if strategy:
            filtered = [e for e in filtered if e.get("strategy") == strategy]

        closed = [e for e in filtered if e["status"] == "closed"]
        wins = [e for e in closed if e.get("pnl") and e["pnl"] > 0]
        losses = [e for e in closed if e.get("pnl") and e["pnl"] < 0]

        total_pnl = sum(e.get("pnl", 0) or 0 for e in closed)
        avg_win = sum(e["pnl"] for e in wins) / len(wins) if wins else 0
        avg_loss = sum(e["pnl"] for e in losses) / len(losses) if losses else 0

        analysis = {
            "timestamp": timestamp,
            "filters": {"desk": desk, "strategy": strategy},
            "statistics": {
                "total_trades": len(closed),
                "open_trades": len([e for e in filtered if e["status"] == "open"]),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(closed) if closed else 0,
                "total_pnl": total_pnl,
                "average_win": avg_win,
                "average_loss": avg_loss,
                "profit_factor": abs(avg_win / avg_loss) if avg_loss else 0,
            },
            "by_symbol": {},
            "by_strategy": {},
        }
        return str(analysis)
