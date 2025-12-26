"""Trade Journal Tool voor loggen en reviewen van trades."""

from datetime import datetime
from typing import Any

from crewai.tools import BaseTool


class TradeJournalTool(BaseTool):
    """Tool voor bijhouden van een trade journal met entries en analyse.

    Ondersteunt het loggen van trades, toevoegen van notities, en reviewen van historische entries.
    """

    name: str = "trade_journal"
    description: str = (
        "Log trades met rationale, notities, en uitkomsten. Review eerdere trades voor "
        "patroon analyse en leren. Essentieel voor trading discipline en verbetering."
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
        """Voer trade journal actie uit.

        Args:
            action: Een van 'log_entry', 'log_exit', 'add_note', 'list', 'get', 'analyze'
            trade_id: Externe trade ID van exchange
            entry_type: Type entry ('entry', 'exit', 'adjustment', 'note')
            symbol: Trading symbool
            side: Trade richting ('buy', 'sell')
            size: Positie grootte
            price: Uitvoerings prijs
            rationale: Reden voor de trade
            notes: Aanvullende notities
            outcome: Trade uitkomst beschrijving
            pnl: Winst/verlies bedrag
            desk: Trading desk ('spot', 'futures')
            strategy: Strategie naam
            entry_id: Journal entry ID voor updates
            limit: Aantal entries om te tonen voor list actie
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
            return f"Onbekende actie: {action}. Beschikbaar: log_entry, log_exit, add_note, list, get, analyze"

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
        """Log een trade entry."""
        if not symbol:
            return "Fout: symbool is vereist"

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
        return f"Trade entry gelogd: {entry}"

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
        """Log een trade exit."""
        # Vind overeenkomende open entry
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
                return f"Trade exit gelogd: {entry}"

        # Als geen overeenkomende entry, creëer standalone exit record
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
        return f"Trade exit gelogd (geen overeenkomende entry): {entry}"

    def _add_note(self, entry_id: int | None, notes: str | None, timestamp: str) -> str:
        """Voeg een notitie toe aan een bestaande entry."""
        if not entry_id:
            return "Fout: entry_id is vereist"
        if not notes:
            return "Fout: notities is vereist"

        for entry in self._entries:
            if entry["id"] == entry_id:
                entry["notes"].append({"timestamp": timestamp, "note": notes})
                return f"Notitie toegevoegd aan entry {entry_id}"

        return f"Entry {entry_id} niet gevonden"

    def _list_entries(
        self,
        desk: str | None,
        strategy: str | None,
        symbol: str | None,
        limit: int,
        timestamp: str,
    ) -> str:
        """Toon journal entries met optionele filters."""
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
        """Haal een specifieke journal entry op."""
        if not entry_id:
            return "Fout: entry_id is vereist"

        for entry in self._entries:
            if entry["id"] == entry_id:
                return {"timestamp": timestamp, "entry": entry}

        return f"Entry {entry_id} niet gevonden"

    def _analyze_entries(self, desk: str | None, strategy: str | None, timestamp: str) -> str:
        """Analyseer journal entries voor patronen en statistieken."""
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
