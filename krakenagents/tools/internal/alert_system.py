"""Alert System Tool voor escalatie en notificaties."""

from datetime import datetime
from enum import Enum
from typing import Any

from crewai.tools import BaseTool


class AlertSeverity(str, Enum):
    """Alert ernstniveaus."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertSystemTool(BaseTool):
    """Tool voor beheer van alerts en escalaties binnen de organisatie.

    Verzorgt alert creatie, routing, bevestiging, en escalatie.
    """

    name: str = "alert_system"
    description: str = (
        "Creëer en beheer alerts voor risico events, systeem problemen, en belangrijke notificaties. "
        "Ondersteunt ernstniveaus (info, warning, critical, emergency) en escalatie routing."
    )

    # In-memory alert storage (would be persistent in production)
    _alerts: list = []
    _alert_id_counter: int = 0

    def _run(
        self,
        action: str = "list",
        severity: str | None = None,
        message: str | None = None,
        source: str | None = None,
        target: str | None = None,
        alert_id: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Voer alert systeem actie uit.

        Args:
            action: Een van 'create', 'list', 'acknowledge', 'escalate', 'resolve'
            severity: Alert ernstniveau (info, warning, critical, emergency)
            message: Alert bericht inhoud
            source: Bron agent/systeem die de alert creëert
            target: Doel agent/rol voor de alert
            alert_id: Alert ID voor acknowledge/escalate/resolve acties
        """
        timestamp = datetime.now().isoformat()

        if action == "create":
            return self._create_alert(severity, message, source, target, timestamp)
        elif action == "list":
            return self._list_alerts(severity, timestamp)
        elif action == "acknowledge":
            return self._acknowledge_alert(alert_id, source, timestamp)
        elif action == "escalate":
            return self._escalate_alert(alert_id, target, timestamp)
        elif action == "resolve":
            return self._resolve_alert(alert_id, source, timestamp)
        else:
            return f"Onbekende actie: {action}. Beschikbaar: create, list, acknowledge, escalate, resolve"

    def _create_alert(
        self,
        severity: str | None,
        message: str | None,
        source: str | None,
        target: str | None,
        timestamp: str,
    ) -> str:
        """Creëer een nieuwe alert."""
        if not message:
            return "Fout: bericht is vereist voor het creëren van een alert"

        self._alert_id_counter += 1
        alert = {
            "id": self._alert_id_counter,
            "severity": severity or "info",
            "message": message,
            "source": source or "onbekend",
            "target": target,
            "status": "actief",
            "created_at": timestamp,
            "acknowledged_at": None,
            "resolved_at": None,
            "escalation_history": [],
        }
        self._alerts.append(alert)
        return f"Alert aangemaakt: {alert}"

    def _list_alerts(self, severity: str | None, timestamp: str) -> str:
        """Toon alerts, optioneel gefilterd op ernstniveau."""
        filtered = self._alerts
        if severity:
            filtered = [a for a in self._alerts if a["severity"] == severity]

        active = [a for a in filtered if a["status"] == "actief"]
        return {
            "timestamp": timestamp,
            "filter": {"severity": severity},
            "active_count": len(active),
            "total_count": len(filtered),
            "alerts": active[:20],  # Limiet tot 20 meest recente
        }

    def _acknowledge_alert(self, alert_id: int | None, source: str | None, timestamp: str) -> str:
        """Bevestig een alert."""
        if not alert_id:
            return "Fout: alert_id is vereist"

        for alert in self._alerts:
            if alert["id"] == alert_id:
                alert["status"] = "bevestigd"
                alert["acknowledged_at"] = timestamp
                alert["acknowledged_by"] = source
                return f"Alert {alert_id} bevestigd"

        return f"Alert {alert_id} niet gevonden"

    def _escalate_alert(self, alert_id: int | None, target: str | None, timestamp: str) -> str:
        """Escaleer een alert naar een hoger niveau."""
        if not alert_id:
            return "Fout: alert_id is vereist"
        if not target:
            return "Fout: target is vereist voor escalatie"

        for alert in self._alerts:
            if alert["id"] == alert_id:
                # Verhoog ernstniveau indien mogelijk
                severity_order = ["info", "warning", "critical", "emergency"]
                current_idx = severity_order.index(alert["severity"])
                if current_idx < len(severity_order) - 1:
                    alert["severity"] = severity_order[current_idx + 1]

                alert["target"] = target
                alert["escalation_history"].append({
                    "timestamp": timestamp,
                    "new_target": target,
                    "new_severity": alert["severity"],
                })
                return f"Alert {alert_id} geëscaleerd naar {target} met ernstniveau {alert['severity']}"

        return f"Alert {alert_id} niet gevonden"

    def _resolve_alert(self, alert_id: int | None, source: str | None, timestamp: str) -> str:
        """Los een alert op."""
        if not alert_id:
            return "Fout: alert_id is vereist"

        for alert in self._alerts:
            if alert["id"] == alert_id:
                alert["status"] = "opgelost"
                alert["resolved_at"] = timestamp
                alert["resolved_by"] = source
                return f"Alert {alert_id} opgelost"

        return f"Alert {alert_id} niet gevonden"
