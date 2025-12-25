"""Alert System Tool for escalation and notifications."""

from datetime import datetime
from enum import Enum
from typing import Any

from crewai.tools import BaseTool


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertSystemTool(BaseTool):
    """Tool for managing alerts and escalations within the organization.

    Handles alert creation, routing, acknowledgment, and escalation.
    """

    name: str = "alert_system"
    description: str = (
        "Create and manage alerts for risk events, system issues, and important notifications. "
        "Supports severity levels (info, warning, critical, emergency) and escalation routing."
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
        """Execute alert system action.

        Args:
            action: One of 'create', 'list', 'acknowledge', 'escalate', 'resolve'
            severity: Alert severity (info, warning, critical, emergency)
            message: Alert message content
            source: Source agent/system creating the alert
            target: Target agent/role for the alert
            alert_id: Alert ID for acknowledge/escalate/resolve actions
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
            return f"Unknown action: {action}. Available: create, list, acknowledge, escalate, resolve"

    def _create_alert(
        self,
        severity: str | None,
        message: str | None,
        source: str | None,
        target: str | None,
        timestamp: str,
    ) -> str:
        """Create a new alert."""
        if not message:
            return "Error: message is required for creating an alert"

        self._alert_id_counter += 1
        alert = {
            "id": self._alert_id_counter,
            "severity": severity or "info",
            "message": message,
            "source": source or "unknown",
            "target": target,
            "status": "active",
            "created_at": timestamp,
            "acknowledged_at": None,
            "resolved_at": None,
            "escalation_history": [],
        }
        self._alerts.append(alert)
        return f"Alert created: {alert}"

    def _list_alerts(self, severity: str | None, timestamp: str) -> str:
        """List alerts, optionally filtered by severity."""
        filtered = self._alerts
        if severity:
            filtered = [a for a in self._alerts if a["severity"] == severity]

        active = [a for a in filtered if a["status"] == "active"]
        return {
            "timestamp": timestamp,
            "filter": {"severity": severity},
            "active_count": len(active),
            "total_count": len(filtered),
            "alerts": active[:20],  # Limit to 20 most recent
        }

    def _acknowledge_alert(self, alert_id: int | None, source: str | None, timestamp: str) -> str:
        """Acknowledge an alert."""
        if not alert_id:
            return "Error: alert_id is required"

        for alert in self._alerts:
            if alert["id"] == alert_id:
                alert["status"] = "acknowledged"
                alert["acknowledged_at"] = timestamp
                alert["acknowledged_by"] = source
                return f"Alert {alert_id} acknowledged"

        return f"Alert {alert_id} not found"

    def _escalate_alert(self, alert_id: int | None, target: str | None, timestamp: str) -> str:
        """Escalate an alert to a higher level."""
        if not alert_id:
            return "Error: alert_id is required"
        if not target:
            return "Error: target is required for escalation"

        for alert in self._alerts:
            if alert["id"] == alert_id:
                # Increase severity if possible
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
                return f"Alert {alert_id} escalated to {target} with severity {alert['severity']}"

        return f"Alert {alert_id} not found"

    def _resolve_alert(self, alert_id: int | None, source: str | None, timestamp: str) -> str:
        """Resolve an alert."""
        if not alert_id:
            return "Error: alert_id is required"

        for alert in self._alerts:
            if alert["id"] == alert_id:
                alert["status"] = "resolved"
                alert["resolved_at"] = timestamp
                alert["resolved_by"] = source
                return f"Alert {alert_id} resolved"

        return f"Alert {alert_id} not found"
