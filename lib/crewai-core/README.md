# crewai-core

Shared utilities used by both `crewai` and `crewai-cli`: version lookup, storage
paths, user-data helpers, telemetry, and the printer.

This package is a leaf — it has no dependency on the `crewai` framework — and is
pulled in transitively by `crewai` and `crewai-cli`. End users do not normally
install it directly.
