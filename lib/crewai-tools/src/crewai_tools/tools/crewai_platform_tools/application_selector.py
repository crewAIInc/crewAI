from uuid import UUID


class ApplicationSelector:
    """Parse an application selector.

    Selectors use the ``application[/action][@connection_uuid]`` syntax.

    Raises:
        ValueError: If the selector does not follow the supported syntax.
    """

    name: str
    action: str | None
    connection_id: UUID | None

    def __init__(self, value: str) -> None:
        if not value:
            raise ValueError(f"Invalid application selector {value!r}: cannot be empty")
        if "@" in value and "/" in value and value.index("@") < value.index("/"):
            raise ValueError(
                f"Invalid application selector {value!r}: "
                "connection ID must be the last segment"
            )

        app, connection_separator, connection_id = value.partition("@")
        name, action_separator, action = app.partition("/")

        if not name:
            raise ValueError(
                f"Invalid application selector {value!r}: application cannot be empty"
            )
        if action_separator and not action:
            raise ValueError(
                f"Invalid application selector {value!r}: action cannot be empty"
            )
        if connection_separator and not connection_id:
            raise ValueError(
                f"Invalid application selector {value!r}: connection ID cannot be empty"
            )

        parsed_connection_id = None
        if connection_id:
            try:
                parsed_connection_id = UUID(connection_id)
            except ValueError as error:
                raise ValueError(
                    f"Invalid application selector {value!r}: "
                    "connection ID must be a valid UUID"
                ) from error

        self.name = name
        self.action = action if action_separator else None
        self.connection_id = parsed_connection_id
