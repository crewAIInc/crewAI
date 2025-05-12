class EntityMemoryItem:
    def __init__(
        self,
        name: str,
        type: str,
        description: str,
        relationships: str,
    ) -> None:
        self.name = name
        self.type = type
        self.description = description
        self.metadata = {"relationships": relationships}
