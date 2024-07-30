from typing import Any, Callable, Dict, Tuple

from crewai.pipeline.pipeline import Pipeline

Route = Tuple[Callable[[Dict[str, Any]], bool], Pipeline]
