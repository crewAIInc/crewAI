import os
import json
import requests

from typing import Type, Any, cast, Optional
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool

class LlamaIndexTool(BaseTool):
    """Tool to wrap LlamaIndex tools/query engines."""
    llama_index_tool: Any

    def _run(
		self,
        *args: Any,
		**kwargs: Any,
	) -> Any:
        """Run tool."""
        from llama_index.core.tools import BaseTool as LlamaBaseTool
        tool = cast(LlamaBaseTool, self.llama_index_tool)
        return tool(*args, **kwargs)
	
    @classmethod
    def from_tool(
        cls,
        tool: Any,
        **kwargs: Any
    ) -> "LlamaIndexTool":
        from llama_index.core.tools import BaseTool as LlamaBaseTool
        
        if not isinstance(tool, LlamaBaseTool):
            raise ValueError(f"Expected a LlamaBaseTool, got {type(tool)}")
        tool = cast(LlamaBaseTool, tool)

        if tool.metadata.fn_schema is None:
            raise ValueError("The LlamaIndex tool does not have an fn_schema specified.")
        args_schema = cast(Type[BaseModel], tool.metadata.fn_schema)
        
        return cls(
            name=tool.metadata.name,
            description=tool.metadata.description,
            args_schema=args_schema,
            llama_index_tool=tool,
            **kwargs
        )


    @classmethod
    def from_query_engine(
        cls,
        query_engine: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        **kwargs: Any
    ) -> "LlamaIndexTool":
        from llama_index.core.query_engine import BaseQueryEngine
        from llama_index.core.tools import QueryEngineTool

        if not isinstance(query_engine, BaseQueryEngine):
            raise ValueError(f"Expected a BaseQueryEngine, got {type(query_engine)}")

        # NOTE: by default the schema expects an `input` variable. However this 
        # confuses crewAI so we are renaming to `query`.
        class QueryToolSchema(BaseModel):
            """Schema for query tool."""
            query: str = Field(..., description="Search query for the query tool.")

        # NOTE: setting `resolve_input_errors` to True is important because the schema expects `input` but we are using `query`
        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine,
            name=name,
            description=description,
            return_direct=return_direct,
            resolve_input_errors=True,  
        )
        # HACK: we are replacing the schema with our custom schema
        query_engine_tool.metadata.fn_schema = QueryToolSchema
        
        return cls.from_tool(
            query_engine_tool,
            **kwargs
        )
        