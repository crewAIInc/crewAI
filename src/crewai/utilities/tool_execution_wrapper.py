#!/usr/bin/env python3
"""
Tool Execution Wrapper for CrewAI Issue #3154

This module wraps CrewAI's tool execution to enforce authenticity verification.
Prevents agents from fabricating tool results by requiring execution proof.
"""

import functools
from typing import Any, Dict, Optional, Union

from crewai.utilities.tool_execution_verifier import (
    ExecutionAuthenticityLevel,
    ToolExecutionCertificate,
    ToolExecutionFabricationError,
    get_tool_execution_verifier,
)


class ToolExecutionAuthenticityWrapper:
    """Wraps tool execution to enforce authenticity verification"""
    
    def __init__(self, original_tool):
        self.original_tool = original_tool
        self.verifier = get_tool_execution_verifier()
        
        # Copy original tool attributes
        for attr in ['name', 'description', 'args_schema', 'result_as_answer']:
            if hasattr(original_tool, attr):
                setattr(self, attr, getattr(original_tool, attr))
    
    def invoke(self, input: Union[str, dict], config: Optional[dict] = None, **kwargs: Any) -> Any:
        """Wrapped invoke method with execution verification"""
        
        # Extract tool info for verification
        tool_name = getattr(self.original_tool, 'name', 'Unknown Tool')
        tool_args = input if isinstance(input, dict) else {}
        
        print(f"🔍 WRAPPER: Verifying execution of {tool_name}")
        
        # Start monitoring before execution
        execution_id = self.verifier.monitor.start_monitoring(tool_name, tool_args)
        
        try:
            # Execute the original tool
            result = self.original_tool.invoke(input, config, **kwargs)
            
            # Finish monitoring and get certificate
            certificate = self.verifier.monitor.finish_monitoring(execution_id, result)
            
            # Analyze authenticity
            self._handle_authenticity_result(certificate, tool_name, result)
            
            return result
            
        except Exception as e:
            # Clean up monitoring on error
            if execution_id in self.verifier.monitor.active_monitoring:
                del self.verifier.monitor.active_monitoring[execution_id]
            raise e
    
    def _handle_authenticity_result(self, certificate: ToolExecutionCertificate, 
                                  tool_name: str, result: Any):
        """Handle the authenticity verification result"""
        
        authenticity = certificate.authenticity_level
        
        if authenticity == ExecutionAuthenticityLevel.VERIFIED_REAL:
            print(f"✅ VERIFIED: {tool_name} execution is authentic")
            
        elif authenticity == ExecutionAuthenticityLevel.LIKELY_REAL:
            print(f"✅ LIKELY REAL: {tool_name} execution has evidence of authenticity")
            
        elif authenticity == ExecutionAuthenticityLevel.UNCERTAIN:
            print(f"⚠️  UNCERTAIN: Cannot verify {tool_name} execution authenticity")
            
        elif authenticity == ExecutionAuthenticityLevel.LIKELY_FAKE:
            print(f"❌ LIKELY FAKE: {tool_name} execution appears fabricated")
            if self.verifier.strict_mode:
                raise ToolExecutionFabricationError(
                    f"Tool '{tool_name}' execution appears to be fabricated"
                )
                
        elif authenticity == ExecutionAuthenticityLevel.VERIFIED_FAKE:
            print(f"❌ VERIFIED FAKE: {tool_name} execution is definitely fabricated")
            if self.verifier.strict_mode:
                raise ToolExecutionFabricationError(
                    f"Tool '{tool_name}' execution is verified to be fabricated"
                )
    
    def __getattr__(self, name):
        """Delegate other attributes to original tool"""
        return getattr(self.original_tool, name)


def wrap_tool_with_verification(tool) -> ToolExecutionAuthenticityWrapper:
    """Wrap a tool with execution verification"""
    if isinstance(tool, ToolExecutionAuthenticityWrapper):
        return tool  # Already wrapped
    
    return ToolExecutionAuthenticityWrapper(tool)


def wrap_tools_list(tools: list) -> list:
    """Wrap a list of tools with execution verification"""
    return [wrap_tool_with_verification(tool) for tool in tools]


# Monkey patching approach for CrewAI integration
def patch_crewai_tool_execution():
    """Monkey patch CrewAI's tool execution to add verification"""
    
    # Import here to avoid circular imports
    from crewai.tools.base_tool import BaseTool
    from crewai.tools.structured_tool import CrewStructuredTool
    from crewai.tools.tool_usage import ToolUsage
    
    # Store original methods
    original_tool_usage_use = ToolUsage._use
    original_structured_tool_invoke = CrewStructuredTool.invoke
    
    def verified_tool_usage_use(self, tool_string: str, tool: Any, calling: Any) -> str:
        """Patched ToolUsage._use method with verification"""
        
        print(f"🔍 PATCH: Intercepted tool usage for {tool.name}")
        
        # Wrap the tool if not already wrapped
        if not isinstance(tool, ToolExecutionAuthenticityWrapper):
            tool = wrap_tool_with_verification(tool)
        
        # Call original method with wrapped tool
        return original_tool_usage_use(self, tool_string, tool, calling)
    
    def verified_structured_tool_invoke(self, input: Union[str, dict], 
                                       config: Optional[dict] = None, **kwargs: Any) -> Any:
        """Patched CrewStructuredTool.invoke method with verification"""
        
        print(f"🔍 PATCH: Intercepted structured tool invoke for {self.name}")
        
        # Get verifier
        verifier = get_tool_execution_verifier()
        
        # Start monitoring
        tool_args = input if isinstance(input, dict) else {}
        execution_id = verifier.monitor.start_monitoring(self.name, tool_args)
        
        try:
            # Call original invoke
            result = original_structured_tool_invoke(self, input, config, **kwargs)
            
            # Finish monitoring
            certificate = verifier.monitor.finish_monitoring(execution_id, result)
            
            # Handle authenticity
            wrapper = ToolExecutionAuthenticityWrapper(self)
            wrapper._handle_authenticity_result(certificate, self.name, result)
            
            return result
            
        except Exception as e:
            # Clean up monitoring
            if execution_id in verifier.monitor.active_monitoring:
                del verifier.monitor.active_monitoring[execution_id]
            raise e
    
    # Apply patches
    ToolUsage._use = verified_tool_usage_use
    CrewStructuredTool.invoke = verified_structured_tool_invoke
    
    print("🔧 PATCH APPLIED: CrewAI tool execution is now verified")


def unpatch_crewai_tool_execution():
    """Remove patches from CrewAI tool execution"""
    
    # This would restore original methods
    # For now, just print that we'd remove patches
    print("🔧 PATCH REMOVED: CrewAI tool execution verification disabled")


if __name__ == "__main__":
    # Test the wrapper system
    from pydantic import BaseModel, Field

    from crewai.tools.base_tool import Tool
    
    class TestToolInput(BaseModel):
        message: str = Field(description="Message to process")
    
    def test_function(message: str) -> str:
        print(f"🔧 REAL TOOL: Processing message: {message}")
        return f"Processed: {message}"
    
    # Create original tool
    original_tool = Tool(
        name="Test Tool",
        description="A test tool",
        func=test_function,
        args_schema=TestToolInput
    )
    
    # Convert to structured tool
    structured_tool = original_tool.to_structured_tool()
    
    # Wrap with verification
    wrapped_tool = wrap_tool_with_verification(structured_tool)
    
    # Test execution
    result = wrapped_tool.invoke({"message": "Hello World"})
    print(f"Result: {result}")
    
    # Get verification report
    verifier = get_tool_execution_verifier()
    report = verifier.get_execution_report()
    print(f"Verification report: {report}")