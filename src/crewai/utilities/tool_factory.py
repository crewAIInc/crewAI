"""
Intelligent tool factory with automatic parameter adaptation.

Handles constructor signature differences across tool classes,
providing seamless instantiation without manual parameter matching.
"""

import inspect
from typing import Any, Dict, Type, TypeVar, cast

T = TypeVar('T')


class ToolFactory:
    """
    Factory for creating tool instances with intelligent parameter adaptation.
    
    Automatically adapts parameters based on target class constructor signatures,
    eliminating 'unexpected keyword argument' errors and providing future-proof
    tool instantiation.
    """
    
    @staticmethod
    def create_compatible(
        cls: Type[T], 
        source_tool: Any, 
        args_schema: Type[Any],
        **additional_kwargs: Any
    ) -> T:
        """
        Create tool instance with automatic parameter adaptation.
        
        Args:
            cls: Target tool class to instantiate
            source_tool: Source tool providing base parameters
            args_schema: Schema for tool arguments
            **additional_kwargs: Additional keyword arguments
            
        Returns:
            Instance of target tool class with adapted parameters
        """
        # Get constructor signature
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        
        # Standard parameter mappings from source tool
        standard_mappings = {
            'name': lambda: getattr(source_tool, 'name', 'Unnamed Tool'),
            'description': lambda: getattr(source_tool, 'description', ''),
            'func': lambda: getattr(source_tool, 'func', None),
            'args_schema': lambda: args_schema,
        }
        
        # Build parameter dict intelligently based on what constructor accepts
        params = {}
        
        # Add standard parameters if the constructor accepts them
        for param_name, value_func in standard_mappings.items():
            if param_name in valid_params:
                value = value_func()
                if value is not None:
                    params[param_name] = value
        
        # Add additional kwargs only if constructor accepts them
        for param_name, value in additional_kwargs.items():
            if param_name in valid_params:
                params[param_name] = value
        
        return cast(T, cls(**params))