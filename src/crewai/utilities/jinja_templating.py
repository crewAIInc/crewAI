import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import jinja2

def to_jinja_template(input_string: str) -> str:
    """
    Convert CrewAI-style {var} templates to Jinja2-style {{var}} templates.
    
    This function preserves existing Jinja2 syntax if present and only converts
    CrewAI-style variables.
    
    Args:
        input_string: String containing CrewAI-style templates.
        
    Returns:
        String with CrewAI-style templates converted to Jinja2 syntax.
    """
    if not input_string or ("{" not in input_string and "}" not in input_string):
        return input_string
        
    pattern = r'(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})'
    
    return re.sub(pattern, r'{{\1}}', input_string)

def render_template(
    input_string: Optional[str],
    inputs: Dict[str, Any],
) -> str:
    """
    Render a template string using Jinja2 with the provided inputs.
    
    This function supports:
    - Container types (List, Dict, Set)
    - Standard objects (datetime, time)
    - Custom objects
    - Conditional and loop statements
    - Filtering options
    
    Args:
        input_string: The string containing template variables to interpolate.
                     Can be None or empty, in which case an empty string is returned.
        inputs: Dictionary mapping template variables to their values.
               Supports all types of values.
               
    Returns:
        The rendered template string.
        
    Raises:
        ValueError: If inputs dictionary is empty when interpolating variables.
        jinja2.exceptions.TemplateError: If there's an error in the template syntax.
        KeyError: If a required template variable is missing from inputs.
    """
    if input_string is None or not input_string:
        return ""
        
    if "{" not in input_string and "}" not in input_string:
        return input_string
        
    if not inputs:
        raise ValueError("Inputs dictionary cannot be empty when interpolating variables")
        
    jinja_template = to_jinja_template(input_string)
    
    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined,  # Raise errors for undefined variables
        autoescape=True  # Enable autoescaping for security
    )
    
    env.filters['date'] = lambda d, format='%Y-%m-%d': d.strftime(format) if isinstance(d, datetime) else str(d)
    
    template = env.from_string(jinja_template)
    
    try:
        return template.render(**inputs)
    except jinja2.exceptions.UndefinedError as e:
        var_name = str(e).split("'")[1] if "'" in str(e) else None
        if var_name:
            raise KeyError(f"Template variable '{var_name}' not found in inputs dictionary")
        raise KeyError(f"Missing required template variable: {str(e)}")
