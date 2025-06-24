import re
from typing import Dict, List, Optional, Union


def extract_xml_content(text: str, tag: str) -> Optional[str]:
    """
    Extract content from a specific XML tag.
    
    Args:
        text: The text to search in
        tag: The XML tag name to extract (without angle brackets)
    
    Returns:
        The content inside the first occurrence of the tag, or None if not found
    """
    pattern = rf'<{re.escape(tag)}>(.*?)</{re.escape(tag)}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_all_xml_content(text: str, tag: str) -> List[str]:
    """
    Extract content from all occurrences of a specific XML tag.
    
    Args:
        text: The text to search in
        tag: The XML tag name to extract (without angle brackets)
    
    Returns:
        List of content strings from all occurrences of the tag
    """
    pattern = rf'<{re.escape(tag)}>(.*?)</{re.escape(tag)}>'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def extract_multiple_xml_tags(text: str, tags: List[str]) -> Dict[str, Optional[str]]:
    """
    Extract content from multiple XML tags.
    
    Args:
        text: The text to search in
        tags: List of XML tag names to extract (without angle brackets)
    
    Returns:
        Dictionary mapping tag names to their content (or None if not found)
    """
    result = {}
    for tag in tags:
        result[tag] = extract_xml_content(text, tag)
    return result


def extract_multiple_xml_tags_all(text: str, tags: List[str]) -> Dict[str, List[str]]:
    """
    Extract content from all occurrences of multiple XML tags.
    
    Args:
        text: The text to search in
        tags: List of XML tag names to extract (without angle brackets)
    
    Returns:
        Dictionary mapping tag names to lists of their content
    """
    result = {}
    for tag in tags:
        result[tag] = extract_all_xml_content(text, tag)
    return result


def extract_xml_with_attributes(text: str, tag: str) -> List[Dict[str, Union[str, Dict[str, str]]]]:
    """
    Extract XML tags with their attributes and content.
    
    Args:
        text: The text to search in
        tag: The XML tag name to extract (without angle brackets)
    
    Returns:
        List of dictionaries containing 'content' and 'attributes' for each occurrence
    """
    pattern = rf'<{re.escape(tag)}([^>]*)>(.*?)</{re.escape(tag)}>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    result = []
    for attrs_str, content in matches:
        attributes = {}
        if attrs_str.strip():
            attr_pattern = r'(\w+)=["\']([^"\']*)["\']'
            attributes = dict(re.findall(attr_pattern, attrs_str))
        
        result.append({
            'content': content.strip(),
            'attributes': attributes
        })
    
    return result


def remove_xml_tags(text: str, tags: List[str]) -> str:
    """
    Remove specific XML tags and their content from text.
    
    Args:
        text: The text to process
        tags: List of XML tag names to remove (without angle brackets)
    
    Returns:
        Text with the specified XML tags and their content removed
    """
    result = text
    for tag in tags:
        pattern = rf'<{re.escape(tag)}[^>]*>.*?</{re.escape(tag)}>'
        result = re.sub(pattern, '', result, flags=re.DOTALL)
    return result.strip()


def strip_xml_tags_keep_content(text: str, tags: List[str]) -> str:
    """
    Remove specific XML tags but keep their content.
    
    Args:
        text: The text to process
        tags: List of XML tag names to strip (without angle brackets)
    
    Returns:
        Text with the specified XML tags removed but content preserved
    """
    result = text
    for tag in tags:
        pattern = rf'<{re.escape(tag)}[^>]*>(.*?)</{re.escape(tag)}>'
        result = re.sub(pattern, r'\1', result, flags=re.DOTALL)
    return result.strip()
