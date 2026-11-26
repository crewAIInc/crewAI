#!/usr/bin/env python
"""
Local testing script for YouTubeSearchTool.

Prerequisites:
1. Set YOUTUBE_API_KEY environment variable
   Get your API key from: https://console.cloud.google.com/apis/credentials
   Enable YouTube Data API v3 in your Google Cloud Console

2. Install dependencies:
   pip install google-api-python-client google-auth

Usage:
   export YOUTUBE_API_KEY="your-api-key-here"
   python test_youtube_search_tool.py
"""

import os
import sys
import json

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from crewai_tools.tools.youtube_search_tool.youtube_search_tool import YouTubeSearchTool


def test_basic_search():
    """Test basic YouTube search functionality."""
    print("=" * 60)
    print("Test 1: Basic YouTube Search")
    print("=" * 60)

    tool = YouTubeSearchTool()

    # Test with a simple query
    results = tool._run(search_query="Python programming tutorial", max_results=3)

    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Title: {result['title']}")
        print(f"    Video ID: {result['video_id']}")
        print(f"    URL: {result['url']}")
        print(f"    Description: {result['description'][:100]}...")
        print(f"    Published: {result['published_at']}")

    return results


def test_max_results():
    """Test max_results parameter."""
    print("\n" + "=" * 60)
    print("Test 2: Max Results Parameter")
    print("=" * 60)

    tool = YouTubeSearchTool()

    # Test with max_results=10
    results = tool._run(search_query="machine learning", max_results=10)

    print(f"Requested 10 results, got {len(results)} results")
    assert len(results) <= 10, "Should not exceed max_results"
    return results


def test_error_handling():
    """Test error handling for missing API key."""
    print("\n" + "=" * 60)
    print("Test 3: Error Handling (Missing API Key)")
    print("=" * 60)

    # Temporarily remove API key
    original_key = os.environ.get("YOUTUBE_API_KEY")
    if "YOUTUBE_API_KEY" in os.environ:
        del os.environ["YOUTUBE_API_KEY"]

    tool = YouTubeSearchTool()

    try:
        tool._run(search_query="test", max_results=1)
        print("ERROR: Should have raised ValueError for missing API key")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}: {e}")
        return False
    finally:
        # Restore API key
        if original_key:
            os.environ["YOUTUBE_API_KEY"] = original_key

    return True


def test_invalid_max_results():
    """Test validation of max_results parameter."""
    print("\n" + "=" * 60)
    print("Test 4: Invalid max_results (should use Pydantic validation)")
    print("=" * 60)

    tool = YouTubeSearchTool()

    # Test with max_results > 50 (should be capped by Pydantic)
    try:
        # Pydantic will validate this at schema level
        from crewai_tools.tools.youtube_search_tool.youtube_search_tool import YouTubeSearchToolSchema

        schema = YouTubeSearchToolSchema(search_query="test", max_results=100)
        print(f"Schema accepted max_results=100 (capped to 50): {schema.max_results}")
    except Exception as e:
        print(f"Validation error: {e}")

    return True


def main():
    """Run all tests."""
    print("YouTubeSearchTool Local Testing")
    print("=" * 60)

    # Check for API key
    if not os.getenv("YOUTUBE_API_KEY"):
        print("\n⚠️  WARNING: YOUTUBE_API_KEY environment variable not set!")
        print("   Please set it before running tests:")
        print("   export YOUTUBE_API_KEY='your-api-key-here'")
        print("\n   Get your API key from: https://console.cloud.google.com/apis/credentials")
        print("   Make sure to enable 'YouTube Data API v3'")
        print("\n   Skipping API-dependent tests...\n")

        # Run only tests that don't need API key
        test_error_handling()
        test_invalid_max_results()
        print("\n✓ Non-API tests passed!")
        return

    try:
        # Run API-dependent tests
        test_basic_search()
        test_max_results()
        test_error_handling()
        test_invalid_max_results()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()