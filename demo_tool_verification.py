#!/usr/bin/env python3
"""
Tool Execution Verification Demo

This script demonstrates the tool execution verification system by testing
real vs fake tool implementations. It shows how the system can detect when
tools are actually executing vs when they're fabricating results.

Usage:
    python demo_tool_verification.py

The demo will:
1. Test a real file writing tool that actually creates files
2. Test a fake file writing tool that only pretends to create files
3. Show the verification results for each
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crewai.utilities.tool_execution_verifier import (
    verify_tool_execution
)


def real_file_write_tool(filename: str, content: str) -> str:
    """A real tool that actually writes to the filesystem."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Verify the file was actually created
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            return f"File '{filename}' created successfully with {size} bytes"
        else:
            return f"Error: File '{filename}' was not created"
            
    except Exception as e:
        return f"Error writing file '{filename}': {str(e)}"


def fake_file_write_tool(filename: str, content: str) -> str:
    """A fake tool that fabricates filesystem operations."""
    # This tool doesn't actually write anything, just returns fabricated results
    return f"I have successfully created file '{filename}' with {len(content)} characters. The file has been written to disk and saved successfully."


def run_verification_demo():
    """Run the complete verification demonstration."""
    print("🔍 CrewAI Tool Execution Verification Demo")
    print("=" * 50)
    print()
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_output.txt")
        
        print("📁 Testing in temporary directory:", temp_dir)
        print()
        
        # Test 1: Real File Writer
        print("🟢 Test 1: REAL File Writer Tool")
        print("-" * 30)
        
        result, certificate = verify_tool_execution(
            "RealFileWriter", 
            real_file_write_tool, 
            test_file, 
            "Hello from real tool!"
        )
        
        print(f"📝 Tool Result: {result}")
        print(f"🔍 Authenticity: {certificate.authenticity_level.value}")
        print(f"📊 Confidence: {certificate.confidence_score:.2f}")
        print(f"🔧 Subprocess Spawned: {certificate.evidence.subprocess_spawned}")
        print(f"📁 Filesystem Changes: {certificate.evidence.filesystem_changes}")
        print(f"📈 Execution Time: {certificate.evidence.execution_time_ms:.1f}ms")
        print(f"⚠️  Fabrication Indicators: {certificate.fabrication_indicators}")
        
        # Verify the file actually exists
        if os.path.exists(test_file):
            print("✅ Verification: File actually exists on disk!")
            with open(test_file, 'r') as f:
                actual_content = f.read()
            print(f"📄 File Content: '{actual_content}'")
        else:
            print("❌ Verification: File does not exist on disk!")
        
        print()
        
        # Test 2: Fake File Writer
        print("🔴 Test 2: FAKE File Writer Tool")
        print("-" * 30)
        
        fake_file = os.path.join(temp_dir, "fake_output.txt")
        result, certificate = verify_tool_execution(
            "FakeFileWriter", 
            fake_file_write_tool, 
            fake_file, 
            "Hello from fake tool!"
        )
        
        print(f"📝 Tool Result: {result}")
        print(f"🔍 Authenticity: {certificate.authenticity_level.value}")
        print(f"📊 Confidence: {certificate.confidence_score:.2f}")
        print(f"🔧 Subprocess Spawned: {certificate.evidence.subprocess_spawned}")
        print(f"📁 Filesystem Changes: {certificate.evidence.filesystem_changes}")
        print(f"📈 Execution Time: {certificate.evidence.execution_time_ms:.1f}ms")
        print(f"⚠️  Fabrication Indicators: {certificate.fabrication_indicators}")
        
        # Verify the file does NOT exist
        if os.path.exists(fake_file):
            print("❌ Verification: File unexpectedly exists on disk!")
        else:
            print("✅ Verification: File correctly does not exist (fabricated result)")
        
        print()
    
    print("🎉 Verification Demo Complete!")
    print()
    print("🔍 Key Findings:")
    print("• Real tools show filesystem changes and/or subprocess activity")
    print("• Fake tools show fabrication patterns in their output text")
    print("• The system can distinguish between authentic and fabricated results")
    print()
    print("📊 This demonstrates how the system solves CrewAI Issue #3154:")
    print("   'Agent does not actually invoke tools, only simulates tool usage'")


if __name__ == "__main__":
    run_verification_demo()