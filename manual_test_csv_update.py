"""
Manual test script to verify CSV knowledge source update functionality.
This script creates a CSV file, loads it as a knowledge source, updates the file,
and verifies that the updated content is detected and loaded.
"""

import os
import time
from pathlib import Path
import tempfile
import sys

from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource


def test_csv_knowledge_source_updates():
    """Test that CSVKnowledgeSource properly detects and loads updates to CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_updates.csv"
        
        initial_csv_content = [
            ["name", "age", "city"],
            ["John", "30", "New York"],
            ["Alice", "25", "San Francisco"],
            ["Bob", "28", "Chicago"],
        ]
        
        with open(csv_path, "w") as f:
            for row in initial_csv_content:
                f.write(",".join(row) + "\n")
        
        print(f"Created CSV file at {csv_path}")
        
        csv_source = CSVKnowledgeSource(file_paths=[csv_path])
        
        if not hasattr(csv_source, 'files_have_changed'):
            print("❌ TEST FAILED: files_have_changed method not found in CSVKnowledgeSource")
            return False
        
        if not hasattr(csv_source, '_file_mtimes'):
            print("❌ TEST FAILED: _file_mtimes attribute not found in CSVKnowledgeSource")
            return False
        
        knowledge = Knowledge(sources=[csv_source], collection_name="test_updates")
        
        if not hasattr(knowledge, '_check_and_reload_sources'):
            print("❌ TEST FAILED: _check_and_reload_sources method not found in Knowledge")
            return False
        
        print("✅ All required methods and attributes exist")
        
        updated_csv_content = [
            ["name", "age", "city"],
            ["John", "30", "Boston"],  # Changed city
            ["Alice", "25", "San Francisco"],
            ["Bob", "28", "Chicago"],
            ["Eve", "22", "Miami"],  # Added new person
        ]
        
        print("\nWaiting for 1 second before updating file...")
        time.sleep(1)
        
        with open(csv_path, "w") as f:
            for row in updated_csv_content:
                f.write(",".join(row) + "\n")
        
        print(f"Updated CSV file at {csv_path}")
        
        if not csv_source.files_have_changed():
            print("❌ TEST FAILED: files_have_changed did not detect file modification")
            return False
        
        print("✅ files_have_changed correctly detected file modification")
        
        csv_source._record_file_mtimes()
        csv_source.content = csv_source.load_content()
        
        content_str = str(csv_source.content)
        if "Boston" in content_str and "Eve" in content_str and "Miami" in content_str:
            print("✅ Content was correctly updated with new data")
        else:
            print("❌ TEST FAILED: Content was not updated with new data")
            return False
        
        print("\n✅ TEST PASSED: CSV knowledge source correctly detects and loads file updates")
        return True


if __name__ == "__main__":
    success = test_csv_knowledge_source_updates()
    sys.exit(0 if success else 1)
