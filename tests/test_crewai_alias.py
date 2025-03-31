import unittest

class TestCrewaiAlias(unittest.TestCase):
    """Test the Crewai alias for backward compatibility."""

    def test_crewai_alias_import(self):
        """Test that Crewai can be imported from crewai.crew."""
        try:
            from crewai.crew import Crewai
            from crewai.crew import Crew
            
            self.assertEqual(Crewai, Crew)
        except ImportError:
            self.fail("Failed to import Crewai from crewai.crew")

if __name__ == "__main__":
    unittest.main()
