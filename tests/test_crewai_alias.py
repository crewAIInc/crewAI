import unittest


class TestCrewaiAlias(unittest.TestCase):
    """Tests validating the Crewai alias and its backward compatibility.
    
    These tests ensure that the Crewai alias works correctly for both
    import scenarios and practical usage, providing backward compatibility
    for existing code that uses the 'Crewai' name.
    """

    def test_crewai_alias_import(self):
        """Test that Crewai can be imported from crewai.crew."""
        try:
            from crewai.crew import Crew, Crewai
            
            self.assertEqual(Crewai, Crew)
        except ImportError:
            self.fail("Failed to import Crewai from crewai.crew")
    
    def test_crewai_instance_creation(self):
        """Ensure Crewai can be instantiated just like Crew."""
        from crewai.agent import Agent
        from crewai.crew import Crew, Crewai
        
        test_agent = Agent(
            role="Test Agent",
            goal="Testing",
            backstory="Created for testing"
        )
        
        crewai_instance = Crewai(agents=[test_agent], tasks=[])
        crew_instance = Crew(agents=[test_agent], tasks=[])
        
        self.assertIsInstance(crewai_instance, Crew)
        self.assertEqual(type(crewai_instance), type(crew_instance))
    
    def test_crewai_deprecation_warning(self):
        """Test that using Crewai emits a deprecation warning."""
        import importlib
        import crewai.crew
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            importlib.reload(crewai.crew)
            
            self.assertTrue(len(w) > 0, "No deprecation warning was captured")
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w), 
                           "No DeprecationWarning was found")
            self.assertTrue(any("Crewai is deprecated" in str(warning.message) for warning in w),
                           "Warning message doesn't contain expected text")


if __name__ == "__main__":
    unittest.main()
