import unittest
import warnings


class BackwardCompatibilityTest(unittest.TestCase):
    def test_telemtry_typo_compatibility(self):
        """Test that the backward compatibility for the telemtry typo works."""
        from crewai.telemetry import Telemetry
        from crewai.telemtry import Telemetry as MisspelledTelemetry
        
        self.assertIs(MisspelledTelemetry, Telemetry)
    
    def test_functionality_preservation(self):
        """Test that the re-exported Telemetry class preserves all functionality."""
        from crewai.telemetry import Telemetry
        from crewai.telemtry import Telemetry as MisspelledTelemetry
        
        self.assertEqual(dir(MisspelledTelemetry), dir(Telemetry))
    
    def test_deprecation_warning(self):
        """Test that importing from the misspelled module raises a deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from crewai.telemtry import Telemetry  # noqa: F401
            
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("crewai.telemtry", str(w[0].message))
            self.assertIn("crewai.telemetry", str(w[0].message))
