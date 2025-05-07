import sys
import unittest
import warnings


class BackwardCompatibilityTest(unittest.TestCase):
    def setUp(self):
        if "crewai.telemtry" in sys.modules:
            del sys.modules["crewai.telemtry"]
        warnings.resetwarnings()
    
    def test_deprecation_warning(self):
        """Test that importing from the misspelled module raises a deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            
            import importlib
            import crewai.telemtry
            importlib.reload(crewai.telemtry)
            
            self.assertGreaterEqual(len(w), 1)
            warning_messages = [str(warning.message) for warning in w]
            warning_categories = [warning.category for warning in w]
            
            has_deprecation_warning = False
            for msg, cat in zip(warning_messages, warning_categories):
                if (issubclass(cat, DeprecationWarning) and 
                    "crewai.telemtry" in msg and 
                    "crewai.telemetry" in msg):
                    has_deprecation_warning = True
                    break
            
            self.assertTrue(has_deprecation_warning, 
                           f"No matching deprecation warning found. Warnings: {warning_messages}")
    
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
