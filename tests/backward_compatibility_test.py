import unittest


class BackwardCompatibilityTest(unittest.TestCase):
    def test_telemtry_typo_compatibility(self):
        """Test that the backward compatibility for the telemtry typo works."""
        from crewai.telemtry import Telemetry as MisspelledTelemetry
        from crewai.telemetry import Telemetry
        
        self.assertIs(MisspelledTelemetry, Telemetry)
