"""Tests for dependency compatibility, specifically for issue #3413.

This module tests that CrewAI can be installed alongside Google Cloud SDKs
without protobuf dependency conflicts.
"""

import subprocess
import sys
import tempfile
import pytest


class TestDependencyCompatibility:
    """Test dependency compatibility with Google Cloud SDKs."""

    def test_opentelemetry_protobuf_compatibility(self):
        """Test that opentelemetry versions work with protobuf<5.0."""
        try:
            from opentelemetry.sdk.trace import TracerProvider
            
            tracer_provider = TracerProvider()
            tracer = tracer_provider.get_tracer("test")
            
            with tracer.start_as_current_span("test_span") as span:
                span.set_attribute("test", "value")
                assert span is not None
                
        except ImportError as e:
            pytest.fail(f"Failed to import opentelemetry modules: {e}")

    def test_google_cloud_sdk_compatibility_simulation(self):
        """Simulate Google Cloud SDK protobuf requirements."""
        try:
            import google.protobuf
            version_parts = google.protobuf.__version__.split('.')
            major_version = int(version_parts[0])
            
            assert major_version < 5, f"protobuf version {google.protobuf.__version__} should be <5.0 for Google Cloud SDK compatibility"
            
        except ImportError:
            pytest.skip("protobuf not installed, skipping compatibility test")

    def test_crewai_telemetry_functionality(self):
        """Test that CrewAI telemetry functionality works with downgraded opentelemetry."""
        try:
            from crewai.telemetry.telemetry import Telemetry
            from crewai.utilities.crew.crew_context import get_crew_context
            
            telemetry = Telemetry()
            assert telemetry is not None
            
            get_crew_context()
            
        except ImportError as e:
            pytest.fail(f"Failed to import CrewAI telemetry modules: {e}")

    def test_dry_run_installation_compatibility(self):
        """Test that CrewAI and Google Cloud SDKs can be installed together."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "--dry-run", "--quiet",
                    "opentelemetry-api>=1.27.0,<1.30.0",
                    "opentelemetry-sdk>=1.27.0,<1.30.0", 
                    "opentelemetry-exporter-otlp-proto-http>=1.27.0,<1.30.0",
                    "google-cloud-storage"
                ], capture_output=True, text=True, cwd=temp_dir)
                
                assert result.returncode == 0, f"Dry run installation failed: {result.stderr}"
                
                assert "protobuf" in result.stdout.lower(), "protobuf should be in installation plan"
                
            except Exception as e:
                pytest.fail(f"Dry run installation test failed: {e}")

    def test_protobuf_version_constraint_resolution(self):
        """Test that protobuf version constraints are properly resolved."""
        try:
            import google.protobuf
            version = google.protobuf.__version__
            
            version_parts = [int(x) for x in version.split('.')]
            major, minor = version_parts[0], version_parts[1]
            
            assert major >= 3, f"protobuf version {version} should be >=3.19"
            if major == 3:
                assert minor >= 19, f"protobuf version {version} should be >=3.19"
            assert major < 5, f"protobuf version {version} should be <5.0 for Google Cloud SDK compatibility"
            
        except ImportError:
            pytest.skip("protobuf not installed, skipping version constraint test")
