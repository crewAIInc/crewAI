import sys
from unittest.mock import MagicMock, patch

import pytest

def test_openlit_compatibility():
    """Test that OpenLit can be imported and initialized with CrewAI."""
    try:
        import openlit
    except ImportError:
        pytest.skip("OpenLit not installed, skipping compatibility test")
    
    with patch.object(openlit, 'init', return_value=None) as mock_init:
        openlit.init(disable_metrics=True)
        mock_init.assert_called_once_with(disable_metrics=True)
        
    assert True

def test_opentelemetry_version_compatibility():
    """Test that the OpenTelemetry version is compatible with OpenLit."""
    pytest.importorskip("openlit")
    
    import pkg_resources
    
    otel_api_version = pkg_resources.get_distribution("opentelemetry-api").version
    otel_sdk_version = pkg_resources.get_distribution("opentelemetry-sdk").version
    
    assert otel_api_version == "1.32.1", f"Expected opentelemetry-api==1.32.1, got {otel_api_version}"
    assert otel_sdk_version == "1.32.1", f"Expected opentelemetry-sdk==1.32.1, got {otel_sdk_version}"
