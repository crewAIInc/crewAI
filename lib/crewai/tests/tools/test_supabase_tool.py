import os
import pytest

from crewai.tools.supabase_tool import SupabaseTool

def test_missing_env_vars():
    # Temporarily remove env vars
    os.environ.pop("SUPABASE_URL", None)
    os.environ.pop("SUPABASE_KEY", None)

    with pytest.raises(ValueError):
        SupabaseTool()
