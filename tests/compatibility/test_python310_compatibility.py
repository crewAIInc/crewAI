def test_self_import_compatibility():
    """
    Test that the Self type is properly imported and used
    in a way that supports Python 3.10+.
    
    This test will pass as long as the module imports successfully.
    The actual failure would happen during import if the compatibility
    fix is not working.
    """
    try:
        from crewai.memory.memory import Self
        assert True, "Self type imported successfully"
    except ImportError:
        assert False, "Failed to import Self type from crewai.memory.memory"
