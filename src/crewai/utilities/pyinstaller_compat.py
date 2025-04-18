import os
import sys


def is_bundled():
    """Check if the application is running from a PyInstaller bundle."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

def get_bundle_dir():
    """Get the PyInstaller bundle directory if the application is bundled."""
    if is_bundled():
        return sys._MEIPASS
    return None
