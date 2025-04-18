import sys
import unittest
from unittest.mock import patch

from crewai.utilities.pyinstaller_compat import get_bundle_dir, is_bundled


class TestPyInstallerCompat(unittest.TestCase):
    def test_is_bundled_normal(self):
        self.assertFalse(is_bundled())
    
    @patch.object(sys, 'frozen', True, create=True)
    @patch.object(sys, '_MEIPASS', '/path/to/bundle', create=True)
    def test_is_bundled_pyinstaller(self):
        self.assertTrue(is_bundled())
        self.assertEqual(get_bundle_dir(), '/path/to/bundle')
