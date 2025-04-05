import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from crewai.utilities.path_utils import add_project_to_path


class TestPathUtils(unittest.TestCase):
    
    @patch('os.getcwd')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_dir')
    def test_add_project_to_path_with_src(self, mock_is_dir, mock_exists, mock_getcwd):
        mock_getcwd.return_value = "/home/user/project"
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        
        original_sys_path = sys.path.copy()
        
        try:
            if "/home/user/project" in sys.path:
                sys.path.remove("/home/user/project")
            if "/home/user/project/src" in sys.path:
                sys.path.remove("/home/user/project/src")
            
            add_project_to_path()
            
            self.assertIn("/home/user/project", sys.path)
            self.assertIn("/home/user/project/src", sys.path)
            
            self.assertTrue(
                sys.path.index("/home/user/project/src") <= 1 and 
                sys.path.index("/home/user/project") <= 1
            )
        finally:
            sys.path = original_sys_path
    
    @patch('os.getcwd')
    @patch('pathlib.Path.exists')
    def test_add_project_to_path_without_src(self, mock_exists, mock_getcwd):
        mock_getcwd.return_value = "/home/user/project"
        mock_exists.return_value = False
        
        original_sys_path = sys.path.copy()
        
        try:
            if "/home/user/project" in sys.path:
                sys.path.remove("/home/user/project")
            
            add_project_to_path()
            
            self.assertIn("/home/user/project", sys.path)
            
            self.assertEqual(sys.path.index("/home/user/project"), 0)
        finally:
            sys.path = original_sys_path
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_dir')
    def test_add_project_to_path_with_custom_dir(self, mock_is_dir, mock_exists):
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        
        original_sys_path = sys.path.copy()
        
        try:
            custom_dir = "/home/user/custom_project"
            if custom_dir in sys.path:
                sys.path.remove(custom_dir)
            if os.path.join(custom_dir, "src") in sys.path:
                sys.path.remove(os.path.join(custom_dir, "src"))
            
            add_project_to_path(custom_dir)
            
            self.assertIn(custom_dir, sys.path)
            self.assertIn(os.path.join(custom_dir, "src"), sys.path)
        finally:
            sys.path = original_sys_path
