import unittest
import os
import sys
import tkinter as tk
import customtkinter as ctk

# Add the parent directory to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import ImageEditorApp

class TestApp(unittest.TestCase):
    """Test cases for the main application"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a root window for testing
        self.root = ctk.CTk()
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Destroy the root window
        self.root.destroy()
    
    def test_app_initialization(self):
        """Test that the app initializes correctly"""
        # Create the app
        app = ImageEditorApp(self.root)
        
        # Check that the app has the expected attributes
        self.assertIsNotNone(app.image_processor)
        self.assertIsNotNone(app.ui)
        self.assertIsNone(app.cv_image)
        self.assertIsNone(app.original_image)
        self.assertEqual(app.original_extension, ".png")
        self.assertEqual(app.undo_stack, [])
        self.assertEqual(app.redo_stack, [])
        self.assertEqual(app.history_list, [])
        self.assertEqual(app.redo_history_list, [])

if __name__ == '__main__':
    unittest.main()
