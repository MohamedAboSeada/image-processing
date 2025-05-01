import unittest
import os
import sys
import tkinter as tk
import customtkinter as ctk

# Add the parent directory to the path so we can import the UI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui import ImageEditorUI
from main import ImageEditorApp

class TestUI(unittest.TestCase):
    """Test cases for the UI components"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a root window for testing
        self.root = ctk.CTk()
        self.app = ImageEditorApp(self.root)
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Destroy the root window
        self.root.destroy()
    
    def test_ui_initialization(self):
        """Test that the UI initializes correctly"""
        # Check that the UI has the expected attributes
        self.assertIsNotNone(self.app.ui.frame_left)
        self.assertIsNotNone(self.app.ui.image_frame)
        self.assertIsNotNone(self.app.ui.edited_container)
        self.assertIsNotNone(self.app.ui.original_container)
        self.assertIsNotNone(self.app.ui.history_listbox)
        self.assertIsNotNone(self.app.ui.toggle_button)
        self.assertTrue(self.app.ui.show_original)
    
    def test_toggle_original(self):
        """Test the toggle_original method"""
        # Initially, show_original should be True
        self.assertTrue(self.app.ui.show_original)
        
        # Toggle to hide the original
        self.app.ui.toggle_original()
        self.assertFalse(self.app.ui.show_original)
        
        # Toggle again to show the original
        self.app.ui.toggle_original()
        self.assertTrue(self.app.ui.show_original)
    
    def test_refresh_history(self):
        """Test the refresh_history method"""
        # Initially, the history should be empty
        initial_text = self.app.ui.history_listbox.get("1.0", "end").strip()
        self.assertEqual(initial_text, "History:")
        
        # Add some history items
        history = ["Operation 1", "Operation 2", "Operation 3"]
        self.app.ui.refresh_history(history)
        
        # Check that the history was updated
        updated_text = self.app.ui.history_listbox.get("1.0", "end").strip()
        expected_text = "History:\n- Operation 1\n- Operation 2\n- Operation 3"
        self.assertEqual(updated_text, expected_text)
    
    def test_update_window_title(self):
        """Test the update_window_title method"""
        # Initially, the title should be the default
        self.assertEqual(self.root.title(), "Advanced Image Editor")
        
        # Update the title with an image name
        self.app.ui.update_window_title("test.png")
        self.assertEqual(self.root.title(), "Advanced Image Editor - test.png")
        
        # Reset the title
        self.app.ui.update_window_title(None)
        self.assertEqual(self.root.title(), "Advanced Image Editor")

if __name__ == '__main__':
    unittest.main()
