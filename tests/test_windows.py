import unittest
import os
import sys
import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np

# Add the parent directory to the path so we can import the windows
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from windows.base_window import BaseWindow
from main import ImageEditorApp

class TestWindows(unittest.TestCase):
    """Test cases for the window classes"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a root window for testing
        self.root = ctk.CTk()
        self.app = ImageEditorApp(self.root)
        
        # Create a test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = [255, 255, 255]  # White square
        
        # Set the test image as the current image
        self.app.cv_image = self.test_image.copy()
        self.app.original_image = self.test_image.copy()
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Destroy the root window
        self.root.destroy()
    
    def test_base_window(self):
        """Test the BaseWindow class"""
        # Create a base window
        window = BaseWindow(self.root, self.app)
        
        # Check that the window has the expected attributes
        self.assertIsNotNone(window.window)
        self.assertEqual(window.window.title(), "Window")
        self.assertEqual(window.image_processor, self.app)
        
        # Clean up
        window.window.destroy()
    
    def test_base_window_methods(self):
        """Test the methods of the BaseWindow class"""
        # Create a base window
        window = BaseWindow(self.root, self.app)
        
        # Create a preview canvas
        window.create_preview_canvas()
        self.assertIsNotNone(window.canvas)
        
        # Create a button frame
        window.create_button_frame()
        self.assertIsNotNone(window.button_frame)
        self.assertIsNotNone(window.confirm_button)
        self.assertIsNotNone(window.cancel_button)
        
        # Test the update_preview method
        window.update_preview(self.test_image)
        
        # Test the confirm and cancel methods
        window.confirm()  # Should close the window
        window.cancel()   # Should close the window
        
        # Clean up
        try:
            window.window.destroy()
        except tk.TclError:
            pass  # Window already destroyed

if __name__ == '__main__':
    unittest.main()
