import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from .base_window import BaseWindow

class CropWindow(BaseWindow):
    """Window for cropping images"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Crop Image", geometry="800x600")

        # Store original image
        self.original_image = self.image_processor.cv_image.copy()
        self.original_height, self.original_width = self.original_image.shape[:2]

        # Create a preview canvas
        self.create_preview_canvas()

        # Create controls frame
        self.controls_frame = ctk.CTkFrame(self.controls_container)
        self.controls_frame.pack(pady=10)

        # X position input
        self.x_frame = ctk.CTkFrame(self.controls_frame)
        self.x_frame.pack(pady=5)

        self.x_label = ctk.CTkLabel(self.x_frame, text="X:")
        self.x_label.pack(side="left", padx=5)

        self.x_entry = ctk.CTkEntry(self.x_frame, width=100)
        self.x_entry.insert(0, "0")
        self.x_entry.pack(side="left", padx=10)

        # Y position input
        self.y_frame = ctk.CTkFrame(self.controls_frame)
        self.y_frame.pack(pady=5)

        self.y_label = ctk.CTkLabel(self.y_frame, text="Y:")
        self.y_label.pack(side="left", padx=5)

        self.y_entry = ctk.CTkEntry(self.y_frame, width=100)
        self.y_entry.insert(0, "0")
        self.y_entry.pack(side="left", padx=10)

        # Width input
        self.width_frame = ctk.CTkFrame(self.controls_frame)
        self.width_frame.pack(pady=5)

        self.width_label = ctk.CTkLabel(self.width_frame, text="Width:")
        self.width_label.pack(side="left", padx=5)

        self.width_entry = ctk.CTkEntry(self.width_frame, width=100)
        self.width_entry.insert(0, str(self.original_width))
        self.width_entry.pack(side="left", padx=10)

        # Height input
        self.height_frame = ctk.CTkFrame(self.controls_frame)
        self.height_frame.pack(pady=5)

        self.height_label = ctk.CTkLabel(self.height_frame, text="Height:")
        self.height_label.pack(side="left", padx=5)

        self.height_entry = ctk.CTkEntry(self.height_frame, width=100)
        self.height_entry.insert(0, str(self.original_height))
        self.height_entry.pack(side="left", padx=10)

        # Original dimensions label
        self.original_dimensions_label = ctk.CTkLabel(
            self.controls_frame,
            text=f"Original dimensions: {self.original_width} x {self.original_height}"
        )
        self.original_dimensions_label.pack(pady=5)

        # Create button frame
        self.create_button_frame()

        # Bind entry fields to update function
        self.x_entry.bind("<KeyRelease>", self.update_preview_crop)
        self.y_entry.bind("<KeyRelease>", self.update_preview_crop)
        self.width_entry.bind("<KeyRelease>", self.update_preview_crop)
        self.height_entry.bind("<KeyRelease>", self.update_preview_crop)

        # Initial preview
        self.update_preview_crop()

    def update_preview_crop(self, event=None):
        """Update the preview with the current crop settings"""
        try:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())

            # Validate values
            x = max(0, min(x, self.original_width - 1))
            y = max(0, min(y, self.original_height - 1))
            width = max(1, min(width, self.original_width - x))
            height = max(1, min(height, self.original_height - y))

            # Update entries with validated values
            self.x_entry.delete(0, ctk.END)
            self.x_entry.insert(0, str(x))
            self.y_entry.delete(0, ctk.END)
            self.y_entry.insert(0, str(y))
            self.width_entry.delete(0, ctk.END)
            self.width_entry.insert(0, str(width))
            self.height_entry.delete(0, ctk.END)
            self.height_entry.insert(0, str(height))

            # Create a copy of the original image with a rectangle showing the crop area
            preview_image = self.original_image.copy()
            cv2.rectangle(preview_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Show the preview
            self.update_preview(preview_image)
        except ValueError:
            # Invalid input, don't update
            pass

    def confirm(self):
        """Apply the crop and close the window"""
        try:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())

            # Validate values
            x = max(0, min(x, self.original_width - 1))
            y = max(0, min(y, self.original_height - 1))
            width = max(1, min(width, self.original_width - x))
            height = max(1, min(height, self.original_height - y))

            # Crop the image
            result_image = self.original_image[y:y+height, x:x+width]

            # Apply the changes
            self.image_processor.apply_operation(result_image, f"Crop ({x},{y},{width},{height})")

            # Close the window
            super().confirm()
        except ValueError:
            # Invalid input, don't close
            pass
