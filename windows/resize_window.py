import customtkinter as ctk
import cv2
from .base_window import BaseWindow

class ResizeWindow(BaseWindow):
    """Window for resizing images"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Resize Image", geometry="800x600")

        # Create a preview canvas
        self.create_preview_canvas()

        # Store original image
        self.original_image = self.image_processor.cv_image.copy()
        self.original_height, self.original_width = self.original_image.shape[:2]

        # Create controls frame
        self.controls_frame = ctk.CTkFrame(self.controls_container)
        self.controls_frame.pack(pady=10)

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

        # Maintain aspect ratio checkbox
        self.aspect_ratio_var = ctk.BooleanVar(value=True)
        self.aspect_ratio_checkbox = ctk.CTkCheckBox(
            self.controls_frame,
            text="Maintain Aspect Ratio",
            variable=self.aspect_ratio_var,
            command=self.toggle_aspect_ratio
        )
        self.aspect_ratio_checkbox.pack(pady=5)

        # Original dimensions label
        self.original_dimensions_label = ctk.CTkLabel(
            self.controls_frame,
            text=f"Original dimensions: {self.original_width} x {self.original_height}"
        )
        self.original_dimensions_label.pack(pady=5)

        # Create button frame
        self.create_button_frame()

        # Bind entry fields to update function
        self.width_entry.bind("<KeyRelease>", self.width_changed)
        self.height_entry.bind("<KeyRelease>", self.height_changed)

        # Initial preview
        self.update_preview_resize()

    def width_changed(self, event=None):
        """Handle width entry changes"""
        try:
            new_width = int(self.width_entry.get())
            if self.aspect_ratio_var.get():
                # Calculate new height based on aspect ratio
                aspect_ratio = self.original_height / self.original_width
                new_height = int(new_width * aspect_ratio)
                self.height_entry.delete(0, ctk.END)
                self.height_entry.insert(0, str(new_height))
            self.update_preview_resize()
        except ValueError:
            # Invalid input, don't update
            pass

    def height_changed(self, event=None):
        """Handle height entry changes"""
        try:
            new_height = int(self.height_entry.get())
            if self.aspect_ratio_var.get():
                # Calculate new width based on aspect ratio
                aspect_ratio = self.original_width / self.original_height
                new_width = int(new_height * aspect_ratio)
                self.width_entry.delete(0, ctk.END)
                self.width_entry.insert(0, str(new_width))
            self.update_preview_resize()
        except ValueError:
            # Invalid input, don't update
            pass

    def toggle_aspect_ratio(self):
        """Handle aspect ratio checkbox toggle"""
        if self.aspect_ratio_var.get():
            # Recalculate height based on current width
            self.width_changed()

    def update_preview_resize(self):
        """Update the preview with the current resize settings"""
        try:
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())

            if width <= 0 or height <= 0:
                return

            preview_image = cv2.resize(self.original_image, (width, height))
            self.update_preview(preview_image)
        except ValueError:
            # Invalid input, don't update
            pass

    def confirm(self):
        """Apply the resize and close the window"""
        try:
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())

            if width <= 0 or height <= 0:
                return

            result_image = cv2.resize(self.original_image, (width, height))

            # Apply the changes
            self.image_processor.apply_operation(result_image, f"Resize to {width}x{height}")

            # Close the window
            super().confirm()
        except ValueError:
            # Invalid input, don't close
            pass
