import customtkinter as ctk
import cv2
from .base_window import BaseWindow

class FlipWindow(BaseWindow):
    """Window for flipping images"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Flip Adjustment", geometry="800x600")

        # Create a preview canvas
        self.create_preview_canvas()

        # Create radio buttons for flip options
        self.controls_frame = ctk.CTkFrame(self.controls_container)
        self.controls_frame.pack(pady=10)

        self.flip_var = ctk.StringVar(value="horizontal")

        self.horizontal_radio = ctk.CTkRadioButton(
            self.controls_frame,
            text="Horizontal Flip",
            variable=self.flip_var,
            value="horizontal"
        )
        self.horizontal_radio.pack(pady=5)

        self.vertical_radio = ctk.CTkRadioButton(
            self.controls_frame,
            text="Vertical Flip",
            variable=self.flip_var,
            value="vertical"
        )
        self.vertical_radio.pack(pady=5)

        self.both_radio = ctk.CTkRadioButton(
            self.controls_frame,
            text="Both (Horizontal & Vertical)",
            variable=self.flip_var,
            value="both"
        )
        self.both_radio.pack(pady=5)

        # Create button frame
        self.create_button_frame()

        # Store original image
        self.original_image = self.image_processor.cv_image.copy()

        # Connect the radio buttons to the update function
        self.flip_var.trace_add("write", self.update_preview_flip)

        # Initial preview
        self.update_preview_flip()

    def update_preview_flip(self, *args):
        """Update the preview with the current flip setting"""
        flip_mode = self.flip_var.get()

        preview_image = self.original_image.copy()

        if flip_mode == "horizontal":
            preview_image = cv2.flip(preview_image, 1)  # 1 for horizontal flip
        elif flip_mode == "vertical":
            preview_image = cv2.flip(preview_image, 0)  # 0 for vertical flip
        elif flip_mode == "both":
            preview_image = cv2.flip(preview_image, -1)  # -1 for both

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the flip and close the window"""
        flip_mode = self.flip_var.get()

        if flip_mode == "horizontal":
            result_image = cv2.flip(self.original_image, 1)
            operation_name = "Horizontal Flip"
        elif flip_mode == "vertical":
            result_image = cv2.flip(self.original_image, 0)
            operation_name = "Vertical Flip"
        elif flip_mode == "both":
            result_image = cv2.flip(self.original_image, -1)
            operation_name = "Flip Both Directions"

        # Apply the changes
        self.image_processor.apply_operation(result_image, operation_name)

        # Close the window
        super().confirm()
