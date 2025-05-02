import customtkinter as ctk
import cv2
import numpy as np
from .base_window import BaseWindow

class BrightnessWindow(BaseWindow):
    """Window for adjusting image brightness"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Brightness Control", geometry="800x600")

        # Create a preview canvas
        self.create_preview_canvas()

        # Create controls frame
        self.controls_frame = ctk.CTkFrame(self.controls_container)
        self.controls_frame.pack(pady=10)

        # Create a slider for brightness adjustment
        self.slider_label = ctk.CTkLabel(self.controls_frame, text="Brightness:")
        self.slider_label.pack(pady=(0, 5))

        self.slider = ctk.CTkSlider(self.controls_frame, from_=0.1, to=3.0,
                                   number_of_steps=290, orientation="horizontal")
        self.slider.set(1.0)  # Default brightness factor
        self.slider.pack(pady=5)

        # Create button frame
        self.create_button_frame()

        # Store original image
        self.original_image = self.image_processor.cv_image.copy()

        # Connect the slider to the update function
        self.slider.configure(command=self.update_preview_brightness)

        # Initial preview
        self.update_preview_brightness(self.slider.get())

    def update_preview_brightness(self, value):
        """Update the preview with the current brightness setting"""
        factor = float(value)

        # Convert to HSV for brightness adjustment
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        preview_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the brightness adjustment and close the window"""
        factor = float(self.slider.get())

        # Convert to HSV for brightness adjustment
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        result_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply the changes
        self.image_processor.apply_operation(result_image, "Adjust Brightness")

        # Close the window
        super().confirm()
