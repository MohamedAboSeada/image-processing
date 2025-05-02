import customtkinter as ctk
import cv2
from .base_window import BaseWindow

class ThresholdWindow(BaseWindow):
    """Window for threshold adjustment"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Threshold Control", geometry="800x600")

        # Create a preview canvas
        self.create_preview_canvas()

        # Create controls frame
        self.controls_frame = ctk.CTkFrame(self.controls_container)
        self.controls_frame.pack(pady=10)

        # Create a slider for threshold adjustment
        self.slider_frame = ctk.CTkFrame(self.controls_frame)
        self.slider_frame.pack(pady=10)

        self.threshold_label = ctk.CTkLabel(self.slider_frame, text="Threshold Value:")
        self.threshold_label.pack(side="left", padx=5)

        self.threshold_slider = ctk.CTkSlider(self.slider_frame, from_=0, to=255,
                                             number_of_steps=255, orientation="horizontal")
        self.threshold_slider.set(127)  # Default threshold value
        self.threshold_slider.pack(side="left", padx=10)

        self.threshold_value_label = ctk.CTkLabel(self.slider_frame, text="127")
        self.threshold_value_label.pack(side="left", padx=5)

        # Create button frame
        self.create_button_frame()

        # Store original image
        self.original_image = self.image_processor.cv_image.copy()

        # Connect the slider to the update function
        self.threshold_slider.configure(command=self.update_preview_threshold)

        # Initial preview
        self.update_preview_threshold(self.threshold_slider.get())

    def update_preview_threshold(self, value):
        """Update the preview with the current threshold setting"""
        threshold_value = int(value)
        self.threshold_value_label.configure(text=str(threshold_value))

        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Convert back to BGR for display
        preview_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the threshold and close the window"""
        threshold_value = int(self.threshold_slider.get())

        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Convert back to BGR for display
        result_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Apply the changes
        self.image_processor.apply_operation(result_image, "Apply Threshold")

        # Close the window
        super().confirm()
