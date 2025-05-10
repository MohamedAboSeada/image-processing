import customtkinter as ctk
import cv2
import numpy as np
from .filter_windows import FilterWindow

class CustomLaplacianWindow(FilterWindow):
    """Window for custom Laplacian kernel filter"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Custom Laplacian Filter Control")
        # Remove kernel slider and controls (not needed)
        self.kernel_slider.pack_forget()
        self.kernel_label.pack_forget()
        self.kernel_value_label.pack_forget()
        self.kernel_frame.pack_forget()
        # Initial preview
        self.update_preview_filter(None)

    def update_preview_filter(self, value):
        preview_image = self.original_image.copy()
        preview_image = self.image_processor.image_processor.apply_custom_laplacian(preview_image)
        self.update_preview(preview_image)

    def confirm(self):
        result_image = self.image_processor.image_processor.apply_custom_laplacian(self.original_image)
        self.image_processor.apply_operation(
            result_image,
            "Custom Laplacian Filter ([1,1,1],[1,-8,1],[1,1,1])"
        )
        super().confirm() 