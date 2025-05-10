import customtkinter as ctk
import cv2
import numpy as np
from .filter_windows import FilterWindow

class LaplacianWindow(FilterWindow):
    """Window for Laplacian filter"""

    def __init__(self, root, image_processor):
        # Initialize alpha value before calling parent's __init__
        self.alpha_value = 1.0

        # Call parent's __init__ which will call update_preview_filter
        super().__init__(root, image_processor, title="Laplacian Filter Control")

        # Add alpha control
        self.alpha_frame = ctk.CTkFrame(self.controls_frame)
        self.alpha_frame.pack(pady=10)

        self.alpha_label = ctk.CTkLabel(self.alpha_frame, text="Laplacian Alpha:")
        self.alpha_label.pack(side="left", padx=5)

        self.alpha_slider = ctk.CTkSlider(self.alpha_frame, from_=0.0, to=1.0,
                                        number_of_steps=20, orientation="horizontal")
        self.alpha_slider.set(self.alpha_value)  # Default alpha
        self.alpha_slider.pack(side="left", padx=5)

        self.alpha_value_label = ctk.CTkLabel(self.alpha_frame, text=f"{self.alpha_value:.2f}")
        self.alpha_value_label.pack(side="left", padx=5)

        # Connect the alpha slider to the update function
        self.alpha_slider.configure(command=self.update_alpha)

        # Update preview again now that alpha slider is available
        self.update_preview_filter(self.kernel_slider.get())

    def update_alpha(self, value):
        """Update the alpha value and refresh the preview"""
        self.alpha_value = float(value)
        self.alpha_value_label.configure(text=f"{self.alpha_value:.2f}")
        self.update_preview_filter(self.kernel_slider.get())

    def update_preview_filter(self, value):
        """Update the preview with the current Laplacian filter setting"""
        kernel_size = self.get_kernel_size()
        if hasattr(self, 'alpha_slider'):
            alpha = float(self.alpha_slider.get())
        else:
            alpha = self.alpha_value
        preview_image = self.original_image.copy()
        preview_image = self.image_processor.image_processor.apply_laplacian(
            preview_image,
            kernel_size=kernel_size,
            alpha=alpha
        )
        self.update_preview(preview_image)

    def confirm(self):
        """Apply the Laplacian filter and close the window"""
        kernel_size = self.get_kernel_size()
        alpha = float(self.alpha_slider.get())
        result_image = self.image_processor.image_processor.apply_laplacian(
            self.original_image,
            kernel_size=kernel_size,
            alpha=alpha
        )
        self.image_processor.apply_operation(
            result_image,
            f"Laplacian Filter (k={kernel_size}, α={alpha:.2f})"
        )
        super().confirm() 