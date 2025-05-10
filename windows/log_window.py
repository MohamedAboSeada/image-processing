import customtkinter as ctk
import cv2
import numpy as np
from .filter_windows import FilterWindow

class LogWindow(FilterWindow):
    """Window for Laplacian of Gaussian (LoG) filter"""

    def __init__(self, root, image_processor):
        self.sigma_value = 1.0
        super().__init__(root, image_processor, title="LoG (Laplacian of Gaussian) Filter Control")

        # Add sigma control
        self.sigma_frame = ctk.CTkFrame(self.controls_frame)
        self.sigma_frame.pack(pady=10)

        self.sigma_label = ctk.CTkLabel(self.sigma_frame, text="Sigma:")
        self.sigma_label.pack(side="left", padx=5)

        self.sigma_slider = ctk.CTkSlider(self.sigma_frame, from_=0.1, to=5.0,
                                        number_of_steps=49, orientation="horizontal")
        self.sigma_slider.set(self.sigma_value)
        self.sigma_slider.pack(side="left", padx=5)

        self.sigma_value_label = ctk.CTkLabel(self.sigma_frame, text=f"{self.sigma_value:.2f}")
        self.sigma_value_label.pack(side="left", padx=5)

        self.sigma_slider.configure(command=self.update_sigma)
        self.update_preview_filter(self.kernel_slider.get())

    def update_sigma(self, value):
        self.sigma_value = float(value)
        self.sigma_value_label.configure(text=f"{self.sigma_value:.2f}")
        self.update_preview_filter(self.kernel_slider.get())

    def update_preview_filter(self, value):
        kernel_size = self.get_kernel_size()
        sigma = float(self.sigma_slider.get()) if hasattr(self, 'sigma_slider') else self.sigma_value
        preview_image = self.original_image.copy()
        preview_image = self.image_processor.image_processor.apply_log(
            preview_image,
            kernel_size=kernel_size,
            sigma=sigma
        )
        self.update_preview(preview_image)

    def confirm(self):
        kernel_size = self.get_kernel_size()
        sigma = float(self.sigma_slider.get())
        result_image = self.image_processor.image_processor.apply_log(
            self.original_image,
            kernel_size=kernel_size,
            sigma=sigma
        )
        self.image_processor.apply_operation(
            result_image,
            f"LoG Filter (k={kernel_size}, σ={sigma:.2f})"
        )
        super().confirm() 