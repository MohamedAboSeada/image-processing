import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from .base_window import BaseWindow

class FilterWindow(BaseWindow):
    """Base class for filter windows"""

    def __init__(self, root, image_processor, title="Filter", geometry="800x600"):
        super().__init__(root, image_processor, title=title, geometry=geometry)

        # Create a preview canvas
        self.create_preview_canvas()

        # Create a frame for kernel size control
        self.kernel_frame = ctk.CTkFrame(self.window)
        self.kernel_frame.pack(pady=10)

        self.kernel_label = ctk.CTkLabel(self.kernel_frame, text="Kernel Size:")
        self.kernel_label.pack(side="left", padx=5)

        # Kernel size must be odd
        self.kernel_slider = ctk.CTkSlider(self.kernel_frame, from_=1, to=15,
                                         number_of_steps=7, orientation="horizontal")
        self.kernel_slider.set(3)  # Default kernel size
        self.kernel_slider.pack(side="left", padx=5)

        self.kernel_value_label = ctk.CTkLabel(self.kernel_frame, text="3")
        self.kernel_value_label.pack(side="left", padx=5)

        # Create button frame
        self.create_button_frame()

        # Store original image
        self.original_image = self.image_processor.cv_image.copy()

        # Connect the slider to the update function
        self.kernel_slider.configure(command=self.update_preview_filter)

        # Initial preview
        self.update_preview_filter(self.kernel_slider.get())

    def get_kernel_size(self):
        """Get the current kernel size (ensuring it's odd)"""
        size = int(self.kernel_slider.get())
        # Ensure kernel size is odd
        if size % 2 == 0:
            size += 1
        self.kernel_value_label.configure(text=str(size))
        return size

    def update_preview_filter(self, value):
        """Update the preview with the current filter setting"""
        # To be implemented by subclasses
        pass

    def confirm(self):
        """Apply the filter and close the window"""
        # To be implemented by subclasses
        super().confirm()


class MedianFilterWindow(FilterWindow):
    """Window for median filter"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Median Filter Control")

    def update_preview_filter(self, value):
        """Update the preview with the current median filter setting"""
        kernel_size = self.get_kernel_size()

        preview_image = self.original_image.copy()
        preview_image = cv2.medianBlur(preview_image, kernel_size)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the median filter and close the window"""
        kernel_size = self.get_kernel_size()

        result_image = cv2.medianBlur(self.original_image, kernel_size)

        # Apply the changes
        self.image_processor.apply_operation(result_image, f"Median Filter (k={kernel_size})")

        # Close the window
        super().confirm()


class MaxFilterWindow(FilterWindow):
    """Window for max filter"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Max Filter Control")

    def update_preview_filter(self, value):
        """Update the preview with the current max filter setting"""
        kernel_size = self.get_kernel_size()

        preview_image = self.original_image.copy()
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        preview_image = cv2.dilate(preview_image, kernel)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the max filter and close the window"""
        kernel_size = self.get_kernel_size()

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result_image = cv2.dilate(self.original_image, kernel)

        # Apply the changes
        self.image_processor.apply_operation(result_image, f"Max Filter (k={kernel_size})")

        # Close the window
        super().confirm()


class MinFilterWindow(FilterWindow):
    """Window for min filter"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Min Filter Control")

    def update_preview_filter(self, value):
        """Update the preview with the current min filter setting"""
        kernel_size = self.get_kernel_size()

        preview_image = self.original_image.copy()
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        preview_image = cv2.erode(preview_image, kernel)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the min filter and close the window"""
        kernel_size = self.get_kernel_size()

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result_image = cv2.erode(self.original_image, kernel)

        # Apply the changes
        self.image_processor.apply_operation(result_image, f"Min Filter (k={kernel_size})")

        # Close the window
        super().confirm()


class AveragingFilterWindow(FilterWindow):
    """Window for averaging filter"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Averaging Filter Control")

    def update_preview_filter(self, value):
        """Update the preview with the current averaging filter setting"""
        kernel_size = self.get_kernel_size()

        preview_image = self.original_image.copy()
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        preview_image = cv2.filter2D(preview_image, -1, kernel)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the averaging filter and close the window"""
        kernel_size = self.get_kernel_size()

        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        result_image = cv2.filter2D(self.original_image, -1, kernel)

        # Apply the changes
        self.image_processor.apply_operation(result_image, f"Averaging Filter (k={kernel_size})")

        # Close the window
        super().confirm()


class WeightedAveragingFilterWindow(FilterWindow):
    """Window for weighted averaging filter"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Weighted Averaging Filter Control")

    def update_preview_filter(self, value):
        """Update the preview with the current weighted averaging filter setting"""
        kernel_size = self.get_kernel_size()

        preview_image = self.original_image.copy()
        preview_image = cv2.GaussianBlur(preview_image, (kernel_size, kernel_size), 0)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the weighted averaging filter and close the window"""
        kernel_size = self.get_kernel_size()

        result_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)

        # Apply the changes
        self.image_processor.apply_operation(result_image, f"Weighted Averaging Filter (k={kernel_size})")

        # Close the window
        super().confirm()


class GaussianFilterWindow(FilterWindow):
    """Window for Gaussian filter"""

    def __init__(self, root, image_processor):
        # Initialize sigma value before calling parent's __init__
        self.sigma_value = 1.0

        # Call parent's __init__ which will call update_preview_filter
        super().__init__(root, image_processor, title="Gaussian Filter Control")

        # Add sigma control
        self.sigma_frame = ctk.CTkFrame(self.window)
        self.sigma_frame.pack(pady=10)

        self.sigma_label = ctk.CTkLabel(self.sigma_frame, text="Sigma:")
        self.sigma_label.pack(side="left", padx=5)

        self.sigma_slider = ctk.CTkSlider(self.sigma_frame, from_=0.1, to=5.0,
                                        number_of_steps=49, orientation="horizontal")
        self.sigma_slider.set(self.sigma_value)  # Default sigma
        self.sigma_slider.pack(side="left", padx=5)

        self.sigma_value_label = ctk.CTkLabel(self.sigma_frame, text=f"{self.sigma_value:.1f}")
        self.sigma_value_label.pack(side="left", padx=5)

        # Connect the sigma slider to the update function
        self.sigma_slider.configure(command=self.update_sigma)

        # Update preview again now that sigma slider is available
        self.update_preview_filter(self.kernel_slider.get())

    def update_sigma(self, value):
        """Update the sigma value and refresh the preview"""
        self.sigma_value = float(value)
        self.sigma_value_label.configure(text=f"{self.sigma_value:.1f}")
        self.update_preview_filter(self.kernel_slider.get())

    def update_preview_filter(self, value):
        """Update the preview with the current Gaussian filter setting"""
        kernel_size = self.get_kernel_size()

        # Use self.sigma_value if sigma_slider is not yet created
        if hasattr(self, 'sigma_slider'):
            sigma = float(self.sigma_slider.get())
        else:
            sigma = self.sigma_value

        preview_image = self.original_image.copy()
        preview_image = cv2.GaussianBlur(preview_image, (kernel_size, kernel_size), sigma)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the Gaussian filter and close the window"""
        kernel_size = self.get_kernel_size()
        sigma = float(self.sigma_slider.get())

        result_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), sigma)

        # Apply the changes
        self.image_processor.apply_operation(result_image, f"Gaussian Filter (k={kernel_size}, σ={sigma:.1f})")

        # Close the window
        super().confirm()



