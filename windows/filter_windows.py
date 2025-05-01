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
        super().__init__(root, image_processor, title="Gaussian Filter Control")

        # Add sigma control
        self.sigma_frame = ctk.CTkFrame(self.window)
        self.sigma_frame.pack(pady=10)

        self.sigma_label = ctk.CTkLabel(self.sigma_frame, text="Sigma:")
        self.sigma_label.pack(side="left", padx=5)

        self.sigma_slider = ctk.CTkSlider(self.sigma_frame, from_=0.1, to=5.0,
                                        number_of_steps=49, orientation="horizontal")
        self.sigma_slider.set(1.0)  # Default sigma
        self.sigma_slider.pack(side="left", padx=5)

        self.sigma_value_label = ctk.CTkLabel(self.sigma_frame, text="1.0")
        self.sigma_value_label.pack(side="left", padx=5)

        # Connect the sigma slider to the update function
        self.sigma_slider.configure(command=self.update_sigma)

    def update_sigma(self, value):
        """Update the sigma value and refresh the preview"""
        sigma = float(value)
        self.sigma_value_label.configure(text=f"{sigma:.1f}")
        self.update_preview_filter(self.kernel_slider.get())

    def update_preview_filter(self, value):
        """Update the preview with the current Gaussian filter setting"""
        kernel_size = self.get_kernel_size()
        sigma = float(self.sigma_slider.get())

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


class NoiseWindow(BaseWindow):
    """Base class for noise windows"""

    def __init__(self, root, image_processor, title="Noise Control", geometry="800x600"):
        super().__init__(root, image_processor, title=title, geometry=geometry)

        # Create a preview canvas
        self.create_preview_canvas()

        # Create button frame
        self.create_button_frame()

        # Store original image
        self.original_image = self.image_processor.cv_image.copy()

        # Initial preview
        self.update_preview(self.original_image)

    def create_button_frame(self):
        """Create the button frame with confirm and cancel buttons"""
        self.button_frame = ctk.CTkFrame(self.window)
        self.button_frame.pack(side="bottom", pady=10)

        self.confirm_button = ctk.CTkButton(self.button_frame, text="Confirm", command=self.confirm)
        self.confirm_button.pack(side="left", padx=10)

        self.cancel_button = ctk.CTkButton(self.button_frame, text="Cancel", command=self.cancel)
        self.cancel_button.pack(side="left", padx=10)

    def update_preview(self, image):
        """Update the preview with the given image"""
        # Convert to RGB for display
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a PIL image
        pil_image = Image.fromarray(rgb_image)

        # Resize if needed to fit the window
        max_size = min(self.window.winfo_width() - 40, self.window.winfo_height() - 100)
        if max_size > 0:  # Ensure window has been drawn
            width, height = pil_image.size
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        # Convert to CTkImage
        ctk_image = ctk.CTkImage(pil_image, size=pil_image.size)

        # Update the canvas
        self.canvas.configure(image=ctk_image)
        self.canvas.image = ctk_image  # Keep a reference

    def confirm(self):
        """Apply the changes and close the window"""
        self.window.destroy()

    def cancel(self):
        """Close the window without applying changes"""
        self.window.destroy()


class SaltPepperNoiseWindow(NoiseWindow):
    """Window for salt and pepper noise"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Salt & Pepper Noise Control")

        # Create sliders for salt and pepper probabilities
        self.salt_frame = ctk.CTkFrame(self.window)
        self.salt_frame.pack(pady=10)

        self.salt_label = ctk.CTkLabel(self.salt_frame, text="Salt Probability:")
        self.salt_label.pack(side="left", padx=5)

        self.salt_slider = ctk.CTkSlider(self.salt_frame, from_=0.0, to=0.1,
                                       number_of_steps=100, orientation="horizontal")
        self.salt_slider.set(0.01)  # Default salt probability
        self.salt_slider.pack(side="left", padx=5)

        self.salt_value_label = ctk.CTkLabel(self.salt_frame, text="0.01")
        self.salt_value_label.pack(side="left", padx=5)

        self.pepper_frame = ctk.CTkFrame(self.window)
        self.pepper_frame.pack(pady=10)

        self.pepper_label = ctk.CTkLabel(self.pepper_frame, text="Pepper Probability:")
        self.pepper_label.pack(side="left", padx=5)

        self.pepper_slider = ctk.CTkSlider(self.pepper_frame, from_=0.0, to=0.1,
                                         number_of_steps=100, orientation="horizontal")
        self.pepper_slider.set(0.01)  # Default pepper probability
        self.pepper_slider.pack(side="left", padx=5)

        self.pepper_value_label = ctk.CTkLabel(self.pepper_frame, text="0.01")
        self.pepper_value_label.pack(side="left", padx=5)

        # Connect sliders to update function
        self.salt_slider.configure(command=self.update_preview_noise)
        self.pepper_slider.configure(command=self.update_preview_noise)

        # Initial preview
        self.update_preview_noise(0.01)

    def update_preview_noise(self, value):
        """Update the preview with the current noise settings"""
        salt_prob = float(self.salt_slider.get())
        pepper_prob = float(self.pepper_slider.get())

        self.salt_value_label.configure(text=f"{salt_prob:.3f}")
        self.pepper_value_label.configure(text=f"{pepper_prob:.3f}")

        preview_image = self.original_image.copy()
        preview_image = self.image_processor.image_processor.add_salt_pepper_noise(preview_image, salt_prob, pepper_prob)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the salt and pepper noise and close the window"""
        salt_prob = float(self.salt_slider.get())
        pepper_prob = float(self.pepper_slider.get())

        result_image = self.image_processor.image_processor.add_salt_pepper_noise(self.original_image, salt_prob, pepper_prob)

        # Apply the changes
        self.image_processor.apply_operation(result_image, f"Salt & Pepper Noise (s={salt_prob:.3f}, p={pepper_prob:.3f})")

        # Close the window
        super().confirm()


class SpeckleNoiseWindow(NoiseWindow):
    """Window for speckle noise"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Speckle Noise Control")

        # Create slider for intensity
        self.intensity_frame = ctk.CTkFrame(self.window)
        self.intensity_frame.pack(pady=10)

        self.intensity_label = ctk.CTkLabel(self.intensity_frame, text="Intensity:")
        self.intensity_label.pack(side="left", padx=5)

        self.intensity_slider = ctk.CTkSlider(self.intensity_frame, from_=0.01, to=0.5,
                                            number_of_steps=49, orientation="horizontal")
        self.intensity_slider.set(0.1)  # Default intensity
        self.intensity_slider.pack(side="left", padx=5)

        self.intensity_value_label = ctk.CTkLabel(self.intensity_frame, text="0.10")
        self.intensity_value_label.pack(side="left", padx=5)

        # Connect slider to update function
        self.intensity_slider.configure(command=self.update_preview_noise)

        # Initial preview
        self.update_preview_noise(0.1)

    def update_preview_noise(self, value):
        """Update the preview with the current noise settings"""
        intensity = float(value)
        self.intensity_value_label.configure(text=f"{intensity:.2f}")

        preview_image = self.original_image.copy()
        preview_image = self.image_processor.image_processor.apply_speckle_noise(preview_image, intensity)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the speckle noise and close the window"""
        intensity = float(self.intensity_slider.get())

        result_image = self.image_processor.image_processor.apply_speckle_noise(self.original_image, intensity)

        # Apply the changes
        self.image_processor.apply_operation(result_image, f"Speckle Noise (i={intensity:.2f})")

        # Close the window
        super().confirm()


class UniformNoiseWindow(NoiseWindow):
    """Window for uniform noise"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Uniform Noise Control")

        # Create sliders for low and high bounds
        self.low_frame = ctk.CTkFrame(self.window)
        self.low_frame.pack(pady=10)

        self.low_label = ctk.CTkLabel(self.low_frame, text="Low Bound:")
        self.low_label.pack(side="left", padx=5)

        self.low_slider = ctk.CTkSlider(self.low_frame, from_=-50, to=0,
                                      number_of_steps=50, orientation="horizontal")
        self.low_slider.set(-10)  # Default low bound
        self.low_slider.pack(side="left", padx=5)

        self.low_value_label = ctk.CTkLabel(self.low_frame, text="-10")
        self.low_value_label.pack(side="left", padx=5)

        self.high_frame = ctk.CTkFrame(self.window)
        self.high_frame.pack(pady=10)

        self.high_label = ctk.CTkLabel(self.high_frame, text="High Bound:")
        self.high_label.pack(side="left", padx=5)

        self.high_slider = ctk.CTkSlider(self.high_frame, from_=0, to=50,
                                       number_of_steps=50, orientation="horizontal")
        self.high_slider.set(10)  # Default high bound
        self.high_slider.pack(side="left", padx=5)

        self.high_value_label = ctk.CTkLabel(self.high_frame, text="10")
        self.high_value_label.pack(side="left", padx=5)

        # Connect sliders to update function
        self.low_slider.configure(command=self.update_preview_noise)
        self.high_slider.configure(command=self.update_preview_noise)

        # Initial preview
        self.update_preview_noise(0)

    def update_preview_noise(self, value):
        """Update the preview with the current noise settings"""
        low = int(self.low_slider.get())
        high = int(self.high_slider.get())

        self.low_value_label.configure(text=str(low))
        self.high_value_label.configure(text=str(high))

        preview_image = self.original_image.copy()
        preview_image = self.image_processor.image_processor.apply_uniform_noise(preview_image, low, high)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the uniform noise and close the window"""
        low = int(self.low_slider.get())
        high = int(self.high_slider.get())

        result_image = self.image_processor.image_processor.apply_uniform_noise(self.original_image, low, high)

        # Apply the changes
        self.image_processor.apply_operation(result_image, f"Uniform Noise (low={low}, high={high})")

        # Close the window
        super().confirm()


class PoissonNoiseWindow(NoiseWindow):
    """Window for Poisson noise"""

    def __init__(self, root, image_processor):
        super().__init__(root, image_processor, title="Poisson Noise Control")

        # Create a label explaining Poisson noise
        self.info_label = ctk.CTkLabel(self.window,
                                     text="Poisson noise is based on the image intensity values.\nNo parameters to adjust.")
        self.info_label.pack(pady=20)

        # Generate preview with Poisson noise
        preview_image = self.original_image.copy()
        preview_image = self.image_processor.image_processor.apply_poisson_noise(preview_image)

        self.update_preview(preview_image)

    def confirm(self):
        """Apply the Poisson noise and close the window"""
        result_image = self.image_processor.image_processor.apply_poisson_noise(self.original_image)

        # Apply the changes
        self.image_processor.apply_operation(result_image, "Poisson Noise")

        # Close the window
        super().confirm()
