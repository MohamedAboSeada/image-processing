import customtkinter as ctk
import cv2
import os
import sys
from PIL import Image
from tkinter import filedialog, messagebox
from utils import resource_path

from ui import ImageEditorUI
from imageProcessing import ImageProcessor
from windows.rotate_window import RotateWindow
from windows.brightness_window import BrightnessWindow
from windows.color_conversion_window import ColorConversionWindow
from windows.threshold_window import ThresholdWindow
from windows.flip_window import FlipWindow
from windows.filter_windows import (
    MedianFilterWindow, MaxFilterWindow, MinFilterWindow,
    AveragingFilterWindow, WeightedAveragingFilterWindow, GaussianFilterWindow
)
from windows.double_exposure_window import DoubleExposureWindow
from windows.compare_window import CompareWindow
from windows.resize_window import ResizeWindow
from windows.crop_window import CropWindow
from windows.laplacian_window import LaplacianWindow
from windows.log_window import LogWindow
from windows.custom_laplacian_window import CustomLaplacianWindow

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Editor")
        self.root.iconbitmap(resource_path("assets/logo.ico"))

        # Initialize image processor
        self.image_processor = ImageProcessor()

        # Initialize UI
        self.ui = ImageEditorUI(root, self)

        # Image state variables
        self.cv_image = None
        self.original_image = None
        self.original_extension = ".png"  # Default extension

        # History variables
        self.undo_stack = []
        self.redo_stack = []
        self.history_list = []
        self.redo_history_list = []

        # Bind keyboard shortcuts
        self.root.bind('<Control-z>', lambda event: self.undo())
        self.root.bind('<Control-y>', lambda event: self.redo())

    def open_image(self):
        """Open an image file"""
        path = filedialog.askopenfilename()
        if path:
            self.cv_image = cv2.imread(path)
            if self.cv_image is None:
                messagebox.showerror("Error", f"Failed to open image: {path}")
                return

            self.original_image = self.cv_image.copy()
            self.undo_stack = []
            self.redo_stack.clear()
            self.history_list.clear()
            self.redo_history_list.clear()

            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)

            image_name = os.path.basename(path)
            self.ui.update_window_title(image_name)

            # Store the original file extension
            _, self.original_extension = os.path.splitext(path)
            if not self.original_extension:
                self.original_extension = ".png"  # Default to PNG if no extension

    def save_image(self):
        """Save the current image"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Use the original file extension as the default
        filetypes = [
            ("Original Format", f"*{self.original_extension}"),
            ("PNG Files", "*.png"),
            ("JPEG Files", "*.jpg"),
            ("All Files", "*.*")
        ]

        path = filedialog.asksaveasfilename(
            defaultextension=self.original_extension,
            filetypes=filetypes
        )

        if path:
            try:
                # Convert path to absolute path and normalize it
                abs_path = os.path.abspath(path)

                # Make sure the directory exists
                directory = os.path.dirname(abs_path)
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)

                # Get the file extension from the path
                _, file_extension = os.path.splitext(abs_path)
                if not file_extension:
                    # If no extension provided, use the original extension
                    abs_path += self.original_extension
                    file_extension = self.original_extension

                # For PNG format
                if file_extension.lower() == '.png':
                    # Convert image to RGB if it's not already
                    if len(self.cv_image.shape) == 3 and self.cv_image.shape[2] == 3:
                        # Save using PIL/Pillow which handles paths with special characters better
                        img_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        pil_img.save(abs_path)
                        messagebox.showinfo("Success", f"Image saved successfully to:\n{abs_path}")
                    else:
                        # For grayscale or other formats
                        result = cv2.imwrite(abs_path, self.cv_image)
                        if result:
                            messagebox.showinfo("Success", f"Image saved successfully to:\n{abs_path}")
                        else:
                            messagebox.showerror("Error", f"Failed to save image to:\n{abs_path}")
                # For JPEG format
                elif file_extension.lower() in ('.jpg', '.jpeg'):
                    # Save using PIL/Pillow
                    img_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    pil_img.save(abs_path, quality=95)  # High quality JPEG
                    messagebox.showinfo("Success", f"Image saved successfully to:\n{abs_path}")
                else:
                    # For other formats, use PIL if it's a color image
                    if len(self.cv_image.shape) == 3 and self.cv_image.shape[2] == 3:
                        img_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        try:
                            pil_img.save(abs_path)
                            messagebox.showinfo("Success", f"Image saved successfully to:\n{abs_path}")
                        except Exception:
                            # Fallback to OpenCV if PIL fails
                            result = cv2.imwrite(abs_path, self.cv_image)
                            if result:
                                messagebox.showinfo("Success", f"Image saved successfully to:\n{abs_path}")
                            else:
                                messagebox.showerror("Error", f"Failed to save image to:\n{abs_path}")
                    else:
                        # Fallback to OpenCV for grayscale or other formats
                        result = cv2.imwrite(abs_path, self.cv_image)
                        if result:
                            messagebox.showinfo("Success", f"Image saved successfully to:\n{abs_path}")
                        else:
                            messagebox.showerror("Error", f"Failed to save image to:\n{abs_path}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while saving the image:\n{str(e)}")

    def undo(self):
        """Undo the last operation"""
        if self.undo_stack:
            self.redo_stack.append(self.cv_image.copy())  # Save current state to redo stack
            operation = self.history_list.pop()  # Get the operation name before removing it
            self.redo_history_list.append(operation)  # Save the operation name for redo
            self.cv_image = self.undo_stack.pop()
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)

    def redo(self):
        """Redo the last undone operation"""
        if self.redo_stack:
            self.undo_stack.append(self.cv_image.copy())  # Save current state to undo stack
            self.cv_image = self.redo_stack.pop()
            if self.redo_history_list:
                operation = self.redo_history_list.pop()  # Get the original operation name
                self.history_list.append(operation)  # Add the original operation name to history
            else:
                self.history_list.append("Redo")  # Fallback if no operation name is available
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)

    def reset_image(self):
        """Reset the image to its original state"""
        if self.original_image is not None:
            self.cv_image = self.original_image.copy()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.redo_history_list.clear()
            self.history_list.clear()
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)

    def close_image(self):
        """Close the current image"""
        if self.cv_image is not None:
            confirm = messagebox.askyesno("Confirm Close", "Are you sure you want to close the current image?")
            if confirm:
                self.cv_image = None
                self.original_image = None
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.redo_history_list.clear()
                self.history_list.clear()
                self.ui.refresh_history(self.history_list)
                self.ui.reset_interface()

    def apply_operation(self, result_image, operation_name):
        """
        Apply an image processing operation and update the UI

        Args:
            result_image: The resulting image after the operation
            operation_name: The name of the operation for the history
        """
        if self.cv_image is not None:
            self.push_undo(operation_name)
            self.cv_image = result_image
            self.ui.update_display(self.cv_image, self.original_image)

    def push_undo(self, operation=""):
        """
        Push the current image state to the undo stack

        Args:
            operation: The name of the operation
        """
        if self.cv_image is not None:
            self.undo_stack.append(self.cv_image.copy())
            self.redo_stack.clear()
            self.redo_history_list.clear()
            self.history_list.append(operation)
            self.ui.refresh_history(self.history_list)

    # Window opening methods
    def open_rotate_window(self):
        """Open the rotate window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        RotateWindow(self.root, self)

    def open_brightness_window(self):
        """Open the brightness window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        BrightnessWindow(self.root, self)

    def open_color_conversion_window(self):
        """Open the color conversion window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        ColorConversionWindow(self.root, self)

    def open_threshold_window(self):
        """Open the threshold window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        ThresholdWindow(self.root, self)

    def open_flip_window(self):
        """Open the flip window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        FlipWindow(self.root, self)

    def open_median_filter_window(self):
        """Open the median filter window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        MedianFilterWindow(self.root, self)

    def open_max_filter_window(self):
        """Open the max filter window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        MaxFilterWindow(self.root, self)

    def open_min_filter_window(self):
        """Open the min filter window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        MinFilterWindow(self.root, self)

    def open_double_exposure_window(self):
        """Open the double exposure window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        DoubleExposureWindow(self.root, self)

    def open_compare_window(self):
        """Open the compare window"""
        if self.cv_image is None or self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        CompareWindow(self.root, self)

    # Simple operations (no window needed)
    def resize_image(self):
        """Open the resize window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        ResizeWindow(self.root, self)

    def crop_image(self):
        """Open the crop window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        CropWindow(self.root, self)

    def normalize_image(self):
        """Normalize the current image"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        result_image = self.image_processor.normalize(self.cv_image)
        self.apply_operation(result_image, "Normalize")

    def equalize_histogram(self):
        """Equalize the histogram of the current image"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        result_image = self.image_processor.equalize_histogram(self.cv_image)
        self.apply_operation(result_image, "Equalize Histogram")

    def add_salt_pepper_noise(self):
        """Add salt and pepper noise to the current image"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        result_image = self.image_processor.add_salt_pepper_noise(self.cv_image)
        self.apply_operation(result_image, "Add Salt & Pepper Noise")

    def apply_speckle_noise(self):
        """Add speckle noise to the current image"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        result_image = self.image_processor.apply_speckle_noise(self.cv_image)
        self.apply_operation(result_image, "Add Speckle Noise")

    def apply_poisson_noise(self):
        """Add Poisson noise to the current image"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        result_image = self.image_processor.apply_poisson_noise(self.cv_image)
        self.apply_operation(result_image, "Add Poisson Noise")

    def apply_uniform_noise(self):
        """Add uniform noise to the current image"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        result_image = self.image_processor.apply_uniform_noise(self.cv_image)
        self.apply_operation(result_image, "Add Uniform Noise")

    def apply_averaging_filter(self):
        """Open the averaging filter window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        AveragingFilterWindow(self.root, self)

    def apply_weighted_averaging_filter(self):
        """Open the weighted averaging filter window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        WeightedAveragingFilterWindow(self.root, self)

    def apply_gaussian_filter(self):
        """Open the Gaussian filter window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        GaussianFilterWindow(self.root, self)

    def open_laplacian_window(self):
        """Open the Laplacian window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        LaplacianWindow(self.root, self)

    def open_log_window(self):
        """Open the LoG (Laplacian of Gaussian) window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        LogWindow(self.root, self)

    def open_custom_laplacian_window(self):
        """Open the custom Laplacian window"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        CustomLaplacianWindow(self.root, self)

    def split_channels(self):
        """Split the current image into its color channels"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        self.image_processor.split_channels(self.cv_image)

    def show_histogram(self):
        """Show the histogram of the current image"""
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        self.image_processor.show_histogram(self.cv_image)

if __name__ == '__main__':
    root = ctk.CTk()
    app = ImageEditorApp(root)
    root.mainloop()
