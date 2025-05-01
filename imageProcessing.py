import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, simpledialog, messagebox
import customtkinter as ctk
import imutils
import os
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, ui):
        self.ui = ui
        self.cv_image = None
        self.original_image = None
        self.undo_stack = []
        self.redo_stack = []  # Add redo stack
        self.history_list = []
        self.original_visible = True

    def open_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.cv_image = cv2.imread(path)
            self.original_image = self.cv_image.copy()
            self.undo_stack = []
            self.redo_stack.clear()  # Clear redo stack on new image load
            self.history_list.clear()
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)
            image_name = os.path.basename(path)
            self.ui.update_window_title(image_name)

    def save_image(self):
        if self.cv_image is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png",
                                              filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])
            if path:
                cv2.imwrite(path, self.cv_image)

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.cv_image.copy())  # Save current state to redo stack
            self.cv_image = self.undo_stack.pop()
            self.history_list.pop()
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.cv_image.copy())  # Save current state to undo stack
            self.cv_image = self.redo_stack.pop()
            self.history_list.append("Redo")  # Add a placeholder for redo in history
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)

    def reset_image(self):
        if self.original_image is not None:
            self.cv_image = self.original_image.copy()
            self.undo_stack.clear()
            self.redo_stack.clear()  # Clear redo stack on reset
            self.history_list.clear()
            self.ui.refresh_history(self.history_list)
            self.ui.update_display(self.cv_image, self.original_image)

    def delete_image(self):
        if self.cv_image is not None:
            confirm = messagebox.askyesno("Confirm Delete", "Are you sure you want to delete the current image?")
            if confirm:
                self.cv_image = None
                self.original_image = None
                self.undo_stack.clear()
                self.redo_stack.clear()  # Clear redo stack on delete
                self.history_list.clear()
                self.ui.refresh_history(self.history_list)
                self.ui.image_label.configure(image="", text="")
                self.ui.image_label.image = None
                self.ui.image_label_original.configure(image="", text="")
                self.ui.image_label_original.image = None

    def toggle_original_visibility(self):
        if self.original_visible:
            self.ui.image_label_original.pack_forget()
            self.original_visible = False
            self.ui.update_toggle_button(False)
        else:
            self.ui.image_label_original.pack(side="left", expand=True, fill="both", padx=5, pady=5)
            self.original_visible = True
            self.ui.update_toggle_button(True)

    def push_undo(self, operation=""):
        if self.cv_image is not None:
            self.undo_stack.append(self.cv_image.copy())
            self.redo_stack.clear()  # Clear redo stack on new operation
            self.history_list.append(operation)
            self.ui.refresh_history(self.history_list)

    def resize_image(self):
        values = simpledialog.askstring("Resize", "Enter width, height (comma separated):")
        w, h = map(int, values.split(","))
        if w and h:
            self.push_undo("Resize")
            self.cv_image = cv2.resize(self.cv_image, (w, h))
            self.ui.update_display(self.cv_image, self.original_image)

    def crop_image(self):
        values = simpledialog.askstring("Crop", "Enter x, y, width, height (comma separated):")
        if values:
            try:
                x, y, w, h = map(int, values.split(","))
                self.push_undo('Crop')
                self.cv_image = self.cv_image[y:y+h, x:x+w]
                self.ui.update_display(self.cv_image, self.original_image)  # Fixed method call
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter four valid integers separated by commas.")

    def rotate_image_with_crop(self):
        angle = simpledialog.askfloat("Rotate with Crop", "Enter angle")
        if angle is not None:
            self.push_undo("Rotate with Crop")
            self.cv_image = imutils.rotate_bound(self.cv_image, angle)
            self.ui.update_display(self.cv_image, self.original_image)

    def rotate_image_without_crop(self):
        angle = simpledialog.askfloat("Rotate without Crop", "Enter angle")
        if angle is not None:
            self.push_undo("Rotate without Crop")
            self.cv_image = imutils.rotate(self.cv_image, angle)
            self.ui.update_display(self.cv_image, self.original_image)

    def adjust_brightness_window(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for brightness control
        brightness_window = ctk.CTkToplevel(self.ui.root)
        brightness_window.title("Brightness Control")
        brightness_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(brightness_window, text="")  # Remove default "ctk label" text
        canvas.pack(expand=True, fill="both")

        # Create a slider for brightness adjustment
        slider = ctk.CTkSlider(brightness_window, from_=0.1, to=3.0, number_of_steps=290, orientation="horizontal")
        slider.set(1.0)  # Default brightness factor
        slider.pack(pady=10)

        # Function to update the preview
        def update_preview(value):
            factor = float(value)
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            preview_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        slider.configure(command=update_preview)
        update_preview(slider.get())

        # Confirm and Cancel buttons
        def confirm():
            factor = float(slider.get())
            self.push_undo("Adjust Brightness")
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            self.cv_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.ui.update_display(self.cv_image, self.original_image)
            brightness_window.destroy()

        def cancel():
            brightness_window.destroy()

        button_frame = ctk.CTkFrame(brightness_window)
        button_frame.pack(pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    def convert_Rgb(self):
        self.push_undo("Convert to RGB")
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.ui.update_display(self.cv_image, self.original_image)

    def convert_Hsv(self):
        self.push_undo("Convert to HSV")
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        self.ui.update_display(self.cv_image, self.original_image)

    def convert_Gray(self):
        self.push_undo("Convert to Gray")
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        self.cv_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.ui.update_display(self.cv_image, self.original_image)

    def split_channels(self):
        b, g, r = cv2.split(self.cv_image)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(b, cmap='gray')
        plt.title("Blue Channel")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(g, cmap='gray')
        plt.title("Green Channel")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(r, cmap='gray')
        plt.title("Red Channel")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def normalize_image(self):
        self.push_undo("Normalize")
        self.cv_image = cv2.normalize(self.cv_image, None, 0, 255, cv2.NORM_MINMAX)
        self.ui.update_display(self.cv_image, self.original_image)

    def show_histogram(self):
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([self.cv_image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title("Histogram")
        plt.show()

    def equalize_histogram(self):
        self.push_undo("Equalize Histogram")
        yuv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        self.cv_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        self.ui.update_display(self.cv_image, self.original_image)

    def image_addition(self):
        path = filedialog.askopenfilename()
        if path:
            other = cv2.imread(path)
            other = cv2.resize(other, (self.cv_image.shape[1], self.cv_image.shape[0]))
            self.push_undo("Image Addition")
            self.cv_image = cv2.add(self.cv_image, other)
            self.ui.update_display(self.cv_image, self.original_image)

    def weighted_addition(self):
        path = filedialog.askopenfilename()
        if path:
            alpha = simpledialog.askfloat("Alpha", "Enter alpha")
            beta = 1 - alpha
            other = cv2.imread(path)
            other = cv2.resize(other, (self.cv_image.shape[1], self.cv_image.shape[0]))
            self.push_undo("Weighted Addition")
            self.cv_image = cv2.addWeighted(self.cv_image, alpha, other, beta, 0)
            self.ui.update_display(self.cv_image, self.original_image)

    def image_subtraction(self):
        path = filedialog.askopenfilename()
        if path:
            other = cv2.imread(path)
            other = cv2.resize(other, (self.cv_image.shape[1], self.cv_image.shape[0]))
            self.push_undo("Image Subtraction")
            self.cv_image = cv2.subtract(self.cv_image, other)
            self.ui.update_display(self.cv_image, self.original_image)

    def add_salt_pepper_noise(self, amount=0.02, salt_vs_pepper=0.5):
        self.push_undo("Add Salt & Pepper Noise")
        noisy = np.copy(self.cv_image)
        row, col, ch = noisy.shape
        num_salt = np.ceil(amount * noisy.size * salt_vs_pepper).astype(int)
        num_pepper = np.ceil(amount * noisy.size * (1.0 - salt_vs_pepper)).astype(int)

        # Salt (white) noise
        coords = [np.random.randint(0, i - 1, num_salt) for i in noisy.shape[:2]]
        noisy[coords[0], coords[1]] = [255, 255, 255]

        # Pepper (black) noise
        coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy.shape[:2]]
        noisy[coords[0], coords[1]] = [0, 0, 0]

        self.cv_image = noisy
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_speckle_noise(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        row, col, ch = self.cv_image.shape
        gauss = np.random.normal(0, 1, (row, col, ch)).reshape(row, col, ch)
        noisy = np.clip(self.cv_image + self.cv_image * gauss, 0, 255).astype(np.uint8)
        self.push_undo("Applied Speckle Noise")
        self.cv_image = noisy
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_poisson_noise(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        noisy = np.random.poisson(self.cv_image / 255.0 * 255).astype(np.uint8)
        self.push_undo("Applied Poisson Noise")
        self.cv_image = np.clip(noisy, 0, 255)
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_uniform_noise(self, low=0, high=50):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        row, col, ch = self.cv_image.shape
        noise = np.random.uniform(low, high, (row, col, ch)).reshape(row, col, ch)
        noisy = np.clip(self.cv_image + noise, 0, 255).astype(np.uint8)
        self.push_undo("Applied Uniform Noise")
        self.cv_image = noisy
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_averaging_filter(self):
        self.push_undo("Averaging Filter")
        kernel = np.ones((5, 5), np.float32) / 25
        self.cv_image = cv2.filter2D(self.cv_image, -1, kernel)
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_weighted_averaging_filter(self):
        self.push_undo("Weighted Averaging Filter")
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32)
        kernel /= np.sum(kernel)
        self.cv_image = cv2.filter2D(self.cv_image, -1, kernel)
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_gaussian_filter(self):
        self.push_undo("Gaussian Filter")
        self.cv_image = cv2.GaussianBlur(self.cv_image, (5, 5), 0)
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_median_filter(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for median filter control
        median_window = ctk.CTkToplevel(self.ui.root)
        median_window.title("Median Filter Control")
        median_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(median_window, text="")
        canvas.pack(expand=True, fill="both")

        # Create a slider for kernel size adjustment
        kernel_frame = ctk.CTkFrame(median_window)
        kernel_frame.pack(pady=10)

        kernel_label = ctk.CTkLabel(kernel_frame, text="Kernel Size:")
        kernel_label.pack(side="left", padx=5)

        # Kernel size must be odd
        kernel_sizes = [3, 5, 7, 9, 11, 13, 15]
        kernel_var = ctk.StringVar(value="5")

        kernel_dropdown = ctk.CTkOptionMenu(
            kernel_frame,
            values=[str(k) for k in kernel_sizes],
            variable=kernel_var
        )
        kernel_dropdown.pack(side="left", padx=5)

        # Function to update the preview
        def update_preview(*args):
            kernel_size = int(kernel_var.get())
            preview_image = self.cv_image.copy()
            preview_image = cv2.medianBlur(preview_image, kernel_size)
            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        kernel_var.trace_add("write", update_preview)
        update_preview()  # Initial preview

        # Confirm and Cancel buttons
        def confirm():
            kernel_size = int(kernel_var.get())
            self.push_undo("Median Filter")
            self.cv_image = cv2.medianBlur(self.cv_image, kernel_size)
            self.ui.update_display(self.cv_image, self.original_image)
            median_window.destroy()

        def cancel():
            median_window.destroy()

        button_frame = ctk.CTkFrame(median_window)
        button_frame.pack(pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    def apply_max_filter(self):
        self.push_undo("Max Filter")
        self.cv_image = cv2.dilate(self.cv_image, np.ones((5,5), np.uint8))
        self.ui.update_display(self.cv_image, self.original_image)

    def apply_min_filter(self):
        self.push_undo("Min Filter")
        self.cv_image = cv2.erode(self.cv_image, np.ones((5,5), np.uint8))
        self.ui.update_display(self.cv_image, self.original_image)

    def interactive_compare(self):
        if self.cv_image is None or self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        compare_window = ctk.CTkToplevel(self.ui.root)
        compare_window.title("Compare Original vs Edited")
        compare_window.geometry("800x600")

        canvas = ctk.CTkLabel(compare_window)
        canvas.pack(expand=True, fill="both")

        slider = ctk.CTkSlider(compare_window, from_=0, to=1, number_of_steps=100, orientation="horizontal")
        slider.set(0.5)
        slider.pack(pady=10)

        def update_image(value):
            alpha = float(value)
            beta = 1.0 - alpha

            img1 = cv2.resize(self.original_image.copy(), (600, 400))
            img2 = cv2.resize(self.cv_image.copy(), (600, 400))

            blended = cv2.addWeighted(img2, alpha, img1, beta, 0)
            blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            canvas.configure(image=blended)
            canvas.image = blended

        slider.configure(command=update_image)
        update_image(slider.get())

    def adjust_threshold(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for threshold control
        threshold_window = ctk.CTkToplevel(self.ui.root)
        threshold_window.title("Threshold Control")
        threshold_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(threshold_window, text="")  # Remove default "ctk label" text
        canvas.pack(expand=True, fill="both")

        # Create a slider for threshold adjustment
        slider = ctk.CTkSlider(threshold_window, from_=0, to=255, number_of_steps=255, orientation="horizontal")
        slider.set(127)  # Default threshold value
        slider.pack(pady=10)

        # Function to update the preview
        def update_preview(value):
            threshold_value = int(float(value))
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            preview_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        slider.configure(command=update_preview)
        update_preview(slider.get())

        # Confirm and Cancel buttons
        def confirm():
            threshold_value = int(slider.get())
            self.push_undo("Adjust Threshold")
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            self.cv_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            self.ui.update_display(self.cv_image, self.original_image)
            threshold_window.destroy()

        def cancel():
            threshold_window.destroy()

        button_frame = ctk.CTkFrame(threshold_window)
        button_frame.pack(pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    def adjust_flip(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for flip adjustment
        flip_window = ctk.CTkToplevel(self.ui.root)
        flip_window.title("Flip Adjustment")
        flip_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(flip_window, text="")  # Remove default "ctk label" text
        canvas.pack(expand=True, fill="both")

        # Create radio buttons for flip options
        flip_option = ctk.StringVar(value="horizontal")

        def update_preview(*args):
            option = flip_option.get()
            preview_image = self.cv_image.copy()
            if option == "horizontal":
                preview_image = cv2.flip(preview_image, 1)
            elif option == "vertical":
                preview_image = cv2.flip(preview_image, 0)
            elif option == "both":
                preview_image = cv2.flip(preview_image, -1)

            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        # Trace the flip_option variable to call update_preview on change
        flip_option.trace_add("write", update_preview)

        horizontal_button = ctk.CTkRadioButton(flip_window, text="Flip Horizontally", variable=flip_option, value="horizontal")
        horizontal_button.pack(anchor="w", padx=10, pady=5)

        vertical_button = ctk.CTkRadioButton(flip_window, text="Flip Vertically", variable=flip_option, value="vertical")
        vertical_button.pack(anchor="w", padx=10, pady=5)

        both_button = ctk.CTkRadioButton(flip_window, text="Flip Both", variable=flip_option, value="both")
        both_button.pack(anchor="w", padx=10, pady=5)

        # Update preview immediately for the default option
        update_preview()

        # Confirm and Cancel buttons
        def confirm():
            option = flip_option.get()
            self.push_undo("Flip")
            if option == "horizontal":
                self.cv_image = cv2.flip(self.cv_image, 1)
            elif option == "vertical":
                self.cv_image = cv2.flip(self.cv_image, 0)
            elif option == "both":
                self.cv_image = cv2.flip(self.cv_image, -1)
            self.ui.update_display(self.cv_image, self.original_image)
            flip_window.destroy()

        def cancel():
            flip_window.destroy()

        button_frame = ctk.CTkFrame(flip_window)
        button_frame.pack(pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    def double_exposure_window(self):
        if self.cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create a new window for double exposure
        double_exposure_window = ctk.CTkToplevel(self.ui.root)
        double_exposure_window.title("Double Exposure Effect")
        double_exposure_window.geometry("800x600")

        # Create a preview canvas
        canvas = ctk.CTkLabel(double_exposure_window, text="")
        canvas.pack(expand=True, fill="both")

        # Variable to store the second image
        self.second_image = None

        # Function to upload second image
        def upload_second_image():
            path = filedialog.askopenfilename()
            if path:
                second_img = cv2.imread(path)
                if second_img is not None:
                    # Resize second image to match dimensions of the first image
                    self.second_image = cv2.resize(second_img, (self.cv_image.shape[1], self.cv_image.shape[0]))
                    # Show mode selection after image is uploaded
                    modes_frame.pack(pady=10)
                    # Update preview with default mode
                    update_preview()
                else:
                    messagebox.showerror("Error", "Could not load the image.")

        # Upload button
        upload_button = ctk.CTkButton(double_exposure_window, text="Upload Second Image", command=upload_second_image)
        upload_button.pack(pady=20)

        # Mode selection frame (initially hidden)
        modes_frame = ctk.CTkFrame(double_exposure_window)

        # Mode variable
        mode_var = ctk.StringVar(value="add")

        # Alpha values frame for weighted mode (initially hidden)
        alpha_frame = ctk.CTkFrame(double_exposure_window)

        # Alpha entry fields
        alpha_label1 = ctk.CTkLabel(alpha_frame, text="Alpha for First Image:")
        alpha_label1.pack(side="left", padx=5)
        alpha_entry1 = ctk.CTkEntry(alpha_frame, width=60)
        alpha_entry1.insert(0, "0.5")
        alpha_entry1.pack(side="left", padx=5)

        alpha_label2 = ctk.CTkLabel(alpha_frame, text="Alpha for Second Image:")
        alpha_label2.pack(side="left", padx=5)
        alpha_entry2 = ctk.CTkEntry(alpha_frame, width=60)
        alpha_entry2.insert(0, "0.5")
        alpha_entry2.pack(side="left", padx=5)

        # Apply button for alpha values
        apply_alpha_button = ctk.CTkButton(alpha_frame, text="Apply", width=60, command=lambda: update_preview())
        apply_alpha_button.pack(side="left", padx=10)

        # Function to update preview based on selected mode
        def update_preview():
            if self.second_image is None:
                return

            mode = mode_var.get()
            if mode == "add":
                preview_image = cv2.add(self.cv_image, self.second_image)
            elif mode == "weighted":
                try:
                    alpha1 = float(alpha_entry1.get())
                    alpha2 = float(alpha_entry2.get())
                    preview_image = cv2.addWeighted(self.cv_image, alpha1, self.second_image, alpha2, 0)
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid alpha values.")
                    return
            elif mode == "subtract":
                preview_image = cv2.subtract(self.cv_image, self.second_image)

            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
            preview_image = cv2.resize(preview_image, (600, 400))
            preview_image = Image.fromarray(preview_image)
            preview_image = ImageTk.PhotoImage(preview_image)
            canvas.configure(image=preview_image)
            canvas.image = preview_image

        def on_mode_change():
            if mode_var.get() == "weighted":
                alpha_frame.pack(pady=10)
                double_exposure_window.geometry("900x650")  # Increase window size
            else:
                alpha_frame.pack_forget()
                double_exposure_window.geometry("800x600")  # Reset window size
            update_preview()

        modes = [
            ("Add Two Images", "add"),
            ("Add Weighted", "weighted"),
            ("Subtract", "subtract")
        ]

        for text, value in modes:
            radio = ctk.CTkRadioButton(modes_frame, text=text, variable=mode_var, value=value, command=on_mode_change)
            radio.pack(anchor="w", padx=10, pady=5)

        # Confirm and Cancel buttons
        def confirm():
            if self.second_image is None:
                messagebox.showwarning("No Second Image", "Please upload a second image first.")
                return

            mode = mode_var.get()
            self.push_undo("Double Exposure")

            if mode == "add":
                self.cv_image = cv2.add(self.cv_image, self.second_image)
            elif mode == "weighted":
                try:
                    alpha1 = float(alpha_entry1.get())
                    alpha2 = float(alpha_entry2.get())
                    self.cv_image = cv2.addWeighted(self.cv_image, alpha1, self.second_image, alpha2, 0)
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid alpha values.")
                    return
            elif mode == "subtract":
                self.cv_image = cv2.subtract(self.cv_image, self.second_image)

            self.ui.update_display(self.cv_image, self.original_image)
            double_exposure_window.destroy()

        def cancel():
            double_exposure_window.destroy()

        button_frame = ctk.CTkFrame(double_exposure_window)
        button_frame.pack(side="bottom", pady=10)

        confirm_button = ctk.CTkButton(button_frame, text="Confirm", command=confirm)
        confirm_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel)
        cancel_button.pack(side="right", padx=5)

    def close_image(self):
        if self.cv_image is not None:
            confirm = messagebox.askyesno("Close Image", "Are you sure you want to close the current image?")
            if confirm:
                self.cv_image = None
                self.original_image = None
                self.undo_stack.clear()
                self.redo_stack.clear()  # Clear redo stack on close
                self.history_list.clear()
                self.ui.reset_interface()